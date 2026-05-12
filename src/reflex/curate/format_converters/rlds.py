"""RLDS / TFRecord format converter (Phase 1.5 — full impl).

Spec: features/08_curate/_curation/format-converters/rlds.md
RLDS schema: https://github.com/google-research/rlds

RLDS is Google's reinforcement-learning dataset format built on TFRecord.
Each Episode is serialized as one tf.train.SequenceExample whose
feature_lists carry per-step observation + action + standard RL metadata
(is_first / is_last / is_terminal / discount / reward / language_instruction).

This Phase 1.5 implementation produces output that:
  - Can be read by `tfds.builder_from_directory(<output>)` + iterated as a
    standard `tf.data.Dataset`
  - Has a `dataset_info.json` + `features.json` declaring the per-step
    feature schema for downstream consumers
  - Sharded into N TFRecord files for parallel read scaling

Phase 2 deferrals (per spec):
  - Image observations (skipped — same image-data caveat as LeRobot v3)
  - Streaming write (whole-dataset materialization for now)
  - tfds.dataset_builder integration (would let consumers use
    `tfds.load("reflex_curate_dataset")`; we ship the directory format
    that `tfds.builder_from_directory` accepts which is the practical
    equivalent for self-hosted datasets)

Deps gated behind [curate-rlds] extra (tensorflow + tensorflow_datasets).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflex.curate.format_converters.base import (
    ConversionResult,
    FormatConverter,
    _group_by_episode,
    _iter_jsonl,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)

RLDS_VERSION = "1.0.0"
DEFAULT_SHARD_SIZE = 100  # episodes per TFRecord shard


class RLDSConverter(FormatConverter):
    """Convert Reflex JSONL traces → RLDS / TFRecord format."""

    FORMAT_NAME = "rlds"

    def __init__(
        self,
        *,
        dataset_name: str = "reflex_curate_dataset",
        version: str = RLDS_VERSION,
        license: str = "CC-BY-4.0",
        shard_size: int = DEFAULT_SHARD_SIZE,
        embodiment: str = "unknown",
    ):
        self.dataset_name = dataset_name
        self.version = version
        self.license = license
        self.shard_size = max(1, int(shard_size))
        self.embodiment = embodiment

    def convert(
        self,
        *,
        input_jsonl: str | Path | list[str | Path],
        output_dir: str | Path,
        min_quality: float | None = None,
        canonical_only: bool = False,
    ) -> ConversionResult:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "RLDS conversion requires tensorflow + tensorflow_datasets: "
                "pip install 'reflex-vla[curate-rlds]'"
            ) from exc

        result = ConversionResult(
            output_dir=str(output_dir),
            format=self.FORMAT_NAME,
            started_at=_utc_now_iso(),
        )
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)

        # Group rows by episode across all input JSONLs.
        input_paths = self._resolve_inputs(input_jsonl)
        all_episodes: dict[str, list[dict[str, Any]]] = {}
        for p in input_paths:
            for ep_id, rows in _group_by_episode(_iter_jsonl(p)).items():
                all_episodes.setdefault(ep_id, []).extend(rows)

        # Filter + collect kept episodes.
        kept: list[tuple[str, list[dict[str, Any]]]] = []
        for ep_id, rows in sorted(all_episodes.items()):
            keep, reason = self._filter_episode(
                rows, min_quality=min_quality, canonical_only=canonical_only,
            )
            if not keep:
                result.skipped_episodes += 1
                result.skipped_reasons[reason] += 1
                continue
            kept.append((ep_id, rows))

        if not kept:
            result.warnings.append("no_episodes_passed_filter")
            result.completed_at = _utc_now_iso()
            return result

        # Probe action_dim + state_dim from the first kept episode.
        action_dim = max(
            len(a) for _, rows in kept[:1] for a in
            [chunk for r in rows for chunk in (r.get("action_chunk") or []) if isinstance(chunk, list)]
        )
        state_dim_candidates = [
            len(r.get("state_vec") or [])
            for _, rows in kept[:1] for r in rows
            if isinstance(r.get("state_vec"), list)
        ]
        state_dim = max(state_dim_candidates) if state_dim_candidates else 0

        # Write TFRecord shards.
        shard_paths: list[str] = []
        shard_lengths: list[int] = []
        current_shard_episodes: list[bytes] = []
        shard_idx = 0
        for ep_index, (ep_id, rows) in enumerate(kept):
            example_bytes = self._build_episode_example(
                tf=tf,
                rows=rows,
                ep_id=ep_id,
                action_dim=action_dim,
                state_dim=state_dim,
            )
            current_shard_episodes.append(example_bytes)
            if len(current_shard_episodes) >= self.shard_size:
                path = self._write_shard(
                    tf, output, shard_idx, current_shard_episodes,
                )
                shard_paths.append(path.name)
                shard_lengths.append(len(current_shard_episodes))
                result.bytes_written += path.stat().st_size
                current_shard_episodes = []
                shard_idx += 1

        # Final partial shard.
        if current_shard_episodes:
            path = self._write_shard(
                tf, output, shard_idx, current_shard_episodes,
            )
            shard_paths.append(path.name)
            shard_lengths.append(len(current_shard_episodes))
            result.bytes_written += path.stat().st_size

        result.episode_count = len(kept)
        result.step_count = sum(
            sum(len(r.get("action_chunk") or []) for r in rows)
            for _, rows in kept
        )

        # Write dataset_info.json + features.json.
        self._write_metadata(
            output=output,
            episode_count=result.episode_count,
            shard_lengths=shard_lengths,
            shard_paths=shard_paths,
            action_dim=action_dim,
            state_dim=state_dim,
        )

        result.completed_at = _utc_now_iso()
        return result

    def _build_episode_example(
        self,
        *,
        tf: Any,
        rows: list[dict[str, Any]],
        ep_id: str,
        action_dim: int,
        state_dim: int,
    ) -> bytes:
        """Build one tf.train.SequenceExample = one RLDS Episode.

        Structure follows the RLDS canonical format: per-step features go
        into feature_lists; per-episode metadata goes into context features.
        """
        actions, states = self._flatten_actions_and_steps(rows)
        if not actions:
            # Caller filtered already, but defensive.
            return b""

        n = len(actions)
        instruction = rows[0].get("instruction_raw") or rows[0].get("instruction_hash") or ""

        # Pad actions / states to consistent dim across the episode.
        actions_padded = [
            list(a) + [0.0] * (action_dim - len(a)) for a in actions
        ]
        if state_dim > 0:
            states_padded = [
                (list(s) if s is not None else [0.0] * state_dim) + [0.0] * max(0, state_dim - len(s or []))
                for s in states
            ]
        else:
            states_padded = [[]] * n

        is_first = [True] + [False] * (n - 1)
        is_last = [False] * (n - 1) + [True]
        is_terminal = [False] * (n - 1) + [True]
        rewards = [0.0] * (n - 1) + [1.0]  # Sparse terminal reward
        discounts = [1.0] * n

        feature_lists: dict[str, Any] = {
            "action": tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=a))
                for a in actions_padded
            ]),
            "reward": tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=[r]))
                for r in rewards
            ]),
            "discount": tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=[d]))
                for d in discounts
            ]),
            "is_first": tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(b)]))
                for b in is_first
            ]),
            "is_last": tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(b)]))
                for b in is_last
            ]),
            "is_terminal": tf.train.FeatureList(feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(b)]))
                for b in is_terminal
            ]),
            "language_instruction": tf.train.FeatureList(feature=[
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[instruction.encode("utf-8")]))
            ] * n),
        }
        if state_dim > 0:
            feature_lists["observation/state"] = tf.train.FeatureList(feature=[
                tf.train.Feature(float_list=tf.train.FloatList(value=s))
                for s in states_padded
            ])

        # Episode-level context features.
        md0 = rows[0].get("metadata", {}) or {}
        context_features: dict[str, Any] = {
            "episode_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[ep_id.encode("utf-8")]),
            ),
        }
        if md0.get("quality_score") is not None:
            context_features["quality_score"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(md0["quality_score"])]),
            )

        seq_example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features),
            feature_lists=tf.train.FeatureLists(feature_list=feature_lists),
        )
        return seq_example.SerializeToString()

    def _write_shard(
        self,
        tf: Any,
        output: Path,
        shard_idx: int,
        episode_bytes_list: list[bytes],
    ) -> Path:
        """Write a TFRecord file containing N serialized episodes."""
        path = output / f"{self.dataset_name}-train.tfrecord-{shard_idx:05d}"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for example_bytes in episode_bytes_list:
                if example_bytes:
                    writer.write(example_bytes)
        return path

    def _write_metadata(
        self,
        *,
        output: Path,
        episode_count: int,
        shard_lengths: list[int],
        shard_paths: list[str],
        action_dim: int,
        state_dim: int,
    ) -> None:
        """Emit dataset_info.json + features.json.

        Schema mirrors what `tfds.builder_from_directory` expects so the
        directory is loadable via standard tfds tooling.
        """
        feature_spec: dict[str, Any] = {
            "steps": {
                "type": "Dataset",
                "feature_spec": {
                    "action": {
                        "type": "Tensor", "dtype": "float32",
                        "shape": [action_dim],
                    },
                    "reward": {"type": "Tensor", "dtype": "float32", "shape": [1]},
                    "discount": {"type": "Tensor", "dtype": "float32", "shape": [1]},
                    "is_first": {"type": "Tensor", "dtype": "int64", "shape": [1]},
                    "is_last": {"type": "Tensor", "dtype": "int64", "shape": [1]},
                    "is_terminal": {"type": "Tensor", "dtype": "int64", "shape": [1]},
                    "language_instruction": {"type": "Text", "encoding": "utf-8"},
                },
            },
            "episode_metadata": {
                "type": "FeaturesDict",
                "feature_spec": {
                    "episode_id": {"type": "Text", "encoding": "utf-8"},
                    "quality_score": {"type": "Tensor", "dtype": "float32", "shape": [1]},
                },
            },
        }
        if state_dim > 0:
            feature_spec["steps"]["feature_spec"]["observation"] = {
                "type": "FeaturesDict",
                "feature_spec": {
                    "state": {
                        "type": "Tensor", "dtype": "float32",
                        "shape": [state_dim],
                    },
                },
            }

        info = {
            "name": self.dataset_name,
            "version": self.version,
            "description": (
                "Reflex Curate dataset in RLDS format. Generated from contributed "
                "deployment data; anonymized at source, quality-scored, and exported."
            ),
            "license": self.license,
            "embodiment": self.embodiment,
            "splits": [{
                "name": "train",
                "num_examples": episode_count,
                "shard_lengths": shard_lengths,
                "filepaths": shard_paths,
            }],
            "features": feature_spec,
            "format_version": "rlds-1.0",
            "generated_at": _utc_now_iso(),
        }
        with open(output / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        with open(output / "features.json", "w") as f:
            json.dump(feature_spec, f, indent=2)


__all__ = [
    "RLDS_VERSION",
    "DEFAULT_SHARD_SIZE",
    "RLDSConverter",
]
