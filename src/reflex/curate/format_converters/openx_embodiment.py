"""Open X-Embodiment format converter (Phase 1.5 — full impl on top of RLDS).

Spec: features/08_curate/_curation/format-converters/openx-embodiment.md

OXE is the Google / DeepMind cross-robot dataset standard. Built on top of
RLDS / TFRecord. Adds:
  - `embodiment_id` field per episode (Reflex slug → OXE canonical name,
    e.g. franka → franka_emika_panda)
  - Per-OXE-convention action naming: actions are stored RAW (no
    cross-embodiment normalization at the converter layer; consumers
    normalize per their training config)
  - `dataset_name` published as a recognizable OXE-style name
    ("reflex_curate_<embodiment>")

This Phase 1.5 implementation extends RLDSConverter — same TFRecord shard
output, same feature schema, plus the embodiment-id metadata.

Deps gated behind [curate-rlds] extra (shared with RLDS).
"""
from __future__ import annotations

import json
import logging
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


# Reflex embodiment slug → OXE canonical embodiment name.
# Sourced from the OXE robot taxonomy.
EMBODIMENT_OXE_MAP: dict[str, str] = {
    "franka": "franka_emika_panda",
    "so100": "so_arm100",
    "ur5": "ur5",
    "aloha": "aloha_static",
    "thor": "google_robot",
    "agx_orin": "google_robot",
}


class OpenXEmbodimentConverter(FormatConverter):
    """Convert Reflex JSONL traces → Open X-Embodiment dataset directory.

    Wraps RLDSConverter — emits the same TFRecord layout + adds OXE
    embodiment_id + dataset-name conventions.
    """

    FORMAT_NAME = "openx-embodiment"

    def __init__(
        self,
        *,
        embodiment: str = "unknown",
        license: str = "CC-BY-4.0",
        shard_size: int = 100,
    ):
        self.embodiment = embodiment
        self.oxe_embodiment = EMBODIMENT_OXE_MAP.get(
            embodiment.lower(), "unknown",
        )
        self.license = license
        self.shard_size = max(1, int(shard_size))
        self.dataset_name = f"reflex_curate_{embodiment.lower()}"

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
                "Open X-Embodiment conversion requires tensorflow + tensorflow_datasets: "
                "pip install 'reflex-vla[curate-rlds]'"
            ) from exc

        # Reuse RLDSConverter for the bulk of the work; we'll patch the
        # episode SequenceExample to carry the OXE-specific embodiment_id
        # context feature, and override the dataset_info.json / features.json
        # writers to surface the OXE schema.
        from reflex.curate.format_converters.rlds import RLDSConverter

        rlds = _OXEPatchedRLDS(
            dataset_name=self.dataset_name,
            license=self.license,
            shard_size=self.shard_size,
            embodiment=self.embodiment,
            oxe_embodiment=self.oxe_embodiment,
        )
        return rlds.convert(
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            min_quality=min_quality,
            canonical_only=canonical_only,
        )


class _OXEPatchedRLDS:
    """Internal RLDSConverter subclass that adds OXE embodiment_id + naming.

    Not exported. Constructed by OpenXEmbodimentConverter.convert().
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        license: str,
        shard_size: int,
        embodiment: str,
        oxe_embodiment: str,
    ):
        from reflex.curate.format_converters.rlds import RLDSConverter
        self._rlds = RLDSConverter(
            dataset_name=dataset_name,
            license=license,
            shard_size=shard_size,
            embodiment=embodiment,
        )
        self._oxe_embodiment = oxe_embodiment
        self._embodiment = embodiment

    def convert(self, **kwargs: Any) -> ConversionResult:
        # Monkey-patch the episode-example builder to inject embodiment_id.
        original_build = self._rlds._build_episode_example
        oxe_emb = self._oxe_embodiment

        def patched_build_episode_example(*, tf, rows, ep_id, action_dim, state_dim):
            # Same logic as RLDSConverter._build_episode_example but with
            # an extra `embodiment_id` context feature. Easier to reproduce
            # the build here than to extend the base method.
            actions, states = self._rlds._flatten_actions_and_steps(rows)
            if not actions:
                return b""

            n = len(actions)
            instruction = (
                rows[0].get("instruction_raw") or rows[0].get("instruction_hash") or ""
            )

            actions_padded = [
                list(a) + [0.0] * (action_dim - len(a)) for a in actions
            ]
            if state_dim > 0:
                states_padded = [
                    (list(s) if s is not None else [0.0] * state_dim) +
                    [0.0] * max(0, state_dim - len(s or []))
                    for s in states
                ]
            else:
                states_padded = [[]] * n

            is_first = [True] + [False] * (n - 1)
            is_last = [False] * (n - 1) + [True]
            is_terminal = [False] * (n - 1) + [True]
            rewards = [0.0] * (n - 1) + [1.0]
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

            md0 = rows[0].get("metadata", {}) or {}
            context_features: dict[str, Any] = {
                "episode_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[ep_id.encode("utf-8")]),
                ),
                # OXE-specific: embodiment_id context feature.
                "embodiment_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[oxe_emb.encode("utf-8")]),
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

        # Patch the metadata writer to add the embodiment_id schema entry.
        original_write_metadata = self._rlds._write_metadata

        def patched_write_metadata(
            *, output, episode_count, shard_lengths, shard_paths,
            action_dim, state_dim,
        ):
            original_write_metadata(
                output=output, episode_count=episode_count,
                shard_lengths=shard_lengths, shard_paths=shard_paths,
                action_dim=action_dim, state_dim=state_dim,
            )
            # Patch dataset_info.json with OXE-specific fields.
            info_path = output / "dataset_info.json"
            features_path = output / "features.json"
            with open(info_path) as f:
                info = json.load(f)
            info["oxe_embodiment"] = self._oxe_embodiment
            info["reflex_embodiment"] = self._embodiment
            info["format_version"] = "openx-embodiment-1.0"
            info["features"]["episode_metadata"]["feature_spec"]["embodiment_id"] = {
                "type": "Text", "encoding": "utf-8",
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            # Patch features.json the same way.
            with open(features_path) as f:
                features = json.load(f)
            features["episode_metadata"]["feature_spec"]["embodiment_id"] = {
                "type": "Text", "encoding": "utf-8",
            }
            with open(features_path, "w") as f:
                json.dump(features, f, indent=2)

        self._rlds._build_episode_example = patched_build_episode_example
        self._rlds._write_metadata = patched_write_metadata

        return self._rlds.convert(**kwargs)


__all__ = [
    "EMBODIMENT_OXE_MAP",
    "OpenXEmbodimentConverter",
]
