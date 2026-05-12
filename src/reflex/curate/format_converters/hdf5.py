"""HDF5 format converter (Phase 1 — generic scientific format).

Spec: features/08_curate/_curation/format-converters/hdf5.md
HDF5 spec: https://www.hdfgroup.org/solutions/hdf5/

Output layout (single-file mode, default):

    <output>.h5
    ├── meta/                                    (group)
    │   ├── dataset_id, reflex_version, created_at, num_episodes
    │   ├── tasks  (variable-length str array)
    │   └── attrs  (license, contributor_count, ...)
    └── episodes/                                (group, one subgroup per episode)
        ├── episode_000000/
        │   ├── observation/
        │   │   └── state              (T, state_dim) float32
        │   ├── action                   (T, action_dim) float32
        │   ├── timestamp               (T,) float64
        │   └── attrs                    success_flag, quality_score, length, ...
        ├── episode_000001/
        └── ...

Multi-file mode (`split_episodes=True`): one .h5 per episode, identical
structure within each file.

Phase 1 ships compression: lzf for arrays (fast read/write), no compression
for state/action (small + already compact).

Image arrays NOT included in v1 — same caveat as LeRobot v3 converter
(image_redaction='full' + ffmpeg deps required for full pipeline).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from reflex.curate.format_converters.base import (
    ConversionResult,
    FormatConverter,
    _group_by_episode,
    _iter_jsonl,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)


class HDF5Converter(FormatConverter):
    """Convert Reflex JSONL traces → HDF5 hierarchical container."""

    FORMAT_NAME = "hdf5"

    def __init__(
        self,
        *,
        dataset_id: str = "reflex_curate_dataset",
        license: str = "CC-BY-4.0",
        compression: str = "lzf",  # "lzf" | "gzip" | None
        split_episodes: bool = False,
    ):
        if compression not in (None, "lzf", "gzip"):
            raise ValueError(
                f"compression must be None|lzf|gzip, got {compression!r}"
            )
        self.dataset_id = dataset_id
        self.license = license
        self.compression = compression
        self.split_episodes = bool(split_episodes)

    def convert(
        self,
        *,
        input_jsonl: str | Path | list[str | Path],
        output_dir: str | Path,
        min_quality: float | None = None,
        canonical_only: bool = False,
    ) -> ConversionResult:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py required for HDF5 conversion: pip install 'reflex-vla[curate-hdf5]'"
            ) from exc

        result = ConversionResult(
            output_dir=str(output_dir),
            format=self.FORMAT_NAME,
            started_at=_utc_now_iso(),
        )
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)

        # Read + group across all input JSONLs.
        input_paths = self._resolve_inputs(input_jsonl)
        all_episodes: dict[str, list[dict[str, Any]]] = {}
        for p in input_paths:
            for ep_id, rows in _group_by_episode(_iter_jsonl(p)).items():
                all_episodes.setdefault(ep_id, []).extend(rows)

        # Filter + sort.
        kept_episodes: list[tuple[str, list[dict[str, Any]]]] = []
        all_tasks: list[str] = []
        task_indices: dict[str, int] = {}
        for ep_id, rows in sorted(all_episodes.items()):
            keep, reason = self._filter_episode(
                rows, min_quality=min_quality, canonical_only=canonical_only,
            )
            if not keep:
                result.skipped_episodes += 1
                result.skipped_reasons[reason] += 1
                continue
            kept_episodes.append((ep_id, rows))

            instruction = rows[0].get("instruction_raw") or rows[0].get("instruction_hash") or "unknown"
            if instruction not in task_indices:
                task_indices[instruction] = len(all_tasks)
                all_tasks.append(instruction)

        if not kept_episodes:
            result.warnings.append("no_episodes_passed_filter")
            result.completed_at = _utc_now_iso()
            return result

        # Single-file vs multi-file output.
        if self.split_episodes:
            for episode_index, (ep_id, rows) in enumerate(kept_episodes):
                file_path = output / f"episode_{episode_index:06d}.h5"
                self._write_single_file(
                    h5py, file_path, [(ep_id, rows)],
                    base_episode_index=episode_index,
                    task_indices=task_indices, all_tasks=all_tasks,
                    result=result,
                )
        else:
            file_path = output / f"{self.dataset_id}.h5"
            self._write_single_file(
                h5py, file_path, kept_episodes,
                base_episode_index=0,
                task_indices=task_indices, all_tasks=all_tasks,
                result=result,
            )

        result.completed_at = _utc_now_iso()
        return result

    def _write_single_file(
        self,
        h5py,
        file_path: Path,
        episodes: list[tuple[str, list[dict[str, Any]]]],
        *,
        base_episode_index: int,
        task_indices: dict[str, int],
        all_tasks: list[str],
        result: ConversionResult,
    ) -> None:
        compression_kwargs = {"compression": self.compression} if self.compression else {}

        with h5py.File(file_path, "w") as h5:
            # /meta group
            meta = h5.create_group("meta")
            meta.attrs["dataset_id"] = self.dataset_id
            meta.attrs["license"] = self.license
            meta.attrs["created_at"] = result.started_at
            meta.attrs["num_episodes"] = len(episodes)
            meta.attrs["reflex_version"] = self._reflex_version()
            # tasks as variable-length string dataset
            dt_str = h5py.string_dtype(encoding="utf-8")
            meta.create_dataset(
                "tasks", data=np.asarray(all_tasks, dtype=object), dtype=dt_str,
            )

            episodes_group = h5.create_group("episodes")
            for offset, (ep_id, rows) in enumerate(episodes):
                episode_index = base_episode_index + offset
                actions_list, states_list = self._flatten_actions_and_steps(rows)
                if not actions_list:
                    continue
                action_arr = np.asarray(actions_list, dtype=np.float32)
                state_dim = max(
                    (len(s) for s in states_list if s is not None), default=0,
                )
                if state_dim > 0:
                    state_arr = np.asarray(
                        [
                            (s if s is not None else [0.0] * state_dim) +
                            [0.0] * max(0, state_dim - (len(s) if s is not None else 0))
                            for s in states_list
                        ],
                        dtype=np.float32,
                    )
                else:
                    state_arr = None

                ep_group = episodes_group.create_group(f"episode_{episode_index:06d}")

                # action
                ep_group.create_dataset(
                    "action", data=action_arr, **compression_kwargs,
                )

                # observation/state
                if state_arr is not None:
                    obs = ep_group.create_group("observation")
                    obs.create_dataset(
                        "state", data=state_arr, **compression_kwargs,
                    )

                # timestamps (synthetic — derived from row index)
                timestamps = np.arange(action_arr.shape[0], dtype=np.float64) / 30.0
                ep_group.create_dataset("timestamp", data=timestamps)

                # episode attrs
                md0 = rows[0].get("metadata", {}) or {}
                instruction = (
                    rows[0].get("instruction_raw")
                    or rows[0].get("instruction_hash") or "unknown"
                )
                ep_group.attrs["task_index"] = task_indices[instruction]
                ep_group.attrs["episode_id"] = str(ep_id)
                ep_group.attrs["length"] = action_arr.shape[0]
                if md0.get("quality_score") is not None:
                    ep_group.attrs["quality_score"] = float(md0["quality_score"])
                if md0.get("is_failure") is not None:
                    ep_group.attrs["is_failure"] = bool(md0["is_failure"])
                if md0.get("primary_failure_mode"):
                    ep_group.attrs["primary_failure_mode"] = str(md0["primary_failure_mode"])
                if md0.get("dedup_cluster_id"):
                    ep_group.attrs["dedup_cluster_id"] = str(md0["dedup_cluster_id"])
                if md0.get("is_canonical") is not None:
                    ep_group.attrs["is_canonical"] = bool(md0["is_canonical"])

                result.step_count += action_arr.shape[0]
                result.episode_count += 1

        result.bytes_written += file_path.stat().st_size

    def _reflex_version(self) -> str:
        try:
            from reflex import __version__
            return str(__version__)
        except Exception:  # noqa: BLE001
            return "unknown"


__all__ = [
    "HDF5Converter",
]
