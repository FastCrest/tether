"""LeRobot v3 format converter.

Spec: https://github.com/huggingface/lerobot/blob/main/docs/source/dataset_v3.md

Output layout:

    <output_dir>/
    ├── README.md                              auto-generated dataset card
    ├── meta/
    │   ├── info.json                          codebase version + features schema
    │   ├── stats.json                         global feature statistics
    │   ├── tasks.parquet                      task index, indexed by task text
    │   └── episodes/
    │       └── chunk-000/file-000.parquet     episode metadata + offsets
    └── data/
        └── chunk-000/
            └── file-000.parquet               consolidated frame rows

Phase 1 v1 ships parquet + metadata. Video materialization (mp4 in
`videos/` subdir) requires ffmpeg-python + imageio-ffmpeg deps; deferred
until first customer with `image_redaction='full'` data asks for it.

Spec said "round-trip parity vs reference dataset" is a Phase 1 target.
This module ships the converter; round-trip parity validation against a
real reference dataset is its own follow-up task (depends on a known
LeRobot v3 reference dataset for comparison, which we'd have to download
+ test against).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from tether.curate.format_converters.base import (
    ConversionResult,
    FormatConverter,
    _group_by_episode,
    _iter_jsonl,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)

LEROBOT_V3_VERSION = "v3.0"


class LeRobotV3Converter(FormatConverter):
    """Convert Tether JSONL traces → LeRobot v3 dataset directory."""

    FORMAT_NAME = "lerobot-v3"

    def __init__(
        self,
        *,
        robot_type: str = "unknown",
        fps: int = 30,
        action_names: list[str] | None = None,
        state_names: list[str] | None = None,
        license: str = "CC-BY-4.0",
        video_camera_name: str = "cam_main",
        encode_videos: bool = True,
    ):
        self.robot_type = robot_type
        self.fps = int(fps)
        self.action_names = action_names
        self.state_names = state_names
        self.license = license
        self.video_camera_name = video_camera_name
        self.encode_videos = bool(encode_videos)

    def convert(
        self,
        *,
        input_jsonl: str | Path | list[str | Path],
        output_dir: str | Path,
        min_quality: float | None = None,
        canonical_only: bool = False,
    ) -> ConversionResult:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow required for LeRobot v3 conversion: "
                "pip install pyarrow (already installed via huggingface_hub)"
            ) from exc

        result = ConversionResult(
            output_dir=str(output_dir),
            format=self.FORMAT_NAME,
            started_at=_utc_now_iso(),
        )
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        (output / "meta").mkdir(exist_ok=True)
        (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # Read + group by episode across all input JSONLs.
        input_paths = self._resolve_inputs(input_jsonl)
        all_episodes: dict[str, list[dict[str, Any]]] = {}
        for p in input_paths:
            for episode_id, rows in _group_by_episode(_iter_jsonl(p)).items():
                # Concat across files when same episode_id appears in multiple
                # JSONLs (rare but possible if a session spans uploaders).
                all_episodes.setdefault(episode_id, []).extend(rows)

        # Build the task index by hash-deduping instructions.
        task_index_by_text: dict[str, int] = {}
        task_rows: list[dict[str, Any]] = []
        pending_episodes: list[dict[str, Any]] = []
        action_dim = 0
        state_dim = 0

        # Sort episodes for deterministic episode_index assignment. First pass
        # discovers global action/state dims so every row in the consolidated
        # v3 data shard has one stable schema.
        for episode_id, rows in sorted(all_episodes.items()):
            keep, reason = self._filter_episode(
                rows, min_quality=min_quality, canonical_only=canonical_only,
            )
            if not keep:
                result.skipped_episodes += 1
                result.skipped_reasons[reason] += 1
                continue

            actions, states = self._flatten_actions_and_steps(rows)
            if not actions:
                result.skipped_episodes += 1
                result.skipped_reasons["no_actions"] += 1
                continue
            action_dim = max(action_dim, max(len(a) for a in actions))
            state_dim = max(state_dim, max((len(s) for s in states if s is not None), default=0))

            # Task index lookup (one row in tasks.parquet per unique instruction).
            instruction = rows[0].get("instruction_raw")
            if not isinstance(instruction, str):
                # Fall back to hash-only when raw instruction was redacted.
                instruction = rows[0].get("instruction_hash") or "unknown_task"
            if instruction not in task_index_by_text:
                task_index_by_text[instruction] = len(task_index_by_text)
                task_rows.append({
                    "task_index": task_index_by_text[instruction],
                    "task": instruction,
                })
            task_idx = task_index_by_text[instruction]
            pending_episodes.append({
                "episode_id": episode_id,
                "rows": rows,
                "actions": actions,
                "states": states,
                "instruction": instruction,
                "task_idx": task_idx,
            })

        if not pending_episodes:
            result.warnings.append("no_episodes_passed_filter")
            result.completed_at = _utc_now_iso()
            return result

        data_rows: list[dict[str, Any]] = []
        episode_meta_rows: list[dict[str, Any]] = []
        all_actions: list[list[float]] = []
        all_states: list[list[float]] = []

        global_step_index = 0
        for episode_index, episode in enumerate(pending_episodes):
            rows = episode["rows"]
            actions = episode["actions"]
            states = episode["states"]
            task_idx = episode["task_idx"]
            instruction = episode["instruction"]
            step_count = len(actions)

            # Pad missing states with zeros (state_vec replicated across chunk
            # rows; if state_vec is None for a row, we keep None → fill zeros).
            states_filled = [
                s if s is not None else [0.0] * state_dim for s in states
            ]
            # Coerce all action / state vectors to consistent shape.
            actions_out = [list(a) + [0.0] * (action_dim - len(a)) for a in actions]
            states_out = [
                (list(s) + [0.0] * (state_dim - len(s))) if state_dim > 0 else []
                for s in states_filled
            ]

            for i, action in enumerate(actions_out):
                data_row = {
                    "frame_index": i,
                    "episode_index": episode_index,
                    "index": global_step_index + i,
                    "timestamp": float(i) / self.fps,
                    "task_index": task_idx,
                    "action": action,
                }
                if state_dim > 0:
                    data_row["observation.state"] = states_out[i]
                data_rows.append(data_row)

            all_actions.extend(actions_out)
            all_states.extend(states_out)
            episode_stats = self._build_feature_stats(
                actions=actions_out,
                states=states_out if state_dim > 0 else [],
            )
            episode_meta = {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": step_count,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_step_index,
                "dataset_to_index": global_step_index + step_count,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            episode_meta.update(self._flatten_stats(episode_stats))
            episode_meta_rows.append(episode_meta)

            # Video materialization (per [curate-video] extra). Skips when
            # frames aren't decodable (hash-only image_b64) or when the
            # encoder dep isn't installed.
            if self.encode_videos:
                self._maybe_encode_episode_video(
                    output=output,
                    episode_index=episode_index,
                    rows=rows,
                    result=result,
                )

            result.episode_count += 1
            result.step_count += step_count
            global_step_index += step_count

        # Write the v3 file-based shards and metadata.
        data_path = output / "data" / "chunk-000" / "file-000.parquet"
        data_schema_fields = [
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("timestamp", pa.float32()),
            ("task_index", pa.int64()),
            ("action", pa.list_(pa.float32(), list_size=action_dim)),
        ]
        if state_dim > 0:
            data_schema_fields.append(
                ("observation.state", pa.list_(pa.float32(), list_size=state_dim))
            )
        data_table = pa.Table.from_pylist(data_rows, schema=pa.schema(data_schema_fields))
        pq.write_table(data_table, str(data_path), compression="snappy")
        result.bytes_written += data_path.stat().st_size

        episodes_path = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        pq.write_table(pa.Table.from_pylist(episode_meta_rows), str(episodes_path), compression="snappy")
        result.bytes_written += episodes_path.stat().st_size

        tasks_path = output / "meta" / "tasks.parquet"
        self._write_tasks_parquet(pa=pa, pq=pq, path=tasks_path, tasks=task_rows)
        result.bytes_written += tasks_path.stat().st_size

        stats = self._build_feature_stats(
            actions=all_actions,
            states=all_states if state_dim > 0 else [],
        )
        stats_path = output / "meta" / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        result.bytes_written += stats_path.stat().st_size

        # Write info.json.
        info = self._build_info_json(
            action_dim=action_dim,
            state_dim=state_dim,
            episode_count=result.episode_count,
            frame_count=result.step_count,
            task_count=len(task_rows),
        )
        with open(output / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        result.bytes_written += (output / "meta" / "info.json").stat().st_size

        # Write README.md (dataset card).
        readme = self._build_readme(result, task_rows)
        with open(output / "README.md", "w") as f:
            f.write(readme)
        result.bytes_written += (output / "README.md").stat().st_size

        result.completed_at = _utc_now_iso()
        return result

    def _maybe_encode_episode_video(
        self,
        *,
        output: Path,
        episode_index: int,
        rows: list[dict[str, Any]],
        result: ConversionResult,
    ) -> tuple[int, int] | None:
        """Encode the episode's frames to mp4 if image bytes are available
        and the [curate-video] extra is installed. Returns (width, height) on
        success, None when skipped."""
        try:
            from tether.curate.format_converters.shared.video_encoder import (
                VideoEncoderUnavailable,
                collect_frames_from_rows,
                encode_frames_to_mp4,
            )
        except ImportError:
            # Shared module always imports; this branch is unreachable but
            # keeps the type-checker honest.
            return None

        frames = collect_frames_from_rows(rows)
        if not frames:
            # image_b64 is hash-only or absent — typical for default
            # `--record-images hash_only` mode. Note once per converter pass.
            if "videos_skipped_hash_only" not in result.skipped_reasons:
                result.warnings.append(
                    "videos_skipped:image_b64_is_hash_only_or_absent"
                )
            result.skipped_reasons["videos_skipped_hash_only"] += 1
            return None

        video_path = (
            output / "videos"
            / f"observation.images.{self.video_camera_name}"
            / "chunk-000"
            / f"file-{episode_index:03d}.mp4"
        )
        try:
            bytes_written = encode_frames_to_mp4(
                frames=frames,
                output_path=video_path,
                fps=self.fps,
            )
        except VideoEncoderUnavailable as exc:
            if "video_encoder_unavailable" not in result.skipped_reasons:
                result.warnings.append(
                    f"videos_skipped:install [curate-video] extra ({exc})"
                )
            result.skipped_reasons["video_encoder_unavailable"] += 1
            return None
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(
                f"videos_episode_{episode_index:06d}_failed:{exc}"
            )
            return None

        result.bytes_written += bytes_written
        # Read back the dimensions for info.json features schema.
        try:
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(frames[0])).convert("RGB")
            return img.size
        except Exception:  # noqa: BLE001
            return None

    def _build_info_json(
        self,
        *,
        action_dim: int,
        state_dim: int,
        episode_count: int,
        frame_count: int,
        task_count: int,
    ) -> dict[str, Any]:
        action_names = self.action_names or [
            f"axis_{i}" for i in range(action_dim)
        ]
        state_names = self.state_names or [
            f"axis_{i}" for i in range(state_dim)
        ]
        info: dict[str, Any] = {
            "codebase_version": LEROBOT_V3_VERSION,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": episode_count,
            "total_frames": frame_count,
            "total_tasks": task_count,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "splits": {"train": f"0:{episode_count}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [action_dim],
                    "names": action_names,
                },
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        }
        if state_dim > 0:
            info["features"]["observation.state"] = {
                "dtype": "float32",
                "shape": [state_dim],
                "names": state_names,
            }
        return info

    @staticmethod
    def _build_feature_stats(
        *,
        actions: list[list[float]],
        states: list[list[float]],
    ) -> dict[str, dict[str, list[float] | list[int]]]:
        stats: dict[str, dict[str, list[float] | list[int]]] = {}
        if actions:
            stats["action"] = LeRobotV3Converter._vector_stats(actions)
        if states:
            stats["observation.state"] = LeRobotV3Converter._vector_stats(states)
        return stats

    @staticmethod
    def _vector_stats(vectors: list[list[float]]) -> dict[str, list[float] | list[int]]:
        arr = np.asarray(vectors, dtype=np.float32)
        return {
            "min": arr.min(axis=0).astype(float).tolist(),
            "max": arr.max(axis=0).astype(float).tolist(),
            "mean": arr.mean(axis=0).astype(float).tolist(),
            "std": arr.std(axis=0).astype(float).tolist(),
            "count": [int(arr.shape[0])],
            "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).astype(float).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).astype(float).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).astype(float).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
        }

    @staticmethod
    def _flatten_stats(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for feature_name, feature_stats in stats.items():
            for stat_name, value in feature_stats.items():
                flat[f"stats/{feature_name}/{stat_name}"] = value
        return flat

    @staticmethod
    def _write_tasks_parquet(
        *,
        pa: Any,
        pq: Any,
        path: Path,
        tasks: list[dict[str, Any]],
    ) -> None:
        table = pa.table(
            {
                "task_index": [int(task["task_index"]) for task in tasks],
                "task": [str(task["task"]) for task in tasks],
            }
        )
        pandas_metadata = {
            "index_columns": ["task"],
            "column_indexes": [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ],
            "columns": [
                {
                    "name": "task_index",
                    "field_name": "task_index",
                    "pandas_type": "int64",
                    "numpy_type": "int64",
                    "metadata": None,
                },
                {
                    "name": "task",
                    "field_name": "task",
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                },
            ],
            "attributes": {},
            "creator": {
                "library": "pyarrow",
                "version": getattr(pa, "__version__", "unknown"),
            },
            "pandas_version": "2.0.0",
        }
        metadata = dict(table.schema.metadata or {})
        metadata[b"pandas"] = json.dumps(pandas_metadata).encode()
        pq.write_table(table.replace_schema_metadata(metadata), str(path), compression="snappy")

    def _build_readme(
        self,
        result: ConversionResult,
        tasks: list[dict[str, Any]],
    ) -> str:
        task_lines = "\n".join(f"  - {t['task']}" for t in tasks[:20])
        if len(tasks) > 20:
            task_lines += f"\n  - ... ({len(tasks) - 20} more)"
        return f"""---
dataset_info:
  num_episodes: {result.episode_count}
  num_tasks: {len(tasks)}
  num_steps: {result.step_count}
license: {self.license}
robot_type: {self.robot_type}
fps: {self.fps}
generator: tether curate / lerobot-v3 converter
generated_at: {result.started_at}
---

# Tether Curate dataset

Generated from contributed Tether deployment data. Anonymized at source
(face-blurred + instruction-hashed by default), quality-scored, deduped,
auto-tagged, and exported in HuggingFace LeRobot v3 format.

## Statistics

- Episodes: {result.episode_count}
- Steps: {result.step_count}
- Tasks: {len(tasks)}
- Robot: {self.robot_type}
- Frame rate: {self.fps} Hz
- Output size: {result.bytes_written / (1024 * 1024):.2f} MB

## Tasks

{task_lines}

## License

{self.license}

## Notes

This v1 export does NOT include video frames (mp4 files under `videos/`).
Frames require `image_redaction=full` recording mode + the [curate-video]
install extra. Action + state + metadata are complete.
"""


__all__ = [
    "LEROBOT_V3_VERSION",
    "LeRobotV3Converter",
]
