"""LeRobot v3 format converter.

Spec: https://github.com/huggingface/lerobot/blob/main/docs/source/dataset_v3.md

Output layout:

    <output_dir>/
    ├── README.md                              auto-generated dataset card
    ├── meta/
    │   ├── info.json                          codebase version + features schema
    │   ├── tasks.jsonl                        {task_index, task} per line
    │   └── episodes.jsonl                     {episode_index, tasks, length}
    └── data/
        └── chunk-000/
            ├── episode_000000.parquet         (state + action + frame_index + ...)
            └── episode_000001.parquet

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

import hashlib
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

LEROBOT_V3_VERSION = "v3.0"


class LeRobotV3Converter(FormatConverter):
    """Convert Reflex JSONL traces → LeRobot v3 dataset directory."""

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
        tasks_jsonl_lines: list[dict[str, Any]] = []
        episodes_jsonl_lines: list[dict[str, Any]] = []

        # Sort episodes for deterministic episode_index assignment.
        episode_index = 0
        global_step_index = 0
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

            # Task index lookup (one row in tasks.jsonl per unique instruction).
            instruction = rows[0].get("instruction_raw")
            if not isinstance(instruction, str):
                # Fall back to hash-only when raw instruction was redacted.
                instruction = rows[0].get("instruction_hash") or "unknown_task"
            if instruction not in task_index_by_text:
                task_index_by_text[instruction] = len(task_index_by_text)
                tasks_jsonl_lines.append({
                    "task_index": task_index_by_text[instruction],
                    "task": instruction,
                })
            task_idx = task_index_by_text[instruction]

            # Build per-step rows for parquet.
            step_count = len(actions)
            frame_indices = list(range(step_count))
            global_indices = [global_step_index + i for i in range(step_count)]
            episode_indices = [episode_index] * step_count
            timestamps = [float(i) / self.fps for i in range(step_count)]
            task_indices = [task_idx] * step_count

            # Pad missing states with zeros (state_vec replicated across chunk
            # rows; if state_vec is None for a row, we keep None → fill zeros).
            state_dim = max((len(s) for s in states if s is not None), default=0)
            action_dim = max(len(a) for a in actions)
            states_filled = [
                s if s is not None else [0.0] * state_dim for s in states
            ]
            # Coerce all action / state vectors to consistent shape.
            actions_out = [list(a) + [0.0] * (action_dim - len(a)) for a in actions]
            states_out = [
                (list(s) + [0.0] * (state_dim - len(s))) if state_dim > 0 else []
                for s in states_filled
            ]

            table_data = {
                "frame_index": frame_indices,
                "episode_index": episode_indices,
                "index": global_indices,
                "timestamp": timestamps,
                "task_index": task_indices,
                "action": actions_out,
            }
            if state_dim > 0:
                table_data["observation.state"] = states_out

            table = pa.table(table_data)
            parquet_path = output / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            pq.write_table(table, str(parquet_path))
            result.bytes_written += parquet_path.stat().st_size
            result.episode_count += 1
            result.step_count += step_count
            global_step_index += step_count

            episodes_jsonl_lines.append({
                "episode_index": episode_index,
                "tasks": [task_idx],
                "length": step_count,
            })

            # Video materialization (per [curate-video] extra). Skips when
            # frames aren't decodable (hash-only image_b64) or when the
            # encoder dep isn't installed.
            video_dims = None
            if self.encode_videos:
                video_dims = self._maybe_encode_episode_video(
                    output=output,
                    episode_index=episode_index,
                    rows=rows,
                    result=result,
                )

            episode_index += 1

        if result.episode_count == 0:
            result.warnings.append("no_episodes_passed_filter")
            result.completed_at = _utc_now_iso()
            return result

        # Write tasks.jsonl + episodes.jsonl.
        with open(output / "meta" / "tasks.jsonl", "w") as f:
            for line in tasks_jsonl_lines:
                f.write(json.dumps(line) + "\n")
        with open(output / "meta" / "episodes.jsonl", "w") as f:
            for line in episodes_jsonl_lines:
                f.write(json.dumps(line) + "\n")

        # Write info.json.
        info = self._build_info_json(
            action_dim=action_dim,
            state_dim=state_dim,
            episode_count=result.episode_count,
        )
        with open(output / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Write README.md (dataset card).
        readme = self._build_readme(result, tasks_jsonl_lines)
        with open(output / "README.md", "w") as f:
            f.write(readme)

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
            from reflex.curate.format_converters.shared.video_encoder import (
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
            output / "videos" / "chunk-000"
            / f"observation.images.{self.video_camera_name}"
            / f"episode_{episode_index:06d}.mp4"
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
            "splits": {"train": f"0:{episode_count}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
generator: reflex-vla curate / lerobot-v3 converter
generated_at: {result.started_at}
---

# Reflex Curate dataset

Generated from contributed Reflex deployment data. Anonymized at source
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
