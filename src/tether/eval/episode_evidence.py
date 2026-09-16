"""Bounded, durable episode evidence for checkpoint evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
CAPTURE_SEMANTICS = {
    "step_observation": (
        "Exact post-env-step proprio record: end-effector position/quaternion, gripper qpos, "
        "and source image shapes. It is the state resulting from the recorded applied action."
    ),
    "step_action": "Exact 7-D environment action applied immediately before the recorded observation.",
    "policy_request": (
        "Monotonic request start/finish/duration plus planned-action count and action offset. "
        "One request may be referenced by several executed chunk actions."
    ),
    "task_events": "Observed evaluator events such as episode_started, task_success, or step_limit_reached.",
    "camera_frame": (
        "Sampled post-env-step agent-view frame at the configured stride. Wrist pixels are not retained by this capture level."
    ),
    "terminal_reason": "Observed evaluator terminal classification: success, timeout, adapter_error, or interruption.",
    "not_captured": [
        "Raw wrist-camera pixels",
        "The raw policy-input image tensor after preprocessing",
        "Language tokens/attention masks",
        "Noise tensors or model-internal action chunks beyond the applied action",
        "A separate raw pre-action observation on step zero",
    ],
}


@dataclass(frozen=True)
class CaptureLimits:
    max_bytes: int = 256 * 1024 * 1024
    max_steps: int = 2_000
    max_frames: int = 600
    frame_stride: int = 2
    jpeg_quality: int = 80
    flush_steps: int = 10

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_steps", "max_frames", "frame_stride", "flush_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class EpisodeEvidenceWriter:
    """Append evidence without retaining trajectories or frames in memory."""

    def __init__(
        self, root: str | Path, *, provenance: dict[str, Any], limits: CaptureLimits | None = None
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames = self.root / "frames"
        self.frames.mkdir(exist_ok=True)
        self.steps_path = self.root / "steps.jsonl"
        self.manifest_path = self.root / "episode-manifest.json"
        self.limits = limits or CaptureLimits()
        self.provenance = provenance
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.step_count = 0
        self.frame_count = 0
        self.bytes_written = 0
        self.errors: list[str] = []
        self.truncated = False
        self.truncation_reasons: list[str] = []
        self._stream = self.steps_path.open("a", encoding="utf-8")
        self._write_manifest(complete=False, terminal_reason=None)

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def _atomic_json(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def _mark_truncated(self, reason: str) -> None:
        self.truncated = True
        if reason not in self.truncation_reasons:
            self.truncation_reasons.append(reason)

    def _write_manifest(self, *, complete: bool, terminal_reason: str | None) -> None:
        artifacts = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path == self.manifest_path or path.name.endswith(".tmp"):
                continue
            artifacts.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _digest(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "tether-checkpoint-episode-evidence",
            "complete": complete,
            "terminal_reason": terminal_reason,
            "started_at_unix_s": self.started_wall,
            "duration_s": max(0.0, time.monotonic() - self.started_mono),
            "provenance": self.provenance,
            "limits": asdict(self.limits),
            "counts": {"steps": self.step_count, "frames": self.frame_count},
            "truncated": self.truncated,
            "truncation_reasons": self.truncation_reasons,
            "recording_errors": self.errors,
            "capture_semantics": CAPTURE_SEMANTICS,
            "artifacts": artifacts,
        }
        self._atomic_json(self.manifest_path, manifest)

    def record_step(
        self,
        *,
        step_index: int,
        phase: str,
        observation: dict[str, Any],
        action: list[float] | None,
        policy_request: dict[str, Any] | None,
        task_events: list[dict[str, Any]],
        frame: Any | None = None,
    ) -> bool:
        if self.step_count >= self.limits.max_steps:
            self._mark_truncated("step limit")
            return False
        frame_path = None
        if frame is not None and step_index % self.limits.frame_stride == 0:
            if self.frame_count >= self.limits.max_frames:
                self._mark_truncated("frame limit")
            else:
                try:
                    from io import BytesIO
                    from PIL import Image

                    frame_path = f"frames/{step_index:06d}.jpg"
                    target = self.root / frame_path
                    encoded_frame = BytesIO()
                    Image.fromarray(frame).save(
                        encoded_frame, format="JPEG", quality=self.limits.jpeg_quality
                    )
                    frame_bytes = encoded_frame.getvalue()
                    if self.bytes_written + len(frame_bytes) > self.limits.max_bytes:
                        self._mark_truncated("byte limit")
                        frame_path = None
                    else:
                        target.write_bytes(frame_bytes)
                        self.frame_count += 1
                        self.bytes_written += len(frame_bytes)
                except Exception as exc:
                    self.errors.append(f"frame {step_index}: {type(exc).__name__}: {exc}")
                    frame_path = None
        record = {
            "step_index": step_index,
            "timestamp_s": max(0.0, time.monotonic() - self.started_mono),
            "phase": phase,
            "observation": observation,
            "observation_semantics": "post-env-step",
            "action": action,
            "action_semantics": "applied-before-recorded-observation",
            "policy_request": policy_request,
            "task_events": task_events,
            "camera_frame": frame_path,
            "camera_frame_semantics": "post-env-step-agentview-sampled" if frame_path else None,
        }
        encoded = json.dumps(record, separators=(",", ":")) + "\n"
        if self.bytes_written + len(encoded.encode()) > self.limits.max_bytes:
            self._mark_truncated("byte limit")
            return False
        self._stream.write(encoded)
        self.bytes_written += len(encoded.encode())
        self.step_count += 1
        if self.step_count % self.limits.flush_steps == 0:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._write_manifest(complete=False, terminal_reason=None)
        return True

    def finish(self, terminal_reason: str) -> dict[str, Any]:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._write_manifest(complete=True, terminal_reason=terminal_reason)
        return json.loads(self.manifest_path.read_text())

    def interrupt(self, reason: str = "interrupted") -> dict[str, Any]:
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
        self._write_manifest(complete=False, terminal_reason=reason)
        return json.loads(self.manifest_path.read_text())


def validate_episode_evidence(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "episode-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema_version")
    if (
        schema not in SUPPORTED_SCHEMA_VERSIONS
        or manifest.get("kind") != "tether-checkpoint-episode-evidence"
    ):
        raise ValueError("unsupported episode evidence manifest")
    artifact_paths = set()
    for item in manifest.get("artifacts", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe artifact path")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("artifact leaves evidence directory")
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _digest(path) != item["sha256"]
        ):
            raise ValueError(f"artifact validation failed: {relative}")
        artifact_paths.add(relative.as_posix())

    if schema >= 2:
        if manifest.get("capture_semantics") != CAPTURE_SEMANTICS:
            raise ValueError("schema-2 capture semantics are missing or changed")
        steps_path = root / "steps.jsonl"
        rows = []
        if steps_path.is_file():
            for line in steps_path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        if len(rows) != int(manifest.get("counts", {}).get("steps", -1)):
            raise ValueError("step count does not match steps.jsonl")
        for index, row in enumerate(rows):
            if row.get("step_index") != index:
                raise ValueError("captured steps are not a contiguous prefix")
            if row.get("observation_semantics") != "post-env-step":
                raise ValueError("step observation semantics are missing")
            if row.get("action_semantics") != "applied-before-recorded-observation":
                raise ValueError("step action semantics are missing")
            if not isinstance(row.get("observation"), dict):
                raise ValueError("captured step is missing its proprio observation")
            action = row.get("action")
            if not isinstance(action, list) or len(action) != 7:
                raise ValueError("captured step must retain the exact 7-D applied action")
            frame = row.get("camera_frame")
            if frame is not None and frame not in artifact_paths:
                raise ValueError("step camera frame is not covered by the artifact manifest")
        frame_artifacts = [path for path in artifact_paths if path.startswith("frames/")]
        if len(frame_artifacts) != int(manifest.get("counts", {}).get("frames", -1)):
            raise ValueError("frame count does not match retained frame artifacts")
    return manifest


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "CAPTURE_SEMANTICS",
    "CaptureLimits",
    "EpisodeEvidenceWriter",
    "validate_episode_evidence",
]
