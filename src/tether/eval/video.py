"""Per-episode MP4 encoder for `tether eval --video`.

Per ADR 2026-04-25-eval-as-a-service-architecture decision #7:
- Phase 1 ships --video with local-only output (Phase 2 layers HF Hub
  upload via the self-distilling-serve HF token plumbing)
- ffmpeg encode at quality cap that stays under ~10MB per episode

The encoder is a thin wrapper around ffmpeg subprocess. Frames are
piped as raw RGB24 (no PNG/JPEG round-trip in-process). If ffmpeg
isn't on PATH, fail loud with a remediation pointing at the install
docs (NEVER silent fallback).

Tests stub the ffmpeg call via the `ffmpeg_caller` injection point so
they don't require a real ffmpeg binary.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


# Per-episode size cap. Chosen to keep a 90-task × 50-episode run
# under 45GB on disk (90 × 50 × 10MB). Customers running --video on
# 100+ eps per task should bump this OR write a pruning step.
DEFAULT_MAX_BYTES_PER_EPISODE = 10 * 1024 * 1024  # 10 MB

# Default frames-per-second matches LIBERO env.render() default.
DEFAULT_FPS = 20

# Initial CRF (Constant Rate Factor) for libx264. Higher = smaller +
# lower quality. 28 = good baseline; if over cap we fall back to 35.
INITIAL_CRF = 28
FALLBACK_CRF = 35


class VideoEncodeError(Exception):
    """Base for video encoding failures."""


class FfmpegMissingError(VideoEncodeError):
    """ffmpeg binary not found on PATH."""


@dataclass(frozen=True)
class VideoEncodeResult:
    """Frozen output of encode_episode_video()."""

    path: str
    bytes_written: int
    n_frames: int
    fps: int
    crf_used: int
    elapsed_s: float

    def __post_init__(self) -> None:
        if self.bytes_written < 0:
            raise ValueError(
                f"bytes_written must be >= 0, got {self.bytes_written}"
            )
        if self.n_frames < 0:
            raise ValueError(f"n_frames must be >= 0, got {self.n_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")


# Type alias for the ffmpeg subprocess injection point. Production
# wires this to a real subprocess.run; tests stub it.
FfmpegCaller = Callable[[list[str], bytes], subprocess.CompletedProcess]


def _real_ffmpeg_caller(cmd: list[str], frames_bytes: bytes) -> subprocess.CompletedProcess:
    """Production caller: run ffmpeg subprocess, pipe frames via stdin."""
    return subprocess.run(
        cmd,
        input=frames_bytes,
        capture_output=True,
        timeout=120.0,
    )


def encode_episode_video(
    *,
    frames: Sequence,  # list[np.ndarray] HxWx3 uint8 -- typed loosely to avoid hard numpy dep
    output_path: str | Path,
    fps: int = DEFAULT_FPS,
    max_bytes: int = DEFAULT_MAX_BYTES_PER_EPISODE,
    ffmpeg_caller: FfmpegCaller | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> VideoEncodeResult:
    """Encode `frames` as MP4 at quality cap; write to `output_path`.

    Args:
        frames: list of HxWx3 uint8 numpy arrays (or any sequence of arrays
            with .tobytes() method).
        output_path: where to write the .mp4. Parent dirs created if missing.
        fps: target frames-per-second.
        max_bytes: cap per-episode output size. If first encode exceeds,
            re-encodes with higher CRF (lower quality).
        ffmpeg_caller: subprocess wrapper. None = real ffmpeg subprocess.
        ffmpeg_binary: name/path of ffmpeg binary. Used for PATH check.

    Raises:
        FfmpegMissingError: ffmpeg not on PATH (and no caller injected).
        VideoEncodeError: ffmpeg ran but failed.
        ValueError: empty frames list, invalid fps, or invalid frame shape.

    Returns VideoEncodeResult with path + bytes + n_frames + crf_used.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
    if len(frames) == 0:
        raise ValueError("frames cannot be empty")

    # Probe the first frame for dimensions
    first = frames[0]
    if not hasattr(first, "shape") or len(first.shape) != 3:
        raise ValueError(
            f"frames[0] must be HxWx3 array, got shape "
            f"{getattr(first, 'shape', 'no .shape')}"
        )
    height, width, channels = first.shape
    if channels != 3:
        raise ValueError(
            f"frames must be RGB (3 channels), got {channels}"
        )

    # Resolve caller — real ffmpeg requires the binary on PATH
    if ffmpeg_caller is None:
        if shutil.which(ffmpeg_binary) is None:
            raise FfmpegMissingError(
                f"ffmpeg binary {ffmpeg_binary!r} not found on PATH. "
                f"Install via your system package manager (apt/brew/etc) "
                f"OR pass --no-video to skip video output."
            )
        ffmpeg_caller = _real_ffmpeg_caller

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate all frame bytes once (avoids re-iterating on retry)
    frames_bytes = b"".join(f.tobytes() for f in frames)

    import time
    t0 = time.perf_counter()

    # First attempt with INITIAL_CRF
    crf_used, bytes_written = _encode_attempt(
        ffmpeg_caller=ffmpeg_caller,
        ffmpeg_binary=ffmpeg_binary,
        frames_bytes=frames_bytes,
        width=width,
        height=height,
        fps=fps,
        crf=INITIAL_CRF,
        output=output,
    )

    # If over cap, retry with stronger compression
    if bytes_written > max_bytes:
        logger.info(
            "encode_episode_video: first pass %d bytes > cap %d, "
            "retrying with crf=%d",
            bytes_written, max_bytes, FALLBACK_CRF,
        )
        crf_used, bytes_written = _encode_attempt(
            ffmpeg_caller=ffmpeg_caller,
            ffmpeg_binary=ffmpeg_binary,
            frames_bytes=frames_bytes,
            width=width,
            height=height,
            fps=fps,
            crf=FALLBACK_CRF,
            output=output,
        )

    elapsed = time.perf_counter() - t0
    return VideoEncodeResult(
        path=str(output),
        bytes_written=bytes_written,
        n_frames=len(frames),
        fps=fps,
        crf_used=crf_used,
        elapsed_s=elapsed,
    )


def _encode_attempt(
    *,
    ffmpeg_caller: FfmpegCaller,
    ffmpeg_binary: str,
    frames_bytes: bytes,
    width: int,
    height: int,
    fps: int,
    crf: int,
    output: Path,
) -> tuple[int, int]:
    """Run one ffmpeg encode pass. Returns (crf_used, bytes_written).

    Raises VideoEncodeError if ffmpeg returns non-zero.
    """
    cmd = _build_ffmpeg_cmd(
        ffmpeg_binary=ffmpeg_binary,
        width=width, height=height, fps=fps, crf=crf,
        output_path=output,
    )
    result = ffmpeg_caller(cmd, frames_bytes)
    if result.returncode != 0:
        stderr = (
            result.stderr.decode("utf-8", errors="replace")
            if isinstance(result.stderr, bytes)
            else (result.stderr or "")
        )
        raise VideoEncodeError(
            f"ffmpeg exited with code {result.returncode}. "
            f"stderr (last 500 chars): {stderr[-500:]}"
        )
    if not output.exists():
        raise VideoEncodeError(
            f"ffmpeg returncode=0 but output {output} not created"
        )
    return crf, output.stat().st_size


def _build_ffmpeg_cmd(
    *,
    ffmpeg_binary: str,
    width: int,
    height: int,
    fps: int,
    crf: int,
    output_path: Path,
) -> list[str]:
    """Build the ffmpeg command-line. Frames piped via stdin as raw RGB24."""
    return [
        ffmpeg_binary,
        "-y",  # overwrite output without prompting
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # stdin
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",  # Quicktime/iOS compatible
        "-loglevel", "error",
        str(output_path),
    ]


def episode_video_path(
    *,
    output_dir: str | Path,
    task_id: str,
    episode_index: int,
) -> Path:
    """Conventional output path: <output>/videos/<task_id>_episode_<N>.mp4.

    Pure helper — does NOT create the file.
    """
    safe_task = task_id.replace("/", "_")
    return Path(output_dir) / "videos" / f"{safe_task}_episode_{episode_index}.mp4"


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_MAX_BYTES_PER_EPISODE",
    "FALLBACK_CRF",
    "FfmpegCaller",
    "FfmpegMissingError",
    "INITIAL_CRF",
    "VideoEncodeError",
    "VideoEncodeResult",
    "encode_episode_video",
    "episode_video_path",
]
