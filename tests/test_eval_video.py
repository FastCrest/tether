"""Tests for src/tether/eval/video.py — Phase 1 eval-as-a-service Day 5."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tether.eval.video import (
    DEFAULT_FPS,
    DEFAULT_MAX_BYTES_PER_EPISODE,
    FALLBACK_CRF,
    INITIAL_CRF,
    FfmpegMissingError,
    VideoEncodeError,
    VideoEncodeResult,
    encode_episode_video,
    episode_video_path,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_max_bytes_default_is_10mb():
    assert DEFAULT_MAX_BYTES_PER_EPISODE == 10 * 1024 * 1024


def test_default_fps_positive():
    assert DEFAULT_FPS > 0


def test_initial_crf_lower_than_fallback_crf():
    """Higher CRF = stronger compression; fallback compresses more."""
    assert INITIAL_CRF < FALLBACK_CRF


# ---------------------------------------------------------------------------
# VideoEncodeResult dataclass
# ---------------------------------------------------------------------------


def test_video_result_is_frozen():
    r = VideoEncodeResult(
        path="/tmp/x.mp4", bytes_written=100, n_frames=5,
        fps=20, crf_used=28, elapsed_s=0.1,
    )
    with pytest.raises(AttributeError):
        r.bytes_written = 999  # type: ignore[misc]


def test_video_result_rejects_negative_bytes():
    with pytest.raises(ValueError, match="bytes_written"):
        VideoEncodeResult(
            path="x", bytes_written=-1, n_frames=1, fps=20, crf_used=28,
            elapsed_s=0.0,
        )


def test_video_result_rejects_negative_frames():
    with pytest.raises(ValueError, match="n_frames"):
        VideoEncodeResult(
            path="x", bytes_written=0, n_frames=-1, fps=20, crf_used=28,
            elapsed_s=0.0,
        )


def test_video_result_rejects_zero_fps():
    with pytest.raises(ValueError, match="fps"):
        VideoEncodeResult(
            path="x", bytes_written=0, n_frames=0, fps=0, crf_used=28,
            elapsed_s=0.0,
        )


# ---------------------------------------------------------------------------
# encode_episode_video — input validation
# ---------------------------------------------------------------------------


class _FakeFrame:
    """Minimal numpy-array stand-in: shape + tobytes()."""

    def __init__(self, height: int = 8, width: int = 8, channels: int = 3):
        self.shape = (height, width, channels)
        self._bytes = b"\x00" * (height * width * channels)

    def tobytes(self) -> bytes:
        return self._bytes


def test_encode_rejects_zero_fps(tmp_path):
    with pytest.raises(ValueError, match="fps"):
        encode_episode_video(
            frames=[_FakeFrame()], output_path=tmp_path / "x.mp4",
            fps=0, ffmpeg_caller=_dummy_caller,
        )


def test_encode_rejects_zero_max_bytes(tmp_path):
    with pytest.raises(ValueError, match="max_bytes"):
        encode_episode_video(
            frames=[_FakeFrame()], output_path=tmp_path / "x.mp4",
            max_bytes=0, ffmpeg_caller=_dummy_caller,
        )


def test_encode_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError, match="frames cannot be empty"):
        encode_episode_video(
            frames=[], output_path=tmp_path / "x.mp4",
            ffmpeg_caller=_dummy_caller,
        )


def test_encode_rejects_2d_frames(tmp_path):
    """Frames must be HxWx3, not HxW."""
    bad = _FakeFrame()
    bad.shape = (8, 8)  # 2D, not 3D
    with pytest.raises(ValueError, match="HxWx3"):
        encode_episode_video(
            frames=[bad], output_path=tmp_path / "x.mp4",
            ffmpeg_caller=_dummy_caller,
        )


def test_encode_rejects_non_rgb_frames(tmp_path):
    """Frames must be 3-channel (RGB), not RGBA or grayscale-3D."""
    bad = _FakeFrame(channels=4)
    with pytest.raises(ValueError, match="3 channels"):
        encode_episode_video(
            frames=[bad], output_path=tmp_path / "x.mp4",
            ffmpeg_caller=_dummy_caller,
        )


# ---------------------------------------------------------------------------
# encode_episode_video — happy path with stub ffmpeg_caller
# ---------------------------------------------------------------------------


def _make_caller(returncode: int = 0, *, file_size: int = 1000, stderr: bytes = b""):
    """Create a stub FfmpegCaller that pretends to write a file of given size."""
    def _caller(cmd: list[str], frames_bytes: bytes) -> subprocess.CompletedProcess:
        # Simulate file creation (find -i then output_path == cmd[-1])
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * file_size)
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout=b"", stderr=stderr,
        )

    _caller.calls = []  # type: ignore[attr-defined]
    return _caller


def _dummy_caller(cmd: list[str], frames_bytes: bytes) -> subprocess.CompletedProcess:
    """Tiny success caller for validation tests that don't exercise the call path."""
    return _make_caller(returncode=0, file_size=10)(cmd, frames_bytes)


def test_encode_happy_path_writes_file(tmp_path):
    out = tmp_path / "ep.mp4"
    caller = _make_caller(returncode=0, file_size=500)
    result = encode_episode_video(
        frames=[_FakeFrame() for _ in range(5)],
        output_path=out, ffmpeg_caller=caller,
    )
    assert isinstance(result, VideoEncodeResult)
    assert result.bytes_written == 500
    assert result.n_frames == 5
    assert result.crf_used == INITIAL_CRF
    assert out.exists()


def test_encode_creates_parent_dir(tmp_path):
    out = tmp_path / "deep" / "dir" / "ep.mp4"
    caller = _make_caller(returncode=0, file_size=10)
    encode_episode_video(
        frames=[_FakeFrame()],
        output_path=out, ffmpeg_caller=caller,
    )
    assert out.parent.exists()


def test_encode_retries_with_fallback_crf_when_over_cap(tmp_path):
    """First pass over cap → re-encode at FALLBACK_CRF."""
    out = tmp_path / "huge.mp4"
    sizes = iter([20_000_000, 5_000_000])  # 20MB then 5MB

    def _shrinking_caller(cmd, frames_bytes):
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * next(sizes))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b"",
        )

    result = encode_episode_video(
        frames=[_FakeFrame()],
        output_path=out, max_bytes=10 * 1024 * 1024,
        ffmpeg_caller=_shrinking_caller,
    )
    assert result.crf_used == FALLBACK_CRF
    assert result.bytes_written == 5_000_000


def test_encode_skips_retry_when_first_pass_under_cap(tmp_path):
    """If first pass fits, do NOT re-encode."""
    out = tmp_path / "small.mp4"
    call_count = {"n": 0}

    def _counting_caller(cmd, frames_bytes):
        call_count["n"] += 1
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * 100)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b"",
        )

    encode_episode_video(
        frames=[_FakeFrame()],
        output_path=out, max_bytes=10_000,
        ffmpeg_caller=_counting_caller,
    )
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# encode_episode_video — failure paths
# ---------------------------------------------------------------------------


def test_encode_raises_on_ffmpeg_nonzero_returncode(tmp_path):
    out = tmp_path / "ep.mp4"

    def _failing_caller(cmd, frames_bytes):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout=b"", stderr=b"some ffmpeg error",
        )

    with pytest.raises(VideoEncodeError, match="exited with code 1"):
        encode_episode_video(
            frames=[_FakeFrame()],
            output_path=out, ffmpeg_caller=_failing_caller,
        )


def test_encode_raises_when_output_not_created(tmp_path):
    """ffmpeg returncode=0 but file wasn't written → VideoEncodeError."""
    out = tmp_path / "ep.mp4"

    def _liar_caller(cmd, frames_bytes):
        # Return success but DON'T write the file
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b"",
        )

    with pytest.raises(VideoEncodeError, match="output .* not created"):
        encode_episode_video(
            frames=[_FakeFrame()],
            output_path=out, ffmpeg_caller=_liar_caller,
        )


def test_encode_raises_ffmpegmissing_when_binary_absent(tmp_path, monkeypatch):
    """No ffmpeg on PATH + no caller injected → FfmpegMissingError."""
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: None)
    with pytest.raises(FfmpegMissingError, match="not found on PATH"):
        encode_episode_video(
            frames=[_FakeFrame()],
            output_path=tmp_path / "ep.mp4",
            # ffmpeg_caller=None → triggers the PATH check
        )


# ---------------------------------------------------------------------------
# Command construction (no actual ffmpeg invocation)
# ---------------------------------------------------------------------------


def test_command_contains_expected_flags(tmp_path):
    captured = {}

    def _spy(cmd, frames_bytes):
        captured["cmd"] = cmd
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * 100)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b"",
        )

    encode_episode_video(
        frames=[_FakeFrame(height=480, width=640)],
        output_path=tmp_path / "ep.mp4", fps=30,
        ffmpeg_caller=_spy,
    )
    cmd = captured["cmd"]
    assert "ffmpeg" in cmd[0] or cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert "rawvideo" in cmd
    assert "rgb24" in cmd
    assert "640x480" in cmd
    assert "30" in cmd  # fps
    assert "libx264" in cmd
    assert "-crf" in cmd
    assert str(INITIAL_CRF) in cmd


# ---------------------------------------------------------------------------
# episode_video_path helper
# ---------------------------------------------------------------------------


def test_episode_video_path_format(tmp_path):
    p = episode_video_path(
        output_dir=tmp_path,
        task_id="libero_spatial",
        episode_index=3,
    )
    assert p == tmp_path / "videos" / "libero_spatial_episode_3.mp4"


def test_episode_video_path_sanitizes_slashes(tmp_path):
    p = episode_video_path(
        output_dir=tmp_path,
        task_id="libero/spatial/task1",  # path-traversal style
        episode_index=0,
    )
    # Slashes replaced with underscores
    assert "/" not in p.name
    assert "libero_spatial_task1_episode_0.mp4" == p.name


def test_episode_video_path_does_not_create_file(tmp_path):
    p = episode_video_path(
        output_dir=tmp_path, task_id="t", episode_index=0,
    )
    assert not p.exists()
    # And doesn't create the parent either
    assert not p.parent.exists()
