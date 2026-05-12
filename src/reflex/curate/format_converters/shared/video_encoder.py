"""H.264 mp4 encoder for the LeRobot v3 video output path.

Inputs: a list of image frames (each as bytes — base64-decoded already, OR
raw image bytes from disk OR PIL.Image objects).
Output: a single .mp4 file at the requested fps.

Implementation: imageio-ffmpeg ships its own ffmpeg binary so the customer
doesn't need a system-wide install. Pillow decodes per-frame; imageio's
writer pipes raw RGB frames to ffmpeg via stdin.

Phase 1: H.264 only (broad compat). Phase 1.5 can add AV1 for storage-
conscious customers (per LeRobot v3 spec recommendations + planned-
improvements section).
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_CODEC = "libx264"          # H.264 (royalty-free for OSS distribution)
DEFAULT_PIXEL_FORMAT = "yuv420p"   # broadest compat
DEFAULT_QUALITY = 18                # CRF 18 ≈ visually lossless


class VideoEncoderUnavailable(ImportError):
    """Raised when ffmpeg-python / imageio-ffmpeg are missing.

    Caller should handle gracefully — the LeRobot v3 converter falls back
    to data-only output when this fires.
    """


def encode_frames_to_mp4(
    *,
    frames: list[bytes],
    output_path: str | Path,
    fps: int = 30,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    quality: int = DEFAULT_QUALITY,
) -> int:
    """Encode a list of image frames (raw bytes — PNG/JPEG/etc.) to H.264 mp4.

    Args:
        frames: list of image bytes — each element is one frame, decodable
            by Pillow. Order = display order.
        output_path: target .mp4 path. Parent dir is created if missing.
        fps: target frame rate (matches the source recording fps).
        codec: ffmpeg codec name.
        pixel_format: ffmpeg pixel format.
        quality: H.264 CRF (lower = higher quality; 18 ≈ visually lossless,
            23 = default, 28 = lower quality / smaller file).

    Returns the bytes written.

    Raises:
        VideoEncoderUnavailable: if imageio-ffmpeg / Pillow not installed.
        ValueError: on empty frames or undecodable frames.
    """
    if not frames:
        raise ValueError("encode_frames_to_mp4: no frames to encode")

    try:
        import imageio_ffmpeg
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise VideoEncoderUnavailable(
            "video encoding requires imageio-ffmpeg + Pillow + numpy: "
            "pip install 'reflex-vla[curate-video]'"
        ) from exc

    # Decode first frame to learn the size + sanity-check.
    try:
        first = Image.open(io.BytesIO(frames[0])).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"first frame undecodable: {exc}") from exc
    width, height = first.size

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(out),
        size=(width, height),
        fps=int(fps),
        codec=codec,
        pix_fmt_out=pixel_format,
        quality=None,
        bitrate=None,
        macro_block_size=None,  # disables yuv420p auto-padding warning on odd dims
        ffmpeg_log_level="error",
        # ffmpeg input args. -crf controls H.264 quality; map our `quality`.
        output_params=["-crf", str(int(quality)), "-preset", "veryfast"],
    )
    writer.send(None)  # initialize the generator

    try:
        # Send each frame as a flat RGB byte buffer.
        writer.send(np.asarray(first, dtype=np.uint8).tobytes())
        for raw in frames[1:]:
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("video_encoder.skipping_undecodable_frame: %s", exc)
                continue
            if img.size != (width, height):
                img = img.resize((width, height))
            writer.send(np.asarray(img, dtype=np.uint8).tobytes())
    finally:
        writer.close()

    return out.stat().st_size


def decode_image_field(image_b64: Any) -> bytes | None:
    """Decode the JSONL `image_b64` field to raw image bytes.

    Returns None when the field is absent / hash-only / undecodable. Hash-only
    redaction in the recorder writes a SHA-256 hex (~64 chars), so anything
    below 200 chars is heuristically not a real image.
    """
    if not isinstance(image_b64, str) or len(image_b64) < 200:
        return None
    try:
        return base64.b64decode(image_b64, validate=False)
    except Exception:  # noqa: BLE001
        return None


def collect_frames_from_rows(rows: list[dict[str, Any]]) -> list[bytes]:
    """Per-row image_b64 → raw frames, in input order. Skips undecodable rows."""
    out: list[bytes] = []
    for r in rows:
        decoded = decode_image_field(r.get("image_b64"))
        if decoded is not None:
            out.append(decoded)
    return out


__all__ = [
    "DEFAULT_CODEC",
    "DEFAULT_PIXEL_FORMAT",
    "DEFAULT_QUALITY",
    "VideoEncoderUnavailable",
    "collect_frames_from_rows",
    "decode_image_field",
    "encode_frames_to_mp4",
]
