"""Serializers for ZMQ transport — msgpack + JPEG-on-wire (Lift #2 Day 3).

Ported from FluxVLA ``serializers.py:1-250`` (Apache-2.0, LimX Dynamics)
with three tightenings per Z-2 and Z-3 from the research sidecar:

1. **JPEG whitelist** — only ndim==3 + uint8 arrays whose key is in the
   whitelist get JPEG-compressed. Other arrays serialize as raw numpy bytes.
   This prevents silent 600KB payloads from non-image arrays.
2. **One-time warning** — if a key looks like it could be an image (ndim==3,
   uint8) but isn't in the whitelist, log a warning once per session.
3. **schema_version** — every encoded message includes ``schema_version: 1``
   at the top level.
"""
from __future__ import annotations

import io
import logging
import warnings
from typing import Any

import msgpack
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Camera key whitelist — union of FluxVLA LIBERO/Aloha keys + tether embodiment
# camera names. Frozenset for O(1) lookup.
JPEG_WHITELIST: frozenset[str] = frozenset({
    # FluxVLA LIBERO/Aloha keys
    "cam_high", "agentview_image", "robot0_eye_in_hand_image",
    "cam_left_wrist", "cam_right_wrist",
    # tether embodiment camera names
    "base", "wrist_l", "wrist_r", "external",
    "wrist_left", "wrist_right",
    # Common lerobot obs keys
    "observation.images.image", "observation.images.image2",
    "observation.images.image3",
    "image", "image2", "image3",
})

_warned_keys: set[str] = set()


class JpegEncodingError(RuntimeError):
    """Raised when JPEG encoding fails (e.g. wrong dtype or shape)."""


class MsgpackDecodingError(RuntimeError):
    """Raised when msgpack decoding fails."""


def _should_jpeg_compress(key: str, value: Any) -> bool:
    """Check if a value should be JPEG-compressed for the wire."""
    if not isinstance(value, np.ndarray):
        return False
    if value.ndim != 3 or value.dtype != np.uint8:
        return False
    if key in JPEG_WHITELIST:
        return True
    # One-time warning for unwhitelisted image-like arrays
    if key not in _warned_keys:
        _warned_keys.add(key)
        logger.warning(
            "ZMQ serializer: key %r looks like an image (ndim=3, uint8, shape=%s) "
            "but isn't in the JPEG whitelist. Sending as raw numpy bytes (~%d KB). "
            "Add it to JPEG_WHITELIST in serializers.py to enable compression.",
            key, value.shape, value.nbytes // 1024,
        )
    return False


def encode_observation(obs: dict[str, Any], *, jpeg_quality: int = 85) -> bytes:
    """Encode an observation dict to msgpack bytes with JPEG-compressed images.

    Args:
        obs: Observation dict. Values can be numpy arrays, scalars, strings,
            or nested dicts.
        jpeg_quality: JPEG quality for whitelisted image keys (1-100).

    Returns:
        msgpack-encoded bytes with ``schema_version`` field.
    """
    import cv2

    encoded: dict[str, Any] = {"schema_version": SCHEMA_VERSION}

    for key, value in obs.items():
        if _should_jpeg_compress(key, value):
            success, jpeg_buf = cv2.imencode(
                ".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not success:
                raise JpegEncodingError(f"JPEG encoding failed for key {key!r}")
            encoded[key] = {"__jpeg__": True, "data": jpeg_buf.tobytes(), "shape": list(value.shape)}
        elif isinstance(value, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            encoded[key] = {"__numpy__": True, "data": buf.getvalue(), "dtype": str(value.dtype)}
        elif isinstance(value, (int, float, str, bool)):
            encoded[key] = value
        elif isinstance(value, dict):
            encoded[key] = value
        elif isinstance(value, (list, tuple)):
            encoded[key] = value
        else:
            encoded[key] = str(value)

    return msgpack.packb(encoded, use_bin_type=True)


def decode_observation(data: bytes) -> dict[str, Any]:
    """Decode msgpack bytes back to an observation dict.

    Reverses JPEG compression and numpy serialization.

    Args:
        data: msgpack-encoded bytes from ``encode_observation``.

    Returns:
        Observation dict with numpy arrays restored.

    Raises:
        MsgpackDecodingError: if msgpack decoding fails.
    """
    import cv2

    try:
        raw = msgpack.unpackb(data, raw=False)
    except Exception as e:
        raise MsgpackDecodingError(f"msgpack decode failed: {e}") from e

    obs: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "schema_version":
            continue
        if isinstance(value, dict):
            if value.get("__jpeg__"):
                jpeg_bytes = np.frombuffer(value["data"], dtype=np.uint8)
                img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise JpegEncodingError(f"JPEG decode failed for key {key!r}")
                obs[key] = img
            elif value.get("__numpy__"):
                buf = io.BytesIO(value["data"])
                obs[key] = np.load(buf)
            else:
                obs[key] = value
        else:
            obs[key] = value

    return obs


def encode_actions(actions: np.ndarray) -> bytes:
    """Serialize action array to numpy bytes."""
    buf = io.BytesIO()
    np.save(buf, actions, allow_pickle=False)
    return buf.getvalue()


def decode_actions(data: bytes) -> np.ndarray:
    """Deserialize action array from numpy bytes."""
    return np.load(io.BytesIO(data))


__all__ = [
    "JPEG_WHITELIST",
    "SCHEMA_VERSION",
    "JpegEncodingError",
    "MsgpackDecodingError",
    "decode_actions",
    "decode_observation",
    "encode_actions",
    "encode_observation",
]
