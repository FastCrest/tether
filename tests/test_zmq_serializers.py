"""Tests for ZMQ serializers (Lift #2 Day 3).

Validates JPEG round-trip quality, numpy serialization, schema_version,
whitelist behavior, and error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")

from tether.runtime.transports.zmq.serializers import (
    JPEG_WHITELIST,
    SCHEMA_VERSION,
    JpegEncodingError,
    MsgpackDecodingError,
    decode_observation,
    encode_observation,
)


# ── JPEG round-trip ──────────────────────────────────────────────────


def test_jpeg_round_trip_quality():
    """Whitelisted image key → JPEG compress → decode. cos ≥ 0.99 at q=85.

    Note: pure random noise gets cos ~0.91 due to JPEG's lossy DCT; real
    camera images (smooth gradients) get cos ~0.999+. Use a gradient pattern.
    """
    # Gradient pattern — realistic for camera images (smooth, not noise)
    x = np.linspace(0, 255, 224, dtype=np.uint8)
    img = np.stack([np.tile(x, (224, 1))] * 3, axis=-1)
    obs = {"agentview_image": img}
    encoded = encode_observation(obs)
    decoded = decode_observation(encoded)

    result = decoded["agentview_image"]
    assert result.shape == img.shape
    assert result.dtype == np.uint8

    flat_orig = img.flatten().astype(np.float32)
    flat_result = result.flatten().astype(np.float32)
    cos = np.dot(flat_orig, flat_result) / (np.linalg.norm(flat_orig) * np.linalg.norm(flat_result))
    assert cos >= 0.999, f"JPEG cos similarity = {cos:.6f} (expected >= 0.999)"


def test_jpeg_compression_ratio():
    """JPEG at q=85 should be significantly smaller than raw pixels."""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    raw_size = img.nbytes
    encoded = encode_observation({"agentview_image": img})
    assert len(encoded) < raw_size * 0.5  # at least 2× compression


# ── numpy round-trip ─────────────────────────────────────────────────


def test_float32_array_round_trip():
    arr = np.random.randn(1, 50, 7).astype(np.float32)
    obs = {"actions": arr}
    decoded = decode_observation(encode_observation(obs))
    np.testing.assert_array_equal(decoded["actions"], arr)


def test_float16_array_round_trip():
    arr = np.random.randn(10, 32).astype(np.float16)
    obs = {"embeddings": arr}
    decoded = decode_observation(encode_observation(obs))
    np.testing.assert_array_equal(decoded["embeddings"], arr)


# ── scalar / string round-trip ───────────────────────────────────────


def test_scalar_and_string_round_trip():
    obs = {"task": "pick up the cup", "step": 42, "done": True, "reward": 0.5}
    decoded = decode_observation(encode_observation(obs))
    assert decoded["task"] == "pick up the cup"
    assert decoded["step"] == 42
    assert decoded["done"] is True
    assert decoded["reward"] == 0.5


# ── whitelist behavior ───────────────────────────────────────────────


def test_whitelisted_key_gets_jpeg():
    """Keys in JPEG_WHITELIST get JPEG-compressed (smaller encoded size)."""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs_whitelisted = {"agentview_image": img}
    obs_non_whitelisted = {"custom_sensor": img}

    enc_wl = encode_observation(obs_whitelisted)
    enc_nwl = encode_observation(obs_non_whitelisted)

    # JPEG-compressed should be smaller than raw numpy serialization
    assert len(enc_wl) < len(enc_nwl)


def test_non_whitelisted_uint8_3d_warns(caplog):
    """Non-whitelisted uint8 3D array triggers a one-time warning."""
    from tether.runtime.transports.zmq.serializers import _warned_keys
    _warned_keys.discard("mystery_camera")  # reset for this test

    import logging
    with caplog.at_level(logging.WARNING):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        encode_observation({"mystery_camera": img})

    assert any("mystery_camera" in r.message for r in caplog.records)


def test_non_image_array_not_jpeg():
    """float32 arrays are never JPEG-compressed regardless of ndim."""
    arr = np.random.randn(224, 224, 3).astype(np.float32)
    obs = {"agentview_image": arr}
    decoded = decode_observation(encode_observation(obs))
    np.testing.assert_array_almost_equal(decoded["agentview_image"], arr)


# ── schema_version ───────────────────────────────────────────────────


def test_encoded_includes_schema_version():
    import msgpack
    obs = {"state": np.zeros(8, dtype=np.float32)}
    raw = msgpack.unpackb(encode_observation(obs), raw=False)
    assert raw["schema_version"] == SCHEMA_VERSION


# ── error paths ──────────────────────────────────────────────────────


def test_msgpack_decode_error():
    with pytest.raises(MsgpackDecodingError):
        decode_observation(b"not valid msgpack \x00\x01\x02")


# ── 3-camera dict round-trip ─────────────────────────────────────────


def test_multi_camera_round_trip():
    obs = {
        "agentview_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "cam_high": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "task": "pick up the red cup",
    }
    decoded = decode_observation(encode_observation(obs))

    assert decoded["agentview_image"].shape == (224, 224, 3)
    assert decoded["robot0_eye_in_hand_image"].shape == (224, 224, 3)
    assert decoded["cam_high"].shape == (224, 224, 3)
    np.testing.assert_array_equal(decoded["robot0_eef_pos"], obs["robot0_eef_pos"])
    assert decoded["task"] == "pick up the red cup"
