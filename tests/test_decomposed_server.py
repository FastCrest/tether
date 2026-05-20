"""Tests for src/reflex/runtime/decomposed_server.py — Pi05DecomposedServer
wrapper that closes the B.4/B.5 measurement gap.

Per ADR 2026-04-25-decomposed-dispatch-via-reflex-serve: the wrapper
exposes the ReflexServer interface (predict_from_base64_async,
run_batch, _action_guard, etc.) so decomposed exports can serve through
create_app + the existing /act handler + all wedges.

These tests stub Pi05DecomposedInference so they don't require ORT or
GPU; they exercise the prep + dispatch contract.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest

from reflex.runtime.decomposed_server import (
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_LANG_SEQ_LEN,
    DEFAULT_MAX_ACTION_DIM,
    Pi05DecomposedServer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_hf_normalizer_fallback(monkeypatch):
    """Unit tests should not fetch teacher normalizers from HuggingFace."""
    monkeypatch.setattr(Pi05DecomposedServer, "_infer_teacher_ref", lambda self: None)


def _make_export_dir(
    tmp_path: Path,
    *,
    export_kind: str = "decomposed",
    action_dim: int = 7,
    chunk_size: int = 50,
) -> Path:
    """Build a minimal export dir with a reflex_config.json."""
    p = tmp_path / "export"
    p.mkdir()
    config = {
        # Keep this fixture local-only. A real HF repo id triggers teacher
        # normalizer fallback during server.load(), which makes these unit
        # tests depend on network/cache state.
        "model_id": "test-pi05-decomposed",
        "model_type": "pi05_decomposed_student",
        "target": "desktop",
        "num_denoising_steps": 1,
        "chunk_size": chunk_size,
        "action_chunk_size": chunk_size,
        "action_dim": action_dim,
        "opset": 19,
        "export_kind": export_kind,
        "decomposed": {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "max_action_dim": 32,
        },
    }
    (p / "reflex_config.json").write_text(json.dumps(config))
    return p


def _make_b64_image(width: int = 100, height: int = 80) -> str:
    """Generate a synthetic JPEG-encoded base64 image."""
    from PIL import Image
    arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _StubInference:
    """Minimal stand-in for Pi05DecomposedInference. Captures call args
    + returns a known-shape actions array."""

    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []
        self.export_dir = kwargs.get("export_dir")
        self.providers = kwargs.get("providers")
        # Mimic the _expert_session lang_seq probe -- present but None
        # so the probe falls back to default DEFAULT_LANG_SEQ_LEN.
        self._expert_session = None

    def predict_action_chunk(self, **kwargs):
        self.calls.append(kwargs)
        # Return (B=1, chunk=50, max_action_dim=32) zeros
        return np.zeros((1, 50, 32), dtype=np.float32)


class _StubTokenizer:
    """Minimal HF-tokenizer-like stub."""

    vocab_size = 1024

    def __call__(self, instruction, **kwargs):
        seq = kwargs.get("max_length", 16)
        # All-zeros tokens + zero mask (shapes correct for the prep test)
        return {
            "input_ids": np.zeros((1, seq), dtype=np.int64),
            "attention_mask": np.zeros((1, seq), dtype=bool),
        }


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_load_rejects_missing_config(tmp_path):
    p = tmp_path / "no-config"
    p.mkdir()
    server = Pi05DecomposedServer(p)
    with pytest.raises(FileNotFoundError, match="reflex_config.json"):
        server.load()


def test_load_rejects_wrong_export_kind(tmp_path, monkeypatch):
    p = _make_export_dir(tmp_path, export_kind="monolithic")
    monkeypatch.setattr(
        "reflex.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        _StubInference,
    )
    server = Pi05DecomposedServer(p)
    with pytest.raises(ValueError, match="export_kind='decomposed'"):
        server.load()


def test_load_succeeds_with_valid_config(tmp_path, monkeypatch):
    p = _make_export_dir(tmp_path)
    monkeypatch.setattr(
        "reflex.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        _StubInference,
    )
    server = Pi05DecomposedServer(p)
    server.load()
    assert server.ready
    assert server.action_dim == 7
    assert server.chunk_size == 50
    assert server.max_action_dim == 32
    assert server._inference is not None


def test_load_reads_action_dim_from_config(tmp_path, monkeypatch):
    p = _make_export_dir(tmp_path, action_dim=14)
    monkeypatch.setattr(
        "reflex.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        _StubInference,
    )
    server = Pi05DecomposedServer(p)
    server.load()
    assert server.action_dim == 14


# ---------------------------------------------------------------------------
# predict_from_base64 -- happy path (with stub tokenizer + inference)
# ---------------------------------------------------------------------------


def _build_loaded_server(tmp_path, monkeypatch) -> Pi05DecomposedServer:
    p = _make_export_dir(tmp_path)
    stub = _StubInference()
    monkeypatch.setattr(
        "reflex.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        lambda **kwargs: stub,
    )
    server = Pi05DecomposedServer(p)
    server.load()
    # Inject stub tokenizer (skip real HF load)
    server._tokenizer = _StubTokenizer()
    return server


def test_predict_from_base64_returns_actions_dict(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    result = server.predict_from_base64(
        image_b64=_make_b64_image(),
        instruction="pick up the cup",
        state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert "actions" in result
    assert "action_dim" in result
    assert "latency_ms" in result
    assert result["action_dim"] == 7
    # Shape: (chunk_size, action_dim) = (50, 7)
    assert len(result["actions"]) == 50
    assert len(result["actions"][0]) == 7


def test_predict_from_base64_invalid_image_returns_error(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    result = server.predict_from_base64(
        image_b64="not-valid-base64-data!!!",
        instruction="x",
        state=None,
    )
    assert "error" in result
    assert "Failed to decode image" in result["error"]


def test_predict_handles_missing_image(tmp_path, monkeypatch):
    """When image is None, the server pads with -1 (matching the missing-
    camera convention used by SmolVLA training)."""
    server = _build_loaded_server(tmp_path, monkeypatch)
    result = server.predict_from_base64(
        image_b64=None, instruction="x", state=None,
    )
    # Doesn't error -- pads + still returns actions
    assert "actions" in result


def test_predict_inference_call_passes_correct_shapes(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    server.predict_from_base64(
        image_b64=_make_b64_image(width=64, height=128),
        instruction="x",
        state=[0.1, 0.2, 0.3],
    )
    call = server._inference.calls[0]
    # Image shape: (1, 3, 224, 224)
    assert call["img_base"].shape == (1, 3, DEFAULT_CAMERA_RESOLUTION, DEFAULT_CAMERA_RESOLUTION)
    assert call["img_base"].dtype == np.float32
    # Mask shape: (1,)
    assert call["mask_base"].shape == (1,)
    assert call["mask_base"].dtype == bool
    # Padded cameras: shape match base; image=-1; mask=False
    assert call["img_wrist_l"].shape == call["img_base"].shape
    assert (call["img_wrist_l"] == -1.0).all()
    assert not call["mask_wrist_l"].any()
    # Lang shape: (1, lang_seq_len)
    assert call["lang_tokens"].shape == (1, DEFAULT_LANG_SEQ_LEN)
    assert call["lang_tokens"].dtype == np.int64
    # State shape: (1, max_action_dim=32)
    assert call["state"].shape == (1, DEFAULT_MAX_ACTION_DIM)
    # Padded with zeros
    assert (call["state"][0, 3:] == 0).all()
    # State is np.float32 (per the assertion above on the array's dtype, and
    # the upstream marshal layer's float32 cast). Comparing np.float32(0.1)
    # to Python's 0.1 (which is float64) yields False under == because the
    # bit patterns differ at the boundary precision. Use np.isclose.
    assert np.isclose(call["state"][0, 0], 0.1)


def test_predict_state_truncated_when_too_long(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    long_state = list(range(40))  # >32
    server.predict_from_base64(
        image_b64=_make_b64_image(),
        instruction="x",
        state=[float(x) for x in long_state],
    )
    call = server._inference.calls[0]
    # Truncated to max_action_dim=32
    assert call["state"].shape == (1, 32)
    assert call["state"][0, 31] == 31.0  # last allowed value


# ---------------------------------------------------------------------------
# Async + run_batch interface
# ---------------------------------------------------------------------------


def test_predict_from_base64_async_returns_same_shape(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    result = asyncio.run(server.predict_from_base64_async(
        image_b64=_make_b64_image(), instruction="x", state=None,
    ))
    assert "actions" in result
    assert len(result["actions"]) == 50


def test_run_batch_returns_one_result_per_request(tmp_path, monkeypatch):
    """PolicyRuntime's run_batch_callback contract: in-order results."""
    server = _build_loaded_server(tmp_path, monkeypatch)

    class _Req:
        def __init__(self, image, instruction, state):
            self.image = image
            self.instruction = instruction
            self.state = state

    requests = [
        _Req(_make_b64_image(), "x", None),
        _Req(_make_b64_image(), "y", None),
        _Req(_make_b64_image(), "z", None),
    ]
    results = asyncio.run(server.run_batch(requests))
    assert len(results) == 3
    for r in results:
        assert "actions" in r


def test_predict_returns_error_when_not_ready(tmp_path):
    """predict before load() -> returns error envelope."""
    p = _make_export_dir(tmp_path)
    server = Pi05DecomposedServer(p)
    # Don't call load()
    result = server.predict_from_base64(image_b64=None, instruction="", state=None)
    assert "error" in result
    assert "not ready" in result["error"]


# ---------------------------------------------------------------------------
# Inference mode reporting
# ---------------------------------------------------------------------------


def test_inference_mode_uninitialized_before_load(tmp_path):
    p = _make_export_dir(tmp_path)
    server = Pi05DecomposedServer(p)
    assert server._inference_mode == "uninitialized"


def test_inference_mode_cuda_after_load(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    assert server._inference_mode == "onnx_cuda_decomposed"


def test_inference_mode_cpu_after_load(tmp_path, monkeypatch):
    p = _make_export_dir(tmp_path)
    monkeypatch.setattr(
        "reflex.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        _StubInference,
    )
    server = Pi05DecomposedServer(p, device="cpu")
    server.load()
    assert server._inference_mode == "onnx_cpu_decomposed"


# ---------------------------------------------------------------------------
# Image prep -- standalone unit
# ---------------------------------------------------------------------------


def test_prep_image_resizes_to_camera_resolution(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    arr = (np.random.rand(80, 200, 3) * 255).astype(np.uint8)
    img, mask = server._prep_image(arr)
    assert img.shape == (1, 3, DEFAULT_CAMERA_RESOLUTION, DEFAULT_CAMERA_RESOLUTION)
    assert img.dtype == np.float32
    assert (img >= 0.0).all() and (img <= 1.0).all()
    assert mask.tolist() == [True]


def test_prep_image_none_returns_padded_neg_one(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    img, mask = server._prep_image(None)
    assert img.shape == (1, 3, DEFAULT_CAMERA_RESOLUTION, DEFAULT_CAMERA_RESOLUTION)
    assert (img == -1.0).all()
    assert not mask.any()


def test_prep_image_rejects_non_3channel(tmp_path, monkeypatch):
    server = _build_loaded_server(tmp_path, monkeypatch)
    bad = np.zeros((100, 100), dtype=np.uint8)  # 2D
    with pytest.raises(ValueError, match="HxWx3"):
        server._prep_image(bad)
