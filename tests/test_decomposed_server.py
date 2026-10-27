"""Tests for src/tether/runtime/decomposed_server.py — Pi05DecomposedServer
wrapper that closes the B.4/B.5 measurement gap.

Per ADR 2026-04-25-decomposed-dispatch-via-tether-serve: the wrapper
exposes the TetherServer interface (predict_from_base64_async,
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

from tether.correction.bid import BIDConfig
from tether.runtime.decomposed_server import (
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
    export_kind: str = "decomposed_onnx",
    action_dim: int = 7,
    chunk_size: int = 50,
    decomposed: dict | None = None,
) -> Path:
    """Build a minimal export dir with a tether_config.json."""
    p = tmp_path / "export"
    p.mkdir()
    (p / "vlm_prefix.onnx").write_bytes(b"prefix")
    (p / "expert_denoise.onnx").write_bytes(b"expert")
    config = {
        "schema_version": 1,
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
        "artifacts": [
            {"path": "vlm_prefix.onnx", "role": "model"},
            {"path": "expert_denoise.onnx", "role": "model"},
        ],
        "io_contract": {
            "inputs": [{"name": "noise", "dtype": "float32", "shape": [1, chunk_size, 32]}],
            "outputs": [{"name": "actions", "dtype": "float32", "shape": [1, chunk_size, 32]}],
        },
        "decomposed": decomposed if decomposed is not None else {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "max_action_dim": 32,
        },
    }
    (p / "tether_config.json").write_text(json.dumps(config))
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
        # The wrapper validates actual per-stage registration before readiness.
        # These are explicit fixture session facts, never hardware evidence.
        from types import SimpleNamespace
        from tether.runtime.ort_providers import provider_name
        stage_providers = kwargs.get("providers_by_stage") or {
            name: self.providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            for name in ("vlm_prefix", "expert_denoise")
        }
        for name, attribute in (("vlm_prefix", "_sess_prefix"), ("expert_denoise", "_sess_expert")):
            active = [provider_name(value) for value in stage_providers[name]]
            setattr(self, attribute, SimpleNamespace(
                get_providers=lambda active=active: active,
                get_provider_options=lambda: {}, get_inputs=lambda: [],
            ))
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
    with pytest.raises(FileNotFoundError, match="tether_config.json"):
        server.load()


def test_load_rejects_wrong_export_kind(tmp_path, monkeypatch):
    p = _make_export_dir(tmp_path, export_kind="monolithic_onnx")
    monkeypatch.setattr(
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        _StubInference,
    )
    server = Pi05DecomposedServer(p)
    with pytest.raises(ValueError, match="export_kind='decomposed_onnx'"):
        server.load()


def test_load_succeeds_with_valid_config(tmp_path, monkeypatch):
    p = _make_export_dir(tmp_path)
    monkeypatch.setattr(
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
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
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
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
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
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


def test_bid_wins_over_a2c2_during_prediction(tmp_path, monkeypatch):
    """BID selection must never be followed by A2C2 correction in Phase 1."""
    server = _build_loaded_server(tmp_path, monkeypatch)

    class _ExplodingA2C2Hook:
        def __init__(self):
            from types import SimpleNamespace

            self.calls = 0
            self.head = SimpleNamespace(
                config=SimpleNamespace(action_dim=7, obs_dim=256),
            )

        def maybe_apply_to_chunk(self, *, actions):
            self.calls += 1
            raise AssertionError("A2C2 must not run when BID is enabled")

    hook = _ExplodingA2C2Hook()
    server.set_a2c2_hook(hook)
    server.set_bid_config(BIDConfig(n_candidates=2))
    assert server._a2c2_hook is None

    # The mutex is order-independent: a later hook bind is ignored too.
    server.set_a2c2_hook(hook)
    assert server._a2c2_hook is None

    result = server.predict_from_base64(
        image_b64=_make_b64_image(),
        instruction="pick up the cup",
        state=[0.0] * 7,
    )
    assert "error" not in result
    assert result["bid_n_candidates"] == 2
    assert "a2c2_applied" not in result
    assert hook.calls == 0


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
        image_b64=None, instruction="x", state=[0.0] * 8,
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
        image_b64=_make_b64_image(), instruction="x", state=[0.0] * 8,
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
        _Req(_make_b64_image(), "x", [0.0] * 8),
        _Req(_make_b64_image(), "y", [0.0] * 8),
        _Req(_make_b64_image(), "z", [0.0] * 8),
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
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
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
    assert (img >= -1.0).all() and (img <= 1.0).all()  # SigLIP range, as LeRobot
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


# ---------------------------------------------------------------------------
# LeRobot pi0.5 input/output contract (prompt state, padding, images, action dim)
# ---------------------------------------------------------------------------


class _RecordingTokenizer(_StubTokenizer):
    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, instruction, **kwargs):
        self.prompts.append(instruction)
        return super().__call__(instruction, **kwargs)


def _server_with(tmp_path, monkeypatch, *, action_dim=7, decomposed=None, stats=None):
    p = _make_export_dir(tmp_path, action_dim=action_dim, decomposed=decomposed)
    if stats is not None:
        from safetensors.numpy import save_file
        save_file({k: np.asarray(v, np.float32) for k, v in stats.items()},
                  str(p / "policy_preprocessor_step_2_normalizer_processor.safetensors"))
    stub = _StubInference()
    monkeypatch.setattr(
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        lambda **kwargs: stub,
    )
    server = Pi05DecomposedServer(p)
    server.load()
    server._tokenizer = _RecordingTokenizer()
    return server


_BASE_BLOCK = {"vlm_prefix_onnx": "vlm_prefix.onnx", "expert_denoise_onnx": "expert_denoise.onnx"}


def test_prompt_carries_discretized_normalized_state_like_lerobot(tmp_path, monkeypatch):
    """LeRobot pi0.5 (processor_pi05.py) builds 'Task: ..., State: <256-bin ints>;\\nAction: '
    from the NORMALIZED state. Expected ints computed by hand from
    digitize(x, linspace(-1, 1, 257)[:-1]) - 1."""
    server = _server_with(
        tmp_path, monkeypatch,
        decomposed={**_BASE_BLOCK, "max_action_dim": 32, "state_in_prompt": True},
        stats={"observation.state.mean": [1.0] * 5, "observation.state.std": [2.0] * 5},
    )
    # normalized: (x - 1) / 2 -> [0.0, 0.5, -1.0, 1.0, -2.0]
    result = server.predict_from_base64(
        image_b64=_make_b64_image(), instruction="pick_up the\ncup", state=[1.0, 2.0, -1.0, 3.0, -3.0],
    )
    assert "actions" in result, result
    assert server._tokenizer.prompts == ["Task: pick up the cup, State: 128 192 0 255 -1;\nAction: "]


def test_legacy_export_without_field_puts_state_in_prompt(tmp_path, monkeypatch):
    server = _server_with(tmp_path, monkeypatch, decomposed={**_BASE_BLOCK, "max_action_dim": 32})
    server.predict_from_base64(image_b64=_make_b64_image(), instruction="x", state=[0.0, 0.0])
    assert server._tokenizer.prompts == ["Task: x, State: 128 128;\nAction: "]


@pytest.mark.parametrize("block", [
    {**_BASE_BLOCK, "max_action_dim": 32, "state_in_prompt": False},
    {**_BASE_BLOCK, "max_action_dim": 32, "expert_takes_state": True},  # legacy state-out export
])
def test_state_out_exports_keep_state_out_of_prompt(tmp_path, monkeypatch, block):
    server = _server_with(tmp_path, monkeypatch, decomposed=block)
    server.predict_from_base64(image_b64=_make_b64_image(), instruction="x", state=[0.3, 0.1])
    assert server._tokenizer.prompts == ["Task: x;\nAction: "]


def test_state_in_prompt_export_refuses_missing_state(tmp_path, monkeypatch):
    server = _server_with(tmp_path, monkeypatch, decomposed={**_BASE_BLOCK, "state_in_prompt": True})
    result = server.predict_from_base64(image_b64=_make_b64_image(), instruction="x", state=None)
    assert "must include state" in result["error"]


def test_tokenizer_pads_right_like_lerobot(tmp_path, monkeypatch):
    """PaliGemma's tokenizer defaults to left padding; LeRobot pads right."""
    from types import SimpleNamespace
    import transformers

    loaded = SimpleNamespace(padding_side="left", vocab_size=1)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: loaded)
    server = _server_with(tmp_path, monkeypatch)
    server._tokenizer = None
    server._ensure_tokenizer()
    assert server._tokenizer.padding_side == "right"


def test_prep_image_matches_lerobot_resize_with_pad_and_siglip_range(tmp_path, monkeypatch):
    """Reference: LeRobot resize_with_pad_torch (bilinear, align_corners=False,
    centered zero pad) then *2-1. A 2x4 image to 2x2 keeps aspect -> 1x2
    content row, one pad row on the bottom (divmod(1, 2) = (0, 1)).
    torch bilinear 4->2 samples src x = (d + 0.5) * 2 - 0.5 = 0.5, 2.5."""
    server = _build_loaded_server(tmp_path, monkeypatch)
    server.camera_resolution = 2
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    img[0, :, 0] = [0, 100, 200, 250]
    img[1, :, 0] = [50, 150, 250, 0]
    out, mask = server._prep_image(img)
    assert out.shape == (1, 3, 2, 2) and out.dtype == np.float32 and mask.tolist() == [True]
    # y: 2->1, src y = 0.5 -> rows averaged; x: src 0.5 and 2.5
    row = (img[0, :, 0].astype(np.float32) + img[1, :, 0]) / 2 / 255
    expected = np.array([(row[0] + row[1]) / 2, (row[2] + row[3]) / 2]) * 2 - 1
    np.testing.assert_allclose(out[0, 0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(out[0, 0, 1], [-1.0, -1.0])  # zero pad -> -1
    np.testing.assert_allclose(out[0, 1], [[-1.0, -1.0], [-1.0, -1.0]])  # black channel
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_legacy_padded_action_dim_returns_dataset_action_dim(tmp_path, monkeypatch):
    """Exports before decomposed.max_action_dim wrote action_dim=32 (padded).
    LeRobot slices to output_features['action'] (7); the normalizer stats say 7."""
    server = _server_with(
        tmp_path, monkeypatch, action_dim=32, decomposed=dict(_BASE_BLOCK),
        stats={"action.mean": [0.0] * 7, "action.std": [1.0] * 7},
    )
    assert server.action_dim == 7
    result = server.predict_from_base64(image_b64=_make_b64_image(), instruction="x", state=[0.0])
    assert result["action_dim"] == 7
    assert np.asarray(result["actions"]).shape == (50, 7)


def test_action_dim_disagreeing_with_normalizer_fails_load(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="disagrees with the action normalizer"):
        _server_with(
            tmp_path, monkeypatch, action_dim=6, decomposed={**_BASE_BLOCK, "max_action_dim": 32},
            stats={"action.mean": [0.0] * 7, "action.std": [1.0] * 7},
        )
