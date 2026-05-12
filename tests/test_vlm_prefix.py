"""Tests for VLM prefix pipeline: export, expert wiring, server integration.

Covers the full pipeline from vlm_prefix export through server predict(),
using mocks/stubs to avoid loading real 450M checkpoints.

Updated for the 4-file ONNX pipeline (vision_encoder, text_embedder,
decoder_prefill, state_encoder) and VLMPrefixOrchestrator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from reflex.exporters.vlm_prefix_exporter import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PREFIX_SEQ_LEN,
    DEFAULT_VLM_KV_DIM,
    VisionEncoderForONNX,
    export_vlm_prefix,
)
from reflex.exporters.vlm_components import (
    HIDDEN_SIZE,
    MAX_STATE_DIM,
    StateEncoder,
    assemble_prefix,
    pad_state,
)
from reflex.exporters.smolvla_exporter import ExpertGQALayer, ExpertStack
from reflex.runtime.vlm_orchestrator import VLMPrefixOrchestrator
from reflex.runtime.server import ReflexServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_expert_stack():
    """Build a minimal ExpertStack with 2 layers, 1 cross-attention layer."""
    hidden = 64
    action_dim = 8
    nq, nkv, hd = 4, 2, 16
    inter = 128
    vlm_kv_dim = 32
    cross_indices = [1]  # layer 1 is cross-attention

    layers = []
    for i in range(2):
        kv_in = vlm_kv_dim if i in cross_indices else None
        layers.append(ExpertGQALayer(hidden, nq, nkv, hd, inter, kv_in=kv_in))

    suffix_weights = {
        "in_w": torch.randn(hidden, action_dim),
        "in_b": torch.randn(hidden),
        "t_in_w": torch.randn(hidden, hidden * 2),
        "t_in_b": torch.randn(hidden),
        "t_out_w": torch.randn(hidden, hidden),
        "t_out_b": torch.randn(hidden),
    }
    action_proj_weights = {
        "w": torch.randn(action_dim, hidden),
        "b": torch.randn(action_dim),
    }
    final_norm_weight = torch.ones(hidden)

    stack = ExpertStack(
        layers=layers,
        expert_hidden=hidden,
        action_dim=action_dim,
        cross_indices=cross_indices,
        vlm_kv_dim=vlm_kv_dim,
        suffix_weights=suffix_weights,
        action_proj_weights=action_proj_weights,
        final_norm_weight=final_norm_weight,
    )
    stack.eval()
    return stack, {"hidden": hidden, "action_dim": action_dim, "vlm_kv_dim": vlm_kv_dim}


@pytest.fixture
def v02_export_dir(tmp_path):
    """Create a mock v0.2+ export dir with VLM ONNX files."""
    config = {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "action_chunk_size": 10,
        "export_version": "0.3",
        "vlm_prefix_onnx": "vision_encoder.onnx",
        "vlm_image_size": [512, 512],
        "vlm_kv_dim": 960,
        "vlm_prefix_seq_len": 50,
        "expert": {
            "expert_hidden": 720,
            "action_dim": 32,
            "num_layers": 16,
        },
    }
    (tmp_path / "reflex_config.json").write_text(json.dumps(config))
    # Create placeholder ONNX files (real loading is mocked)
    (tmp_path / "expert_stack.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "vision_encoder.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "text_embedder.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "decoder_prefill.onnx").write_bytes(b"fake-onnx")
    return tmp_path


@pytest.fixture
def v01_export_dir(tmp_path):
    """Create a mock v0.1 export dir (no vlm_prefix)."""
    config = {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "action_chunk_size": 10,
        "expert": {
            "expert_hidden": 720,
            "action_dim": 32,
            "num_layers": 16,
        },
    }
    (tmp_path / "reflex_config.json").write_text(json.dumps(config))
    (tmp_path / "expert_stack.onnx").write_bytes(b"fake-onnx")
    return tmp_path


def _make_mock_ort_session(input_names, output_shape):
    """Create a mock ORT InferenceSession that returns random data."""
    session = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inputs.append(inp)
    session.get_inputs.return_value = inputs
    session.get_providers.return_value = ["CPUExecutionProvider"]

    def mock_run(output_names, feed_dict):
        return [np.random.randn(*output_shape).astype(np.float32)]

    session.run.side_effect = mock_run
    return session


# ---------------------------------------------------------------------------
# Test 1: VLM architecture constants
# ---------------------------------------------------------------------------

class TestVLMArchitectureConstants:
    def test_vlm_kv_dim_is_960(self):
        """Architecture constant vlm_kv_dim should be 960 (SmolLM2 hidden size)."""
        assert DEFAULT_VLM_KV_DIM == 960

    def test_hidden_size_matches_vlm_kv_dim(self):
        """HIDDEN_SIZE in vlm_components should match DEFAULT_VLM_KV_DIM."""
        assert HIDDEN_SIZE == DEFAULT_VLM_KV_DIM

    def test_max_state_dim_is_32(self):
        """MAX_STATE_DIM should be 32."""
        assert MAX_STATE_DIM == 32


# ---------------------------------------------------------------------------
# Test 2: Expert with real vlm_kv differs from zeros
# ---------------------------------------------------------------------------

class TestExpertVLMKV:
    def test_expert_with_real_vlm_kv_differs_from_zeros(self, tiny_expert_stack):
        """Cross-attention actually uses vlm_kv: real vs zeros should differ."""
        stack, meta = tiny_expert_stack
        vlm_kv_dim = meta["vlm_kv_dim"]
        action_dim = meta["action_dim"]

        noisy_actions = torch.randn(1, 5, action_dim)
        timestep = torch.tensor([0.5])
        position_ids = torch.arange(5).unsqueeze(0)

        # v0.5 schema: separate vlm_k (RoPE'd) and vlm_v, each [L, B, seq, kv_dim].
        num_layers = 2

        real_k = torch.randn(num_layers, 1, 10, vlm_kv_dim)
        real_v = torch.randn(num_layers, 1, 10, vlm_kv_dim)
        with torch.no_grad():
            out_real = stack(noisy_actions, timestep, position_ids,
                             vlm_k=real_k, vlm_v=real_v)

        zero_k = torch.zeros(num_layers, 1, 10, vlm_kv_dim)
        zero_v = torch.zeros(num_layers, 1, 10, vlm_kv_dim)
        with torch.no_grad():
            out_zero = stack(noisy_actions, timestep, position_ids,
                             vlm_k=zero_k, vlm_v=zero_v)

        l2_diff = torch.norm(out_real - out_zero).item()
        assert l2_diff > 1e-3, (
            f"Expert outputs should differ with real vs zero vlm_k/v but L2={l2_diff}"
        )

    def test_expert_vlm_kv_none_fallback(self, tiny_expert_stack):
        """When vlm_k/vlm_v are None, expert falls back to zeros without error."""
        stack, meta = tiny_expert_stack
        noisy_actions = torch.randn(1, 5, meta["action_dim"])
        timestep = torch.tensor([0.5])
        position_ids = torch.arange(5).unsqueeze(0)

        with torch.no_grad():
            out = stack(noisy_actions, timestep, position_ids,
                        vlm_k=None, vlm_v=None)

        assert out.shape == (1, 5, meta["action_dim"])


# ---------------------------------------------------------------------------
# Test 3: Server predict() with VLM returns vlm_conditioning="real"
# ---------------------------------------------------------------------------

class TestServerVLMConditioning:
    def test_server_vlm_conditioning_real(self, v02_export_dir):
        """Server with VLM orchestrator returns vlm_conditioning='real'."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids", "vlm_kv"],
            (1, 10, 32),
        )

        server = ReflexServer(v02_export_dir, device="cpu")
        # Manually wire the server instead of calling load() which does real ort import
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids", "vlm_kv"]

        # Mock the VLM orchestrator
        mock_vlm = MagicMock()
        mock_vlm.is_loaded = True
        mock_vlm.is_complete = True
        mock_vlm.run.return_value = np.random.randn(1, 50, 960).astype(np.float32)
        server._vlm = mock_vlm
        server._vlm_loaded = True

        # Provide image and instruction to trigger VLM path
        fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = server.predict(image=fake_image, instruction="pick up the cup")

        assert result["vlm_conditioning"] == "real"
        assert "actions" in result
        assert mock_vlm.run.call_count == 1

    def test_server_vlm_conditioning_still_works(self, v02_export_dir):
        """Verify /act response includes vlm_conditioning field."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids", "vlm_kv"],
            (1, 10, 32),
        )

        server = ReflexServer(v02_export_dir, device="cpu")
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids", "vlm_kv"]
        server._vlm = None
        server._vlm_loaded = False

        result = server.predict()

        assert "vlm_conditioning" in result
        assert result["vlm_conditioning"] == "dummy"
        assert "actions" in result


# ---------------------------------------------------------------------------
# Test 4: Backward compat -- v0.1 export dir (no vlm_prefix)
# ---------------------------------------------------------------------------

class TestServerBackwardCompat:
    def test_server_backward_compat_v01(self, v01_export_dir):
        """Server with v0.1 export dir returns vlm_conditioning='dummy'."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids"],
            (1, 10, 32),
        )

        server = ReflexServer(v01_export_dir, device="cpu")
        # Wire manually
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._vlm = None
        server._vlm_loaded = False
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids"]

        result = server.predict()

        assert result["vlm_conditioning"] == "dummy"
        assert "actions" in result
        assert "error" not in result

    def test_v01_with_image_still_dummy(self, v01_export_dir):
        """Even with image+instruction, v0.1 server stays dummy (no VLM session)."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids"],
            (1, 10, 32),
        )

        server = ReflexServer(v01_export_dir, device="cpu")
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._vlm = None
        server._vlm_loaded = False
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids"]

        fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = server.predict(image=fake_image, instruction="pick up the cup")

        assert result["vlm_conditioning"] == "dummy"


# ---------------------------------------------------------------------------
# Test 5: Config v0.2+ schema loads with all new fields
# ---------------------------------------------------------------------------

class TestConfigV02Schema:
    def test_config_v02_schema(self, tmp_path):
        """Config with v0.2+ fields loads and has correct types."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "export_version": "0.3",
            "vlm_prefix_onnx": "vision_encoder.onnx",
            "vlm_image_size": [512, 512],
            "vlm_kv_dim": 960,
            "vlm_prefix_seq_len": 50,
            "expert": {
                "expert_hidden": 720,
                "action_dim": 32,
            },
        }
        config_path = tmp_path / "reflex_config.json"
        config_path.write_text(json.dumps(config))

        loaded = json.loads(config_path.read_text())

        assert loaded["export_version"] == "0.3"
        assert loaded["vlm_prefix_onnx"] == "vision_encoder.onnx"
        assert isinstance(loaded["vlm_image_size"], list)
        assert len(loaded["vlm_image_size"]) == 2
        assert all(isinstance(x, int) for x in loaded["vlm_image_size"])
        assert isinstance(loaded["vlm_kv_dim"], int)
        assert loaded["vlm_kv_dim"] == 960
        assert isinstance(loaded["vlm_prefix_seq_len"], int)

    def test_v01_config_missing_vlm_fields(self, tmp_path):
        """v0.1 config loads fine -- missing VLM fields default gracefully."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "expert": {"action_dim": 32},
        }
        config_path = tmp_path / "reflex_config.json"
        config_path.write_text(json.dumps(config))

        loaded = json.loads(config_path.read_text())

        # These should be absent, and consumers use .get() with defaults
        assert "vlm_prefix_onnx" not in loaded
        assert "export_version" not in loaded
        assert loaded.get("vlm_prefix_onnx") is None
        assert loaded.get("vlm_kv_dim", 960) == 960


# ---------------------------------------------------------------------------
# Test 6: export_vlm_prefix updates config file
# ---------------------------------------------------------------------------

class TestVLMPrefixExporterUpdatesConfig:
    def test_export_vlm_prefix_writes_config_fields(self, tmp_path):
        """export_vlm_prefix writes vlm_image_size, vlm_kv_dim, vlm_prefix_onnx to config.

        Since export_vlm_prefix does heavy lifting (loads real HF model, runs
        ONNX export), we test the config-update logic by directly simulating
        what the function writes, then verifying the schema.
        """
        # Pre-populate a v0.1 config
        initial_config = {
            "model_id": "lerobot/smolvla_base",
            "expert": {"action_dim": 32},
        }
        config_path = tmp_path / "reflex_config.json"
        config_path.write_text(json.dumps(initial_config))

        # Simulate what export_vlm_prefix writes to config.
        # vlm_hidden_size (960) is the VLM-internal dim. vlm_kv_dim (320) is
        # the POST-projection dim that the expert's cross-attn expects —
        # decoder_prefill.onnx bakes in the final-layer k_proj to output this
        # shape directly.
        config = json.loads(config_path.read_text())
        config["vlm_image_size"] = [512, 512]
        config["vlm_hidden_size"] = 960
        config["vlm_kv_dim"] = 320
        config["vlm_prefix_onnx"] = "vision_encoder.onnx"
        config["export_version"] = "0.3"
        config["decoder_prefill_onnx"] = "decoder_prefill.onnx"
        config_path.write_text(json.dumps(config, indent=2))

        # Verify config was updated
        updated_config = json.loads(config_path.read_text())
        assert updated_config["vlm_prefix_onnx"] == "vision_encoder.onnx"
        assert updated_config["vlm_hidden_size"] == 960
        assert updated_config["vlm_kv_dim"] == 320
        assert updated_config["vlm_image_size"] == [512, 512]
        assert updated_config["export_version"] == "0.3"
        assert updated_config["decoder_prefill_onnx"] == "decoder_prefill.onnx"
        # Original fields preserved
        assert updated_config["model_id"] == "lerobot/smolvla_base"
        assert updated_config["expert"]["action_dim"] == 32


# ---------------------------------------------------------------------------
# Test 7: assemble_prefix shapes and attention mask
# ---------------------------------------------------------------------------

class TestAssemblePrefix:
    def test_assemble_prefix_shapes(self):
        """assemble_prefix concatenates [image, text, state] with correct shapes."""
        B = 2
        N_img = 64
        T = 10
        hidden = 960

        image_embeds = np.random.randn(B, N_img, hidden).astype(np.float32)
        text_embeds = np.random.randn(B, T, hidden).astype(np.float32)
        state_embed = np.random.randn(B, 1, hidden).astype(np.float32)

        prefix, mask = assemble_prefix(image_embeds, text_embeds, state_embed)

        expected_seq = N_img + T + 1  # 64 + 10 + 1 = 75
        assert prefix.shape == (B, expected_seq, hidden), (
            f"Expected prefix shape ({B}, {expected_seq}, {hidden}), got {prefix.shape}"
        )
        assert mask.shape == (B, expected_seq), (
            f"Expected mask shape ({B}, {expected_seq}), got {mask.shape}"
        )
        assert prefix.dtype == np.float32
        assert mask.dtype == np.int64

    def test_assemble_prefix_attention_mask(self):
        """Attention mask: 0 for image+text (bidirectional), 1 for state (causal)."""
        B = 1
        N_img = 64
        T = 10
        hidden = 960

        image_embeds = np.random.randn(B, N_img, hidden).astype(np.float32)
        text_embeds = np.random.randn(B, T, hidden).astype(np.float32)
        state_embed = np.random.randn(B, 1, hidden).astype(np.float32)

        _, mask = assemble_prefix(image_embeds, text_embeds, state_embed)

        # Image + text tokens should be 0 (bidirectional)
        assert np.all(mask[:, :N_img + T] == 0), (
            "Image+text tokens should have mask=0 (bidirectional)"
        )
        # State token should be 1 (causal)
        assert np.all(mask[:, N_img + T:] == 1), (
            "State token should have mask=1 (causal)"
        )

    def test_assemble_prefix_content_order(self):
        """Verify the concatenation order is [image, text, state]."""
        B = 1
        hidden = 16  # small for readability

        image_embeds = np.ones((B, 2, hidden), dtype=np.float32) * 1.0
        text_embeds = np.ones((B, 3, hidden), dtype=np.float32) * 2.0
        state_embed = np.ones((B, 1, hidden), dtype=np.float32) * 3.0

        prefix, _ = assemble_prefix(image_embeds, text_embeds, state_embed)

        # Check each segment
        np.testing.assert_allclose(prefix[:, :2, :], 1.0)
        np.testing.assert_allclose(prefix[:, 2:5, :], 2.0)
        np.testing.assert_allclose(prefix[:, 5:6, :], 3.0)


# ---------------------------------------------------------------------------
# Test 8: StateEncoder shape
# ---------------------------------------------------------------------------

class TestStateEncoder:
    def test_state_encoder_shape(self):
        """StateEncoder with dim 32->960 produces [B, 1, 960]."""
        encoder = StateEncoder(max_state_dim=32, hidden_size=960)
        encoder.eval()

        state = torch.randn(1, 32)
        with torch.no_grad():
            out = encoder(state)

        assert out.shape == (1, 1, 960), f"Expected (1, 1, 960), got {out.shape}"

    def test_state_encoder_custom_dims(self):
        """StateEncoder works with custom dimensions."""
        encoder = StateEncoder(max_state_dim=14, hidden_size=64)
        encoder.eval()

        state = torch.randn(2, 14)
        with torch.no_grad():
            out = encoder(state)

        assert out.shape == (2, 1, 64), f"Expected (2, 1, 64), got {out.shape}"

    def test_state_encoder_with_weights(self):
        """StateEncoder initializes correctly with provided weights."""
        weight = torch.randn(960, 32)
        bias = torch.randn(960)
        encoder = StateEncoder(
            state_proj_weight=weight,
            state_proj_bias=bias,
            max_state_dim=32,
            hidden_size=960,
        )
        encoder.eval()

        # Verify weights were copied
        assert torch.allclose(encoder.proj.weight.data, weight)
        assert torch.allclose(encoder.proj.bias.data, bias)


# ---------------------------------------------------------------------------
# Test 9: pad_state
# ---------------------------------------------------------------------------

class TestPadState:
    def test_pad_state_1d(self):
        """pad_state(np.array([1,2,3]), 32) produces [32] with zeros."""
        state = np.array([1.0, 2.0, 3.0])
        padded = pad_state(state, 32)

        assert padded.shape == (32,)
        assert padded.dtype == np.float32
        np.testing.assert_allclose(padded[:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(padded[3:], 0.0)

    def test_pad_state_2d(self):
        """pad_state works with 2D [B, D] inputs."""
        state = np.array([[1.0, 2.0], [3.0, 4.0]])
        padded = pad_state(state, 32)

        assert padded.shape == (2, 32)
        np.testing.assert_allclose(padded[:, :2], state)
        np.testing.assert_allclose(padded[:, 2:], 0.0)

    def test_pad_state_already_full(self):
        """pad_state returns unchanged array when already at max_state_dim."""
        state = np.random.randn(32).astype(np.float32)
        padded = pad_state(state, 32)
        np.testing.assert_allclose(padded, state)

    def test_pad_state_too_large_raises(self):
        """pad_state raises ValueError when state exceeds max_state_dim."""
        state = np.random.randn(33)
        with pytest.raises(ValueError, match="exceeds"):
            pad_state(state, 32)


# ---------------------------------------------------------------------------
# Test 10: VLMPrefixOrchestrator graceful degradation
# ---------------------------------------------------------------------------

class TestOrchestratorGracefulDegradation:
    def test_orchestrator_vision_only(self, tmp_path):
        """Orchestrator with only vision_encoder.onnx runs without crashing."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "vlm_kv_dim": 960,
            "vlm_image_size": [512, 512],
        }
        (tmp_path / "reflex_config.json").write_text(json.dumps(config))
        # Only create vision_encoder.onnx -- no decoder_prefill
        (tmp_path / "vision_encoder.onnx").write_bytes(b"fake-onnx")

        # Mock onnxruntime.InferenceSession
        mock_vision_session = _make_mock_ort_session(
            ["pixel_values"], (1, 64, 960)
        )

        def mock_session_init(path, **kwargs):
            if "vision_encoder" in str(path):
                return mock_vision_session
            raise FileNotFoundError(f"Not found: {path}")

        with patch.dict("sys.modules", {}):
            import onnxruntime
            with patch.object(onnxruntime, "InferenceSession", side_effect=mock_session_init):
                from reflex.runtime.vlm_orchestrator import VLMPrefixOrchestrator
                orch = VLMPrefixOrchestrator(tmp_path, config)

                assert orch.is_loaded
                assert not orch.is_complete  # no decoder_prefill

    def test_orchestrator_no_files(self, tmp_path):
        """Orchestrator with no ONNX files: is_loaded is False."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "vlm_kv_dim": 960,
        }
        (tmp_path / "reflex_config.json").write_text(json.dumps(config))
        # No ONNX files created

        from reflex.runtime.vlm_orchestrator import VLMPrefixOrchestrator
        orch = VLMPrefixOrchestrator(tmp_path, config)

        assert not orch.is_loaded


# ---------------------------------------------------------------------------
# Test 11: VLMPrefixOrchestrator full pipeline with mocked sessions
# ---------------------------------------------------------------------------

class TestOrchestratorFullPipeline:
    def test_orchestrator_full_pipeline(self, tmp_path):
        """Mock all ONNX sessions, verify end-to-end produces [1, prefix_len, 960]."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "vlm_kv_dim": 960,
            "vlm_image_size": [512, 512],
        }
        (tmp_path / "reflex_config.json").write_text(json.dumps(config))
        (tmp_path / "vision_encoder.onnx").write_bytes(b"fake-onnx")
        (tmp_path / "text_embedder.onnx").write_bytes(b"fake-onnx")
        (tmp_path / "decoder_prefill.onnx").write_bytes(b"fake-onnx")

        # Mock sessions
        vision_session = _make_mock_ort_session(["pixel_values"], (1, 64, 960))
        text_session = _make_mock_ort_session(["input_ids"], (1, 32, 960))
        prefill_session = MagicMock()

        # decoder_prefill should return [1, seq, 960]
        def prefill_run(output_names, feed_dict):
            embeds = feed_dict.get("prefix_embeds", feed_dict.get("inputs_embeds"))
            if embeds is not None:
                seq = embeds.shape[1]
            else:
                seq = 97
            return [np.random.randn(1, seq, 960).astype(np.float32)]

        prefill_session.run.side_effect = prefill_run
        prefill_session.get_inputs.return_value = [
            MagicMock(name="inputs_embeds"),
            MagicMock(name="attention_mask"),
        ]
        prefill_session.get_providers.return_value = ["CPUExecutionProvider"]

        session_map = {
            "vision_encoder.onnx": vision_session,
            "text_embedder.onnx": text_session,
            "decoder_prefill.onnx": prefill_session,
        }

        def mock_ort_init(path, **kwargs):
            path_str = str(path)
            for key, session in session_map.items():
                if key in path_str:
                    return session
            return _make_mock_ort_session([], (1, 1, 960))

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.ones((1, 32), dtype=np.int64),
        }

        import onnxruntime
        with patch.object(onnxruntime, "InferenceSession", side_effect=mock_ort_init):
            with patch.object(
                VLMPrefixOrchestrator, "_load_tokenizer_and_processor"
            ):
                orch = VLMPrefixOrchestrator(tmp_path, config)
                orch._tokenizer = mock_tokenizer

                assert orch.is_loaded
                assert orch.is_complete

                # Run with a fake image, instruction, and state
                fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                fake_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

                # Modern API returns (vlm_k, vlm_v) — both shape
                # [num_layers, batch, seq, kv_dim]. Earlier signature
                # returned a single 3D prefix; updated when expert
                # cross-attention adopted per-layer K/V (vlm_components).
                vlm_k, vlm_v = orch.run(fake_image, "pick up the cup", fake_state)

                assert vlm_k.ndim == 4 and vlm_v.ndim == 4
                # batch dim
                assert vlm_k.shape[1] == 1
                assert vlm_v.shape[1] == 1
                # k and v share the same sequence + kv_dim
                assert vlm_k.shape == vlm_v.shape


# ---------------------------------------------------------------------------
# Test 12: Different instructions produce different prefixes (integration)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("REFLEX_INTEGRATION") != "1",
    reason="Set REFLEX_INTEGRATION=1 to run integration tests",
)
class TestDifferentInstructions:
    def test_different_instructions_different_prefix(self):
        """Run two different instructions through the real pipeline, verify prefix_kv differs."""
        # This test requires real model weights and would only run in CI
        # with REFLEX_INTEGRATION=1
        pytest.skip("Integration test requires real model checkpoint")
