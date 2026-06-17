"""Tests for model type auto-detection from state_dict keys."""

import torch

from tether.checkpoint import detect_model_type


def _dummy_tensor():
    return torch.zeros(1)


class TestDetectModelType:
    def test_detects_smolvla_via_vision_encoder(self):
        sd = {"model.vision_encoder.patch_embed.weight": _dummy_tensor()}
        assert detect_model_type(sd) == "smolvla"

    def test_detects_smolvla_via_vlm_with_expert(self):
        sd = {"model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": _dummy_tensor()}
        assert detect_model_type(sd) == "smolvla"

    def test_detects_pi0(self):
        sd = {
            "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": _dummy_tensor(),
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight": _dummy_tensor(),
            "action_in_proj.weight": _dummy_tensor(),
        }
        assert detect_model_type(sd) == "pi0"

    def test_detects_pi05_via_adarmsnorm(self):
        sd = {
            "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": _dummy_tensor(),
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight": _dummy_tensor(),
            "action_in_proj.weight": _dummy_tensor(),
        }
        assert detect_model_type(sd) == "pi05"

    def test_returns_none_for_unknown(self):
        sd = {"some.random.model.weight": _dummy_tensor()}
        assert detect_model_type(sd) is None


class TestPi0ExpectedStructure:
    """Smoke tests that pi0 exporter's expected keys are in the agreed layout."""

    def test_pi0_action_keys_are_top_level(self):
        """pi0 has action projections at the top level (no 'model.' prefix)."""
        from tether.exporters.pi0 import PI0_ACTION_KEYS

        for key_path in PI0_ACTION_KEYS.values():
            # pi0 keys are top-level, e.g. "action_in_proj.weight", not "model.action_in_proj.weight"
            assert not key_path.startswith("model."), f"pi0 key should be top-level: {key_path}"

    def test_pi0_expert_prefix(self):
        """pi0 expert layers are under 'paligemma_with_expert.gemma_expert.model.'."""
        from tether.exporters.pi0 import PI0_EXPERT_PREFIX

        assert PI0_EXPERT_PREFIX.startswith("paligemma_with_expert")
        assert "gemma_expert" in PI0_EXPERT_PREFIX


class TestPi05ExpectedStructure:
    """pi0.5 has time_mlp (separate from action) and AdaRMSNorm."""

    def test_pi05_action_keys_are_top_level(self):
        from tether.exporters.pi0 import PI05_ACTION_KEYS

        for key_path in PI05_ACTION_KEYS.values():
            assert not key_path.startswith("model."), f"pi0.5 key should be top-level: {key_path}"

    def test_pi05_time_mlp_is_separate(self):
        """pi0.5 uses `time_mlp_in/out` (not `action_time_mlp_in/out` like pi0)."""
        from tether.exporters.pi0 import PI05_ACTION_KEYS

        assert PI05_ACTION_KEYS["t_in_w"].startswith("time_mlp_")
        assert not PI05_ACTION_KEYS["t_in_w"].startswith("action_time_mlp_")


class TestDecomposedAdaRMSNorm:
    def test_forward_shape(self):
        """AdaRMSNorm output preserves input shape."""
        import torch
        from tether.decompose import DecomposedAdaRMSNorm

        hidden = 64
        norm = DecomposedAdaRMSNorm(hidden, time_dim=hidden)
        x = torch.randn(2, 10, hidden)
        time_emb = torch.randn(2, hidden)
        out = norm(x, time_emb)
        assert out.shape == x.shape

    def test_dense_weight_shape(self):
        """dense weight is [3*hidden, time_dim] — scale+shift+gate projections."""
        from tether.decompose import DecomposedAdaRMSNorm

        hidden, time_dim = 1024, 1024
        norm = DecomposedAdaRMSNorm(hidden, time_dim=time_dim)
        assert norm.dense.weight.shape == (3 * hidden, time_dim)
        assert norm.dense.bias.shape == (3 * hidden,)

    def test_return_gate(self):
        """When return_gate=True, returns (modulated, gate) tuple."""
        import torch
        from tether.decompose import DecomposedAdaRMSNorm

        hidden = 64
        norm = DecomposedAdaRMSNorm(hidden, time_dim=hidden)
        x = torch.randn(2, 10, hidden)
        time_emb = torch.randn(2, hidden)
        modulated, gate = norm(x, time_emb, return_gate=True)
        assert modulated.shape == (2, 10, hidden)
        assert gate.shape == (2, 1, hidden)
