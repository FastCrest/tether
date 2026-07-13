"""Regression test: SmolVLA ExpertGQALayer MLP uses SiLU, not gelu-tanh.

The SmolVLA expert is built from config.text_config (SmolLM2 / LlamaConfig),
whose hidden_act defaults to "silu". Prior to the fix, line ~212 of
expert_stack.py erroneously called F.gelu(..., approximate="tanh") (Gemma's
activation), which corrupts parity on the decomposed SmolVLA export path.

This test verifies three things:
1. A forward pass through ExpertGQALayer (self-attention mode) with known
   weights produces output that matches the reference formula using SiLU.
2. The same pass does NOT match the old gelu-tanh formula.
3. The source of ExpertGQALayer no longer contains `approximate="tanh"` in
   its MLP block, while Pi05ExpertGQALayer (Gemma-based) still does.
"""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn.functional as F

from tether.models.heads.expert_stack import ExpertGQALayer, Pi05ExpertGQALayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_layer(seed: int = 0) -> ExpertGQALayer:
    """Tiny ExpertGQALayer (hidden=8, 2 heads, hd=4, inter=16) for fast tests."""
    torch.manual_seed(seed)
    return ExpertGQALayer(hidden=8, nq=2, nkv=2, hd=4, inter=16)


# ---------------------------------------------------------------------------
# 1. Forward-pass produces output matching SiLU reference
# ---------------------------------------------------------------------------

class TestSmolVLAExpertUseSiLU:
    def test_mlp_matches_silu_reference(self):
        """ExpertGQALayer MLP output must equal silu(gate)*up (NOT gelu-tanh)."""
        layer = _make_tiny_layer(seed=42)
        layer.eval()

        torch.manual_seed(7)
        B, S, H = 1, 3, 8
        x = torch.randn(B, S, H)
        pos_ids = torch.arange(S).unsqueeze(0)  # [1, S]

        with torch.no_grad():
            actual = layer(x, pos_ids)

        # ----------------------------------------------------------------
        # Build reference: replicate the MLP sub-block with identical
        # weights using SiLU (the correct SmolLM2 activation).
        # We run the full layer forward so we can isolate the residual.
        # Instead, we directly test the MLP sub-block in isolation.
        # ----------------------------------------------------------------
        # Re-run post_attention_layernorm to get the normed input to MLP.
        with torch.no_grad():
            # Mirror the forward() internals up to the MLP.
            b, s, _ = x.shape
            res0 = x
            x_norm0 = layer.input_layernorm(x)
            q = layer.q_proj(x_norm0).view(b, s, layer.nq, layer.hd).transpose(1, 2)
            k = layer.k_proj(x_norm0).view(b, s, layer.nkv, layer.hd).transpose(1, 2)
            v = layer.v_proj(x_norm0).view(b, s, layer.nkv, layer.hd).transpose(1, 2)
            q = layer.rope.apply(q, pos_ids)
            k = layer.rope.apply(k, pos_ids)

            kv_len = k.shape[2]
            k_exp = k.unsqueeze(2).expand(-1, -1, layer.kv_groups, -1, -1).reshape(b, layer.nq, kv_len, layer.hd)
            v_exp = v.unsqueeze(2).expand(-1, -1, layer.kv_groups, -1, -1).reshape(b, layer.nq, kv_len, layer.hd)

            scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(layer.hd)
            attn = F.softmax(scores, dim=-1)
            x_after_attn = res0 + layer.o_proj(
                torch.matmul(attn, v_exp).transpose(1, 2).contiguous().view(b, s, -1)
            )
            res1 = x_after_attn
            x_mlp_in = layer.post_attention_layernorm(x_after_attn)

            # SiLU reference (expected)
            ref_silu = res1 + layer.down_proj(F.silu(layer.gate_proj(x_mlp_in)) * layer.up_proj(x_mlp_in))

            # gelu-tanh (old, wrong)
            ref_gelu_tanh = res1 + layer.down_proj(
                F.gelu(layer.gate_proj(x_mlp_in), approximate="tanh") * layer.up_proj(x_mlp_in)
            )

        # The layer output must match SiLU reference exactly.
        assert torch.allclose(actual, ref_silu, atol=0.0), (
            "ExpertGQALayer output does not match SiLU reference — "
            "the MLP activation may still be gelu-tanh."
        )

        # The layer output must NOT match gelu-tanh reference.
        # (On inputs where silu(x) != gelu_tanh(x) the outputs will differ.
        #  Verify they differ by at least a small amount.)
        max_diff = (actual - ref_gelu_tanh).abs().max().item()
        assert max_diff > 1e-6, (
            f"ExpertGQALayer output matches the old gelu-tanh formula (max_diff={max_diff:.2e}); "
            "the activation may not have been fixed."
        )

    def test_silu_and_geluTanh_differ_on_known_input(self):
        """Sanity-check: SiLU and gelu-tanh are NOT equivalent on our test input.

        This ensures the forward-pass test above is actually discriminating —
        if silu and gelu_tanh happened to agree on x=0 the previous test
        would be vacuous.
        """
        torch.manual_seed(99)
        x = torch.randn(4, 8)
        silu_out = F.silu(x)
        gelu_tanh_out = F.gelu(x, approximate="tanh")
        max_diff = (silu_out - gelu_tanh_out).abs().max().item()
        assert max_diff > 1e-4, (
            "SiLU and gelu-tanh produced nearly identical outputs on test input; "
            f"max_diff={max_diff:.2e}. The discriminating test may be vacuous."
        )


# ---------------------------------------------------------------------------
# 2. Source-level guard: ExpertGQALayer has no gelu-tanh in its MLP
# ---------------------------------------------------------------------------

class TestActivationSourceGuards:
    def test_expert_gqa_layer_no_gelu_tanh(self):
        """ExpertGQALayer.forward must not call F.gelu(..., approximate='tanh')."""
        src = inspect.getsource(ExpertGQALayer.forward)
        assert 'approximate="tanh"' not in src and "approximate='tanh'" not in src, (
            "ExpertGQALayer.forward still contains gelu-tanh. "
            "SmolVLA expert activation must be SiLU (matches SmolLM2/LlamaConfig)."
        )

    def test_expert_gqa_layer_uses_silu(self):
        """ExpertGQALayer.forward must call F.silu for the MLP gate activation."""
        src = inspect.getsource(ExpertGQALayer.forward)
        assert "F.silu(" in src, (
            "ExpertGQALayer.forward does not call F.silu. "
            "SmolVLA expert activation must be SiLU."
        )

    def test_pi05_expert_still_uses_gelu_tanh(self):
        """Pi05ExpertGQALayer.forward must STILL use gelu-tanh (Gemma-based, correct)."""
        src = inspect.getsource(Pi05ExpertGQALayer.forward)
        assert 'approximate="tanh"' in src or "approximate='tanh'" in src, (
            "Pi05ExpertGQALayer.forward no longer uses gelu-tanh. "
            "The Gemma-based pi0.5 expert should keep gelu-tanh — do not accidentally remove it."
        )
