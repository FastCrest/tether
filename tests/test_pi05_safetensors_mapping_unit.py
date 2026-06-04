"""Unit tests for Pi0.5 safetensors→flat-dict mapping.

Validates ``pi05_safetensors_to_flat()`` and ``expand_tied_pi05()`` against
the known Pi0.5 key structure. No checkpoint download required.
"""
from __future__ import annotations

import torch

from tether.models.vlas._pi05_safetensors_mapping import (
    expand_tied_pi05,
    pi05_safetensors_to_flat,
)


# ── Skip rules ──────────────────────────────────────────────────────────


def test_skip_gemma_expert_lm_head():
    assert pi05_safetensors_to_flat("paligemma_with_expert.gemma_expert.lm_head.weight") is None


def test_skip_plain_layernorm_weight():
    """Plain layernorm weight (buffer) is skipped — only .dense.* is kept."""
    assert pi05_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight") is None
    assert pi05_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.layers.5.post_attention_layernorm.weight") is None


def test_skip_final_norm_plain_weight():
    assert pi05_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.norm.weight") is None


def test_skip_state_proj():
    """Pi0.5 uses state-in-language — no state_proj."""
    assert pi05_safetensors_to_flat("state_proj.weight") is None
    assert pi05_safetensors_to_flat("state_proj.bias") is None


# ── AdaRMSNorm dense weights KEPT ────────────────────────────────────────


def test_keep_expert_input_layernorm_dense():
    """AdaRMSNorm's dense.weight/bias are Parameters — KEEP."""
    src = "paligemma_with_expert.gemma_expert.model.layers.3.input_layernorm.dense.weight"
    result = pi05_safetensors_to_flat(src)
    assert result is not None
    assert "vla_head.expert_stack.layers.3.input_layernorm.dense.weight" == result


def test_keep_expert_post_attn_layernorm_dense():
    src = "paligemma_with_expert.gemma_expert.model.layers.7.post_attention_layernorm.dense.bias"
    result = pi05_safetensors_to_flat(src)
    assert result is not None
    assert "vla_head.expert_stack.layers.7.post_attention_layernorm.dense.bias" == result


def test_keep_final_norm_dense():
    w = pi05_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.norm.dense.weight")
    b = pi05_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.norm.dense.bias")
    assert w == "vla_head.expert_stack.final_norm.dense.weight"
    assert b == "vla_head.expert_stack.final_norm.dense.bias"


# ── Expert projections (same as Pi0) ─────────────────────────────────────


def test_expert_attention_flattened():
    src = "paligemma_with_expert.gemma_expert.model.layers.5.self_attn.k_proj.weight"
    assert pi05_safetensors_to_flat(src) == "vla_head.expert_stack.layers.5.k_proj.weight"


def test_expert_mlp_flattened():
    src = "paligemma_with_expert.gemma_expert.model.layers.12.mlp.down_proj.weight"
    assert pi05_safetensors_to_flat(src) == "vla_head.expert_stack.layers.12.down_proj.weight"


# ── Vision tower (same as Pi0) ───────────────────────────────────────────


def test_vision_tower():
    src = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    assert pi05_safetensors_to_flat(src) == "vision_backbone.model.vision_model.embeddings.patch_embedding.weight"


# ── PaliGemma LLM (same as Pi0) ─────────────────────────────────────────


def test_paligemma_lm_head():
    assert pi05_safetensors_to_flat("paligemma_with_expert.paligemma.lm_head.weight") == "llm_backbone.model.lm_head.weight"


# ── Action projections ───────────────────────────────────────────────────


def test_action_in_proj():
    assert pi05_safetensors_to_flat("action_in_proj.weight") == "vla_head.expert_stack.action_in_proj.weight"


def test_action_out_proj():
    assert pi05_safetensors_to_flat("action_out_proj.bias") == "vla_head.expert_stack.action_out_proj.bias"


# ── Time MLP (Pi0.5 naming: action_time_mlp → time_mlp) ─────────────────


def test_time_mlp_in():
    assert pi05_safetensors_to_flat("action_time_mlp_in.weight") == "vla_head.expert_stack.time_mlp_in.weight"


def test_time_mlp_out():
    assert pi05_safetensors_to_flat("action_time_mlp_out.bias") == "vla_head.expert_stack.time_mlp_out.bias"


# ── Tied weight expansion ───────────────────────────────────────────────


def test_expand_tied_populates_embed_tokens():
    lm_head = torch.randn(2, 2)
    flat = {"llm_backbone.model.lm_head.weight": lm_head}
    out = expand_tied_pi05(flat)
    assert "llm_backbone.model.model.language_model.embed_tokens.weight" in out
    assert out["llm_backbone.model.model.language_model.embed_tokens.weight"] is lm_head
