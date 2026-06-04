"""Unit tests for FluxVLA pi0.5 safetensors→flat-dict mapping.

Validates ``fluxvla_pi05_safetensors_to_flat()`` and
``expand_tied_fluxvla_pi05()`` against FluxVLA's key naming convention.
No checkpoint download required.
"""
from __future__ import annotations

import torch

from tether.models.vlas._fluxvla_pi05_safetensors_mapping import (
    expand_tied_fluxvla_pi05,
    fluxvla_pi05_safetensors_to_flat,
)


# ── Vision tower ──────────────────────────────────────────────────────


def test_vision_tower_strips_vision_level():
    src = "vision_backbone.vision.vision_model.encoder.layers.5.self_attn.q_proj.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vision_backbone.model.vision_model.encoder.layers.5.self_attn.q_proj.weight"
    )


def test_vision_backbone_patch_embedding():
    src = "vision_backbone.vision.vision_model.embeddings.patch_embedding.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vision_backbone.model.vision_model.embeddings.patch_embedding.weight"
    )


# ── LLM backbone ─────────────────────────────────────────────────────


def test_llm_backbone_layers():
    src = "llm_backbone.layers.0.self_attn.q_proj.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "llm_backbone.model.model.language_model.layers.0.self_attn.q_proj.weight"
    )


def test_llm_backbone_embed_tokens():
    src = "llm_backbone.embed_tokens.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "llm_backbone.model.model.language_model.embed_tokens.weight"
    )


def test_llm_backbone_norm():
    src = "llm_backbone.norm.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "llm_backbone.model.model.language_model.norm.weight"
    )


# ── Expert (with flattening) ─────────────────────────────────────────


def test_expert_self_attn_flattened():
    src = "llm_expert.layers.5.self_attn.k_proj.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.layers.5.k_proj.weight"
    )


def test_expert_mlp_flattened():
    src = "llm_expert.layers.12.mlp.down_proj.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.layers.12.down_proj.weight"
    )


def test_expert_ada_layernorm_dense_kept():
    src = "llm_expert.layers.3.input_layernorm.dense.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.layers.3.input_layernorm.dense.weight"
    )


def test_expert_post_attn_layernorm_dense_kept():
    src = "llm_expert.layers.7.post_attention_layernorm.dense.bias"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.layers.7.post_attention_layernorm.dense.bias"
    )


def test_expert_final_norm_dense():
    w = fluxvla_pi05_safetensors_to_flat("llm_expert.norm.dense.weight")
    b = fluxvla_pi05_safetensors_to_flat("llm_expert.norm.dense.bias")
    assert w == "vla_head.expert_stack.final_norm.dense.weight"
    assert b == "vla_head.expert_stack.final_norm.dense.bias"


def test_expert_embed_tokens():
    src = "llm_expert.embed_tokens.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.embed_tokens.weight"
    )


# ── Projector ─────────────────────────────────────────────────────────


def test_projector_strips_inner_projector():
    w = fluxvla_pi05_safetensors_to_flat("projector.projector.weight")
    b = fluxvla_pi05_safetensors_to_flat("projector.projector.bias")
    assert w == "llm_backbone.model.model.multi_modal_projector.linear.weight"
    assert b == "llm_backbone.model.model.multi_modal_projector.linear.bias"


# ── Action projections ────────────────────────────────────────────────


def test_action_in_proj():
    src = "action_in_proj.projector.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.action_in_proj.weight"
    )


def test_action_out_proj():
    src = "action_out_proj.projector.bias"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.action_out_proj.bias"
    )


# ── Time MLP ──────────────────────────────────────────────────────────


def test_time_mlp_in():
    src = "time_mlp_in.projector.weight"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.time_mlp_in.weight"
    )


def test_time_mlp_out():
    src = "time_mlp_out.projector.bias"
    assert fluxvla_pi05_safetensors_to_flat(src) == (
        "vla_head.expert_stack.time_mlp_out.bias"
    )


# ── Tied weight expansion ────────────────────────────────────────────


def test_expand_tied_creates_lm_head():
    embed = torch.randn(2, 2)
    flat = {"llm_backbone.model.model.language_model.embed_tokens.weight": embed}
    out = expand_tied_fluxvla_pi05(flat)
    assert "llm_backbone.model.lm_head.weight" in out
    assert out["llm_backbone.model.lm_head.weight"] is embed


def test_expand_tied_noop_if_lm_head_exists():
    embed = torch.randn(2, 2)
    lm_head = torch.randn(2, 2)
    flat = {
        "llm_backbone.model.model.language_model.embed_tokens.weight": embed,
        "llm_backbone.model.lm_head.weight": lm_head,
    }
    out = expand_tied_fluxvla_pi05(flat)
    assert out["llm_backbone.model.lm_head.weight"] is lm_head
