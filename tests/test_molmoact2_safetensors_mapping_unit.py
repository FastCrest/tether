"""Unit tests for MolmoAct2 safetensors→flat-dict mapping."""
from __future__ import annotations

from tether.models.vlas._molmoact2_safetensors_mapping import (
    molmoact2_safetensors_to_flat,
)


def test_vision_backbone():
    src = "model.vision_backbone.image_pooling_2d.wk.weight"
    assert molmoact2_safetensors_to_flat(src) == "vision_backbone.image_pooling_2d.wk.weight"


def test_transformer_to_vlm():
    src = "model.transformer.blocks.0.attn_norm.weight"
    assert molmoact2_safetensors_to_flat(src) == "vlm_backbone.blocks.0.attn_norm.weight"


def test_action_expert_to_vla_head():
    src = "model.action_expert.blocks.0.cross_attn.out_proj.weight"
    assert molmoact2_safetensors_to_flat(src) == "vla_head.blocks.0.cross_attn.out_proj.weight"


def test_action_embed():
    src = "model.action_expert.action_embed.weight"
    assert molmoact2_safetensors_to_flat(src) == "vla_head.action_embed.weight"


def test_lm_head():
    assert molmoact2_safetensors_to_flat("lm_head.weight") == "vlm_backbone.lm_head.weight"


def test_passthrough():
    assert molmoact2_safetensors_to_flat("some.unknown.key") == "some.unknown.key"
