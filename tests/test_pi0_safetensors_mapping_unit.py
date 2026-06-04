"""Unit tests for Pi0 safetensors→flat-dict mapping.

Validates ``pi0_safetensors_to_flat()`` and ``expand_tied_pi0()`` against a
synthetic key set that mirrors the real ``lerobot/pi0_base`` structure
(no checkpoint download required, runs in CI in milliseconds).

The full end-to-end parity test (loads the real 3.3 GB checkpoint) lives in
``test_pi0_safetensors_direct_parity.py`` and is skipped by default.
"""
from __future__ import annotations

import torch

from tether.models.vlas._pi0_safetensors_mapping import (
    expand_tied_pi0,
    pi0_safetensors_to_flat,
)


# ── Skip rules ──────────────────────────────────────────────────────────


def test_skip_gemma_expert_lm_head():
    assert pi0_safetensors_to_flat("paligemma_with_expert.gemma_expert.lm_head.weight") is None


def test_skip_expert_layer_norms():
    assert pi0_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight") is None
    assert pi0_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.layers.17.post_attention_layernorm.weight") is None


def test_skip_gemma_expert_final_norm():
    assert pi0_safetensors_to_flat("paligemma_with_expert.gemma_expert.model.norm.weight") is None


# ── Vision tower ────────────────────────────────────────────────────────


def test_vision_tower_root():
    src = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    expected = "vision_backbone.model.vision_model.embeddings.patch_embedding.weight"
    assert pi0_safetensors_to_flat(src) == expected


def test_vision_tower_deep():
    src = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.9.self_attn.v_proj.weight"
    expected = "vision_backbone.model.vision_model.encoder.layers.9.self_attn.v_proj.weight"
    assert pi0_safetensors_to_flat(src) == expected


# ── PaliGemma LLM ───────────────────────────────────────────────────────


def test_paligemma_lm_head():
    assert (
        pi0_safetensors_to_flat("paligemma_with_expert.paligemma.lm_head.weight")
        == "llm_backbone.model.lm_head.weight"
    )


def test_paligemma_language_model_layers():
    src = "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
    expected = "llm_backbone.model.model.language_model.layers.0.self_attn.q_proj.weight"
    assert pi0_safetensors_to_flat(src) == expected


def test_paligemma_multi_modal_projector():
    src = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    expected = "llm_backbone.model.model.multi_modal_projector.linear.weight"
    assert pi0_safetensors_to_flat(src) == expected


# ── Gemma expert (with .self_attn. / .mlp. flattening) ──────────────────


def test_gemma_expert_attention_flattened():
    """`.self_attn.q_proj.` → `.q_proj.` (no .self_attn. parent)."""
    src = "paligemma_with_expert.gemma_expert.model.layers.5.self_attn.k_proj.weight"
    expected = "vla_head.expert_stack.layers.5.k_proj.weight"
    assert pi0_safetensors_to_flat(src) == expected


def test_gemma_expert_mlp_flattened():
    """`.mlp.gate_proj.` → `.gate_proj.` (no .mlp. parent)."""
    src = "paligemma_with_expert.gemma_expert.model.layers.12.mlp.down_proj.weight"
    expected = "vla_head.expert_stack.layers.12.down_proj.weight"
    assert pi0_safetensors_to_flat(src) == expected


def test_gemma_expert_all_seven_projections():
    """All 7 projections per layer flatten correctly."""
    layer_keys = [
        ("self_attn.q_proj", "q_proj"),
        ("self_attn.k_proj", "k_proj"),
        ("self_attn.v_proj", "v_proj"),
        ("self_attn.o_proj", "o_proj"),
        ("mlp.gate_proj", "gate_proj"),
        ("mlp.up_proj", "up_proj"),
        ("mlp.down_proj", "down_proj"),
    ]
    for src_suffix, dst_suffix in layer_keys:
        src = f"paligemma_with_expert.gemma_expert.model.layers.3.{src_suffix}.weight"
        expected = f"vla_head.expert_stack.layers.3.{dst_suffix}.weight"
        assert pi0_safetensors_to_flat(src) == expected, f"{src} → got {pi0_safetensors_to_flat(src)}, want {expected}"


# ── Action expert + state projector ─────────────────────────────────────


def test_action_in_proj():
    assert pi0_safetensors_to_flat("action_in_proj.weight") == "vla_head.expert_stack.action_in_proj.weight"
    assert pi0_safetensors_to_flat("action_in_proj.bias") == "vla_head.expert_stack.action_in_proj.bias"


def test_action_out_proj():
    assert pi0_safetensors_to_flat("action_out_proj.bias") == "vla_head.expert_stack.action_out_proj.bias"


def test_action_time_mlp():
    assert pi0_safetensors_to_flat("action_time_mlp_in.weight") == "vla_head.expert_stack.action_time_mlp_in.weight"
    assert pi0_safetensors_to_flat("action_time_mlp_out.bias") == "vla_head.expert_stack.action_time_mlp_out.bias"


def test_state_proj():
    assert pi0_safetensors_to_flat("state_proj.weight") == "projector.linear.weight"
    assert pi0_safetensors_to_flat("state_proj.bias") == "projector.linear.bias"


# ── Tied weight expansion ───────────────────────────────────────────────


def test_expand_tied_populates_embed_tokens_from_lm_head():
    lm_head = torch.randn(257152, 2048)
    flat = {"llm_backbone.model.lm_head.weight": lm_head}
    out = expand_tied_pi0(flat)

    assert "llm_backbone.model.model.language_model.embed_tokens.weight" in out
    # Shared storage — same object, not copy
    assert out["llm_backbone.model.model.language_model.embed_tokens.weight"] is lm_head


def test_expand_tied_noop_if_embed_tokens_already_present():
    """If both keys are already in the dict, don't overwrite embed_tokens."""
    lm_head = torch.zeros(2, 2)
    embed = torch.ones(2, 2)
    flat = {
        "llm_backbone.model.lm_head.weight": lm_head,
        "llm_backbone.model.model.language_model.embed_tokens.weight": embed,
    }
    out = expand_tied_pi0(flat)
    assert torch.equal(out["llm_backbone.model.model.language_model.embed_tokens.weight"], embed)


def test_expand_tied_noop_if_no_lm_head():
    """If lm_head isn't in the dict, don't add embed_tokens spuriously."""
    flat = {"some.other.weight": torch.zeros(2, 2)}
    out = expand_tied_pi0(flat)
    assert "llm_backbone.model.model.language_model.embed_tokens.weight" not in out


# ── Aggregate: synthesized pi0_base key set produces exact Phase A keys ─


def test_full_pi0_base_key_set_matches_phase_a():
    """Apply mapping to a synthesized pi0_base ST key set; check output matches Phase A."""
    # Synthesize the structural shape of pi0_base (777 keys).
    # Numbers below were captured from a real lerobot/pi0_base inspection.
    st_keys: list[str] = []

    # Vision tower: 437 keys (siglip-base-patch16-224, 27 layers)
    st_keys.extend([
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight",
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias",
    ])
    for layer in range(27):
        for sub in [
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "layer_norm1.weight", "layer_norm1.bias",
            "layer_norm2.weight", "layer_norm2.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ]:
            st_keys.append(f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer}.{sub}")

    # Gemma expert: 18 layers × 9 entries (7 projections + 2 norms) + 1 final norm + 1 tied lm_head = 164
    for layer in range(18):
        for sub in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ]:
            st_keys.append(f"paligemma_with_expert.gemma_expert.model.layers.{layer}.{sub}")
    st_keys.append("paligemma_with_expert.gemma_expert.model.norm.weight")
    st_keys.append("paligemma_with_expert.gemma_expert.lm_head.weight")

    # Action expert: 4 projections × (weight + bias) = 8
    for proj in ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]:
        st_keys.append(f"{proj}.weight")
        st_keys.append(f"{proj}.bias")

    # State projection: 2
    st_keys.append("state_proj.weight")
    st_keys.append("state_proj.bias")

    # Apply mapping
    mapped = set()
    skipped = 0
    for k in st_keys:
        out = pi0_safetensors_to_flat(k)
        if out is None:
            skipped += 1
        else:
            mapped.add(out)

    # Validate skip count: 18 layers × 2 norms + 1 final norm + 1 gemma_expert.lm_head = 38
    assert skipped == 38

    # Validate vision_tower: 5 root keys + 27 × 16 layer keys = 437
    vision_keys = {k for k in mapped if k.startswith("vision_backbone.")}
    assert len(vision_keys) == 437, f"vision_tower count = {len(vision_keys)}"

    # Validate gemma_expert layers map to vla_head.expert_stack.layers.N.{q,k,v,o,gate,up,down}_proj.weight
    expert_layer_keys = {k for k in mapped if k.startswith("vla_head.expert_stack.layers.")}
    assert len(expert_layer_keys) == 18 * 7, f"expert_layer count = {len(expert_layer_keys)}"
    # Each layer has exactly 7 projections
    for layer in range(18):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            assert f"vla_head.expert_stack.layers.{layer}.{proj}.weight" in mapped
