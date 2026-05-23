"""Tests for Pi0VLA.from_lerobot_policy + Pi05VLA.from_lerobot_policy.

Lift #3 prereq #1 — promotes the working parity-script pattern (used by
Day 4h pi0 + Day 5b pi0.5 Modal bit-identical fires) to a first-class
spine API.

The naive ``from_pretrained(hf_id)`` path is broken for real
lerobot/pi05_* and lerobot/pi0_* checkpoints — they nest PaliGemma
weights under ``model.paligemma_with_expert.paligemma.*`` which stock
``PaliGemmaForConditionalGeneration.from_pretrained`` can't see, so
weights fall back to random init.

These tests stub a minimal "lerobot policy" + verify the new classmethods
extract the right submodules.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ─── Minimal stub for a lerobot policy ────────────────────────────────


def _build_stub_pi0_policy_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic state_dict matching lerobot pi0's nested key format.

    Keys are prefixed with ``model.`` per lerobot's HF storage layout.
    """
    expert_hidden = 1024
    action_dim = 32
    nq, nkv, hd = 8, 1, 256
    inter = 16384
    num_layers = 18
    expert_base = "model.paligemma_with_expert.gemma_expert.model."

    sd: dict[str, torch.Tensor] = {}

    # Top-level pi0 action keys (lerobot wraps in "model.").
    sd["model.action_in_proj.weight"] = torch.randn(expert_hidden, action_dim)
    sd["model.action_in_proj.bias"] = torch.randn(expert_hidden)
    sd["model.action_out_proj.weight"] = torch.randn(action_dim, expert_hidden)
    sd["model.action_out_proj.bias"] = torch.randn(action_dim)
    sd["model.action_time_mlp_in.weight"] = torch.randn(expert_hidden, expert_hidden * 2)
    sd["model.action_time_mlp_in.bias"] = torch.randn(expert_hidden)
    sd["model.action_time_mlp_out.weight"] = torch.randn(expert_hidden, expert_hidden)
    sd["model.action_time_mlp_out.bias"] = torch.randn(expert_hidden)
    sd["model.state_proj.weight"] = torch.randn(expert_hidden, action_dim)
    sd["model.state_proj.bias"] = torch.randn(expert_hidden)

    # Per-layer expert weights.
    for i in range(num_layers):
        p = f"{expert_base}layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(expert_hidden)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(expert_hidden)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(nq * hd, expert_hidden)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(nkv * hd, expert_hidden)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(nkv * hd, expert_hidden)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(expert_hidden, nq * hd)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, expert_hidden)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, expert_hidden)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(expert_hidden, inter)
    sd[f"{expert_base}norm.weight"] = torch.randn(expert_hidden)
    return sd


def _build_stub_paligemma() -> Any:
    """Mock paligemma that exposes the .model.vision_tower attribute path."""
    paligemma = MagicMock()
    # SigLIPBackbone(model=...) expects an nn.Module; mock with a real one.
    vision_tower = nn.Module()
    paligemma.model = MagicMock()
    paligemma.model.vision_tower = vision_tower
    return paligemma


# ─── Pi0VLA.from_lerobot_policy ──────────────────────────────────────


def test_pi0vla_from_lerobot_policy_extracts_paligemma_submodule():
    """The classmethod must reach into policy.model.paligemma_with_expert.paligemma
    and split it into vision_backbone + llm_backbone slots."""
    from reflex.models.vlas.pi0 import Pi0VLA

    # Build a stub policy whose state_dict() returns lerobot-prefixed keys.
    stub_paligemma = _build_stub_paligemma()
    state_dict = _build_stub_pi0_policy_state_dict()

    policy = MagicMock()
    policy.model.paligemma_with_expert.paligemma = stub_paligemma
    policy.state_dict.return_value = state_dict

    vla = Pi0VLA.from_lerobot_policy(policy)

    # Slot wiring: vision_backbone + llm_backbone + projector + vla_head all set.
    assert vla.vision_backbone is not None
    assert vla.llm_backbone is not None
    assert vla.projector is not None
    assert vla.vla_head is not None

    # The vision_backbone wraps the paligemma's vision_tower.
    assert vla.vision_backbone.model is stub_paligemma.model.vision_tower


# ─── Pi05VLA.from_lerobot_policy ─────────────────────────────────────


def _build_stub_pi05_policy_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic state_dict for pi0.5 — AdaRMSNorm (dense.weight/.bias) keys
    + bias-less RMSNorms, plus the pi0.5 time_mlp_in/out instead of pi0's
    action_time_mlp_in/out.
    """
    expert_hidden = 1024
    action_dim = 32
    nq, nkv, hd = 8, 1, 256
    inter = 16384
    num_layers = 18
    expert_base = "model.paligemma_with_expert.gemma_expert.model."

    sd: dict[str, torch.Tensor] = {}
    sd["model.action_in_proj.weight"] = torch.randn(expert_hidden, action_dim)
    sd["model.action_in_proj.bias"] = torch.randn(expert_hidden)
    sd["model.action_out_proj.weight"] = torch.randn(action_dim, expert_hidden)
    sd["model.action_out_proj.bias"] = torch.randn(action_dim)
    # pi0.5: time_mlp_in/out (not action_time_mlp_*)
    sd["model.time_mlp_in.weight"] = torch.randn(expert_hidden, expert_hidden)
    sd["model.time_mlp_in.bias"] = torch.randn(expert_hidden)
    sd["model.time_mlp_out.weight"] = torch.randn(expert_hidden, expert_hidden)
    sd["model.time_mlp_out.bias"] = torch.randn(expert_hidden)

    for i in range(num_layers):
        p = f"{expert_base}layers.{i}"
        # AdaRMSNorm dense weight+bias (3*hidden output for scale/shift/gate)
        sd[f"{p}.input_layernorm.dense.weight"] = torch.randn(3 * expert_hidden, expert_hidden)
        sd[f"{p}.input_layernorm.dense.bias"] = torch.randn(3 * expert_hidden)
        sd[f"{p}.post_attention_layernorm.dense.weight"] = torch.randn(3 * expert_hidden, expert_hidden)
        sd[f"{p}.post_attention_layernorm.dense.bias"] = torch.randn(3 * expert_hidden)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(nq * hd, expert_hidden)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(nkv * hd, expert_hidden)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(nkv * hd, expert_hidden)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(expert_hidden, nq * hd)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, expert_hidden)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, expert_hidden)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(expert_hidden, inter)
    # Final norm: AdaRMSNorm
    sd[f"{expert_base}norm.dense.weight"] = torch.randn(3 * expert_hidden, expert_hidden)
    sd[f"{expert_base}norm.dense.bias"] = torch.randn(3 * expert_hidden)
    return sd


def test_pi05vla_from_lerobot_policy_extracts_paligemma_submodule():
    from reflex.models.vlas.pi05 import Pi05VLA

    stub_paligemma = _build_stub_paligemma()
    state_dict = _build_stub_pi05_policy_state_dict()

    policy = MagicMock()
    policy.model.paligemma_with_expert.paligemma = stub_paligemma
    policy.state_dict.return_value = state_dict

    vla = Pi05VLA.from_lerobot_policy(policy)

    # Pi05 has 3 REQUIRED slots: vision, llm, head (no projector — state in lang).
    assert vla.vision_backbone is not None
    assert vla.llm_backbone is not None
    assert vla.vla_head is not None
    # Projector unused for pi05.
    assert vla.projector is None


def test_from_lerobot_policy_no_lerobot_dep_at_import_time():
    """Importing the spine module must NOT pull in lerobot — the classmethods
    take the policy as an argument, so lerobot stays an extras-only dep.
    """
    import importlib
    import sys

    # Pre-condition: lerobot is NOT in sys.modules yet (if it is, this test
    # is moot — the dep was pulled in by something else).
    if "lerobot" in sys.modules:
        pytest.skip("lerobot already imported, can't test no-import contract")

    importlib.import_module("reflex.models.vlas.pi0")
    importlib.import_module("reflex.models.vlas.pi05")

    assert "lerobot" not in sys.modules, (
        "Importing Pi0VLA / Pi05VLA pulled in lerobot — from_lerobot_policy "
        "must use 'policy: Any' typing + no import-time lerobot reference."
    )
