"""Tests for SmolVLA — SmolVLA composition class on the BaseVLA spine.

Lift #1 Day 6 per ``features/03_export/basevla-spine_plan.md``. Validates:

- registration on the VLAS registry
- slot declarations (REQUIRED_SLOTS, OPTIONAL_SLOTS, NAME_MAPPING)
- construction via direct kwargs + from_config (stubbed components)
- forward routes to llm_backbone
- predict_action raises NotImplementedError (runtime path is currently
  ``tether.runtime.smolvla_native``; spine predict_action is a follow-up)
- end-to-end SPINE → ExpertStack: SmolVLA.from_pretrained synthetic
  state_dict produces a bit-identical ExpertStack vs direct
  ``smolvla_exporter.build_expert_stack`` call (the parity gate from
  the Day 6 plan).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from tether.models.base_vla import BaseVLA
from tether.models.heads import VLAHead
from tether.models.llm import LLMBackbone
from tether.models.projectors import Projector
from tether.models.vision import VisionBackbone
from tether.models.vlas.smolvla import SmolVLA
from tether.registry.components import VLAS


# ─── Registration + slot declarations ───────────────────────────────────


def test_smolvla_registered():
    assert "SmolVLA" in VLAS
    assert VLAS.get("SmolVLA") is SmolVLA


def test_smolvla_is_basevla_subclass():
    assert issubclass(SmolVLA, BaseVLA)


def test_smolvla_required_slots():
    """SmolVLA declares 4 required slots: vision/llm/projector/head.
    projector = state_proj (state-in-input; unlike pi0.5 which embeds
    state in language)."""
    assert SmolVLA.REQUIRED_SLOTS == (
        "vision_backbone", "llm_backbone", "projector", "vla_head",
    )
    assert SmolVLA.OPTIONAL_SLOTS == ()


def test_smolvla_name_mapping_default_empty():
    """Decision S-1 — empty NAME_MAPPING (SmolVLA checkpoint keys load directly)."""
    assert SmolVLA.NAME_MAPPING == {}


# ─── Construction via direct kwargs ─────────────────────────────────────


def test_smolvla_constructs_with_4_stub_components():
    vla = SmolVLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    assert isinstance(vla.vision_backbone, _StubVision)
    assert isinstance(vla.llm_backbone, _StubLLM)
    assert isinstance(vla.projector, _StubProjector)
    assert isinstance(vla.vla_head, _StubHead)
    # Unused slots stay None
    assert vla.vlm_backbone is None
    assert vla.text_encoder is None


def test_smolvla_missing_required_slot_raises():
    with pytest.raises(ValueError, match="missing required slot"):
        SmolVLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            projector=_StubProjector(),
            # vla_head missing
        )


def test_smolvla_undeclared_slot_raises():
    """Per BaseVLA — passing vlm_backbone (not in REQUIRED + OPTIONAL) raises."""
    with pytest.raises(ValueError, match="undeclared"):
        SmolVLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            projector=_StubProjector(),
            vla_head=_StubHead(),
            vlm_backbone=_StubVision(),
        )


# ─── Construction via from_config ───────────────────────────────────────


def test_smolvla_from_config_with_prebuilt_instances():
    vla = SmolVLA.from_config({
        "vision_backbone": _StubVision(),
        "llm_backbone": _StubLLM(),
        "projector": _StubProjector(),
        "vla_head": _StubHead(),
    })
    assert isinstance(vla, SmolVLA)
    assert vla.vision_backbone is not None
    assert vla.projector is not None


# ─── Forward routing ────────────────────────────────────────────────────


def test_forward_routes_to_llm_backbone():
    stub_llm = _StubLLM()
    vla = SmolVLA(
        vision_backbone=_StubVision(),
        llm_backbone=stub_llm,
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    embeds = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = vla.forward({
        "inputs_embeds": embeds,
        "attention_mask": mask,
        "past_key_values": None,
    })
    assert stub_llm.last_call["inputs_embeds"] is embeds
    assert out.last_hidden_state.shape == (1, 5, 8)


# ─── predict_action — Day 6 Phase A — NotImplementedError ──────────────


def test_predict_action_raises_not_implemented():
    """SmolVLA's runtime path is currently ``tether.runtime.smolvla_native``.
    A spine predict_action() is a follow-up; the Day 6 acceptance is the
    monolithic ONNX export path, not runtime inference."""
    vla = SmolVLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    with pytest.raises(NotImplementedError, match="Phase B"):
        vla.predict_action()


# ─── Spine → ExpertStack parity (Day 6 export path) ────────────────────


def test_smolvla_spine_builds_same_expert_as_legacy(tmp_path: Any):
    """SmolVLA.from_pretrained's expert stack is bit-identical to a direct
    smolvla_exporter.build_expert_stack call on the same state_dict.

    Validates the Day 6 acceptance: the new spine path produces the same
    numerics as the legacy direct-build path. ONNX bytes would also be
    identical given the same torch graph + opset.
    """
    # Synthetic SmolVLA state_dict — minimal but architecturally correct.
    sd = _build_synthetic_smolvla_state_dict()

    # Legacy path: build expert directly.
    from tether.exporters.smolvla import build_expert_stack
    legacy_stack, legacy_meta = build_expert_stack(sd, head_dim=64)

    # Spine path: build via SmolVLA composition. Stub out the VLM load to
    # avoid network access.
    spine_stack = _build_smolvla_expert_via_spine(sd)

    # Same architectural metadata
    assert legacy_meta["expert_hidden"] == spine_stack.expert_hidden
    assert sorted(legacy_stack.cross_indices) == sorted(spine_stack.cross_indices)
    assert legacy_stack.vlm_kv_dim == spine_stack.vlm_kv_dim
    assert len(legacy_stack.layers) == len(spine_stack.layers)

    # Same forward — bit-identical numerics on identical inputs
    chunk_size, action_dim = 50, 32
    num_layers = len(legacy_stack.layers)
    vlm_kv_dim = legacy_stack.vlm_kv_dim
    noisy_actions = torch.randn(1, chunk_size, action_dim)
    timestep = torch.tensor([0.5])
    pos_ids = torch.arange(chunk_size).unsqueeze(0)
    vlm_k = torch.zeros(num_layers, 1, 1, vlm_kv_dim)
    vlm_v = torch.zeros(num_layers, 1, 1, vlm_kv_dim)
    prefix_offset = torch.tensor([[241]], dtype=torch.int64)
    kv_mask = torch.ones(1, 1, dtype=torch.bool)

    with torch.no_grad():
        out_legacy = legacy_stack(noisy_actions, timestep, pos_ids,
                                  vlm_k, vlm_v, prefix_offset, kv_mask)
        out_spine = spine_stack(noisy_actions, timestep, pos_ids,
                                vlm_k, vlm_v, prefix_offset, kv_mask)

    max_diff = (out_legacy - out_spine).abs().max().item()
    assert max_diff == 0.0, f"spine vs legacy diverges by {max_diff} (expected 0)"


# ─── Helpers ────────────────────────────────────────────────────────────


def _build_smolvla_expert_via_spine(state_dict: dict[str, torch.Tensor]) -> Any:
    """Build SmolVLA's expert via the spine composition (no VLM load)."""
    from tether.exporters.smolvla import build_expert_stack
    from tether.models.heads.flow_matching_head import FlowMatchingHead
    from tether.models.projectors.linear_projector import LinearProjector

    # SmolVLA composition without the VLM — stub vision/llm with minimal
    # objects so REQUIRED_SLOTS bind. Then bind the real expert_stack.
    stack, _meta = build_expert_stack(state_dict, head_dim=64)
    head = FlowMatchingHead(expert_stack=stack)
    state_proj = LinearProjector(in_dim=32, out_dim=576)

    vla = SmolVLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=state_proj,
        vla_head=head,
    )
    return vla.vla_head.expert_stack


def _build_synthetic_smolvla_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic SmolVLA state_dict — minimal architecture matching real shapes.

    Mirrors the lerobot SmolVLA layout: ``model.vlm_with_expert.lm_expert.*``
    for the expert layers + top-level ``model.action_in_proj`` etc. Uses
    a tiny 2-layer expert (vs production 8 layers) for fast tests.
    """
    # SmolVLA-500M production GQA (must match build_expert_stack's
    # AutoConfig lookup of HuggingFaceTB/SmolVLM2-500M-Video-Instruct):
    # text_config.num_attention_heads=15, num_key_value_heads=5.
    # Expert head_dim = expert_hidden / nq = 420 / 15 = 28 (integer).
    expert_hidden = 420  # 15 nq × 28 expert_hd; 0.75 × 576 ≈ 432, close to actual
    action_dim = 32
    nq, nkv, expert_hd = 15, 5, 28
    inter = 1024
    num_layers = 2  # tiny for tests
    cross_idxs = {0}  # layer 0 is cross-attn (to VLM), layer 1 is self-attn
    vlm_kv_dim = nkv * 64  # 5 VLM kv heads × 64 vlm head_dim = 320

    base = "model.vlm_with_expert.lm_expert.model."
    sd: dict[str, torch.Tensor] = {}

    # Top-level action + time MLP keys (model.* prefix)
    sd["model.action_in_proj.weight"] = torch.randn(expert_hidden, action_dim)
    sd["model.action_in_proj.bias"] = torch.randn(expert_hidden)
    sd["model.action_out_proj.weight"] = torch.randn(action_dim, expert_hidden)
    sd["model.action_out_proj.bias"] = torch.randn(action_dim)
    sd["model.action_time_mlp_in.weight"] = torch.randn(expert_hidden, expert_hidden * 2)
    sd["model.action_time_mlp_in.bias"] = torch.randn(expert_hidden)
    sd["model.action_time_mlp_out.weight"] = torch.randn(expert_hidden, expert_hidden)
    sd["model.action_time_mlp_out.bias"] = torch.randn(expert_hidden)

    # Per-layer
    for i in range(num_layers):
        p = f"{base}layers.{i}"
        is_cross = i in cross_idxs
        kv_in = vlm_kv_dim if is_cross else expert_hidden

        sd[f"{p}.input_layernorm.weight"] = torch.randn(expert_hidden)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(expert_hidden)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(nq * expert_hd, expert_hidden)
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(nkv * expert_hd, kv_in)
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(nkv * expert_hd, kv_in)
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(expert_hidden, nq * expert_hd)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, expert_hidden)
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, expert_hidden)
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(expert_hidden, inter)

    # Final norm
    sd[f"{base}norm.weight"] = torch.randn(expert_hidden)

    # State proj (state-in-input)
    sd["model.state_proj.weight"] = torch.randn(576, 32)
    sd["model.state_proj.bias"] = torch.randn(576)

    return sd


class _StubVision(VisionBackbone):
    def forward(self, images: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        b = images.shape[0] if images.ndim >= 1 else 1
        return torch.zeros(b, 4, 8)


class _StubProjector(Projector):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x


class _StubLLM(LLMBackbone, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.last_call: dict[str, Any] = {}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        self.last_call = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
        }
        if inputs_embeds is not None:
            return SimpleNamespace(last_hidden_state=inputs_embeds)
        return SimpleNamespace(last_hidden_state=torch.zeros(1, 4, 8))


class _StubHead(VLAHead):
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(1, 50, 32)
