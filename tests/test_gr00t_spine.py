"""Tests for GR00TVLA — GR00T N1.6 on the BaseVLA spine.

Lift #1 Day 7 per ``features/03_export/basevla-spine_plan.md``. This is
**the critical day** — proves the 6-slot design works for fused VLMs
(Eagle is the ONLY VLM that lives in the ``vlm_backbone`` slot; pi0/pi05/
smolvla use ``vision_backbone`` + ``llm_backbone`` instead).

Validates:

- registration of GR00TVLA on VLAS, DITHead on VLA_HEADS, EagleBackbone
  on VLM_BACKBONES
- slot declarations (REQUIRED_SLOTS = (vlm_backbone, vla_head),
  ALL OTHER SLOTS UNUSED — this is the proof)
- construction via direct kwargs + from_config (stubbed components)
- forward routes through Eagle → DiT
- predict_action shape correctness on a stubbed pipeline
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from reflex.models.base_vla import BaseVLA
from reflex.models.heads import VLAHead
from reflex.models.vlas.gr00t import GR00TVLA
from reflex.models.vlm import VLMBackbone
from reflex.registry.components import VLAS, VLA_HEADS, VLM_BACKBONES


# ─── Registration + slot declarations ───────────────────────────────────


def test_gr00t_vla_registered():
    assert "GR00TVLA" in VLAS
    assert VLAS.get("GR00TVLA") is GR00TVLA


def test_dit_head_registered():
    from reflex.models.heads.dit_head import DITHead
    assert "DITHead" in VLA_HEADS
    assert VLA_HEADS.get("DITHead") is DITHead


def test_eagle_backbone_registered():
    from reflex.models.vlm.eagle_backbone import EagleBackbone
    assert "EagleBackbone" in VLM_BACKBONES
    assert VLM_BACKBONES.get("EagleBackbone") is EagleBackbone


def test_gr00t_vla_is_basevla_subclass():
    assert issubclass(GR00TVLA, BaseVLA)


def test_gr00t_vla_required_slots_validates_6_slot_design():
    """THE proof point per the lift #1 plan: GR00TVLA uses ONLY 2 of the
    6 spine slots (vlm_backbone + vla_head). All other slots must be None.

    This validates that the 6-slot taxonomy was the right call — fused
    VLMs like Eagle don't need to be force-fit into the 2-tower
    (vision_backbone + llm_backbone) pattern that pi0/pi05/smolvla use.
    """
    assert GR00TVLA.REQUIRED_SLOTS == ("vlm_backbone", "vla_head")
    assert GR00TVLA.OPTIONAL_SLOTS == ()


def test_gr00t_vla_name_mapping_default_empty():
    """Decision S-1 — GR00T N1.6 keys load directly (no NAME_MAPPING)."""
    assert GR00TVLA.NAME_MAPPING == {}


# ─── Construction via direct kwargs ─────────────────────────────────────


def test_gr00t_vla_constructs_with_2_stub_components():
    vla = GR00TVLA(
        vlm_backbone=_StubVLM(),
        vla_head=_StubDITHead(),
    )
    assert isinstance(vla.vlm_backbone, _StubVLM)
    assert isinstance(vla.vla_head, _StubDITHead)
    # ALL OTHER SLOTS UNUSED — this is the 6-slot proof
    assert vla.vision_backbone is None
    assert vla.llm_backbone is None
    assert vla.projector is None
    assert vla.text_encoder is None


def test_gr00t_vla_missing_required_slot_raises():
    with pytest.raises(ValueError, match="missing required slot"):
        GR00TVLA(vlm_backbone=_StubVLM())  # vla_head missing


def test_gr00t_vla_undeclared_slot_raises():
    """Passing vision_backbone (the slot pi0/pi05/smolvla use but
    GR00T doesn't) must raise — spine catches the wrong taxonomy."""
    with pytest.raises(ValueError, match="undeclared"):
        GR00TVLA(
            vlm_backbone=_StubVLM(),
            vla_head=_StubDITHead(),
            vision_backbone=_StubVLM(),
        )


# ─── Construction via from_config ───────────────────────────────────────


def test_gr00t_vla_from_config_with_prebuilt_instances():
    vla = GR00TVLA.from_config({
        "vlm_backbone": _StubVLM(),
        "vla_head": _StubDITHead(),
    })
    assert isinstance(vla, GR00TVLA)


# ─── Forward routing ────────────────────────────────────────────────────


def test_forward_routes_through_eagle_to_dit():
    """Forward calls Eagle on (images, tokens) then DiT on the result."""
    stub_vlm = _StubVLM(record=True)
    stub_head = _StubDITHead(record=True)
    vla = GR00TVLA(vlm_backbone=stub_vlm, vla_head=stub_head)

    batch_data = {
        "images": torch.randn(1, 3, 224, 224),
        "input_ids": torch.randint(0, 100, (1, 10), dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
        "image_flags": torch.ones(1, dtype=torch.long),
        "noisy_actions": torch.randn(1, 16, 128),
        "timestep": torch.tensor([0.5]),
        "position_ids": torch.arange(16).unsqueeze(0),
        "state": torch.randn(1, 128),
    }
    out = vla.forward(batch_data)
    assert stub_vlm.last_call_seen
    assert stub_head.last_call_seen
    # _StubDITHead returns [1, 16, 128]
    assert out.shape == (1, 16, 128)


# ─── predict_action — end-to-end stubbed denoise loop ──────────────────


def test_predict_action_runs_denoise_loop_with_stubs():
    """predict_action: Eagle prefill → 4-step Euler denoise loop."""
    stub_vlm = _StubVLM(record=True)
    stub_head = _StubDITHead(record=True)
    vla = GR00TVLA(vlm_backbone=stub_vlm, vla_head=stub_head)

    batch, chunk_size, raw_action_dim = 1, 16, 128
    images = torch.randn(batch, 3, 224, 224)
    input_ids = torch.randint(0, 100, (batch, 10), dtype=torch.long)
    state = torch.randn(batch, 128)
    noise = torch.randn(batch, chunk_size, raw_action_dim)

    actions = vla.predict_action(
        images=images, input_ids=input_ids, state=state,
        chunk_size=chunk_size, raw_action_dim=raw_action_dim, num_steps=4,
        noise=noise,
    )
    assert actions.shape == (batch, chunk_size, raw_action_dim)
    # 4 Euler steps means head called 4 times
    assert stub_head.call_count == 4


def test_predict_action_raises_if_required_slot_missing():
    vla = GR00TVLA(vlm_backbone=_StubVLM(), vla_head=_StubDITHead())
    vla.vlm_backbone = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="vlm_backbone is None"):
        vla.predict_action(
            images=torch.randn(1, 3, 224, 224),
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            state=torch.zeros(1, 128),
        )


# ─── Helpers ────────────────────────────────────────────────────────────


class _StubVLM(VLMBackbone, nn.Module):
    def __init__(self, record: bool = False) -> None:
        nn.Module.__init__(self)
        self.record = record
        self.last_call_seen: bool = False

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                *args: Any, image_flags: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        if self.record:
            self.last_call_seen = True
        b = images.shape[0]
        s = input_ids.shape[1]
        return torch.zeros(b, s, 2048)


class _StubDITHead(VLAHead, nn.Module):
    def __init__(self, record: bool = False) -> None:
        nn.Module.__init__(self)
        self.record = record
        self.last_call_seen: bool = False
        self.call_count: int = 0

    def forward(self, noisy_actions: torch.Tensor,
                timestep: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                *args: Any,
                state: torch.Tensor | None = None,
                vlm_kv: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        if self.record:
            self.last_call_seen = True
            self.call_count += 1
        # Return velocity of same shape as noisy_actions
        return torch.zeros_like(noisy_actions)
