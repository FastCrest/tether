"""Tests for Pi05VLA — pi0.5 composition class on the BaseVLA spine.

Lift #1 Day 5 Phase A per `features/03_export/basevla-spine_plan.md`. Validates
the composition shape (mirroring Pi0VLA tests):

- registration on the VLAS registry
- slot declarations (REQUIRED_SLOTS, OPTIONAL_SLOTS, NAME_MAPPING)
- construction via from_config (the spine's primary path)
- predict_action raises NotImplementedError (Phase B will land it)
- forward routes to llm_backbone

Phase B adds the full inference pipeline + parity test vs lerobot PI05Policy.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from reflex.models.base_vla import BaseVLA
from reflex.models.heads import VLAHead
from reflex.models.llm import LLMBackbone
from reflex.models.vision import VisionBackbone
from reflex.models.vlas.pi05 import Pi05VLA
from reflex.registry.components import VLAS


# ─── Registration + slot declarations ───────────────────────────────────


def test_pi05_vla_registered():
    assert "Pi05VLA" in VLAS
    assert VLAS.get("Pi05VLA") is Pi05VLA


def test_pi05_vla_is_basevla_subclass():
    assert issubclass(Pi05VLA, BaseVLA)


def test_pi05_vla_required_slots():
    """Pi05VLA declares 3 required slots: vision/llm/head.
    projector + vlm_backbone + text_encoder unused (state-in-language)."""
    assert Pi05VLA.REQUIRED_SLOTS == ("vision_backbone", "llm_backbone", "vla_head")
    assert Pi05VLA.OPTIONAL_SLOTS == ()


def test_pi05_vla_name_mapping_default_empty():
    """Decision S-1 — empty NAME_MAPPING is the v1 default."""
    assert Pi05VLA.NAME_MAPPING == {}


# ─── Construction via direct kwargs (the test path) ─────────────────────


def test_pi05_vla_constructs_with_3_stub_components():
    vla = Pi05VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        vla_head=_StubHead(),
    )
    assert isinstance(vla.vision_backbone, _StubVision)
    assert isinstance(vla.llm_backbone, _StubLLM)
    assert isinstance(vla.vla_head, _StubHead)
    # Unused slots stay None
    assert vla.projector is None
    assert vla.vlm_backbone is None
    assert vla.text_encoder is None


def test_pi05_vla_missing_required_slot_raises():
    with pytest.raises(ValueError, match="missing required slot"):
        Pi05VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            # vla_head missing
        )


def test_pi05_vla_undeclared_slot_raises():
    """Per BaseVLA — passing vlm_backbone (not in REQUIRED + OPTIONAL) raises."""
    with pytest.raises(ValueError, match="undeclared"):
        Pi05VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            vla_head=_StubHead(),
            vlm_backbone=_StubVision(),
        )


# ─── Construction via from_config ───────────────────────────────────────


def test_pi05_vla_from_config_with_prebuilt_instances():
    vla = Pi05VLA.from_config({
        "vision_backbone": _StubVision(),
        "llm_backbone": _StubLLM(),
        "vla_head": _StubHead(),
    })
    assert isinstance(vla, Pi05VLA)
    assert vla.vision_backbone is not None


# ─── Forward routing ────────────────────────────────────────────────────


def test_forward_routes_to_llm_backbone():
    stub_llm = _StubLLM()
    vla = Pi05VLA(
        vision_backbone=_StubVision(),
        llm_backbone=stub_llm,
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


# ─── predict_action — Phase B deferred ─────────────────────────────────


def test_predict_action_raises_not_implemented():
    """Day 5 Phase A scope — predict_action is the Phase B deliverable."""
    vla = Pi05VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        vla_head=_StubHead(),
    )
    with pytest.raises(NotImplementedError, match="Day 5 Phase B"):
        vla.predict_action(
            images=[torch.randn(1, 3, 224, 224)],
            lang_tokens=torch.zeros(1, 4, dtype=torch.long),
            lang_masks=torch.ones(1, 4, dtype=torch.bool),
        )


# ─── Helpers ────────────────────────────────────────────────────────────


class _StubVision(VisionBackbone):
    def forward(self, images): return images


class _StubLLM(LLMBackbone, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.last_call: dict = {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *args,
        inputs_embeds=None,
        past_key_values=None,
        **kwargs,
    ):
        self.last_call = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _StubHead(VLAHead):
    def forward(self, context, *args, **kwargs): return context
