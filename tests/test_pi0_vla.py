"""Tests for Pi0VLA — pi0 composition class on the BaseVLA spine.

Lift #1 Day 4f per `features/03_export/basevla-spine_plan.md`. Validates
the composition shape:

- registration on the VLAS registry
- slot declarations (REQUIRED_SLOTS, OPTIONAL_SLOTS, NAME_MAPPING)
- construction via from_config (the spine's primary path)
- predict_action raises NotImplementedError per Day 4g deferral
- forward routes to llm_backbone

Day 4g adds the full inference pipeline + parity test vs the legacy
pi0_exporter. This file tests the composition shape only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from reflex.models.base_vla import BaseVLA
from reflex.models.heads import VLAHead
from reflex.models.llm import LLMBackbone
from reflex.models.projectors import Projector
from reflex.models.vision import VisionBackbone
from reflex.models.vlas.pi0 import Pi0VLA
from reflex.registry.components import VLAS


# ─── Registration + slot declarations ───────────────────────────────────


def test_pi0_vla_registered():
    assert "Pi0VLA" in VLAS
    assert VLAS.get("Pi0VLA") is Pi0VLA


def test_pi0_vla_is_basevla_subclass():
    assert issubclass(Pi0VLA, BaseVLA)


def test_pi0_vla_required_slots():
    """Pi0VLA declares 4 required slots: vision/llm/projector/head.
    vlm_backbone + text_encoder unused."""
    assert Pi0VLA.REQUIRED_SLOTS == (
        "vision_backbone", "llm_backbone", "projector", "vla_head",
    )
    assert Pi0VLA.OPTIONAL_SLOTS == ()


def test_pi0_vla_name_mapping_default_empty():
    """Decision S-1 — empty NAME_MAPPING is the v1 default (the lerobot/pi0_base
    checkpoint's keys map directly to component slots via load_state_dict's
    slot-prefix routing)."""
    assert Pi0VLA.NAME_MAPPING == {}


# ─── Construction via direct kwargs (the test path) ─────────────────────


def test_pi0_vla_constructs_with_4_stub_components():
    vla = Pi0VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    assert isinstance(vla.vision_backbone, _StubVision)
    assert isinstance(vla.llm_backbone, _StubLLM)
    assert isinstance(vla.projector, _StubProjector)
    assert isinstance(vla.vla_head, _StubHead)
    # Optional slots stay None
    assert vla.vlm_backbone is None
    assert vla.text_encoder is None


def test_pi0_vla_missing_required_slot_raises():
    """Per BaseVLA contract — missing required slot is a ValueError at
    construction."""
    with pytest.raises(ValueError, match="missing required slot"):
        Pi0VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            projector=_StubProjector(),
            # vla_head missing
        )


def test_pi0_vla_undeclared_slot_raises():
    """Per BaseVLA — passing vlm_backbone (not in REQUIRED + OPTIONAL) raises."""
    with pytest.raises(ValueError, match="undeclared"):
        Pi0VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            projector=_StubProjector(),
            vla_head=_StubHead(),
            vlm_backbone=_StubVision(),  # not in Pi0VLA's slots
        )


# ─── Construction via from_config (Registry path) ───────────────────────


def test_pi0_vla_from_config_with_prebuilt_instances():
    """from_config accepts pre-built component instances directly (the test
    path; from_pretrained handles the full HF load)."""
    vla = Pi0VLA.from_config({
        "vision_backbone": _StubVision(),
        "llm_backbone": _StubLLM(),
        "projector": _StubProjector(),
        "vla_head": _StubHead(),
    })
    assert isinstance(vla, Pi0VLA)
    assert vla.vision_backbone is not None


# ─── Forward routing ────────────────────────────────────────────────────


def test_forward_routes_to_llm_backbone():
    """forward(batch) calls llm_backbone with inputs_embeds + attention_mask
    + past_key_values."""
    stub_llm = _StubLLM()
    vla = Pi0VLA(
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
    assert stub_llm.last_call["attention_mask"] is mask
    assert out.last_hidden_state.shape == (1, 5, 8)


# ─── predict_action deferred ────────────────────────────────────────────


def test_predict_action_raises_not_implemented():
    """Day 4f scope: composition shape only. Day 4g wires the full
    inference pipeline + parity gate. Until then predict_action is an
    explicit NotImplementedError (NOT a silent stub returning None)."""
    vla = Pi0VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    with pytest.raises(NotImplementedError, match="Day 4g"):
        vla.predict_action(images=None, state=None, instruction="x")


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


class _StubProjector(Projector):
    def forward(self, x, *args, **kwargs): return x


class _StubHead(VLAHead):
    def forward(self, context, *args, **kwargs): return context
