"""Lift #3 Day 2 — `BaseVLA.prepare_inference_weights()` composition tests.

Per ``features/01_serve/inference-only-weights_plan.md`` Day 2. The
method aggregates each bound slot's ``prepare_triton`` output into a
single flat ``{key: tensor}`` dict, validating that no two slots'
prefixes collide.

5 test cases per the plan spec.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from reflex.models.base_vla import BaseVLA
from reflex.models.heads import VLAHead
from reflex.models.heads.flow_matching_head import FlowMatchingHead
from reflex.models.llm import LLMBackbone
from reflex.models.projectors.linear_projector import LinearProjector
from reflex.models.vision import VisionBackbone
from reflex.models.vlm import VLMBackbone


# ─── Stubs for the spine slot classes ─────────────────────────────────
# Each stub overrides `prepare_triton` with the canonical real-component
# pattern (named_parameters() under prefix, detached + cloned).


def _named_params_with_prefix(self, prefix: str = ""):
    return {f"{prefix}{n}": p.detach().clone() for n, p in self.named_parameters()}


class _StubVision(VisionBackbone, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, images, *a, **kw):
        return images

    prepare_triton = _named_params_with_prefix


class _StubLLM(LLMBackbone, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.fc = nn.Linear(4, 4)

    def forward(self, *a, **kw):
        return None

    prepare_triton = _named_params_with_prefix


class _StubVLM(VLMBackbone, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.fc = nn.Linear(8, 8)

    def forward(self, *a, **kw):
        return None

    prepare_triton = _named_params_with_prefix


class _StubHead(VLAHead, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.fc = nn.Linear(4, 4)

    def forward(self, *a, **kw):
        return None

    prepare_triton = _named_params_with_prefix


# ─── Tiny concrete BaseVLA for the test ──────────────────────────────


class _TestPi0VLA(BaseVLA):
    REQUIRED_SLOTS = ("vision_backbone", "llm_backbone", "projector", "vla_head")
    OPTIONAL_SLOTS = ()

    def forward(self, batch): ...
    def predict_action(self, *args, **kwargs): ...


class _TestGR00TVLA(BaseVLA):
    """GR00T-style: vlm_backbone + vla_head only."""
    REQUIRED_SLOTS = ("vlm_backbone", "vla_head")
    OPTIONAL_SLOTS = ()

    def forward(self, batch): ...
    def predict_action(self, *args, **kwargs): ...


# ─── Pi0-style composition: vision + llm + projector + head ──────────


def test_pi0_style_composition_builds_flat_dict():
    """4-slot VLA: each component's keys get nested under `{slot}.{...}`."""
    head = FlowMatchingHead(expert_stack=nn.Linear(4, 4))
    vla = _TestPi0VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=LinearProjector(in_dim=4, out_dim=4),
        vla_head=head,
    )
    flat = vla.prepare_inference_weights()

    # The 4 slot prefixes each appear in at least one key.
    prefixes_seen = {k.split(".", 1)[0] for k in flat.keys()}
    assert "vision_backbone" in prefixes_seen
    assert "llm_backbone" in prefixes_seen
    assert "projector" in prefixes_seen
    assert "vla_head" in prefixes_seen


def test_pi0_style_total_param_count_matches():
    """Key count of the flat dict equals the sum of nn.Module
    parameter counts across the 4 bound slots (modulo ABC base-class
    no-ops which return {}).

    For these stubs:
      vision: 1 param (Parameter(4, 4))
      llm: 2 params (fc.weight, fc.bias)
      projector: 2 (linear.weight, linear.bias)
      flow_matching_head wraps nn.Linear: 2 (expert_stack.weight, expert_stack.bias)
    Total = 7.
    """
    head = FlowMatchingHead(expert_stack=nn.Linear(4, 4))
    vla = _TestPi0VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=LinearProjector(in_dim=4, out_dim=4),
        vla_head=head,
    )
    flat = vla.prepare_inference_weights()
    assert len(flat) == 7, f"got {len(flat)} keys; full: {sorted(flat.keys())}"


# ─── GR00T-style: vlm_backbone + vla_head only ───────────────────────


def test_gr00t_style_only_2_slots():
    """REQUIRED_SLOTS=(vlm_backbone, vla_head). All other slots None →
    no entries for them in the flat dict."""
    head = FlowMatchingHead(expert_stack=nn.Linear(4, 4))
    vla = _TestGR00TVLA(
        vlm_backbone=_StubVLM(),
        vla_head=head,
    )
    flat = vla.prepare_inference_weights()

    prefixes_seen = {k.split(".", 1)[0] for k in flat.keys()}
    assert prefixes_seen == {"vlm_backbone", "vla_head"}, (
        f"unexpected prefixes: {prefixes_seen}"
    )
    # No vision_backbone / llm_backbone / projector / text_encoder keys
    forbidden = {"vision_backbone", "llm_backbone", "projector", "text_encoder"}
    leaked = {k for k in flat.keys() if any(k.startswith(f + ".") for f in forbidden)}
    assert not leaked, f"leaked unused-slot keys: {leaked}"


# ─── Prefix collision: defensive raise on duplicate keys ─────────────


def test_duplicate_key_detection_raises():
    """A component that ignores the prefix kwarg would collide with
    another slot. The merge step must raise — silent overwrite would
    corrupt inference."""

    class _BrokenVision(VisionBackbone, nn.Module):
        """Ignores prefix, always returns same keys — collides at merge."""

        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, images, *a, **kw):
            return images

        def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
            # BUG: ignores prefix kwarg + returns exactly the key that
            # FlowMatchingHead.prepare_triton would produce on a
            # nn.Linear-wrapped expert_stack. When vla_head iterates,
            # the merge step will find the collision.
            return {"vla_head.expert_stack.weight": self.weight.detach().clone()}

    head = FlowMatchingHead(expert_stack=nn.Linear(4, 4))
    vla = _TestPi0VLA(
        vision_backbone=_BrokenVision(),
        llm_backbone=_StubLLM(),
        projector=LinearProjector(in_dim=4, out_dim=4),
        vla_head=head,
    )
    with pytest.raises(ValueError, match="duplicate key"):
        vla.prepare_inference_weights()


# ─── Empty composition: no slots → empty dict ────────────────────────


def test_empty_composition_returns_empty_dict():
    """A VLA with no slots bound (other than required-only) should
    return an empty flat dict — used by VLAs where every slot is
    populated with the base ABC no-op class."""

    class _MinimalVLA(BaseVLA):
        REQUIRED_SLOTS = ()
        OPTIONAL_SLOTS = ()

        def forward(self, batch): ...
        def predict_action(self, *args, **kwargs): ...

    flat = _MinimalVLA().prepare_inference_weights()
    assert flat == {}
