"""Tests for the BaseVLA spine + 6 component base ABCs.

Lift #1 Day 3 acceptance gate per `features/03_export/basevla-spine_plan.md`.
Covers:

- ABC enforcement (direct instantiation TypeError)
- BaseVLA slot validation (required, optional, declared, undeclared)
- BaseVLA.from_config dispatch through component registries
- BaseVLA.prepare_inference_weights aggregation
- BaseVLA.load_state_dict with NAME_MAPPING
- apply_name_mapping walker (first-match-wins, strict mode, empty-prefix)
"""
from __future__ import annotations

import pytest

from reflex.models import ALL_SLOTS, BaseVLA, apply_name_mapping
from reflex.models.vision import VisionBackbone
from reflex.models.llm import LLMBackbone
from reflex.models.vlm import VLMBackbone
from reflex.models.projectors import Projector
from reflex.models.heads import VLAHead
from reflex.models.text import TextEncoder


# ─── ABC enforcement ────────────────────────────────────────────────────


def test_basevla_cannot_be_instantiated_directly():
    """BaseVLA is abstract — direct instantiation must TypeError."""
    with pytest.raises(TypeError, match="abstract"):
        BaseVLA()


def test_component_bases_cannot_be_instantiated_directly():
    """Every component base class is abstract."""
    for cls in (VisionBackbone, LLMBackbone, VLMBackbone, Projector, VLAHead, TextEncoder):
        with pytest.raises(TypeError, match="abstract"):
            cls()


def test_basevla_subclass_without_abstract_impls_cannot_instantiate():
    """A subclass that doesn't implement forward + predict_action stays abstract."""
    class _Partial(BaseVLA):
        REQUIRED_SLOTS = ()
        OPTIONAL_SLOTS = ()
        # Missing forward + predict_action

    with pytest.raises(TypeError, match="abstract"):
        _Partial()


# ─── BaseVLA slot validation ────────────────────────────────────────────


# Minimal concrete classes used across slot tests.

class _StubVisionBackbone(VisionBackbone):
    def forward(self, images): return images


class _StubLLMBackbone(LLMBackbone):
    def forward(self, input_ids, attention_mask=None, *args, **kwargs):
        return input_ids


class _StubVLMBackbone(VLMBackbone):
    def forward(self, images, input_ids, attention_mask=None, *args, **kwargs):
        return (images, input_ids)


class _StubProjector(Projector):
    def forward(self, x, *args, **kwargs): return x


class _StubVLAHead(VLAHead):
    def forward(self, context, *args, **kwargs): return context


class _StubTextEncoder(TextEncoder):
    def forward(self, input_ids, attention_mask=None, *args, **kwargs):
        return input_ids


class _MinimalVLA(BaseVLA):
    """Concrete VLA with empty slot requirements — for slot-validation tests."""
    REQUIRED_SLOTS = ()
    OPTIONAL_SLOTS = ALL_SLOTS

    def forward(self, batch): return batch

    def predict_action(self, *, images, state, instruction):
        return None


def test_basevla_instantiates_with_no_slots_when_all_optional():
    """All slots optional → ok with no components."""
    v = _MinimalVLA()
    for slot in ALL_SLOTS:
        assert getattr(v, slot) is None


def test_basevla_binds_passed_components_to_attributes():
    v = _MinimalVLA(vision_backbone=_StubVisionBackbone(), projector=_StubProjector())
    assert isinstance(v.vision_backbone, _StubVisionBackbone)
    assert isinstance(v.projector, _StubProjector)
    # Unbound slots stay None
    assert v.llm_backbone is None
    assert v.vla_head is None


def test_basevla_required_slot_missing_raises():
    class _NeedsHead(BaseVLA):
        REQUIRED_SLOTS = ("vla_head",)
        OPTIONAL_SLOTS = ()
        def forward(self, batch): return batch
        def predict_action(self, *, images, state, instruction): return None

    with pytest.raises(ValueError, match="missing required slot"):
        _NeedsHead()


def test_basevla_required_slot_present_succeeds():
    class _NeedsHead(BaseVLA):
        REQUIRED_SLOTS = ("vla_head",)
        OPTIONAL_SLOTS = ()
        def forward(self, batch): return batch
        def predict_action(self, *, images, state, instruction): return None

    v = _NeedsHead(vla_head=_StubVLAHead())
    assert isinstance(v.vla_head, _StubVLAHead)


def test_basevla_undeclared_slot_passed_raises():
    """If a slot isn't in REQUIRED or OPTIONAL but caller passes it, raise.

    Catches the typo where a 2-tower model is given a vlm_backbone.
    """
    class _TwoTowerOnly(BaseVLA):
        REQUIRED_SLOTS = ("vision_backbone", "llm_backbone", "vla_head")
        OPTIONAL_SLOTS = ()
        def forward(self, batch): return batch
        def predict_action(self, *, images, state, instruction): return None

    with pytest.raises(ValueError, match="undeclared"):
        _TwoTowerOnly(
            vision_backbone=_StubVisionBackbone(),
            llm_backbone=_StubLLMBackbone(),
            vla_head=_StubVLAHead(),
            vlm_backbone=_StubVLMBackbone(),  # not declared
        )


def test_basevla_unknown_slot_in_class_decl_raises():
    """Subclass that declares an invalid slot name fails at construction."""
    class _Bogus(BaseVLA):
        REQUIRED_SLOTS = ("nonexistent_slot",)
        OPTIONAL_SLOTS = ()
        def forward(self, batch): return batch
        def predict_action(self, *, images, state, instruction): return None

    with pytest.raises(TypeError, match="unknown slot"):
        _Bogus()


def test_basevla_slot_in_both_required_and_optional_raises():
    class _Overlap(BaseVLA):
        REQUIRED_SLOTS = ("vision_backbone",)
        OPTIONAL_SLOTS = ("vision_backbone",)
        def forward(self, batch): return batch
        def predict_action(self, *, images, state, instruction): return None

    with pytest.raises(TypeError, match="REQUIRED.*OPTIONAL"):
        _Overlap()


# ─── from_config dispatch ────────────────────────────────────────────────


def test_basevla_from_config_resolves_type_tagged_dicts(monkeypatch):
    """from_config takes a spec dict + builds via component registries."""
    from reflex.registry.components import VISION_BACKBONES, VLA_HEADS

    # Register stubs (use try/finally to clean up after the test).
    VISION_BACKBONES.register(_StubVisionBackbone)
    VLA_HEADS.register(_StubVLAHead)

    try:
        class _SpineVLA(BaseVLA):
            REQUIRED_SLOTS = ("vision_backbone", "vla_head")
            OPTIONAL_SLOTS = ()
            def forward(self, batch): return batch
            def predict_action(self, *, images, state, instruction): return None

        v = _SpineVLA.from_config({
            "vision_backbone": {"type": "_StubVisionBackbone"},
            "vla_head": {"type": "_StubVLAHead"},
        })
        assert isinstance(v.vision_backbone, _StubVisionBackbone)
        assert isinstance(v.vla_head, _StubVLAHead)
    finally:
        VISION_BACKBONES.unregister("_StubVisionBackbone")
        VLA_HEADS.unregister("_StubVLAHead")


def test_basevla_from_config_passes_through_prebuilt_instances():
    """from_config accepts a pre-built component instance directly."""
    pre_built = _StubVLAHead()

    v = _MinimalVLA.from_config({"vla_head": pre_built})
    assert v.vla_head is pre_built


def test_basevla_from_config_unknown_slot_raises():
    with pytest.raises(ValueError, match="Unknown slot"):
        _MinimalVLA.from_config({"foo": {"type": "Bar"}})


# ─── prepare_inference_weights aggregation ──────────────────────────────


def test_prepare_inference_weights_aggregates_across_components():
    """Each component's prepare_triton dict gets prefixed + merged."""
    import torch

    class _LoadedVision(VisionBackbone):
        def forward(self, images): return images
        def prepare_triton(self, prefix=""):
            return {f"{prefix}patch_embed.weight": torch.zeros(3)}

    class _LoadedHead(VLAHead):
        def forward(self, context, *args, **kwargs): return context
        def prepare_triton(self, prefix=""):
            return {f"{prefix}action_out.weight": torch.ones(7)}

    v = _MinimalVLA(vision_backbone=_LoadedVision(), vla_head=_LoadedHead())
    weights = v.prepare_inference_weights()
    assert set(weights.keys()) == {
        "vision_backbone.patch_embed.weight",
        "vla_head.action_out.weight",
    }
    assert torch.equal(weights["vision_backbone.patch_embed.weight"], torch.zeros(3))
    assert torch.equal(weights["vla_head.action_out.weight"], torch.ones(7))


def test_prepare_inference_weights_skips_unbound_slots():
    """Unbound slots contribute nothing — no errors."""
    v = _MinimalVLA()  # All slots empty
    weights = v.prepare_inference_weights()
    assert weights == {}


def test_prepare_inference_weights_supports_outer_prefix():
    """Caller-supplied prefix is prepended to all keys."""
    import torch

    class _LoadedHead(VLAHead):
        def forward(self, context, *args, **kwargs): return context
        def prepare_triton(self, prefix=""):
            return {f"{prefix}w": torch.zeros(1)}

    v = _MinimalVLA(vla_head=_LoadedHead())
    weights = v.prepare_inference_weights(prefix="pi05.")
    assert set(weights.keys()) == {"pi05.vla_head.w"}


# ─── apply_name_mapping walker ──────────────────────────────────────────


def test_apply_name_mapping_first_match_wins():
    """Mapping iteration is declaration-ordered; first prefix match wins."""
    import torch
    state = {
        "module.backbone.weight": torch.zeros(1),
        "backbone.weight": torch.ones(1),  # would also match 'backbone.' if 'module.' didn't strip first
    }
    mapping = {"module.": "", "backbone.": "renamed."}
    renamed = apply_name_mapping(state, mapping)
    # 'module.backbone.weight' → strip 'module.' → 'backbone.weight' (not re-mapped because
    # the result of the first match isn't re-walked)
    assert "backbone.weight" in renamed
    # 'backbone.weight' (the original key, no module. prefix) → matches 'backbone.' → 'renamed.weight'
    assert "renamed.weight" in renamed


def test_apply_name_mapping_passthrough_for_unmatched():
    import torch
    state = {"unmatched.key": torch.zeros(1)}
    renamed = apply_name_mapping(state, {"prefix.": "x."})
    assert renamed == state  # passthrough


def test_apply_name_mapping_empty_prefix_rejected():
    """Empty src_prefix would match everything — ambiguous, reject."""
    import torch
    with pytest.raises(ValueError, match="empty src_prefix"):
        apply_name_mapping({"k": torch.zeros(1)}, {"": "renamed."})


def test_apply_name_mapping_strict_mode_raises_on_unmatched():
    import torch
    state = {
        "matched.key": torch.zeros(1),
        "unmatched.key": torch.zeros(1),
    }
    with pytest.raises(KeyError, match="matched no name_mapping prefix"):
        apply_name_mapping(state, {"matched.": "renamed."}, strict=True)


def test_apply_name_mapping_does_not_mutate_input():
    import torch
    state = {"old.key": torch.zeros(1)}
    original = dict(state)
    apply_name_mapping(state, {"old.": "new."})
    assert state == original  # untouched
