"""Tests for SigLIPBackbone — vision-tower component on the BaseVLA spine.

Lift #1 Day 4b per `features/03_export/basevla-spine_plan.md`. Validates:

- registration on the VISION_BACKBONES registry
- construction via pre-built model (the Day 4c path)
- construction via model_id is documented but NOT tested here (would hit
  HuggingFace at test time, which is brittle + slow)
- forward() shape contract on a tiny stub
- prepare_triton() returns expected key structure
- input validation (must provide exactly one of model_id / model)

The pi0/pi05/smolvla integration parity tests live in their respective
per-VLA test files (Day 4f+).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tether.models.vision import VisionBackbone
from tether.models.vision.siglip_backbone import SigLIPBackbone
from tether.registry.components import VISION_BACKBONES


# ─── Registration ───────────────────────────────────────────────────────


def test_siglip_backbone_registered():
    """The @VISION_BACKBONES.register decorator fires at import time."""
    assert "SigLIPBackbone" in VISION_BACKBONES
    assert VISION_BACKBONES.get("SigLIPBackbone") is SigLIPBackbone


def test_siglip_backbone_is_vision_backbone_subclass():
    """Inherits from the spine ABC."""
    assert issubclass(SigLIPBackbone, VisionBackbone)


# ─── Construction validation ────────────────────────────────────────────


def test_rejects_both_model_and_model_id():
    """Exactly one of model / model_id must be provided."""
    stub = _make_stub_model()
    with pytest.raises(ValueError, match="exactly one"):
        SigLIPBackbone(model_id="anything", model=stub)


def test_rejects_neither_model_nor_model_id():
    with pytest.raises(ValueError, match="exactly one"):
        SigLIPBackbone()


def test_constructs_with_pre_built_model():
    """The model= path used by Day 4c PaliGemma extraction."""
    stub = _make_stub_model()
    backbone = SigLIPBackbone(model=stub)
    assert backbone.model is stub
    assert backbone.output_hidden_states is False  # default


# ─── Forward shape contract ─────────────────────────────────────────────


def test_forward_returns_last_hidden_state():
    """forward(pixel_values) returns the model's last_hidden_state tensor."""
    stub = _make_stub_model(hidden_dim=8, num_patches=4)
    backbone = SigLIPBackbone(model=stub)

    images = torch.randn(2, 3, 16, 16)  # batch=2, dummy 16×16 RGB
    out = backbone(images)
    # Stub returns [batch, num_patches, hidden_dim]
    assert out.shape == (2, 4, 8)


def test_forward_ignores_extra_args():
    """ABC signature has *args/**kwargs — extras dropped, no crash."""
    stub = _make_stub_model(hidden_dim=8, num_patches=4)
    backbone = SigLIPBackbone(model=stub)
    images = torch.randn(1, 3, 16, 16)
    out = backbone(images, "ignored", foo=42)
    assert out.shape == (1, 4, 8)


# ─── prepare_triton (lift #3 interface) ─────────────────────────────────


def test_prepare_triton_returns_all_params():
    """prepare_triton enumerates every parameter under prefix."""
    stub = _make_stub_model(hidden_dim=8, num_patches=4)
    backbone = SigLIPBackbone(model=stub)

    weights = backbone.prepare_triton(prefix="vision.")
    # Stub model has one nn.Parameter named 'patch_embed.weight'
    assert "vision.model.patch_embed.weight" in weights
    assert weights["vision.model.patch_embed.weight"].requires_grad is False


def test_prepare_triton_default_prefix():
    """Default prefix='' produces top-level keys."""
    stub = _make_stub_model(hidden_dim=8, num_patches=4)
    backbone = SigLIPBackbone(model=stub)
    weights = backbone.prepare_triton()
    assert "model.patch_embed.weight" in weights


# ─── Helpers ────────────────────────────────────────────────────────────


def _make_stub_model(hidden_dim: int = 1152, num_patches: int = 256) -> nn.Module:
    """Build a minimal SigLIP-shaped stub for testing without HF downloads.

    Returns an nn.Module that exposes:
    - `.patch_embed.weight` (so prepare_triton has at least one param)
    - `__call__(pixel_values=...)` returning a namespace with `.last_hidden_state`
      of shape `[batch, num_patches, hidden_dim]`
    """
    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.weight = nn.Parameter(torch.zeros(1, 1))

        def forward(self, *, pixel_values, **kwargs):
            batch = pixel_values.shape[0]
            from types import SimpleNamespace
            return SimpleNamespace(
                last_hidden_state=torch.zeros(batch, num_patches, hidden_dim),
            )

    return _Stub()
