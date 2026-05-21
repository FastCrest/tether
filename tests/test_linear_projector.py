"""Tests for LinearProjector — the first concrete spine component.

Lift #1 Day 4a per `features/03_export/basevla-spine_plan.md`. Validates:

- registration on the PROJECTORS registry
- construction via from_config + build_from_cfg
- forward() applies linear projection correctly
- prepare_triton() returns the expected keys + shapes
- bias=False path
- input validation (in_dim/out_dim < 1)

Proves the spine + Registry + Projector ABC compose end-to-end with a
real registered class — foundation that downstream component PRs build
on (SigLIPBackbone, PaliGemmaBackbone, FlowMatchingHead).
"""
from __future__ import annotations

import pytest
import torch

from reflex.models.projectors import Projector
from reflex.models.projectors.linear_projector import LinearProjector
from reflex.registry.components import PROJECTORS
from reflex.registry.builder import build_from_cfg


# ─── Registration + construction ────────────────────────────────────────


def test_linear_projector_registered_in_projectors_registry():
    """The @PROJECTORS.register decorator fires at module import time."""
    assert "LinearProjector" in PROJECTORS
    assert PROJECTORS.get("LinearProjector") is LinearProjector


def test_linear_projector_is_projector_subclass():
    """Inherits from the spine ABC."""
    assert issubclass(LinearProjector, Projector)


def test_linear_projector_constructs_via_kwargs():
    proj = LinearProjector(in_dim=32, out_dim=64)
    assert proj.in_dim == 32
    assert proj.out_dim == 64
    assert proj.linear.in_features == 32
    assert proj.linear.out_features == 64
    assert proj.linear.bias is not None  # default bias=True


def test_linear_projector_constructs_without_bias():
    proj = LinearProjector(in_dim=8, out_dim=16, bias=False)
    assert proj.linear.bias is None


def test_linear_projector_via_build_from_cfg():
    """The intended construction path — type-tagged dict through Registry."""
    proj = build_from_cfg(
        {"type": "LinearProjector", "in_dim": 32, "out_dim": 960},
        PROJECTORS,
    )
    assert isinstance(proj, LinearProjector)
    assert proj.in_dim == 32
    assert proj.out_dim == 960


# ─── Forward + correctness ──────────────────────────────────────────────


def test_forward_applies_linear():
    """forward(x) = x @ W^T + b for nn.Linear semantics."""
    proj = LinearProjector(in_dim=4, out_dim=2)
    # Set weights to known values for determinism
    with torch.no_grad():
        proj.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                                [0.5, -1.0, 0.5, -1.0]]))
        proj.linear.bias.copy_(torch.tensor([0.1, -0.1]))

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # batch=1, in_dim=4
    out = proj(x)
    assert out.shape == (1, 2)
    # row 0: 1+2+3+4+0.1 = 10.1; row 1: 0.5-1+0.5-1-0.1 = -1.1
    assert torch.allclose(out, torch.tensor([[10.1, -1.1]]), atol=1e-6)


def test_forward_ignores_extra_args():
    """ABC signature requires extra args/kwargs slot — ignored, doesn't crash."""
    proj = LinearProjector(in_dim=4, out_dim=4)
    x = torch.randn(2, 4)
    out = proj(x, "extra_arg", kwarg=42)  # extras dropped
    assert out.shape == (2, 4)


# ─── prepare_triton (lift #3 interface) ─────────────────────────────────


def test_prepare_triton_with_bias():
    proj = LinearProjector(in_dim=4, out_dim=2, bias=True)
    weights = proj.prepare_triton()
    assert set(weights.keys()) == {"linear.weight", "linear.bias"}
    assert weights["linear.weight"].shape == (2, 4)
    assert weights["linear.bias"].shape == (2,)


def test_prepare_triton_without_bias():
    proj = LinearProjector(in_dim=4, out_dim=2, bias=False)
    weights = proj.prepare_triton()
    assert set(weights.keys()) == {"linear.weight"}
    assert "linear.bias" not in weights


def test_prepare_triton_with_prefix():
    proj = LinearProjector(in_dim=4, out_dim=2)
    weights = proj.prepare_triton(prefix="state_proj.")
    assert set(weights.keys()) == {"state_proj.linear.weight", "state_proj.linear.bias"}


def test_prepare_triton_returns_detached_tensors():
    """Returned tensors should be detached from any computation graph
    (no autograd link back to nn.Parameter)."""
    proj = LinearProjector(in_dim=4, out_dim=2)
    weights = proj.prepare_triton()
    assert not weights["linear.weight"].requires_grad


# ─── Input validation ───────────────────────────────────────────────────


def test_rejects_zero_in_dim():
    with pytest.raises(ValueError, match="in_dim"):
        LinearProjector(in_dim=0, out_dim=1)


def test_rejects_zero_out_dim():
    with pytest.raises(ValueError, match="out_dim"):
        LinearProjector(in_dim=1, out_dim=0)


def test_rejects_negative_dims():
    with pytest.raises(ValueError):
        LinearProjector(in_dim=-1, out_dim=4)
    with pytest.raises(ValueError):
        LinearProjector(in_dim=4, out_dim=-2)


# ─── PyTorch nn.Module compatibility ────────────────────────────────────


def test_pytorch_state_dict_round_trip():
    """LinearProjector is an nn.Module — standard state_dict semantics work."""
    src = LinearProjector(in_dim=8, out_dim=16)
    sd = src.state_dict()
    assert "linear.weight" in sd
    assert "linear.bias" in sd

    dst = LinearProjector(in_dim=8, out_dim=16)
    dst.load_state_dict(sd)

    # Forward identical
    x = torch.randn(1, 8)
    torch.testing.assert_close(src(x), dst(x))


def test_to_device_moves_params():
    """The nn.Module .to() machinery moves weights."""
    proj = LinearProjector(in_dim=4, out_dim=2)
    # Cast (CPU smoke; we don't have a GPU in CI, just verify the call works)
    proj.to(torch.float32)
    assert proj.linear.weight.dtype == torch.float32
