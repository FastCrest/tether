"""Lift #3 Day 1 — `prepare_triton(prefix)` per-component contract tests.

The substrate that makes `--inference-only-weights` mode possible: every
spine-registered component class exposes a `prepare_triton(prefix)` method
that flattens its weights into a single dict keyed by ``{prefix}{name}``.

Contract (per ``features/01_serve/inference-only-weights_plan.md`` Day 1):

1. Returns a dict[str, torch.Tensor]
2. All keys carry the supplied prefix
3. Empty prefix vs ``"foo."`` prefix produce the same key shapes
4. All values are detached from their original ``nn.Parameter`` — writing
   into the returned tensor must NOT mutate the source Parameter (no aliasing)
5. ABC base classes return ``{}`` (no-op default for unused slots)
6. No duplicate keys within a single component's output

These pin the contract so Lift #5 (Triton kernels) can rely on it.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ─── ABC base-class no-op default ─────────────────────────────────────


def test_vision_backbone_abc_default_is_no_op():
    """The Vision ABC's prepare_triton returns {} — unused slots stay empty."""
    from reflex.models.vision import VisionBackbone

    class _Stub(VisionBackbone):
        def forward(self, images, *a, **kw):
            return images

    stub = _Stub()
    assert stub.prepare_triton("") == {}
    assert stub.prepare_triton("foo.") == {}


def test_llm_backbone_abc_default_is_no_op():
    from reflex.models.llm import LLMBackbone

    class _Stub(LLMBackbone):
        def forward(self, *a, **kw):
            return None

    assert _Stub().prepare_triton("") == {}


def test_vlm_backbone_abc_default_is_no_op():
    from reflex.models.vlm import VLMBackbone

    class _Stub(VLMBackbone):
        def forward(self, *a, **kw):
            return None

    assert _Stub().prepare_triton("") == {}


def test_projector_abc_default_is_no_op():
    from reflex.models.projectors import Projector

    class _Stub(Projector):
        def forward(self, x, *a, **kw):
            return x

    assert _Stub().prepare_triton("") == {}


def test_vla_head_abc_default_is_no_op():
    from reflex.models.heads import VLAHead

    class _Stub(VLAHead):
        def forward(self, *a, **kw):
            return None

    assert _Stub().prepare_triton("") == {}


# ─── Concrete: LinearProjector ────────────────────────────────────────


def test_linear_projector_prepare_triton_with_bias():
    from reflex.models.projectors.linear_projector import LinearProjector

    proj = LinearProjector(in_dim=8, out_dim=16, bias=True)
    out = proj.prepare_triton(prefix="state_proj.")

    assert isinstance(out, dict)
    assert set(out.keys()) == {"state_proj.linear.weight", "state_proj.linear.bias"}
    assert out["state_proj.linear.weight"].shape == (16, 8)
    assert out["state_proj.linear.bias"].shape == (16,)
    # All values are tensors:
    assert all(isinstance(v, torch.Tensor) for v in out.values())


def test_linear_projector_prepare_triton_no_bias():
    from reflex.models.projectors.linear_projector import LinearProjector

    proj = LinearProjector(in_dim=8, out_dim=16, bias=False)
    out = proj.prepare_triton(prefix="")
    assert set(out.keys()) == {"linear.weight"}


def test_linear_projector_prepare_triton_empty_prefix_matches_named():
    from reflex.models.projectors.linear_projector import LinearProjector

    proj = LinearProjector(in_dim=4, out_dim=8)
    out = proj.prepare_triton(prefix="")
    named = dict(proj.named_parameters())
    # Same key set (modulo prefix=""), same shapes.
    assert set(out.keys()) == set(named.keys())
    for k in out:
        assert out[k].shape == named[k].shape


# ─── Aliasing regression: writes to the returned tensor must NOT mutate
# ─── the original nn.Parameter. Catches the Day 1 risk-gate #1 bug.


def test_linear_projector_no_aliasing_on_returned_tensors():
    from reflex.models.projectors.linear_projector import LinearProjector

    proj = LinearProjector(in_dim=4, out_dim=8)
    original_weight = proj.linear.weight.clone()

    out = proj.prepare_triton(prefix="")
    returned = out["linear.weight"]

    # If returned aliases the Parameter's underlying storage, mutating
    # `returned` would also mutate `proj.linear.weight.data`. The contract
    # says it MUST NOT alias.
    with torch.no_grad():
        returned.add_(99.0)  # in-place; would mutate the source if aliased

    # The Parameter should be unchanged.
    assert torch.equal(
        proj.linear.weight, original_weight
    ), "prepare_triton output aliases the source Parameter (mutating one mutates the other) — violates Day 1 contract risk-gate #1"


# ─── Concrete: head/vision/llm wrappers (where we can build a stub) ───


def test_flow_matching_head_prepare_triton_namespaces_under_expert_stack():
    """FlowMatchingHead wraps an ExpertStack; prepare_triton must nest the
    wrapped module's params under ``{prefix}expert_stack.{name}``."""
    from reflex.models.heads.flow_matching_head import FlowMatchingHead

    # Minimum stub: a tiny nn.Module pretending to be an ExpertStack.
    class _StubStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

    head = FlowMatchingHead(expert_stack=_StubStack())
    out = head.prepare_triton(prefix="head.")

    assert set(out.keys()) == {"head.expert_stack.fc.weight", "head.expert_stack.fc.bias"}
    assert all(isinstance(v, torch.Tensor) for v in out.values())


# ─── No duplicate keys within a single component's output ─────────────


def test_linear_projector_no_duplicate_keys():
    from reflex.models.projectors.linear_projector import LinearProjector

    proj = LinearProjector(in_dim=8, out_dim=16, bias=True)
    keys = list(proj.prepare_triton(prefix="any.").keys())
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


def test_flow_matching_head_no_duplicate_keys():
    from reflex.models.heads.flow_matching_head import FlowMatchingHead

    class _StubStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

    head = FlowMatchingHead(expert_stack=_StubStack())
    keys = list(head.prepare_triton(prefix="").keys())
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


# ─── Prefix consistency: every key starts with the supplied prefix ────


@pytest.mark.parametrize("prefix", ["", "foo.", "deeply.nested.prefix."])
def test_prefix_consistency_linear_projector(prefix: str):
    from reflex.models.projectors.linear_projector import LinearProjector

    proj = LinearProjector(in_dim=4, out_dim=8)
    out = proj.prepare_triton(prefix=prefix)
    for k in out:
        assert k.startswith(prefix), f"key {k!r} doesn't start with prefix {prefix!r}"


@pytest.mark.parametrize("prefix", ["", "vla_head."])
def test_prefix_consistency_flow_matching_head(prefix: str):
    from reflex.models.heads.flow_matching_head import FlowMatchingHead

    class _StubStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

    head = FlowMatchingHead(expert_stack=_StubStack())
    out = head.prepare_triton(prefix=prefix)
    for k in out:
        assert k.startswith(prefix), f"key {k!r} doesn't start with prefix {prefix!r}"
