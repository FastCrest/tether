"""Tests for FlowMatchingHead — spine wrapper for pi0/pi05/smolvla ExpertStack.

Lift #1 Day 4e per `features/03_export/basevla-spine_plan.md`. Validates:

- registration on the VLA_HEADS registry
- construction with pre-built expert_stack (the Day 4f composition path)
- construction validation (exactly one of expert_stack / state_dict;
  state_dict requires vla_family; SmolVLA requires head_dim)
- forward() delegates to wrapped expert_stack with correct kwargs
- prepare_triton() flattens wrapped stack's params under "expert_stack." prefix
- ABC subclass

The bit-identical-behavior parity tests vs the existing pi0/pi05/smolvla
exporters live in Day 4g (after Pi0VLA composition lands in Day 4f).
This file tests the wrapper contract.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tether.models.heads import VLAHead
from tether.models.heads.flow_matching_head import FlowMatchingHead
from tether.registry.components import VLA_HEADS


# ─── Registration + ABC ─────────────────────────────────────────────────


def test_flow_matching_head_registered():
    assert "FlowMatchingHead" in VLA_HEADS
    assert VLA_HEADS.get("FlowMatchingHead") is FlowMatchingHead


def test_flow_matching_head_is_vla_head_subclass():
    assert issubclass(FlowMatchingHead, VLAHead)


# ─── Construction validation ────────────────────────────────────────────


def test_rejects_both_expert_stack_and_state_dict():
    stub = _make_stub_expert_stack()
    with pytest.raises(ValueError, match="exactly one"):
        FlowMatchingHead(expert_stack=stub, state_dict={})


def test_rejects_neither_expert_stack_nor_state_dict():
    with pytest.raises(ValueError, match="exactly one"):
        FlowMatchingHead()


def test_state_dict_requires_vla_family():
    """state_dict alone is ambiguous — pi0 / pi05 / smolvla all have
    different builders."""
    with pytest.raises(ValueError, match="vla_family"):
        FlowMatchingHead(state_dict={"some.key": torch.zeros(1)})


def test_state_dict_rejects_unknown_vla_family():
    with pytest.raises(ValueError, match="vla_family"):
        FlowMatchingHead(state_dict={}, vla_family="invented_family")


def test_constructs_with_pre_built_expert_stack():
    stub = _make_stub_expert_stack()
    head = FlowMatchingHead(expert_stack=stub)
    assert head.expert_stack is stub


# ─── forward() delegation ───────────────────────────────────────────────


def test_forward_delegates_to_wrapped_stack():
    """forward() passes through to expert_stack with the canonical kwarg
    set (noisy_actions, timestep, position_ids, vlm_k, vlm_v, prefix_offset,
    kv_mask)."""
    stub = _make_stub_expert_stack()
    head = FlowMatchingHead(expert_stack=stub)

    noisy = torch.randn(2, 50, 7)
    timestep = torch.tensor([0.5, 0.5])
    pos_ids = torch.arange(50).unsqueeze(0).expand(2, -1)

    out = head(noisy, timestep, pos_ids)
    # Stub returns the input shape — verifies forward succeeded
    assert out.shape == (2, 50, 7)
    # Stub recorded the kwargs it received
    assert stub.last_call["noisy_actions"] is noisy
    assert stub.last_call["timestep"] is timestep
    assert stub.last_call["position_ids"] is pos_ids
    # Optional kwargs default to None
    assert stub.last_call["vlm_k"] is None
    assert stub.last_call["vlm_v"] is None


def test_forward_passes_vlm_k_v_when_provided():
    """Cross-attention path: vlm_k + vlm_v get forwarded as kwargs."""
    stub = _make_stub_expert_stack()
    head = FlowMatchingHead(expert_stack=stub)

    noisy = torch.randn(1, 10, 7)
    vlm_k = torch.randn(2, 1, 256, 16)  # [num_layers, B, seq, kv_dim]
    vlm_v = torch.randn(2, 1, 256, 16)
    head(noisy, vlm_k=vlm_k, vlm_v=vlm_v)
    assert stub.last_call["vlm_k"] is vlm_k
    assert stub.last_call["vlm_v"] is vlm_v


def test_forward_ignores_extra_positional_args():
    """ABC signature accepts *args/**kwargs — extras dropped cleanly."""
    stub = _make_stub_expert_stack()
    head = FlowMatchingHead(expert_stack=stub)
    noisy = torch.randn(1, 10, 7)
    head(noisy, None, None, "ignored_positional", random_kwarg=42)
    # No crash; expert_stack called with the named kwargs from the signature
    assert stub.last_call["noisy_actions"] is noisy


# ─── prepare_triton flattens wrapped stack's params ─────────────────────


def test_prepare_triton_returns_expert_stack_params():
    stub = _make_stub_expert_stack()
    head = FlowMatchingHead(expert_stack=stub)
    weights = head.prepare_triton(prefix="vla.head.")

    # Stub has one named param 'action_in_proj.weight'
    expected_key = "vla.head.expert_stack.action_in_proj.weight"
    assert expected_key in weights, (
        f"Expected {expected_key} in {sorted(weights.keys())}"
    )
    assert weights[expected_key].requires_grad is False


def test_prepare_triton_default_prefix_empty():
    stub = _make_stub_expert_stack()
    head = FlowMatchingHead(expert_stack=stub)
    weights = head.prepare_triton()
    assert "expert_stack.action_in_proj.weight" in weights


# ─── SmolVLA-only constraint: head_dim required ─────────────────────────


def test_state_dict_smolvla_requires_head_dim(monkeypatch):
    """SmolVLA's build_expert_stack signature requires the VLM head_dim;
    the head dispatcher passes it through."""
    # Patch the lazy builder import so we don't actually run smolvla build.
    captured = {}

    def fake_build(state_dict, head_dim):
        captured["called"] = True
        captured["head_dim"] = head_dim
        return _make_stub_expert_stack(), {}

    import tether.exporters.smolvla as smolvla_mod
    monkeypatch.setattr(smolvla_mod, "build_expert_stack", fake_build)

    with pytest.raises(ValueError, match="head_dim required"):
        FlowMatchingHead(state_dict={}, vla_family="smolvla")  # no head_dim

    # With head_dim — succeeds + forwards the value
    head = FlowMatchingHead(state_dict={}, vla_family="smolvla", head_dim=64)
    assert captured["called"] is True
    assert captured["head_dim"] == 64
    assert isinstance(head, FlowMatchingHead)


# ─── Helpers ────────────────────────────────────────────────────────────


def _make_stub_expert_stack() -> nn.Module:
    """Minimal ExpertStack-shaped stub that records its last forward kwargs.

    Returns the noisy_actions tensor unchanged so shape-preservation tests
    can verify the call path.
    """

    class _StubStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = nn.Linear(7, 8)
            self.last_call: dict = {}

        def forward(
            self,
            noisy_actions,
            timestep=None,
            position_ids=None,
            vlm_k=None,
            vlm_v=None,
            prefix_offset=None,
            kv_mask=None,
        ):
            self.last_call = dict(
                noisy_actions=noisy_actions,
                timestep=timestep,
                position_ids=position_ids,
                vlm_k=vlm_k,
                vlm_v=vlm_v,
                prefix_offset=prefix_offset,
                kv_mask=kv_mask,
            )
            return noisy_actions

    return _StubStack()
