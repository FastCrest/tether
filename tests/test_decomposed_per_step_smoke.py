"""Local CPU smoke test for the per-step expert wrapper (gate 1).

Validates the ``Pi05ExpertPerStepWrapper`` factory produces a torch nn.Module
that:
- accepts the per-step input shape (36 past_kv tensors + prefix_pad_masks
  + x_t + t)
- calls the model's ``denoise_step`` exactly once with a scalar timestep
- returns v_t (velocity) of shape ``(B, chunk_size, action_dim)`` — NOT
  the fully-denoised actions

This is gate 1 of the 6-gate ship sequence for per-step expert ONNX export.
Gate 1 = local CPU smoke (this file). Gate 2 = 1-NFE+RTC config-time guard
(``test_rtc_adapter_per_step.py``). Gate 3 onwards run on Modal A100.

Uses a mock pi05_model — full ONNX export with a real model is gate 3
(Modal parity test). This gate validates the WRAPPER LOGIC at zero cost.

Spec: ``reflex_context/features/03_export/per-step-expert-export.md``
Research: ``reflex_context/features/03_export/per-step-expert-export_research.md``
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tether.exporters.decomposed import (
    PI05_HEAD_DIM,
    PI05_KV_HEADS,
    PI05_PALIGEMMA_LAYERS,
    Pi05ExpertPerStepWrapper,
)


# ──────────────────────────────────────────────────────────────────
# Mock pi05_model — minimum surface the wrapper accesses.
# ──────────────────────────────────────────────────────────────────
class _DenoiseStepRecorder:
    """Capture the args/kwargs that denoise_step was called with so the
    test can assert the per-step contract was honored."""

    def __init__(self):
        self.calls: list[dict] = []
        self.return_shape = (1, 50, 32)  # (B, chunk_size, action_dim) for pi05

    def __call__(self, **kwargs):
        # Record everything the wrapper passed in
        self.calls.append({k: v for k, v in kwargs.items()})
        # Return a velocity tensor of the expected shape + dtype
        return torch.randn(*self.return_shape, dtype=torch.float32)


class _MockPi05Model(nn.Module):
    """Mock pi05 model exposing only what _Pi05ExpertPerStepWrapper reads:
    - action_in_proj.weight.dtype (used for dtype coercion of x_t)
    - denoise_step (the per-step velocity callable)
    - target_time_embed_mlp / state_proj (presence checks for variant detection)
    """

    def __init__(self, *, snapflow: bool = False, state_out: bool = False):
        super().__init__()
        # action_in_proj: only the weight.dtype matters for the wrapper path
        self.action_in_proj = nn.Linear(32, 256)  # in=action_dim, out=hidden
        # Variant attributes — wrapper does hasattr() checks
        if snapflow:
            self.target_time_embed_mlp = nn.Linear(1, 256)
        if state_out:
            self.state_proj = nn.Linear(7, 256)
        # denoise_step recorder
        self._denoise_recorder = _DenoiseStepRecorder()

    def denoise_step(self, **kwargs):
        return self._denoise_recorder(**kwargs)


# ──────────────────────────────────────────────────────────────────
# Test fixtures: dummy inputs matching the per-step wrapper signature
# ──────────────────────────────────────────────────────────────────
def _make_inputs(B: int = 1, prefix_seq_len: int = 968, chunk: int = 50,
                 action_dim: int = 32, *, with_state: bool = False):
    """Build the input tuple the wrapper's forward(*args) expects."""
    past_kv_shape = (B, PI05_KV_HEADS, prefix_seq_len, PI05_HEAD_DIM)
    past_kvs = [
        torch.randn(past_kv_shape, dtype=torch.float32)
        for _ in range(PI05_PALIGEMMA_LAYERS * 2)
    ]
    prefix_pad_masks = torch.ones(B, prefix_seq_len, dtype=torch.bool)
    x_t = torch.randn(B, chunk, action_dim, dtype=torch.float32)
    t = torch.full((B,), 1.0, dtype=torch.float32)  # scalar timestep
    args = (*past_kvs, prefix_pad_masks, x_t, t)
    if with_state:
        state = torch.randn(B, 7, dtype=torch.float32)
        args = (*args, state)
    return args


class TestPerStepExpertWrapper:
    """Wrapper-level CPU smoke: instantiation, forward, output shape, contract."""

    def test_instantiation_default(self):
        model = _MockPi05Model(snapflow=False, state_out=False)
        wrapper = Pi05ExpertPerStepWrapper(model)
        assert wrapper is not None
        assert wrapper._is_snapflow is False
        assert wrapper._is_state_out is False

    def test_instantiation_snapflow(self):
        model = _MockPi05Model(snapflow=True, state_out=False)
        wrapper = Pi05ExpertPerStepWrapper(model)
        assert wrapper._is_snapflow is True

    def test_instantiation_state_out(self):
        model = _MockPi05Model(snapflow=False, state_out=True)
        wrapper = Pi05ExpertPerStepWrapper(model)
        assert wrapper._is_state_out is True

    def test_forward_returns_v_t_shape(self):
        """Output must be (B, chunk_size, action_dim) — velocity, not actions."""
        model = _MockPi05Model()
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs()
        v_t = wrapper(*args)
        assert v_t.shape == (1, 50, 32), f"got {tuple(v_t.shape)}"
        assert v_t.dtype == torch.float32

    def test_forward_calls_denoise_step_exactly_once(self):
        """Single denoise step per forward call — no internal Euler loop.

        This is the core contract that distinguishes per-step from baked-loop:
        baked wrapper calls denoise_step num_steps times in its forward;
        per-step wrapper calls it exactly once and returns v_t to the caller.
        """
        model = _MockPi05Model()
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs()
        wrapper(*args)
        assert len(model._denoise_recorder.calls) == 1

    def test_denoise_step_called_with_correct_kwargs(self):
        """Wrapper must pass past_key_values + x_t + timestep + prefix_pad_masks."""
        model = _MockPi05Model()
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs()
        wrapper(*args)
        call = model._denoise_recorder.calls[0]
        assert "past_key_values" in call
        assert "x_t" in call
        assert "timestep" in call
        assert "prefix_pad_masks" in call
        # Default variant: no target_time, no state
        assert "target_time" not in call
        assert "state" not in call

    def test_snapflow_passes_target_time(self):
        model = _MockPi05Model(snapflow=True)
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs()
        wrapper(*args)
        call = model._denoise_recorder.calls[0]
        assert "target_time" in call
        # SnapFlow student is 1-NFE — target_time should be 1.0
        assert torch.allclose(call["target_time"], torch.ones(1, dtype=torch.float32))

    def test_state_out_passes_state(self):
        model = _MockPi05Model(state_out=True)
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs(with_state=True)
        wrapper(*args)
        call = model._denoise_recorder.calls[0]
        assert "state" in call
        assert call["state"].shape == (1, 7)

    def test_timestep_is_scalar_per_batch(self):
        """t must be (B,) shape per the per-step contract — not a Python float."""
        model = _MockPi05Model()
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs(B=1)
        wrapper(*args)
        call = model._denoise_recorder.calls[0]
        timestep = call["timestep"]
        assert isinstance(timestep, torch.Tensor)
        assert timestep.shape == (1,)

    def test_past_kv_reconstructed_with_all_layers(self):
        """Wrapper must reconstruct DynamicCache with all 18 paligemma layers."""
        model = _MockPi05Model()
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs()
        wrapper(*args)
        call = model._denoise_recorder.calls[0]
        cache = call["past_key_values"]
        # DynamicCache exposes layers via .layers list (transformers 5.3+)
        # OR via .key_cache (pre-5.3). Either way, we should see 18 layers.
        if hasattr(cache, "layers"):
            assert len(cache.layers) == PI05_PALIGEMMA_LAYERS
        else:
            assert len(cache.key_cache) == PI05_PALIGEMMA_LAYERS

    def test_dtype_coercion_x_t_to_action_dtype(self):
        """If x_t comes in float32 but model wants float16, wrapper coerces."""
        model = _MockPi05Model()
        # Simulate half-precision model
        model.action_in_proj = nn.Linear(32, 256).half()
        wrapper = Pi05ExpertPerStepWrapper(model)
        args = _make_inputs()
        wrapper(*args)
        call = model._denoise_recorder.calls[0]
        # Wrapper should have coerced x_t to float16 before calling denoise_step
        assert call["x_t"].dtype == torch.float16
