"""Unit tests for SnapFlow math + distill registry.

These are GPU-free: we use tiny synthetic velocity functions (3-layer
MLPs, ~100 params) to verify the math invariants. Full VLA integration
lives in the Modal script + the LIBERO drop-gate integration test.

Pins the SnapFlow contracts:
  - flow_matching_interp produces correct linear interpolation
  - two_step_euler_shortcut runs under no_grad + produces right shape
  - snapflow_loss_step returns (tensor, SnapFlowLosses) with valid math
  - ZeroInitTargetTimeEmbedding produces EXACTLY zero at init
  - sinusoidal_time_embedding shapes + basic properties
  - get_method('dmpo' / 'pi_flow' / 'consistency') all raise with
    actionable messages
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tether.distill import get_method
from tether.distill.snapflow import (
    DEFAULT_CONSISTENCY_ALPHA,
    SnapFlowLosses,
    ZeroInitTargetTimeEmbedding,
    flow_matching_interp,
    sinusoidal_time_embedding,
    snapflow_loss_step,
    two_step_euler_shortcut,
)


class TestFlowMatchingInterp:
    def test_t_zero_returns_noise(self):
        noise = torch.randn(2, 5, 4)
        action = torch.randn(2, 5, 4)
        t = torch.zeros(2)
        x_t, v_target = flow_matching_interp(noise, action, t)
        assert torch.allclose(x_t, noise)
        assert torch.allclose(v_target, action - noise)

    def test_t_one_returns_action(self):
        noise = torch.randn(2, 5, 4)
        action = torch.randn(2, 5, 4)
        t = torch.ones(2)
        x_t, v_target = flow_matching_interp(noise, action, t)
        assert torch.allclose(x_t, action)

    def test_t_half_is_midpoint(self):
        noise = torch.zeros(2, 5, 4)
        action = torch.ones(2, 5, 4)
        t = torch.full((2,), 0.5)
        x_t, _ = flow_matching_interp(noise, action, t)
        assert torch.allclose(x_t, torch.full_like(x_t, 0.5))

    def test_v_target_is_t_independent(self):
        """Linear flow-matching: v_target = action - noise regardless of t."""
        noise = torch.randn(3, 2, 2)
        action = torch.randn(3, 2, 2)
        _, v1 = flow_matching_interp(noise, action, torch.rand(3))
        _, v2 = flow_matching_interp(noise, action, torch.rand(3))
        assert torch.allclose(v1, v2)
        assert torch.allclose(v1, action - noise)

    def test_shape_preserved(self):
        for shape in [(1, 10, 32), (4, 50, 7), (8, 1, 128)]:
            noise = torch.randn(*shape)
            action = torch.randn(*shape)
            t = torch.rand(shape[0])
            x_t, v_target = flow_matching_interp(noise, action, t)
            assert x_t.shape == shape
            assert v_target.shape == shape

    def test_mismatched_shapes_rejected(self):
        noise = torch.randn(2, 5, 4)
        action = torch.randn(2, 5, 3)  # different last dim
        t = torch.rand(2)
        with pytest.raises(AssertionError):
            flow_matching_interp(noise, action, t)


class TestTwoStepEulerShortcut:
    def test_runs_under_no_grad(self):
        """Critical: the teacher forward pass must NOT accumulate grads."""
        v_calls = []

        def teacher_fn(x, t, **kw):
            v_calls.append(torch.is_grad_enabled())
            return torch.zeros_like(x)

        x_t = torch.randn(2, 5, 4, requires_grad=True)
        t = torch.rand(2)
        _ = two_step_euler_shortcut(teacher_fn, x_t, t, obs_kwargs={})
        assert v_calls == [False, False], "teacher should run under no_grad"

    def test_teacher_called_twice(self):
        n = {"count": 0}

        def teacher_fn(x, t, **kw):
            n["count"] += 1
            return torch.zeros_like(x)

        _ = two_step_euler_shortcut(
            teacher_fn, torch.randn(2, 5, 4), torch.rand(2), obs_kwargs={},
        )
        assert n["count"] == 2  # t and t+0.5*(1-t)

    def test_zero_velocity_teacher_yields_zero_shortcut(self):
        """If teacher predicts zero velocity, shortcut is zero too."""
        def teacher_fn(x, t, **kw):
            return torch.zeros_like(x)

        x_t = torch.randn(2, 5, 4)
        v = two_step_euler_shortcut(teacher_fn, x_t, torch.rand(2), obs_kwargs={})
        assert torch.allclose(v, torch.zeros_like(v))

    def test_obs_kwargs_passed_through(self):
        received = []

        def teacher_fn(x, t, **kw):
            received.append(kw)
            return torch.zeros_like(x)

        kw = {"image": "img_data", "lang": "prompt"}
        _ = two_step_euler_shortcut(
            teacher_fn, torch.randn(1, 2, 2), torch.rand(1), obs_kwargs=kw,
        )
        assert all(r == kw for r in received)


class TestSnapFlowLossStep:
    def _make_fns(self, shape):
        """Tiny student + teacher callables matching the protocol."""

        def student_fn(x, t, target_time=None, **kw):
            # Returns zero velocity (trivial student for testing)
            return torch.zeros_like(x)

        def teacher_fn(x, t, **kw):
            return torch.zeros_like(x)

        return student_fn, teacher_fn

    def test_returns_tensor_and_snapshot(self):
        student, teacher = self._make_fns((2, 3, 4))
        loss, snap = snapflow_loss_step(
            student,
            teacher,
            action=torch.randn(2, 3, 4),
            noise=torch.randn(2, 3, 4),
            t=torch.rand(2),
            obs_kwargs={},
        )
        assert isinstance(loss, torch.Tensor)
        assert isinstance(snap, SnapFlowLosses)
        assert snap.flow_matching > 0  # student outputs zero, target is nonzero
        assert snap.total == pytest.approx(
            snap.flow_matching + DEFAULT_CONSISTENCY_ALPHA * snap.consistency,
            rel=1e-5,
        )

    def test_perfect_student_yields_zero_loss(self):
        """If student matches the exact analytic velocity, fm_loss = 0.
        Consistency term may still be nonzero unless teacher also perfect."""

        action = torch.randn(2, 3, 4)
        noise = torch.randn(2, 3, 4)

        def perfect_student(x, t, target_time=None, **kw):
            # Student returns v = action - noise (perfect flow-matching answer)
            return (action - noise)

        def trivial_teacher(x, t, **kw):
            return torch.zeros_like(x)

        _, snap = snapflow_loss_step(
            perfect_student,
            trivial_teacher,
            action=action,
            noise=noise,
            t=torch.rand(2),
            obs_kwargs={},
        )
        assert snap.flow_matching == pytest.approx(0.0, abs=1e-6)

    def test_consistency_alpha_scales(self):
        """With a nonzero teacher → student disagreement, total loss
        grows linearly with consistency_alpha."""
        # Student returns zero, teacher returns a constant — disagreement
        # is nonzero on the consistency term.
        def student(x, t, target_time=None, **kw):
            return torch.zeros_like(x)

        def teacher(x, t, **kw):
            return torch.full_like(x, 0.5)

        args = dict(
            action=torch.randn(2, 3, 4),
            noise=torch.randn(2, 3, 4),
            t=torch.rand(2),
            obs_kwargs={},
        )
        _, snap_1 = snapflow_loss_step(
            student, teacher, consistency_alpha=1.0, **args,
        )
        _, snap_2 = snapflow_loss_step(
            student, teacher, consistency_alpha=2.0, **args,
        )
        # Total = fm + alpha * consistency, where consistency > 0 here
        assert snap_1.consistency > 0.0
        assert snap_2.total > snap_1.total
        # The delta should equal (alpha_2 - alpha_1) * consistency
        expected_delta = (2.0 - 1.0) * snap_1.consistency
        assert (snap_2.total - snap_1.total) == pytest.approx(
            expected_delta, rel=1e-5,
        )

    def test_loss_is_differentiable(self):
        """Critical: the returned loss must admit backprop."""

        action = torch.randn(1, 2, 2)
        noise = torch.randn(1, 2, 2)
        t = torch.rand(1)

        w = torch.nn.Parameter(torch.randn(2))

        def student_fn(x, t, target_time=None, **kw):
            return x * w  # trivially differentiable

        def teacher_fn(x, t, **kw):
            return torch.zeros_like(x)

        loss, _ = snapflow_loss_step(
            student_fn, teacher_fn,
            action=action, noise=noise, t=t, obs_kwargs={},
        )
        loss.backward()
        assert w.grad is not None
        assert not torch.isnan(w.grad).any()


class TestZeroInitTargetTimeEmbedding:
    def test_zero_output_at_init(self):
        emb = ZeroInitTargetTimeEmbedding(embedding_dim=16)
        x = torch.randn(4, 16)
        out = emb(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_nonzero_after_training_step(self):
        """After one optimizer step on a nonzero loss, output is nonzero."""
        emb = ZeroInitTargetTimeEmbedding(embedding_dim=16)
        params = list(emb.mlp.parameters())
        opt = torch.optim.SGD(params, lr=0.1)

        x = torch.randn(4, 16)
        target = torch.randn(4, 16)
        out = emb(x)
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        out2 = emb(x)
        assert not torch.allclose(out2, torch.zeros_like(out2))


class TestSinusoidalTimeEmbedding:
    def test_shape(self):
        emb = sinusoidal_time_embedding(torch.rand(4), 32)
        assert emb.shape == (4, 32)

    def test_even_embedding_dim_required(self):
        with pytest.raises(AssertionError):
            sinusoidal_time_embedding(torch.rand(2), 31)  # odd

    def test_values_in_sensible_range(self):
        # sin/cos outputs are in [-1, 1]
        emb = sinusoidal_time_embedding(torch.rand(10), 16)
        assert emb.min() >= -1.0 - 1e-5
        assert emb.max() <= 1.0 + 1e-5

    def test_deterministic_for_same_t(self):
        t = torch.tensor([0.25, 0.5, 0.75])
        e1 = sinusoidal_time_embedding(t, 8)
        e2 = sinusoidal_time_embedding(t, 8)
        assert torch.allclose(e1, e2)


class TestDistillMethodRegistry:
    def test_snapflow_resolvable(self):
        mod = get_method("snapflow")
        assert mod is not None
        assert hasattr(mod, "snapflow_loss_step")

    def test_dmpo_raises_deprecation(self):
        with pytest.raises(ValueError, match="deprecated"):
            get_method("dmpo")

    def test_pi_flow_raises_deprecation(self):
        with pytest.raises(ValueError, match="deprecated"):
            get_method("pi_flow")

    def test_consistency_points_at_v05(self):
        with pytest.raises(ValueError, match="v0.5"):
            get_method("consistency")

    def test_unknown_method_lists_supported(self):
        with pytest.raises(ValueError, match="snapflow") as exc:
            get_method("some_new_method")
        assert "snapflow" in str(exc.value)
