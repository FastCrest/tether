"""Tests for safety guardrails."""

import json

import numpy as np
import pytest

from tether.safety.guard import ActionGuard, SafetyLimits


class TestSafetyLimits:
    def test_default_creates_6dof(self):
        limits = SafetyLimits.default(6)
        assert len(limits.joint_names) == 6
        assert len(limits.position_min) == 6
        assert len(limits.position_max) == 6

    def test_save_and_load(self, tmp_path):
        limits = SafetyLimits.default(4)
        path = tmp_path / "limits.json"
        limits.save(path)
        loaded = SafetyLimits.from_json(path)
        assert loaded.joint_names == limits.joint_names
        assert loaded.position_min == limits.position_min

    def test_custom_limits(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2"],
            position_min=[-1.0, -2.0],
            position_max=[1.0, 2.0],
            velocity_max=[1.5, 1.5],
            effort_max=[50.0, 50.0],
            workspace_indices=[0, 1],
        )
        assert limits.position_min[0] == -1.0
        assert limits.position_max[1] == 2.0
        assert limits.workspace_indices == [0, 1]


class TestActionGuard:
    def test_safe_action_passes(self):
        guard = ActionGuard.default(num_joints=3)
        action = np.array([0.1, 0.2, 0.3])
        result = guard.check_single(action)
        assert result.safe
        assert len(result.violations) == 0
        assert not result.clamped

    def test_out_of_bounds_clamped(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2"],
            position_min=[-1.0, -1.0],
            position_max=[1.0, 1.0],
            velocity_max=[2.0, 2.0],
            effort_max=[50.0, 50.0],
        )
        guard = ActionGuard(limits=limits, mode="clamp")
        action = np.array([5.0, -3.0])
        result = guard.check_single(action)
        assert not result.safe
        assert result.clamped
        assert result.safe_action[0] == 1.0
        assert result.safe_action[1] == -1.0

    def test_reject_mode_zeros(self):
        limits = SafetyLimits(
            joint_names=["j1"],
            position_min=[-1.0],
            position_max=[1.0],
            velocity_max=[2.0],
            effort_max=[50.0],
        )
        guard = ActionGuard(limits=limits, mode="reject")
        action = np.array([5.0])
        result = guard.check_single(action)
        assert not result.safe
        assert result.safe_action[0] == 0.0

    def test_effort_limit_clamps_single_action(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2"],
            position_min=[-10.0, -10.0],
            position_max=[10.0, 10.0],
            velocity_max=[10.0, 10.0],
            effort_max=[2.0, 4.0],
        )
        guard = ActionGuard(limits=limits, mode="clamp")

        result = guard.check_single(np.array([3.0, -6.0]))

        assert not result.safe
        assert result.clamped
        assert result.safe_action == [2.0, -4.0]
        assert any("effort limit" in v for v in result.violations)

    def test_velocity_limit_clamps_between_chunk_actions(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2"],
            position_min=[-10.0, -10.0],
            position_max=[10.0, 10.0],
            velocity_max=[1.0, 0.5],
            effort_max=[50.0, 50.0],
        )
        guard = ActionGuard(limits=limits, mode="clamp")
        actions = np.array([
            [0.0, 0.0],
            [5.0, -5.0],
            [2.0, -1.0],
        ])

        safe_actions, results = guard.check(actions)

        np.testing.assert_allclose(safe_actions[0], [0.0, 0.0])
        np.testing.assert_allclose(safe_actions[1], [1.0, -0.5])
        np.testing.assert_allclose(safe_actions[2], [2.0, -1.0])
        assert any("velocity limit" in v for v in results[1].violations)
        assert results[1].clamped
        assert results[2].safe

    def test_workspace_limit_clamps_explicit_indices(self):
        limits = SafetyLimits(
            joint_names=["x", "unused", "z"],
            position_min=[-10.0, -10.0, -10.0],
            position_max=[10.0, 10.0, 10.0],
            velocity_max=[10.0, 10.0, 10.0],
            effort_max=[50.0, 50.0, 50.0],
            workspace_min=[-1.0, 0.0],
            workspace_max=[1.0, 2.0],
            workspace_indices=[0, 2],
        )
        guard = ActionGuard(limits=limits, mode="clamp")

        result = guard.check_single(np.array([3.0, 9.0, -5.0]))

        assert not result.safe
        assert result.clamped
        assert result.safe_action == [1.0, 9.0, 0.0]
        assert any("workspace_axis_0 above max" in v for v in result.violations)
        assert any("workspace_axis_1 below min" in v for v in result.violations)

    def test_workspace_limits_are_opt_in_for_joint_actions(self):
        limits = SafetyLimits(
            joint_names=["j1", "j2", "j3"],
            position_min=[-10.0, -10.0, -10.0],
            position_max=[10.0, 10.0, 10.0],
            velocity_max=[10.0, 10.0, 10.0],
            effort_max=[50.0, 50.0, 50.0],
            workspace_min=[-1.0, -1.0, 0.0],
            workspace_max=[1.0, 1.0, 1.5],
            workspace_indices=[],
        )
        guard = ActionGuard(limits=limits, mode="clamp")

        result = guard.check_single(np.array([3.0, -3.0, -3.0]))

        assert result.safe
        assert result.safe_action == [3.0, -3.0, -3.0]

    def test_safety_margin_reports_centered_actions_high(self):
        guard = ActionGuard.default(num_joints=2)

        margin = guard.safety_margin(np.array([[0.0, 0.0]], dtype=np.float32))

        assert margin == 1.0

    def test_safety_margin_reports_boundary_as_zero(self):
        guard = ActionGuard.default(num_joints=2)

        margin = guard.safety_margin(np.array([[3.14, 0.0]], dtype=np.float32))

        assert margin == 0.0

    def test_safety_margin_includes_velocity_delta(self):
        limits = SafetyLimits(
            joint_names=["j1"],
            position_min=[-10.0],
            position_max=[10.0],
            velocity_max=[1.0],
            effort_max=[50.0],
        )
        guard = ActionGuard(limits=limits)

        margin = guard.safety_margin(
            np.array([[0.0], [0.8]], dtype=np.float32)
        )

        assert margin == pytest.approx(0.2)

    def test_batch_check(self):
        guard = ActionGuard.default(num_joints=3)
        actions = np.array([
            [0.1, 0.2, 0.3],
            [10.0, -10.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        safe_actions, results = guard.check(actions)
        assert len(results) == 3
        assert results[0].safe
        assert not results[1].safe
        assert results[2].safe
        assert safe_actions.shape == actions.shape

    def test_inference_count(self):
        guard = ActionGuard.default(num_joints=2)
        actions = np.array([[0.1, 0.2]])
        guard.check(actions)
        guard.check(actions)
        assert guard.inference_count == 2

    def test_logging(self, tmp_path):
        guard = ActionGuard.default(num_joints=2, log_dir=tmp_path, model_version="test-v1")
        actions = np.array([[0.1, 0.2], [5.0, -5.0]])
        guard.check(actions)

        log_files = list(tmp_path.glob("inference_log_*.jsonl"))
        assert len(log_files) == 1

        lines = log_files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["model_version"] == "test-v1"
        assert "timestamp" in entry
        assert len(entry["actions_raw"]) == 2

    def test_check_time_is_fast(self):
        guard = ActionGuard.default(num_joints=6)
        action = np.random.randn(6).astype(np.float32)
        result = guard.check_single(action)
        assert result.check_time_ms < 1.0  # Sub-millisecond


class TestNanGuard:
    """NaN/Inf rejection tests — any non-finite value → whole chunk zeroed."""

    def test_nan_action_is_zeroed(self):
        guard = ActionGuard.default(num_joints=3)
        actions = np.array([[0.1, np.nan, 0.3]])
        safe, results = guard.check(actions)
        assert np.all(safe == 0.0)
        assert not results[0].safe
        assert any("non_finite" in v for v in results[0].violations)
        assert results[0].clamped

    def test_inf_action_is_zeroed(self):
        guard = ActionGuard.default(num_joints=3)
        actions = np.array([[0.1, 0.2, np.inf]])
        safe, results = guard.check(actions)
        assert np.all(safe == 0.0)
        assert not results[0].safe

    def test_neg_inf_action_is_zeroed(self):
        guard = ActionGuard.default(num_joints=2)
        actions = np.array([[-np.inf, 0.0], [0.1, 0.2]])
        safe, _ = guard.check(actions)
        # ENTIRE chunk zeroed when any element non-finite
        assert np.all(safe == 0.0)

    def test_clean_action_passes_through(self):
        guard = ActionGuard.default(num_joints=3)
        actions = np.array([[0.1, 0.2, 0.3]])
        safe, results = guard.check(actions)
        np.testing.assert_allclose(safe, actions)
        assert results[0].safe


class TestKillSwitch:
    """Consecutive-clamp kill-switch: after N clamps, guard trips."""

    def test_trips_after_max_consecutive_clamps(self):
        guard = ActionGuard.default(num_joints=2, max_consecutive_clamps=3)
        bad = np.array([[10.0, 10.0]])  # out of bounds → clamps
        assert not guard.tripped
        guard.check(bad)
        assert guard.consecutive_clamps == 1
        assert not guard.tripped
        guard.check(bad)
        assert guard.consecutive_clamps == 2
        assert not guard.tripped
        guard.check(bad)
        assert guard.consecutive_clamps == 3
        assert guard.tripped
        assert "consecutive_clamp_limit_exceeded" in guard.trip_reason

    def test_clean_chunk_resets_counter(self):
        guard = ActionGuard.default(num_joints=2, max_consecutive_clamps=5)
        bad = np.array([[10.0, 10.0]])
        clean = np.array([[0.1, 0.1]])
        guard.check(bad)
        guard.check(bad)
        assert guard.consecutive_clamps == 2
        guard.check(clean)
        assert guard.consecutive_clamps == 0

    def test_nan_chunk_counts_toward_kill_switch(self):
        guard = ActionGuard.default(num_joints=2, max_consecutive_clamps=2)
        guard.check(np.array([[np.nan, 0.1]]))
        guard.check(np.array([[np.inf, 0.1]]))
        assert guard.tripped

    def test_reset_clears_tripped(self):
        guard = ActionGuard.default(num_joints=2, max_consecutive_clamps=1)
        guard.check(np.array([[10.0, 10.0]]))
        assert guard.tripped
        guard.reset()
        assert not guard.tripped
        assert guard.trip_reason is None
        assert guard.consecutive_clamps == 0

    def test_disabled_kill_switch(self):
        guard = ActionGuard.default(num_joints=2, max_consecutive_clamps=0)
        for _ in range(100):
            guard.check(np.array([[10.0, 10.0]]))
        assert not guard.tripped
