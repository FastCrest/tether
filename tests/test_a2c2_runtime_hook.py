"""Tests for src/tether/runtime/a2c2_hook.py — Phase B.5 Day 3 invariants.

Per a2c2-correction execution plan B.5 Day 3 acceptance criteria:
- Auto-skip when latency_p95 < threshold (default 40ms)
- Auto-skip when success_rate > threshold (default 90%)
- Apply when both: high latency AND low success (correction has marginal value)
- Cold-start: skip until min_samples in BOTH windows
- Counters: applied/skipped totals correctly tracked
- Per-action correction across the chunk (not just chunk[0])
- Graceful degrade with zero-vector observation when not provided
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from tether.kernels.a2c2_correction import A2C2Config, A2C2Head
from tether.runtime.a2c2_hook import (
    A2C2Decision,
    A2C2Hook,
    A2C2HookConfig,
)


def _mk_hook(**cfg_kwargs) -> A2C2Hook:
    head = A2C2Head.random_init(seed=0)
    return A2C2Hook(head=head, config=A2C2HookConfig(**cfg_kwargs))


# ---------------------------------------------------------------------------
# A2C2HookConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_zero_latency_threshold():
    with pytest.raises(ValueError, match="latency_threshold_ms"):
        A2C2HookConfig(latency_threshold_ms=0)


def test_config_rejects_negative_latency_threshold():
    with pytest.raises(ValueError, match="latency_threshold_ms"):
        A2C2HookConfig(latency_threshold_ms=-5.0)


def test_config_rejects_success_threshold_out_of_range():
    with pytest.raises(ValueError, match="success_threshold"):
        A2C2HookConfig(success_threshold=-0.1)
    with pytest.raises(ValueError, match="success_threshold"):
        A2C2HookConfig(success_threshold=2.1)  # 2.0 is the upper bound (1.01+ disables skip)


def test_config_accepts_boundary_success_thresholds():
    A2C2HookConfig(success_threshold=0.0)
    A2C2HookConfig(success_threshold=1.0)


def test_config_accepts_disable_skip_sentinel():
    """Values in (1.0, 2.0] disable success-based skip — used in measurement
    runs where the success metric is /act error rate, not task success."""
    h = A2C2HookConfig(success_threshold=1.01)
    assert h.success_threshold == 1.01
    h2 = A2C2HookConfig(success_threshold=2.0)
    assert h2.success_threshold == 2.0


def test_config_rejects_zero_window_sizes():
    with pytest.raises(ValueError, match="latency_window"):
        A2C2HookConfig(latency_window=0)
    with pytest.raises(ValueError, match="success_window"):
        A2C2HookConfig(success_window=0)


def test_config_rejects_zero_min_samples():
    with pytest.raises(ValueError, match="min_samples"):
        A2C2HookConfig(min_samples_for_decision=0)


# ---------------------------------------------------------------------------
# Trackers — record_outcome + percentiles
# ---------------------------------------------------------------------------


def test_record_outcome_drops_negative_latency():
    hook = _mk_hook()
    hook.record_outcome(latency_ms=-1.0, success=True)
    assert hook.sample_count() == (0, 0)


def test_record_outcome_drops_nan_latency():
    hook = _mk_hook()
    hook.record_outcome(latency_ms=float("nan"), success=True)
    assert hook.sample_count() == (0, 0)


def test_record_outcome_appends_valid_sample():
    hook = _mk_hook()
    hook.record_outcome(latency_ms=42.0, success=True)
    assert hook.sample_count() == (1, 1)


def test_latency_p95_zero_when_empty():
    hook = _mk_hook()
    assert hook.latency_p95_ms() == 0.0


def test_success_rate_one_when_empty():
    """Empty success window defaults to 1.0 — assume things are working
    until we have data to prove otherwise."""
    hook = _mk_hook()
    assert hook.success_rate() == 1.0


def test_latency_p95_computes_correctly():
    hook = _mk_hook()
    for v in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        hook.record_outcome(latency_ms=v, success=True)
    p95 = hook.latency_p95_ms()
    # 95th percentile of [10..100] should be ~95.5 (linear interp)
    assert 90.0 <= p95 <= 100.0


def test_success_rate_computes_correctly():
    hook = _mk_hook()
    for s in [True, True, True, False, False]:
        hook.record_outcome(latency_ms=10.0, success=s)
    assert hook.success_rate() == pytest.approx(0.6, abs=0.01)


def test_rolling_window_bounded_to_config_size():
    hook = _mk_hook(latency_window=10, success_window=10, min_samples_for_decision=1)
    for i in range(50):
        hook.record_outcome(latency_ms=float(i), success=True)
    n_lat, n_succ = hook.sample_count()
    assert n_lat == 10
    assert n_succ == 10


# ---------------------------------------------------------------------------
# should_apply — auto-skip logic
# ---------------------------------------------------------------------------


def test_should_apply_cold_start_skips_until_min_samples():
    hook = _mk_hook(min_samples_for_decision=5)
    # 4 samples — under threshold
    for _ in range(4):
        hook.record_outcome(latency_ms=100.0, success=False)
    decision = hook.should_apply()
    assert decision.apply is False
    assert decision.reason == "cold_start"


def test_should_apply_skips_when_latency_low():
    hook = _mk_hook(min_samples_for_decision=3)
    for _ in range(5):
        hook.record_outcome(latency_ms=10.0, success=False)  # low latency
    decision = hook.should_apply()
    assert decision.apply is False
    assert decision.reason == "low_latency"


def test_should_apply_skips_when_success_high():
    hook = _mk_hook(min_samples_for_decision=3)
    for _ in range(5):
        hook.record_outcome(latency_ms=100.0, success=True)  # high success
    decision = hook.should_apply()
    assert decision.apply is False
    assert decision.reason == "high_success"


def test_should_apply_applies_when_high_latency_and_low_success():
    hook = _mk_hook(min_samples_for_decision=3)
    for _ in range(5):
        hook.record_outcome(latency_ms=100.0, success=False)  # bad day
    decision = hook.should_apply()
    assert decision.apply is True
    assert decision.reason == "applied"
    assert decision.latency_p95_ms >= 40.0
    assert decision.success_rate <= 0.90


# ---------------------------------------------------------------------------
# maybe_apply_to_chunk
# ---------------------------------------------------------------------------


def _bad_day_hook() -> A2C2Hook:
    """Hook seeded with samples that trigger should_apply()=True."""
    hook = _mk_hook(min_samples_for_decision=3)
    for _ in range(5):
        hook.record_outcome(latency_ms=100.0, success=False)
    return hook


def _good_day_hook() -> A2C2Hook:
    """Hook seeded with samples that trigger should_apply()=False."""
    hook = _mk_hook(min_samples_for_decision=3)
    for _ in range(5):
        hook.record_outcome(latency_ms=100.0, success=True)  # high success → skip
    return hook


def test_maybe_apply_returns_input_unchanged_when_skipped():
    hook = _good_day_hook()
    actions = np.ones((50, 7), dtype=np.float32)
    out, decision, magnitude = hook.maybe_apply_to_chunk(actions=actions)
    assert decision.apply is False
    assert magnitude == 0.0
    assert out is actions  # same array, no copy on skip path


def test_maybe_apply_corrects_when_applied():
    hook = _bad_day_hook()
    actions = np.ones((50, 7), dtype=np.float32)
    out, decision, magnitude = hook.maybe_apply_to_chunk(actions=actions)
    assert decision.apply is True
    assert out.shape == actions.shape
    assert magnitude > 0.0  # head with random init produces non-zero correction
    assert not np.array_equal(out, actions)


def test_maybe_apply_uses_zero_observation_when_not_provided():
    """When caller doesn't supply an observation, hook gracefully degrades
    to zero vector — Phase 1 mode until VLM-prefix wiring lands."""
    hook = _bad_day_hook()
    actions = np.ones((50, 7), dtype=np.float32)
    # No observation kwarg
    out, decision, _ = hook.maybe_apply_to_chunk(actions=actions)
    assert decision.apply is True
    assert out.shape == actions.shape


def test_maybe_apply_increments_applied_total():
    hook = _bad_day_hook()
    actions = np.ones((50, 7), dtype=np.float32)
    before = hook.applied_total
    hook.maybe_apply_to_chunk(actions=actions)
    assert hook.applied_total == before + 1


def test_maybe_apply_increments_skipped_total_when_skipped():
    hook = _good_day_hook()
    actions = np.ones((50, 7), dtype=np.float32)
    before = hook.skipped_total
    hook.maybe_apply_to_chunk(actions=actions)
    assert hook.skipped_total == before + 1


def test_maybe_apply_rejects_wrong_action_dim():
    hook = _bad_day_hook()
    actions = np.ones((50, 8), dtype=np.float32)  # wrong action dim
    with pytest.raises(ValueError, match="actions shape"):
        hook.maybe_apply_to_chunk(actions=actions)


def test_maybe_apply_rejects_1d_actions():
    hook = _bad_day_hook()
    actions = np.ones(7, dtype=np.float32)  # 1D not 2D
    with pytest.raises(ValueError, match="actions shape"):
        hook.maybe_apply_to_chunk(actions=actions)


def test_maybe_apply_corrects_per_step_in_chunk():
    """Corrections at different positions should differ (positional encoding)
    — verify the head was called per-position, not just once."""
    hook = _bad_day_hook()
    actions = np.zeros((50, 7), dtype=np.float32)
    out, decision, _ = hook.maybe_apply_to_chunk(actions=actions)
    assert decision.apply is True
    # Position 0 vs position 49 should produce different corrections
    # (different positional encoding feeds the head)
    assert not np.allclose(out[0], out[49])


# ---------------------------------------------------------------------------
# from_checkpoint
# ---------------------------------------------------------------------------


def test_from_checkpoint_loads_head_and_wraps_with_default_config(tmp_path):
    head = A2C2Head.random_init(seed=3)
    ckpt = tmp_path / "head.npz"
    head.save(ckpt)
    hook = A2C2Hook.from_checkpoint(ckpt)
    assert hook.head.config == head.config
    assert hook.config.latency_threshold_ms == 40.0  # default


def test_from_checkpoint_accepts_custom_config(tmp_path):
    head = A2C2Head.random_init(seed=3)
    ckpt = tmp_path / "head.npz"
    head.save(ckpt)
    custom = A2C2HookConfig(latency_threshold_ms=20.0, success_threshold=0.8)
    hook = A2C2Hook.from_checkpoint(ckpt, config=custom)
    assert hook.config.latency_threshold_ms == 20.0
    assert hook.config.success_threshold == 0.8


# ---------------------------------------------------------------------------
# A2C2Decision dataclass
# ---------------------------------------------------------------------------


def test_decision_is_frozen_dataclass():
    d = A2C2Decision(apply=True, reason="applied",
                     latency_p95_ms=50.0, success_rate=0.5, samples=10)
    with pytest.raises(AttributeError):
        d.apply = False  # type: ignore[misc]


def test_decision_fields_populated():
    hook = _bad_day_hook()
    d = hook.should_apply()
    assert isinstance(d, A2C2Decision)
    assert d.apply is True
    assert d.reason == "applied"
    assert d.latency_p95_ms > 0
    assert 0.0 <= d.success_rate <= 1.0
    assert d.samples >= 1
