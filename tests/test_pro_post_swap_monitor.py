"""Tests for src/tether/pro/post_swap_monitor.py — Phase 1 Day 7.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #4: 24h
or 500-episode rolling-window watcher with 3 trip signals (T1 safety-
clamp p95, T2 action cos, T3 webhook violations) + N-in-a-row
sensitivity (aggressive=1, normal=2, tolerant=3).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tether.pro.post_swap_monitor import (
    DEFAULT_T1_CLAMP_RATE_RELATIVE,
    DEFAULT_T2_COS_MIN,
    DEFAULT_T3_VIOLATION_COUNT_5MIN,
    MonitorConfig,
    PostSwapMonitor,
    T3_WINDOW_SECONDS,
    TripDecision,
    WINDOW_DURATION_HOURS,
    WINDOW_EPISODE_COUNT,
)


def _utc(year=2026, month=4, day=25, hour=10, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _mk_monitor(**cfg_overrides) -> PostSwapMonitor:
    cfg = MonitorConfig(**cfg_overrides) if cfg_overrides else MonitorConfig()
    return PostSwapMonitor(config=cfg)


# ---------------------------------------------------------------------------
# MonitorConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_sensitivity():
    with pytest.raises(ValueError, match="sensitivity"):
        MonitorConfig(sensitivity="bogus")  # type: ignore[arg-type]


def test_config_rejects_t1_at_or_below_one():
    with pytest.raises(ValueError, match="t1_clamp_rate_relative"):
        MonitorConfig(t1_clamp_rate_relative=1.0)


def test_config_rejects_t2_at_zero():
    with pytest.raises(ValueError, match="t2_cos_min"):
        MonitorConfig(t2_cos_min=0.0)


def test_config_rejects_t2_at_one():
    with pytest.raises(ValueError, match="t2_cos_min"):
        MonitorConfig(t2_cos_min=1.0)


def test_config_rejects_zero_violation_threshold():
    with pytest.raises(ValueError, match="t3_violation"):
        MonitorConfig(t3_violation_count_5min=0)


def test_config_rejects_zero_window_hours():
    with pytest.raises(ValueError, match="window_duration_hours"):
        MonitorConfig(window_duration_hours=0)


def test_config_required_consecutive_trips_by_sensitivity():
    assert MonitorConfig(sensitivity="aggressive").required_consecutive_trips() == 1
    assert MonitorConfig(sensitivity="normal").required_consecutive_trips() == 2
    assert MonitorConfig(sensitivity="tolerant").required_consecutive_trips() == 3


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_monitor_start_window_resets_state():
    m = _mk_monitor()
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    m.record_episode(safety_clamp_count=1)
    assert m.episodes_seen == 1
    # Restart window
    m.start_window(baseline_clamp_rate=0.01, swap_at=_utc(hour=11))
    assert m.episodes_seen == 0


def test_monitor_record_before_start_is_noop():
    m = _mk_monitor()
    m.record_episode(safety_clamp_count=1)
    assert m.episodes_seen == 0


def test_monitor_should_rollback_before_start_returns_in_window_no_op():
    m = _mk_monitor()
    decision = m.should_rollback()
    assert not decision.should_rollback


def test_monitor_start_window_rejects_negative_baseline():
    m = _mk_monitor()
    with pytest.raises(ValueError, match="baseline_clamp_rate"):
        m.start_window(baseline_clamp_rate=-0.1)


# ---------------------------------------------------------------------------
# Window closing
# ---------------------------------------------------------------------------


def test_window_closes_after_24h():
    m = _mk_monitor()
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    # Far future — past 24h window
    later = _utc() + timedelta(hours=25)
    decision = m.should_rollback(now=later)
    assert decision.reason == "window-closed"
    assert not decision.should_rollback


def test_window_closes_after_episode_count():
    m = _mk_monitor(window_episode_count=3)
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    for _ in range(5):
        m.record_episode(safety_clamp_count=0)
    decision = m.should_rollback(now=_utc(hour=10, minute=30))
    assert decision.reason == "window-closed"


def test_is_window_open_property():
    m = _mk_monitor()
    assert not m.is_window_open  # before start
    m.start_window(baseline_clamp_rate=0.005)
    assert m.is_window_open


# ---------------------------------------------------------------------------
# T1: safety-clamp rolling p95
# ---------------------------------------------------------------------------


def test_t1_does_not_fire_when_clamp_rate_normal():
    m = _mk_monitor(sensitivity="normal")
    m.start_window(baseline_clamp_rate=0.5, swap_at=_utc())
    for _ in range(20):
        m.record_episode(safety_clamp_count=0)  # zero clamps
    decision = m.should_rollback(now=_utc(hour=11))
    assert not decision.should_rollback
    assert decision.reason == "in-window"


def test_t1_fires_after_two_consecutive_high_p95():
    """normal sensitivity = 2-in-a-row. Saturate the rolling window with
    spikes so p95 stays elevated for both checks."""
    m = _mk_monitor(sensitivity="normal", t1_clamp_rate_relative=2.0)
    # Baseline clamp rate ~ 0.5 → trigger when p95 > 1.0
    m.start_window(baseline_clamp_rate=0.5, swap_at=_utc())
    for _ in range(20):
        m.record_episode(safety_clamp_count=10)  # way above 1.0
    # First check increments consecutive_trips to 1 (not yet firing)
    d1 = m.should_rollback(now=_utc(hour=11))
    assert not d1.should_rollback
    assert d1.reason == "T1"
    assert d1.consecutive_trips == 1
    # Second check fires
    d2 = m.should_rollback(now=_utc(hour=11, minute=1))
    assert d2.should_rollback
    assert d2.reason == "T1"
    assert d2.consecutive_trips == 2


def test_t1_consecutive_resets_on_normal_episode():
    """A normal-traffic stretch between trips resets the consecutive
    counter. Use a small rolling window so the post-spike dilution
    fully replaces the high samples."""
    m = _mk_monitor(
        sensitivity="normal", t1_clamp_rate_relative=2.0,
        rolling_window_size=10,
    )
    m.start_window(baseline_clamp_rate=0.5, swap_at=_utc())
    # High clamps → trip 1
    for _ in range(10):
        m.record_episode(safety_clamp_count=10)
    d1 = m.should_rollback(now=_utc(hour=11))
    assert d1.consecutive_trips == 1
    # Flood with normal episodes — fully replaces the rolling window of 10
    for _ in range(15):
        m.record_episode(safety_clamp_count=0)
    d2 = m.should_rollback(now=_utc(hour=11, minute=2))
    assert d2.consecutive_trips == 0  # reset
    assert not d2.should_rollback


def test_t1_aggressive_sensitivity_fires_on_first_trip():
    m = _mk_monitor(sensitivity="aggressive", t1_clamp_rate_relative=2.0)
    m.start_window(baseline_clamp_rate=0.5, swap_at=_utc())
    for _ in range(20):
        m.record_episode(safety_clamp_count=10)
    decision = m.should_rollback(now=_utc(hour=11))
    assert decision.should_rollback
    assert decision.consecutive_trips == 1


def test_t1_tolerant_sensitivity_requires_three_trips():
    m = _mk_monitor(sensitivity="tolerant", t1_clamp_rate_relative=2.0)
    m.start_window(baseline_clamp_rate=0.5, swap_at=_utc())
    for _ in range(20):
        m.record_episode(safety_clamp_count=10)
    d1 = m.should_rollback(now=_utc(hour=11))
    d2 = m.should_rollback(now=_utc(hour=11, minute=1))
    d3 = m.should_rollback(now=_utc(hour=11, minute=2))
    assert not d1.should_rollback
    assert not d2.should_rollback
    assert d3.should_rollback


# ---------------------------------------------------------------------------
# T2: action cos to previous model
# ---------------------------------------------------------------------------


def test_t2_fires_when_cos_drops_below_threshold():
    m = _mk_monitor(sensitivity="aggressive", t2_cos_min=0.85)
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    for _ in range(10):
        m.record_episode(safety_clamp_count=0, cos_to_previous_model=0.5)
    decision = m.should_rollback(now=_utc(hour=11))
    assert decision.should_rollback
    assert decision.reason == "T2"


def test_t2_does_not_fire_when_cos_high():
    m = _mk_monitor(sensitivity="aggressive", t2_cos_min=0.85)
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    for _ in range(10):
        m.record_episode(safety_clamp_count=0, cos_to_previous_model=0.95)
    decision = m.should_rollback(now=_utc(hour=11))
    assert not decision.should_rollback


def test_t2_skipped_when_cos_unset():
    """When cos_to_previous_model is None for every episode, T2 doesn't fire."""
    m = _mk_monitor(sensitivity="aggressive", t2_cos_min=0.85)
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    for _ in range(10):
        m.record_episode(safety_clamp_count=0, cos_to_previous_model=None)
    decision = m.should_rollback(now=_utc(hour=11))
    assert not decision.should_rollback


# ---------------------------------------------------------------------------
# T3: webhook violations in 5 min window
# ---------------------------------------------------------------------------


def test_t3_fires_when_violations_exceed_threshold():
    m = _mk_monitor(sensitivity="aggressive", t3_violation_count_5min=5)
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    # Pile up 6 violations in the last minute
    now = _utc(hour=10, minute=5)
    for _ in range(6):
        m.record_episode(
            safety_clamp_count=0, webhook_violations_count=1, now=now,
        )
    decision = m.should_rollback(now=now)
    assert decision.should_rollback
    assert decision.reason == "T3"


def test_t3_does_not_fire_when_violations_outside_window():
    m = _mk_monitor(sensitivity="aggressive", t3_violation_count_5min=5)
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    # Old violations
    old_time = _utc(hour=10, minute=0)
    for _ in range(10):
        m.record_episode(
            safety_clamp_count=0, webhook_violations_count=1, now=old_time,
        )
    # Check 10 minutes later — old violations age out of the 5-min window
    now = old_time + timedelta(minutes=10)
    decision = m.should_rollback(now=now)
    assert not decision.should_rollback


# ---------------------------------------------------------------------------
# TripDecision shape
# ---------------------------------------------------------------------------


def test_decision_is_frozen():
    m = _mk_monitor()
    m.start_window(baseline_clamp_rate=0.005, swap_at=_utc())
    d = m.should_rollback(now=_utc(hour=11))
    with pytest.raises(AttributeError):
        d.should_rollback = True  # type: ignore[misc]
