"""Tests for src/tether/pro/distill_scheduler.py — Phase 1 self-distilling-serve Day 3.

Per ADR 2026-04-25-self-distilling-serve-architecture: pure trigger-decision
primitive with 5 modes (manual / nightly / cron / samples / quality-drop)
+ min-kick-gap protection.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tether.pro.distill_scheduler import (
    ALL_TRIGGER_MODES,
    DEFAULT_MIN_KICK_GAP_S,
    DEFAULT_NIGHTLY_UTC_HOUR,
    DEFAULT_SAMPLES_THRESHOLD,
    DistillScheduler,
    KickDecision,
    SchedulerConfig,
    SchedulerState,
)


def _utc(year=2026, month=4, day=25, hour=10, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _state(
    last_at: datetime | None = None,
    samples: int = 0,
    quality: float | None = None,
) -> SchedulerState:
    return SchedulerState(
        last_kick_at=(
            last_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if last_at else None
        ),
        samples_at_last_kick=samples,
        quality_at_last_kick=quality,
    )


# ---------------------------------------------------------------------------
# SchedulerConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        SchedulerConfig(mode="bogus")


def test_config_accepts_all_known_modes():
    for mode in ALL_TRIGGER_MODES:
        if mode == "cron":
            SchedulerConfig(mode=mode, cron_spec="0 3 * * *")
        else:
            SchedulerConfig(mode=mode)


def test_config_rejects_nightly_hour_out_of_range():
    with pytest.raises(ValueError, match="nightly_utc_hour"):
        SchedulerConfig(mode="nightly", nightly_utc_hour=24)


def test_config_rejects_zero_samples_threshold():
    with pytest.raises(ValueError, match="samples_threshold"):
        SchedulerConfig(mode="samples", samples_threshold=0)


def test_config_rejects_quality_drop_at_zero():
    with pytest.raises(ValueError, match="quality_drop_threshold"):
        SchedulerConfig(mode="quality-drop", quality_drop_threshold=0.0)


def test_config_rejects_quality_drop_above_one():
    with pytest.raises(ValueError, match="quality_drop_threshold"):
        SchedulerConfig(mode="quality-drop", quality_drop_threshold=1.5)


def test_config_rejects_negative_min_kick_gap():
    with pytest.raises(ValueError, match="min_kick_gap_s"):
        SchedulerConfig(mode="manual", min_kick_gap_s=-1)


def test_config_rejects_cron_without_spec():
    with pytest.raises(ValueError, match="cron_spec"):
        SchedulerConfig(mode="cron", cron_spec="")


# ---------------------------------------------------------------------------
# SchedulerState validation
# ---------------------------------------------------------------------------


def test_state_rejects_negative_samples_at_last_kick():
    with pytest.raises(ValueError, match="samples_at_last_kick"):
        SchedulerState(
            last_kick_at=None,
            samples_at_last_kick=-1,
            quality_at_last_kick=None,
        )


# ---------------------------------------------------------------------------
# Min-gap protection (applies to all modes)
# ---------------------------------------------------------------------------


def test_min_gap_blocks_kick_within_cooldown_window():
    sch = DistillScheduler(SchedulerConfig(
        mode="samples", samples_threshold=10, min_kick_gap_s=3600,
    ))
    last = _utc(hour=10)
    now = _utc(hour=10, minute=30)  # 30 min later, well within 1h
    state = _state(last_at=last, samples=0)
    decision = sch.should_kick(
        state=state, current_samples=100, now=now,  # 100 samples > 10 threshold
    )
    assert not decision.kick
    assert decision.reason == "min-gap"


def test_min_gap_zero_allows_back_to_back_kicks():
    sch = DistillScheduler(SchedulerConfig(
        mode="samples", samples_threshold=10, min_kick_gap_s=0,
    ))
    last = _utc(hour=10)
    now = _utc(hour=10, minute=1)
    state = _state(last_at=last, samples=0)
    decision = sch.should_kick(
        state=state, current_samples=100, now=now,
    )
    assert decision.kick
    assert decision.reason == "samples"


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------


def test_manual_mode_never_kicks_automatically():
    sch = DistillScheduler(SchedulerConfig(mode="manual"))
    decision = sch.should_kick(
        state=_state(), current_samples=10_000_000, now=_utc(),
    )
    assert not decision.kick
    assert decision.reason == "manual"


# ---------------------------------------------------------------------------
# Samples mode
# ---------------------------------------------------------------------------


def test_samples_mode_kicks_at_threshold():
    sch = DistillScheduler(SchedulerConfig(
        mode="samples", samples_threshold=100, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(samples=0), current_samples=100, now=_utc(),
    )
    assert decision.kick
    assert decision.reason == "samples"


def test_samples_mode_doesnt_kick_below_threshold():
    sch = DistillScheduler(SchedulerConfig(
        mode="samples", samples_threshold=100, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(samples=0), current_samples=50, now=_utc(),
    )
    assert not decision.kick
    assert decision.reason == "no-trigger"


def test_samples_mode_subtracts_samples_at_last_kick():
    """samples_since_last_kick = current - state.samples_at_last_kick."""
    sch = DistillScheduler(SchedulerConfig(
        mode="samples", samples_threshold=100, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(samples=500),  # last kick was at 500 samples
        current_samples=550,  # +50, below 100 threshold
        now=_utc(),
    )
    assert not decision.kick


# ---------------------------------------------------------------------------
# Nightly mode
# ---------------------------------------------------------------------------


def test_nightly_kicks_after_target_hour_when_no_prior_kick():
    sch = DistillScheduler(SchedulerConfig(
        mode="nightly", nightly_utc_hour=3, min_kick_gap_s=0,
    ))
    now = _utc(hour=4)  # past 03:00 UTC target
    decision = sch.should_kick(
        state=_state(last_at=None), current_samples=100, now=now,
    )
    assert decision.kick
    assert decision.reason == "nightly"


def test_nightly_doesnt_kick_before_target_hour():
    sch = DistillScheduler(SchedulerConfig(
        mode="nightly", nightly_utc_hour=3, min_kick_gap_s=0,
    ))
    now = _utc(hour=2)  # before 03:00 UTC
    decision = sch.should_kick(
        state=_state(last_at=None), current_samples=100, now=now,
    )
    assert not decision.kick
    assert decision.reason == "no-trigger"
    assert decision.next_kick_estimated_at is not None


def test_nightly_skips_when_already_kicked_today():
    sch = DistillScheduler(SchedulerConfig(
        mode="nightly", nightly_utc_hour=3, min_kick_gap_s=0,
    ))
    today_target = _utc(hour=3)
    now = _utc(hour=10)  # later same day
    decision = sch.should_kick(
        state=_state(last_at=today_target),  # already kicked at target
        current_samples=100, now=now,
    )
    assert not decision.kick


def test_nightly_kicks_next_day_after_yesterday_kick():
    sch = DistillScheduler(SchedulerConfig(
        mode="nightly", nightly_utc_hour=3, min_kick_gap_s=0,
    ))
    yesterday_kick = _utc(year=2026, month=4, day=24, hour=3)
    now = _utc(year=2026, month=4, day=25, hour=4)
    decision = sch.should_kick(
        state=_state(last_at=yesterday_kick), current_samples=100, now=now,
    )
    assert decision.kick


# ---------------------------------------------------------------------------
# Quality-drop mode
# ---------------------------------------------------------------------------


def test_quality_drop_kicks_when_drop_exceeds_threshold():
    sch = DistillScheduler(SchedulerConfig(
        mode="quality-drop", quality_drop_threshold=0.05, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(quality=0.90),
        current_samples=100,
        current_quality=0.80,  # 10pp drop, exceeds 5pp threshold
        now=_utc(),
    )
    assert decision.kick
    assert decision.reason == "quality-drop"


def test_quality_drop_doesnt_kick_below_threshold():
    sch = DistillScheduler(SchedulerConfig(
        mode="quality-drop", quality_drop_threshold=0.05, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(quality=0.90),
        current_samples=100,
        current_quality=0.88,  # 2pp drop, below 5pp threshold
        now=_utc(),
    )
    assert not decision.kick


def test_quality_drop_doesnt_kick_when_quality_unset():
    """Without a baseline OR a current measurement, can't compare."""
    sch = DistillScheduler(SchedulerConfig(
        mode="quality-drop", quality_drop_threshold=0.05, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(quality=None),
        current_samples=100, current_quality=0.5, now=_utc(),
    )
    assert not decision.kick


def test_quality_drop_doesnt_kick_when_quality_improved():
    """Improvement = no kick (correct — model is doing fine)."""
    sch = DistillScheduler(SchedulerConfig(
        mode="quality-drop", quality_drop_threshold=0.05, min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(quality=0.80),
        current_samples=100,
        current_quality=0.95,  # improvement
        now=_utc(),
    )
    assert not decision.kick


# ---------------------------------------------------------------------------
# Cron mode (Phase 1 minimal — hour field only)
# ---------------------------------------------------------------------------


def test_cron_kicks_at_target_hour():
    sch = DistillScheduler(SchedulerConfig(
        mode="cron", cron_spec="0 3 * * *", min_kick_gap_s=0,
    ))
    now = _utc(hour=4)  # past 03:00
    decision = sch.should_kick(
        state=_state(last_at=None), current_samples=100, now=now,
    )
    assert decision.kick
    assert decision.reason == "cron"


def test_cron_doesnt_kick_before_target_hour():
    sch = DistillScheduler(SchedulerConfig(
        mode="cron", cron_spec="0 3 * * *", min_kick_gap_s=0,
    ))
    now = _utc(hour=2)
    decision = sch.should_kick(
        state=_state(last_at=None), current_samples=100, now=now,
    )
    assert not decision.kick


def test_cron_rejects_malformed_spec_at_decide_time():
    """Phase 1 minimal cron parser only handles `M H * * *`. Anything else
    silently degrades to no-trigger (operator should validate at config
    time; this is the runtime safety net)."""
    sch = DistillScheduler(SchedulerConfig(
        mode="cron", cron_spec="not a cron", min_kick_gap_s=0,
    ))
    decision = sch.should_kick(
        state=_state(last_at=None), current_samples=100, now=_utc(hour=10),
    )
    assert not decision.kick
    assert decision.reason == "no-trigger"


# ---------------------------------------------------------------------------
# KickDecision shape
# ---------------------------------------------------------------------------


def test_decision_is_frozen_dataclass():
    sch = DistillScheduler(SchedulerConfig(mode="manual"))
    d = sch.should_kick(state=_state(), current_samples=100, now=_utc())
    with pytest.raises(AttributeError):
        d.kick = True  # type: ignore[misc]


def test_decision_fields_populated():
    sch = DistillScheduler(SchedulerConfig(
        mode="samples", samples_threshold=100, min_kick_gap_s=0,
    ))
    d = sch.should_kick(state=_state(), current_samples=100, now=_utc())
    assert isinstance(d, KickDecision)
    assert isinstance(d.kick, bool)
    assert isinstance(d.reason, str)
    assert isinstance(d.samples_since_last_kick, int)
    assert isinstance(d.seconds_since_last_kick, float)


# ---------------------------------------------------------------------------
# Defensive validation
# ---------------------------------------------------------------------------


def test_should_kick_rejects_negative_current_samples():
    sch = DistillScheduler(SchedulerConfig(mode="manual"))
    with pytest.raises(ValueError, match="current_samples"):
        sch.should_kick(state=_state(), current_samples=-1, now=_utc())


def test_should_kick_now_default_uses_current_time():
    """Smoke test — passing now=None doesn't crash."""
    sch = DistillScheduler(SchedulerConfig(mode="manual"))
    decision = sch.should_kick(state=_state(), current_samples=100)
    assert decision.reason == "manual"


# ---------------------------------------------------------------------------
# Dual-loss snapflow_loss_step composition (Day 3 cross-test)
# ---------------------------------------------------------------------------


def test_self_distilling_serve_loss_step_exported_from_snapflow():
    """The new dual-loss function must be importable + callable."""
    from tether.distill.snapflow import (
        DEFAULT_DUAL_LOSS_TEACHER_ALPHA,
        self_distilling_serve_loss_step,
    )
    assert callable(self_distilling_serve_loss_step)
    assert 0 < DEFAULT_DUAL_LOSS_TEACHER_ALPHA < 1
