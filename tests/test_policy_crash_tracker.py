"""Tests for src/tether/runtime/policy_crash_tracker.py — Day 8 substrate.

Per ADR 2026-04-25-policy-versioning-architecture: per-slot
consecutive-crash counter + drain/degrade verdict.
"""
from __future__ import annotations

import pytest

from tether.runtime.policy_crash_tracker import (
    ALL_VERDICTS,
    CrashTrackerVerdict,
    PolicyCrashTracker,
)


# ---------------------------------------------------------------------------
# CrashTrackerVerdict dataclass
# ---------------------------------------------------------------------------


def test_verdict_is_frozen():
    v = CrashTrackerVerdict(
        verdict="healthy", crash_counts={"a": 0, "b": 0},
        threshold=5, reason="ok",
    )
    with pytest.raises(AttributeError):
        v.verdict = "degraded"  # type: ignore[misc]


def test_verdict_rejects_invalid_verdict():
    with pytest.raises(ValueError, match="verdict"):
        CrashTrackerVerdict(
            verdict="invented", crash_counts={}, threshold=1, reason="x",
        )


def test_verdict_rejects_zero_threshold():
    with pytest.raises(ValueError, match="threshold"):
        CrashTrackerVerdict(
            verdict="healthy", crash_counts={}, threshold=0, reason="x",
        )


def test_verdict_should_degrade_property():
    healthy = CrashTrackerVerdict(
        verdict="healthy", crash_counts={}, threshold=1, reason="x",
    )
    degraded = CrashTrackerVerdict(
        verdict="degraded", crash_counts={}, threshold=1, reason="x",
    )
    assert not healthy.should_degrade
    assert degraded.should_degrade


def test_verdict_slot_to_drain_property():
    drain_a = CrashTrackerVerdict(
        verdict="drain-a", crash_counts={}, threshold=1, reason="x",
    )
    drain_b = CrashTrackerVerdict(
        verdict="drain-b", crash_counts={}, threshold=1, reason="x",
    )
    healthy = CrashTrackerVerdict(
        verdict="healthy", crash_counts={}, threshold=1, reason="x",
    )
    assert drain_a.slot_to_drain == "a"
    assert drain_b.slot_to_drain == "b"
    assert healthy.slot_to_drain is None


# ---------------------------------------------------------------------------
# PolicyCrashTracker — construction validation
# ---------------------------------------------------------------------------


def test_tracker_rejects_empty_slots():
    with pytest.raises(ValueError, match="slots"):
        PolicyCrashTracker(slots=(), threshold=1)


def test_tracker_rejects_zero_threshold():
    with pytest.raises(ValueError, match="threshold"):
        PolicyCrashTracker(slots=("a",), threshold=0)


def test_tracker_rejects_duplicate_slots():
    with pytest.raises(ValueError, match="unique"):
        PolicyCrashTracker(slots=("a", "a"), threshold=1)


def test_tracker_initial_counts_are_zero():
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    assert tracker.crash_count("a") == 0
    assert tracker.crash_count("b") == 0


def test_tracker_unknown_slot_raises():
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    with pytest.raises(KeyError):
        tracker.crash_count("c")
    with pytest.raises(KeyError):
        tracker.record_crash(slot="c")
    with pytest.raises(KeyError):
        tracker.record_clean(slot="c")
    with pytest.raises(KeyError):
        tracker.reset(slot="c")


# ---------------------------------------------------------------------------
# Single-policy mode (one slot "prod")
# ---------------------------------------------------------------------------


def test_single_policy_below_threshold_is_healthy():
    tracker = PolicyCrashTracker(slots=("prod",), threshold=5)
    for _ in range(4):
        tracker.record_crash(slot="prod")
    v = tracker.verdict()
    assert v.verdict == "healthy"


def test_single_policy_at_threshold_degrades():
    tracker = PolicyCrashTracker(slots=("prod",), threshold=5)
    for _ in range(5):
        tracker.record_crash(slot="prod")
    v = tracker.verdict()
    assert v.verdict == "degraded"
    assert v.should_degrade


def test_single_policy_clean_resets_counter():
    tracker = PolicyCrashTracker(slots=("prod",), threshold=5)
    for _ in range(4):
        tracker.record_crash(slot="prod")
    tracker.record_clean(slot="prod")
    assert tracker.crash_count("prod") == 0
    assert tracker.verdict().verdict == "healthy"


# ---------------------------------------------------------------------------
# 2-policy mode (slots "a" + "b") -- drain logic
# ---------------------------------------------------------------------------


def test_2policy_both_below_threshold_is_healthy():
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(2):
        tracker.record_crash(slot="a")
    for _ in range(3):
        tracker.record_crash(slot="b")
    v = tracker.verdict()
    assert v.verdict == "healthy"


def test_2policy_a_exceeds_drains_a():
    """a fails 5x; b is clean -> drain-a (100% to b)."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(5):
        tracker.record_crash(slot="a")
    v = tracker.verdict()
    assert v.verdict == "drain-a"
    assert v.slot_to_drain == "a"
    assert not v.should_degrade


def test_2policy_b_exceeds_drains_b():
    """b fails 5x; a is clean -> drain-b (100% to a)."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(5):
        tracker.record_crash(slot="b")
    v = tracker.verdict()
    assert v.verdict == "drain-b"
    assert v.slot_to_drain == "b"
    assert not v.should_degrade


def test_2policy_both_exceed_degrades():
    """Both slots fail >= threshold -> full degraded (problem isn't
    slot-specific)."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(5):
        tracker.record_crash(slot="a")
        tracker.record_crash(slot="b")
    v = tracker.verdict()
    assert v.verdict == "degraded"
    assert v.should_degrade


def test_2policy_clean_on_other_slot_doesnt_reset_first():
    """record_clean(slot=b) does NOT reset slot=a's counter."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(4):
        tracker.record_crash(slot="a")
    tracker.record_clean(slot="b")
    assert tracker.crash_count("a") == 4  # untouched
    assert tracker.crash_count("b") == 0


def test_2policy_drain_persists_until_clean():
    """drain-a verdict persists across additional crashes on a; clean on
    a resets and verdict goes back to healthy."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(7):
        tracker.record_crash(slot="a")
    assert tracker.verdict().verdict == "drain-a"
    tracker.record_clean(slot="a")
    assert tracker.verdict().verdict == "healthy"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_specific_slot():
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(3):
        tracker.record_crash(slot="a")
        tracker.record_crash(slot="b")
    tracker.reset(slot="a")
    assert tracker.crash_count("a") == 0
    assert tracker.crash_count("b") == 3


def test_reset_all_slots():
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(3):
        tracker.record_crash(slot="a")
        tracker.record_crash(slot="b")
    tracker.reset()
    assert tracker.crash_count("a") == 0
    assert tracker.crash_count("b") == 0


# ---------------------------------------------------------------------------
# 3-policy edge: verdict semantics for slot count != 2
# ---------------------------------------------------------------------------


def test_3plus_slots_one_exceeds_degrades():
    """Drain logic only kicks in for the 2-policy {a, b} case. Other
    multi-slot configs treat exceed as degraded (single-slot semantics
    extended)."""
    tracker = PolicyCrashTracker(slots=("a", "b", "c"), threshold=5)
    for _ in range(5):
        tracker.record_crash(slot="b")
    v = tracker.verdict()
    # 3-slot config doesn't get drain-X; falls through to degraded.
    assert v.verdict == "degraded"


def test_2policy_named_other_slots_degrades_on_exceed():
    """Drain logic requires slots == ('a', 'b'). Custom slot names like
    ('prod', 'shadow') don't get drain semantics."""
    tracker = PolicyCrashTracker(slots=("prod", "shadow"), threshold=5)
    for _ in range(5):
        tracker.record_crash(slot="prod")
    v = tracker.verdict()
    # Not the {a, b} pattern -> degraded fallthrough.
    assert v.verdict == "degraded"
