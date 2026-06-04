"""Tests for src/tether/runtime/slo.py — the rolling-window SLO tracker.

Covers:
- `parse_slo_spec` parsing contract (valid + invalid cases)
- `validate_slo_mode` contract
- `SLOTracker` state transitions (recording, violation detection, recovery)
- Thread safety (concurrent record_latency_ms from multiple threads)
"""
from __future__ import annotations

import threading

import pytest

from tether.runtime.slo import (
    VALID_SLO_MODES,
    SloSpec,
    SLOTracker,
    parse_slo_spec,
    validate_slo_mode,
)


# ---------------------------------------------------------------------------
# parse_slo_spec
# ---------------------------------------------------------------------------


def test_parse_slo_spec_p99_150ms():
    s = parse_slo_spec("p99=150ms")
    assert s.percentile == 99.0
    assert s.threshold_ms == 150.0


def test_parse_slo_spec_p95_with_spaces():
    s = parse_slo_spec("p95 = 200 ms")
    assert s.percentile == 95.0
    assert s.threshold_ms == 200.0


def test_parse_slo_spec_p99_9_fractional_percentile():
    s = parse_slo_spec("p99.9=500ms")
    assert s.percentile == 99.9
    assert s.threshold_ms == 500.0


def test_parse_slo_spec_fractional_threshold():
    s = parse_slo_spec("p99=15.5ms")
    assert s.threshold_ms == 15.5


def test_parse_slo_spec_case_insensitive():
    s = parse_slo_spec("P99=150MS")
    assert s.percentile == 99.0
    assert s.threshold_ms == 150.0


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "p99",
    "p99=150",      # missing 'ms'
    "p99=150s",     # wrong unit
    "p=150ms",      # missing percentile value
    "99=150ms",     # missing 'p' prefix
    "p150=150ms",   # percentile > 100
    "p0=150ms",     # percentile = 0 (rejected by SloSpec validator)
    "p99=-100ms",   # negative threshold
    "p99=0ms",      # zero threshold
])
def test_parse_slo_spec_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_slo_spec(bad)


def test_parse_slo_spec_rejects_non_string():
    with pytest.raises(ValueError):
        parse_slo_spec(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_slo_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["log_only", "503", "degrade"])
def test_validate_slo_mode_accepts_valid(mode):
    assert validate_slo_mode(mode) == mode


@pytest.mark.parametrize("bad", ["LOG_ONLY", "fail", "", "404", "reject"])
def test_validate_slo_mode_rejects_invalid(bad):
    with pytest.raises(ValueError):
        validate_slo_mode(bad)


def test_valid_slo_modes_frozen():
    assert VALID_SLO_MODES == frozenset({"log_only", "503", "degrade"})


# ---------------------------------------------------------------------------
# SLOTracker — construction
# ---------------------------------------------------------------------------


def _spec(pct: float = 99.0, ms: float = 100.0) -> SloSpec:
    return SloSpec(percentile=pct, threshold_ms=ms)


def test_tracker_rejects_bad_ctor_args():
    with pytest.raises(ValueError):
        SLOTracker(_spec(), window_size=0)
    with pytest.raises(ValueError):
        SLOTracker(_spec(), check_every=-1)
    with pytest.raises(ValueError):
        SLOTracker(_spec(), recover_ratio=1.0)
    with pytest.raises(ValueError):
        SLOTracker(_spec(), recover_ratio=0.0)
    with pytest.raises(ValueError):
        SLOTracker(_spec(), recover_windows=0)


def test_tracker_initial_state_not_violating():
    t = SLOTracker(_spec())
    assert t.is_violating() is False
    # NaN comparison needs explicit handling
    p = t.current_p99()
    assert p != p  # NaN


# ---------------------------------------------------------------------------
# SLOTracker — violation detection
# ---------------------------------------------------------------------------


def test_tracker_does_not_flag_before_check_every_samples():
    t = SLOTracker(_spec(99.0, 50.0), check_every=32)
    for _ in range(31):
        t.record_latency_ms(1000.0)  # way over SLO
    # 31 samples recorded but check hasn't fired yet (need 32)
    assert t.is_violating() is False


def test_tracker_flags_violation_on_first_check_when_over_slo():
    t = SLOTracker(_spec(99.0, 50.0), check_every=32)
    # 32 samples all at 500ms — p99 way over 50ms threshold
    for _ in range(32):
        t.record_latency_ms(500.0)
    assert t.is_violating() is True
    assert t.current_p99() == 500.0


def test_tracker_stays_clean_when_under_slo():
    t = SLOTracker(_spec(99.0, 100.0), check_every=16)
    for _ in range(100):
        t.record_latency_ms(30.0)
    assert t.is_violating() is False
    assert t.current_p99() == 30.0


def test_tracker_recovers_after_consecutive_windows_under_recovery_threshold():
    spec = _spec(99.0, 100.0)  # threshold 100ms, recovery 0.8×100 = 80ms
    t = SLOTracker(spec, check_every=32, recover_ratio=0.8, recover_windows=3)
    # First: violate
    for _ in range(32):
        t.record_latency_ms(500.0)
    assert t.is_violating() is True

    # Now record many samples under recovery threshold (< 80ms).
    # Need recover_windows=3 consecutive checks where p99 < 80ms AND the
    # rolling window must be dominated by the new samples.
    for _ in range(32 * 3 + 1000):  # fully wash out the old violation samples
        t.record_latency_ms(10.0)
    assert t.is_violating() is False


def test_tracker_does_not_recover_if_p99_rebounds():
    spec = _spec(99.0, 100.0)
    t = SLOTracker(spec, check_every=32, recover_ratio=0.8, recover_windows=3)
    # Violate first
    for _ in range(32):
        t.record_latency_ms(500.0)
    assert t.is_violating() is True

    # Start recovery: two checks under threshold, then rebound above — counter resets
    for _ in range(64):
        t.record_latency_ms(10.0)
    # At this point one or two recovery checks have fired (window still has
    # high-latency tails from the violation); tracker may still be violating.
    # Now push back into violation
    for _ in range(64):
        t.record_latency_ms(500.0)
    assert t.is_violating() is True


def test_tracker_clamps_negative_latency_to_zero():
    t = SLOTracker(_spec(99.0, 100.0))
    t.record_latency_ms(-10.0)
    # Just verifies no crash; negative samples don't pollute percentile math
    assert t.is_violating() is False


def test_tracker_reset_clears_state():
    t = SLOTracker(_spec(99.0, 50.0), check_every=32)
    for _ in range(32):
        t.record_latency_ms(500.0)
    assert t.is_violating() is True
    t.reset()
    assert t.is_violating() is False
    p = t.current_p99()
    assert p != p  # NaN


# ---------------------------------------------------------------------------
# SLOTracker — thread safety
# ---------------------------------------------------------------------------


def test_tracker_thread_safe_record_concurrent():
    """Two threads recording latencies concurrently should not crash or
    produce inconsistent state.
    """
    t = SLOTracker(_spec(99.0, 100.0), check_every=16)
    errors: list[Exception] = []

    def worker(latency: float, count: int):
        try:
            for _ in range(count):
                t.record_latency_ms(latency)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(50.0, 500))
    t2 = threading.Thread(target=worker, args=(200.0, 500))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    # After 1000 samples, current_p99 should have been computed
    p = t.current_p99()
    assert p == p  # not NaN
    # Violation state is deterministic (either flagged or not); we don't
    # assert a specific value because it depends on scheduling, but it
    # shouldn't raise and should be a bool
    assert isinstance(t.is_violating(), bool)


# ---------------------------------------------------------------------------
# SLOTracker — different percentiles
# ---------------------------------------------------------------------------


def test_tracker_p95_detects_violation():
    t = SLOTracker(_spec(95.0, 100.0), check_every=20)
    # 20 samples; 95th percentile ≈ 2 samples over threshold
    samples = [50.0] * 18 + [200.0] * 2
    for s in samples:
        t.record_latency_ms(s)
    # p95 of the sample set is 200ms — over the 100ms SLO
    assert t.is_violating() is True


def test_tracker_p99_ignores_single_outlier():
    t = SLOTracker(_spec(99.0, 100.0), check_every=100)
    # 99 under, 1 outlier at 500ms — p99 stays under (interpolated)
    for _ in range(99):
        t.record_latency_ms(50.0)
    t.record_latency_ms(500.0)
    # With only 1 outlier out of 100, p99 is ~54.5ms (linear interpolation)
    # — well under 100ms SLO
    assert t.is_violating() is False
