"""Tests for src/tether/runtime/batching.py — cost-weighted scheduler primitives.

Covers: cost-model rolling-window correctness, cold-start fallback, threading
safety, scheduler flush conditions at boundaries (empty / single-over-budget
/ budget-reached / timeout / under-budget), shape-homogeneity flag, mode
validation, fallback mode ignores profiled measurements.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from tether.runtime.batching import (
    _DEFAULT_COLD_START_MS,
    _MAX_BUDGET_MS,
    _MIN_BUDGET_MS,
    CostBudgetScheduler,
    CostKey,
    CostMode,
    GpuMsCostModel,
    SchedulerDecision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeReq:
    shape_key: str


def shape_fn(req: FakeReq) -> str:
    return req.shape_key


# ---------------------------------------------------------------------------
# GpuMsCostModel construction invariants
# ---------------------------------------------------------------------------


def test_rejects_zero_default_cold_start():
    with pytest.raises(ValueError, match="default_cold_start_ms"):
        GpuMsCostModel(default_cold_start_ms=0)


def test_rejects_negative_default_cold_start():
    with pytest.raises(ValueError, match="default_cold_start_ms"):
        GpuMsCostModel(default_cold_start_ms=-5.0)


# ---------------------------------------------------------------------------
# GpuMsCostModel.estimate — cold-start + rolling window
# ---------------------------------------------------------------------------


def test_estimate_returns_cold_start_default_when_unseen():
    cm = GpuMsCostModel()
    assert cm.estimate("m1", "franka", "b1_seq968") == _DEFAULT_COLD_START_MS


def test_estimate_returns_cold_start_until_three_measurements():
    cm = GpuMsCostModel()
    cm.record_measurement("m1", "franka", "b1", gpu_ms=42.0)
    assert cm.estimate("m1", "franka", "b1") == _DEFAULT_COLD_START_MS
    cm.record_measurement("m1", "franka", "b1", gpu_ms=43.0)
    assert cm.estimate("m1", "franka", "b1") == _DEFAULT_COLD_START_MS
    # At n=3, median becomes stable enough to trust
    cm.record_measurement("m1", "franka", "b1", gpu_ms=44.0)
    assert cm.estimate("m1", "franka", "b1") == 43.0


def test_estimate_uses_median_of_window():
    cm = GpuMsCostModel()
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        cm.record_measurement("m1", "franka", "b1", gpu_ms=v)
    assert cm.estimate("m1", "franka", "b1") == 30.0


def test_rolling_window_bounded_at_100():
    cm = GpuMsCostModel()
    # 100 values at 100ms, then 100 values at 10ms — median should settle to 10
    for _ in range(100):
        cm.record_measurement("m1", "franka", "b1", gpu_ms=100.0)
    for _ in range(100):
        cm.record_measurement("m1", "franka", "b1", gpu_ms=10.0)
    assert cm.estimate("m1", "franka", "b1") == 10.0


def test_different_shape_keys_are_independent():
    cm = GpuMsCostModel()
    for _ in range(5):
        cm.record_measurement("m1", "franka", "b1", gpu_ms=10.0)
        cm.record_measurement("m1", "franka", "b2", gpu_ms=50.0)
    assert cm.estimate("m1", "franka", "b1") == 10.0
    assert cm.estimate("m1", "franka", "b2") == 50.0


def test_different_embodiments_are_independent():
    cm = GpuMsCostModel()
    for _ in range(5):
        cm.record_measurement("m1", "franka", "b1", gpu_ms=10.0)
        cm.record_measurement("m1", "so100", "b1", gpu_ms=50.0)
    assert cm.estimate("m1", "franka", "b1") == 10.0
    assert cm.estimate("m1", "so100", "b1") == 50.0


# ---------------------------------------------------------------------------
# GpuMsCostModel.record_measurement — defensive validation
# ---------------------------------------------------------------------------


def test_record_drops_negative_measurement():
    cm = GpuMsCostModel()
    cm.record_measurement("m1", "franka", "b1", gpu_ms=-1.0)
    assert not cm.has_measurements("m1", "franka", "b1")


def test_record_drops_zero_measurement():
    cm = GpuMsCostModel()
    cm.record_measurement("m1", "franka", "b1", gpu_ms=0.0)
    assert not cm.has_measurements("m1", "franka", "b1")


def test_record_drops_nan_measurement():
    cm = GpuMsCostModel()
    cm.record_measurement("m1", "franka", "b1", gpu_ms=float("nan"))
    assert not cm.has_measurements("m1", "franka", "b1")


def test_record_accepts_valid_measurement():
    cm = GpuMsCostModel()
    cm.record_measurement("m1", "franka", "b1", gpu_ms=42.5)
    assert cm.has_measurements("m1", "franka", "b1")


# ---------------------------------------------------------------------------
# GpuMsCostModel — thread safety + snapshot
# ---------------------------------------------------------------------------


def test_concurrent_record_from_multiple_threads():
    cm = GpuMsCostModel()

    def writer(tag: float):
        for _ in range(500):
            cm.record_measurement("m1", "franka", "b1", gpu_ms=tag)

    threads = [threading.Thread(target=writer, args=(float(i + 1),)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = cm.export_snapshot()
    matching = [e for e in snap["entries"] if e["shape_key"] == "b1"]
    assert len(matching) == 1
    # 5 writers × 500 writes = 2500 total; window capped at 100
    assert matching[0]["n"] == 100


def test_export_snapshot_empty():
    cm = GpuMsCostModel()
    snap = cm.export_snapshot()
    assert snap["entries"] == []
    assert snap["default_cold_start_ms"] == _DEFAULT_COLD_START_MS


def test_export_snapshot_has_stats():
    cm = GpuMsCostModel()
    for v in [10.0, 20.0, 30.0]:
        cm.record_measurement("m1", "franka", "b1", gpu_ms=v)
    snap = cm.export_snapshot()
    entry = snap["entries"][0]
    assert entry["model_id"] == "m1"
    assert entry["embodiment"] == "franka"
    assert entry["shape_key"] == "b1"
    assert entry["n"] == 3
    assert entry["median_ms"] == 20.0
    assert entry["min_ms"] == 10.0
    assert entry["max_ms"] == 30.0


# ---------------------------------------------------------------------------
# CostBudgetScheduler — construction invariants
# ---------------------------------------------------------------------------


def _mk_scheduler(budget_ms: float = 100.0, mode: str = CostMode.PROFILED, max_wait_ms: float = 5.0) -> CostBudgetScheduler:
    return CostBudgetScheduler(
        max_cost_per_batch_ms=budget_ms,
        cost_model=GpuMsCostModel(),
        max_wait_ms=max_wait_ms,
        mode=mode,
    )


def test_rejects_budget_below_floor():
    with pytest.raises(ValueError, match="max_cost_per_batch_ms"):
        _mk_scheduler(budget_ms=_MIN_BUDGET_MS - 0.1)


def test_rejects_budget_above_ceiling():
    with pytest.raises(ValueError, match="max_cost_per_batch_ms"):
        _mk_scheduler(budget_ms=_MAX_BUDGET_MS + 0.1)


def test_accepts_boundary_budgets():
    _mk_scheduler(budget_ms=_MIN_BUDGET_MS)
    _mk_scheduler(budget_ms=_MAX_BUDGET_MS)


def test_rejects_zero_max_wait_ms():
    with pytest.raises(ValueError, match="max_wait_ms"):
        _mk_scheduler(max_wait_ms=0)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        _mk_scheduler(mode="bogus")


def test_accepts_both_valid_modes():
    _mk_scheduler(mode=CostMode.PROFILED)
    _mk_scheduler(mode=CostMode.FALLBACK)


# ---------------------------------------------------------------------------
# CostBudgetScheduler.should_flush — flush conditions
# ---------------------------------------------------------------------------


def test_empty_batch_does_not_flush():
    sch = _mk_scheduler(budget_ms=100.0)
    decision = sch.should_flush(
        [], model_id="m1", embodiment="franka",
        oldest_wait_ms=0.0, shape_key_fn=shape_fn,
    )
    assert decision.flush is False
    assert decision.reason == "empty"
    assert decision.size == 0


def test_under_budget_and_within_timeout_holds():
    sch = _mk_scheduler(budget_ms=200.0, max_wait_ms=5.0)
    # At cold-start 50ms × 2 = 100ms cost, below 200ms budget
    decision = sch.should_flush(
        [FakeReq("b1"), FakeReq("b1")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=1.0, shape_key_fn=shape_fn,
    )
    assert decision.flush is False
    assert decision.reason == "under_budget"
    assert decision.size == 2
    assert decision.batch_cost_ms == 100.0


def test_budget_reached_flushes():
    sch = _mk_scheduler(budget_ms=100.0, max_wait_ms=5.0)
    # Three requests × 50ms cold-start = 150ms, exceeds 100ms budget
    decision = sch.should_flush(
        [FakeReq("b1"), FakeReq("b1"), FakeReq("b1")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=0.0, shape_key_fn=shape_fn,
    )
    assert decision.flush is True
    assert decision.reason == "budget_reached"
    assert decision.batch_cost_ms == 150.0


def test_timeout_flushes_even_under_budget():
    sch = _mk_scheduler(budget_ms=200.0, max_wait_ms=5.0)
    decision = sch.should_flush(
        [FakeReq("b1")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=6.0, shape_key_fn=shape_fn,
    )
    assert decision.flush is True
    assert decision.reason == "timeout"
    assert decision.size == 1


def test_single_request_over_budget_flushes_immediately():
    cm = GpuMsCostModel()
    # Seed with enough samples to replace cold-start
    for _ in range(5):
        cm.record_measurement("m1", "franka", "heavy", gpu_ms=500.0)
    sch = CostBudgetScheduler(max_cost_per_batch_ms=100.0, cost_model=cm)
    decision = sch.should_flush(
        [FakeReq("heavy")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=0.0, shape_key_fn=shape_fn,
    )
    assert decision.flush is True
    assert decision.reason == "single_request_over_budget"


# ---------------------------------------------------------------------------
# Shape homogeneity signal
# ---------------------------------------------------------------------------


def test_shape_homogeneous_true_for_all_same_shape():
    sch = _mk_scheduler(budget_ms=300.0)
    decision = sch.should_flush(
        [FakeReq("b1"), FakeReq("b1"), FakeReq("b1")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=0.0, shape_key_fn=shape_fn,
    )
    assert decision.shape_homogeneous is True


def test_shape_homogeneous_false_for_mixed():
    sch = _mk_scheduler(budget_ms=300.0)
    decision = sch.should_flush(
        [FakeReq("b1"), FakeReq("b2"), FakeReq("b1")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=0.0, shape_key_fn=shape_fn,
    )
    assert decision.shape_homogeneous is False


# ---------------------------------------------------------------------------
# Fallback mode
# ---------------------------------------------------------------------------


def test_fallback_mode_ignores_profiled_measurements():
    cm = GpuMsCostModel(default_cold_start_ms=50.0)
    # Heavily measure this shape so profiled would return a big number
    for _ in range(20):
        cm.record_measurement("m1", "franka", "b1", gpu_ms=500.0)
    sch = CostBudgetScheduler(
        max_cost_per_batch_ms=200.0,
        cost_model=cm,
        mode=CostMode.FALLBACK,
    )
    # Fallback returns 50ms × 2 requests = 100ms, NOT 500 × 2 = 1000
    cost = sch.batch_cost_ms(
        [FakeReq("b1"), FakeReq("b1")],
        model_id="m1", embodiment="franka", shape_key_fn=shape_fn,
    )
    assert cost == 100.0


def test_profiled_mode_uses_cost_model():
    cm = GpuMsCostModel(default_cold_start_ms=50.0)
    for _ in range(5):
        cm.record_measurement("m1", "franka", "b1", gpu_ms=200.0)
    sch = CostBudgetScheduler(
        max_cost_per_batch_ms=300.0,
        cost_model=cm,
        mode=CostMode.PROFILED,
    )
    cost = sch.batch_cost_ms(
        [FakeReq("b1"), FakeReq("b1")],
        model_id="m1", embodiment="franka", shape_key_fn=shape_fn,
    )
    assert cost == 400.0  # 200 × 2 from profiled median


# ---------------------------------------------------------------------------
# SchedulerDecision fields
# ---------------------------------------------------------------------------


def test_decision_has_all_expected_fields():
    sch = _mk_scheduler(budget_ms=100.0)
    d = sch.should_flush(
        [FakeReq("b1")],
        model_id="m1", embodiment="franka",
        oldest_wait_ms=0.0, shape_key_fn=shape_fn,
    )
    assert isinstance(d, SchedulerDecision)
    assert hasattr(d, "flush")
    assert hasattr(d, "reason")
    assert hasattr(d, "batch_cost_ms")
    assert hasattr(d, "budget_ms")
    assert hasattr(d, "size")
    assert hasattr(d, "shape_homogeneous")
    assert d.budget_ms == 100.0


# ---------------------------------------------------------------------------
# Properties + introspection
# ---------------------------------------------------------------------------


def test_scheduler_exposes_properties():
    cm = GpuMsCostModel()
    sch = CostBudgetScheduler(max_cost_per_batch_ms=150.0, cost_model=cm, max_wait_ms=7.0, mode=CostMode.FALLBACK)
    assert sch.max_cost_ms == 150.0
    assert sch.max_wait_ms == 7.0
    assert sch.mode == CostMode.FALLBACK
    assert sch.cost_model is cm


def test_cost_mode_all_returns_known_values():
    assert set(CostMode.all()) == {"profiled", "fallback"}


def test_cost_key_is_frozen():
    k = CostKey(model_id="m1", embodiment="franka", shape_key="b1")
    with pytest.raises(AttributeError):
        k.model_id = "m2"  # type: ignore[misc]
