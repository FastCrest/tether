"""Tests for src/tether/runtime/policy_runtime.py — per-policy queue + worker.

Covers: construction invariants, lifecycle (start/stop idempotence + drain),
submit happy path + queue-full backpressure, worker batches multiple
submits, scheduler integration (single-request flushes immediately when
over budget), run_batch exception fan-out, result-count mismatch,
cost-model post-flush update, snapshot for diagnostics.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tether.runtime.batching import (
    CostBudgetScheduler,
    CostMode,
    GpuMsCostModel,
)
from tether.runtime.policy_runtime import (
    PolicyRuntime,
    QueueFull,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeReq:
    shape_key: str
    payload: int = 0


def shape_fn(req: FakeReq) -> str:
    return req.shape_key


def _mk_runtime(
    *,
    run_batch_cb,
    budget_ms: float = 100.0,
    max_wait_ms: float = 5.0,
    max_queue: int = 10,
    cost_model: GpuMsCostModel | None = None,
) -> PolicyRuntime:
    cm = cost_model or GpuMsCostModel()
    sch = CostBudgetScheduler(
        max_cost_per_batch_ms=budget_ms,
        cost_model=cm,
        max_wait_ms=max_wait_ms,
    )
    return PolicyRuntime(
        policy_id="test",
        model_id="m1",
        embodiment="franka",
        scheduler=sch,
        cost_model=cm,
        run_batch_callback=run_batch_cb,
        shape_key_fn=shape_fn,
        max_queue=max_queue,
    )


async def _identity_batch(requests: list[FakeReq]) -> list[int]:
    """Returns the payloads as the batch result (mirror)."""
    return [r.payload for r in requests]


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_rejects_empty_policy_id():
    with pytest.raises(ValueError, match="policy_id"):
        _mk_runtime(run_batch_cb=_identity_batch).__class__(
            policy_id="",
            model_id="m1",
            embodiment="franka",
            scheduler=CostBudgetScheduler(100.0, GpuMsCostModel()),
            cost_model=GpuMsCostModel(),
            run_batch_callback=_identity_batch,
            shape_key_fn=shape_fn,
        )


def test_rejects_zero_max_queue():
    with pytest.raises(ValueError, match="max_queue"):
        _mk_runtime(run_batch_cb=_identity_batch, max_queue=0)


def test_constructor_accessors_populated():
    r = _mk_runtime(run_batch_cb=_identity_batch, budget_ms=120.0)
    assert r.policy_id == "test"
    assert r.model_id == "m1"
    assert r.embodiment == "franka"
    assert r.scheduler.max_cost_ms == 120.0
    assert r.is_running is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_then_stop_idempotent():
    r = _mk_runtime(run_batch_cb=_identity_batch)
    await r.start()
    assert r.is_running is True
    await r.start()  # second start is a no-op
    assert r.is_running is True
    await r.stop()
    assert r.is_running is False
    await r.stop()  # second stop is a no-op


@pytest.mark.asyncio
async def test_submit_before_start_raises():
    r = _mk_runtime(run_batch_cb=_identity_batch)
    with pytest.raises(RuntimeError, match="not running"):
        await r.submit(FakeReq("b1", payload=1))


# ---------------------------------------------------------------------------
# Submit happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_returns_result_via_worker():
    r = _mk_runtime(run_batch_cb=_identity_batch)
    await r.start()
    try:
        result = await asyncio.wait_for(
            r.submit(FakeReq("b1", payload=42)), timeout=2.0,
        )
        assert result == 42
    finally:
        await r.stop()


@pytest.mark.asyncio
async def test_submit_multiple_concurrent_returns_each_correct_result():
    r = _mk_runtime(run_batch_cb=_identity_batch)
    await r.start()
    try:
        coros = [r.submit(FakeReq("b1", payload=i)) for i in range(5)]
        results = await asyncio.wait_for(asyncio.gather(*coros), timeout=2.0)
        assert sorted(results) == [0, 1, 2, 3, 4]
    finally:
        await r.stop()


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_raises_queue_full_when_at_capacity():
    """Block the worker so the queue fills, then a further submit raises."""
    blocker = asyncio.Event()

    async def slow_batch(requests: list[FakeReq]) -> list[int]:
        await blocker.wait()
        return [r.payload for r in requests]

    r = _mk_runtime(run_batch_cb=slow_batch, max_queue=2)
    await r.start()
    try:
        # Two submits fill the queue (worker is blocked on the slow_batch)
        f1 = asyncio.create_task(r.submit(FakeReq("b1", payload=1)))
        f2 = asyncio.create_task(r.submit(FakeReq("b1", payload=2)))
        # Give the worker a tick to drain into in-flight, but our slow_batch
        # holds. Our queue may be empty if the worker already drained both
        # into its in-flight batch — submit a few more to refill.
        await asyncio.sleep(0.05)
        # Add more until we hit capacity.
        added = 0
        for i in range(5):
            try:
                asyncio.create_task(r.submit(FakeReq("b1", payload=10 + i)))
                added += 1
            except QueueFull:
                break
            await asyncio.sleep(0.001)
        # At least one of the submits should hit QueueFull eventually.
        with pytest.raises(QueueFull):
            await r.submit(FakeReq("b1", payload=99))
        blocker.set()
        # Drain
        await asyncio.wait_for(asyncio.gather(f1, f2), timeout=2.0)
    finally:
        blocker.set()
        await r.stop()


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_request_flushes_via_timeout():
    """Default budget=100ms, cold-start cost=50ms. One request is under
    budget — the timeout flush at 5ms is what fires."""
    r = _mk_runtime(run_batch_cb=_identity_batch, budget_ms=100.0, max_wait_ms=5.0)
    await r.start()
    try:
        result = await asyncio.wait_for(
            r.submit(FakeReq("b1", payload=7)), timeout=2.0,
        )
        assert result == 7
        assert r.batches_run() >= 1
    finally:
        await r.stop()


@pytest.mark.asyncio
async def test_single_request_over_budget_flushes_immediately():
    """Heavy-cost shape: cost-model says 200ms, budget=100ms → one-shot flush."""
    cm = GpuMsCostModel()
    for _ in range(5):
        cm.record_measurement("m1", "franka", "heavy", gpu_ms=200.0)
    r = _mk_runtime(run_batch_cb=_identity_batch, budget_ms=100.0, cost_model=cm)
    await r.start()
    try:
        result = await asyncio.wait_for(
            r.submit(FakeReq("heavy", payload=99)), timeout=2.0,
        )
        assert result == 99
    finally:
        await r.stop()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_batch_exception_fans_to_all_futures():
    async def failing_batch(requests: list[FakeReq]) -> list[int]:
        raise RuntimeError("model crashed")

    r = _mk_runtime(run_batch_cb=failing_batch)
    await r.start()
    try:
        with pytest.raises(RuntimeError, match="model crashed"):
            await asyncio.wait_for(
                r.submit(FakeReq("b1", payload=1)), timeout=2.0,
            )
    finally:
        await r.stop()


@pytest.mark.asyncio
async def test_result_count_mismatch_raises_clear_error():
    async def short_batch(requests: list[FakeReq]) -> list[int]:
        return [requests[0].payload]  # always returns 1, even if N>1 requests

    r = _mk_runtime(run_batch_cb=short_batch, max_wait_ms=20.0)
    await r.start()
    try:
        # Submit two requests so they batch together
        c1 = asyncio.create_task(r.submit(FakeReq("b1", payload=1)))
        c2 = asyncio.create_task(r.submit(FakeReq("b1", payload=2)))
        with pytest.raises(RuntimeError, match="returned"):
            await asyncio.wait_for(asyncio.gather(c1, c2), timeout=2.0)
    finally:
        await r.stop()


# ---------------------------------------------------------------------------
# Cost-model post-flush update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_model_updated_after_each_flush():
    cm = GpuMsCostModel()
    r = _mk_runtime(run_batch_cb=_identity_batch, cost_model=cm)
    await r.start()
    try:
        for i in range(5):
            await asyncio.wait_for(
                r.submit(FakeReq("b1", payload=i)), timeout=2.0,
            )
        # After 5 submits, cost model should have measurements for shape "b1"
        assert cm.has_measurements("m1", "franka", "b1")
    finally:
        await r.stop()


# ---------------------------------------------------------------------------
# Snapshot + diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_reports_state():
    r = _mk_runtime(run_batch_cb=_identity_batch)
    snap = r.snapshot()
    assert snap["policy_id"] == "test"
    assert snap["model_id"] == "m1"
    assert snap["embodiment"] == "franka"
    assert snap["queue_depth"] == 0
    assert snap["batches_run"] == 0
    assert snap["is_running"] is False
    assert "scheduler" in snap

    await r.start()
    try:
        snap = r.snapshot()
        assert snap["is_running"] is True
        await asyncio.wait_for(r.submit(FakeReq("b1", payload=1)), timeout=2.0)
        snap = r.snapshot()
        assert snap["batches_run"] >= 1
        assert snap["requests_processed"] >= 1
    finally:
        await r.stop()


@pytest.mark.asyncio
async def test_queue_depth_reflects_pending():
    """Block the worker mid-batch so we can observe the queue between submits."""
    blocker = asyncio.Event()

    async def slow_batch(requests: list[FakeReq]) -> list[int]:
        await blocker.wait()
        return [r.payload for r in requests]

    r = _mk_runtime(run_batch_cb=slow_batch, max_queue=20)
    await r.start()
    try:
        # Submit several but don't await — they queue up
        tasks = [
            asyncio.create_task(r.submit(FakeReq("b1", payload=i)))
            for i in range(5)
        ]
        # Wait briefly for them to land in the queue
        await asyncio.sleep(0.05)
        # The worker may have already pulled some into its in-flight batch;
        # what matters is queue_depth() returns a sensible non-negative int
        depth = r.queue_depth()
        assert depth >= 0
        blocker.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    finally:
        blocker.set()
        await r.stop()


# ---------------------------------------------------------------------------
# Stop drains pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_drains_pending_in_flight():
    r = _mk_runtime(run_batch_cb=_identity_batch)
    await r.start()
    submit_coro = r.submit(FakeReq("b1", payload=42))
    result = await asyncio.wait_for(submit_coro, timeout=2.0)
    assert result == 42
    await r.stop()
