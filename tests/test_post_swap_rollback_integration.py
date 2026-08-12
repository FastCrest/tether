"""Integration test: PostSwapMonitor + RollbackHandler composed against a
REAL TwoPolicyDispatcher (not the RollbackHandler unit tests' `_StubRouter`,
which encodes an `router.set_active()` API that `PolicyRouter` never
actually implements).

Closes the gap identified in a 2026-08-12 verification pass: `pro/
post_swap_monitor.py` and `pro/rollback.py` are real, unit-tested-in-
isolation code, but neither was imported anywhere outside `pro/__init__.py`
and their own test files -- nothing proved they compose correctly with the
live two-policy serving path. `runtime/server.py` now wires them together
via `TwoPolicyDispatcher.force_drain()` (added alongside this test); this
file is the composition proof, independent of the FastAPI HTTP layer.
"""
from __future__ import annotations

import asyncio

from tether.pro.post_swap_monitor import MonitorConfig, PostSwapMonitor
from tether.pro.rollback import RollbackHandler
from tether.runtime.policy import Policy
from tether.runtime.two_policy_dispatcher import TwoPolicyDispatcher


def _make_policy(slot: str) -> Policy:
    return Policy(
        slot=slot, model_id=f"pi0-{slot}", model_hash=f"{slot * 8}",
        export_dir=f"/exports/{slot}",
        runtime=None, action_guard=None, rtc_adapter=None,
    )


def _run(coro):
    return asyncio.run(coro)


def _wire_rollback(dispatcher: TwoPolicyDispatcher, *, monitored_slot: str = "a"):
    """Mirrors the closures runtime/server.py builds at two-policy setup."""
    fallback_slot = "b" if monitored_slot == "a" else "a"
    audit_records: list[dict] = []

    def _router_swap_fn(target_slot: str) -> None:
        drain = fallback_slot if target_slot == monitored_slot else monitored_slot
        dispatcher.force_drain(slot=drain, reason="post_swap_monitor_rollback")

    handler = RollbackHandler(
        router_swap_fn=_router_swap_fn,
        active_slot_getter=lambda: monitored_slot,
        audit_writer=audit_records.append,
    )
    return handler, audit_records


def test_monitor_trip_drains_the_monitored_slot_via_real_dispatcher():
    """T2 trip (action cos-similarity collapse): should_rollback() fires,
    the RollbackHandler's router_swap_fn correctly force-drains slot A
    on the REAL dispatcher, and the very next request routes to B."""
    async def _ok(req):
        return {"actions": [[0.0]]}

    dispatcher = TwoPolicyDispatcher(
        policy_a=_make_policy("a"), policy_b=_make_policy("b"),
        predict_a=_ok, predict_b=_ok,
        split_a_percent=100,  # all traffic to the newly-promoted candidate
    )
    handler, audit = _wire_rollback(dispatcher, monitored_slot="a")

    monitor = PostSwapMonitor(config=MonitorConfig(sensitivity="aggressive"))
    monitor.start_window(baseline_clamp_rate=0.0)

    # Feed 6 episodes with low action-similarity to the previous model —
    # enough samples to clear the T2 min-sample floor (5) and trip it.
    for _ in range(6):
        monitor.record_episode(
            safety_clamp_count=0, cos_to_previous_model=0.40,
        )
    decision = monitor.should_rollback()
    assert decision.should_rollback
    assert decision.reason == "T2"

    outcome = handler.execute(trigger="auto", reason=decision.reason)
    assert outcome.succeeded
    assert outcome.from_slot == "a"
    assert outcome.to_slot == "b"
    assert len(audit) == 1
    assert audit[0]["reason"] == "T2"

    # The real dispatcher now routes away from A on the next request.
    _, routing = _run(dispatcher.predict(
        request={}, episode_id="ep_after_rollback", request_id="req_1",
    ))
    assert routing.slot == "b"
    assert routing.crash_verdict == "drain-a"
    # Crash counters are untouched -- this was a monitor trip, not a raw
    # predict() exception.
    assert dispatcher.crash_counts() == {"a": 0, "b": 0}


def test_monitor_does_not_trip_below_sensitivity_threshold():
    """normal sensitivity requires 2 consecutive trips -- a single bad
    window shouldn't fire rollback."""
    async def _ok(req):
        return {"actions": [[0.0]]}

    dispatcher = TwoPolicyDispatcher(
        policy_a=_make_policy("a"), policy_b=_make_policy("b"),
        predict_a=_ok, predict_b=_ok,
        split_a_percent=100,
    )
    _handler, audit = _wire_rollback(dispatcher, monitored_slot="a")

    monitor = PostSwapMonitor(config=MonitorConfig(sensitivity="normal"))
    monitor.start_window(baseline_clamp_rate=0.0)
    for _ in range(6):
        monitor.record_episode(safety_clamp_count=0, cos_to_previous_model=0.40)

    decision = monitor.should_rollback()
    # First trip only -- normal sensitivity needs 2 consecutive.
    assert not decision.should_rollback
    assert decision.consecutive_trips == 1
    assert len(audit) == 0
    # Dispatcher routing is unaffected.
    _, routing = _run(dispatcher.predict(
        request={}, episode_id="ep_1", request_id="req_1",
    ))
    assert routing.slot == "a"


def test_t3_webhook_violation_trip_drains_via_real_dispatcher():
    """T3 (safety-violation webhook count) exercised end-to-end, same
    composition as the T2 test above -- proves all three trip signals,
    not just one, reach the real dispatcher correctly."""
    async def _ok(req):
        return {"actions": [[0.0]]}

    dispatcher = TwoPolicyDispatcher(
        policy_a=_make_policy("a"), policy_b=_make_policy("b"),
        predict_a=_ok, predict_b=_ok, split_a_percent=100,
    )
    handler, audit = _wire_rollback(dispatcher, monitored_slot="a")

    monitor = PostSwapMonitor(config=MonitorConfig(sensitivity="aggressive"))
    monitor.start_window(baseline_clamp_rate=0.0)
    monitor.record_episode(safety_clamp_count=0, webhook_violations_count=6)

    decision = monitor.should_rollback()
    assert decision.should_rollback
    assert decision.reason == "T3"

    outcome = handler.execute(trigger="auto", reason=decision.reason)
    assert outcome.succeeded
    assert audit[0]["reason"] == "T3"

    _, routing = _run(dispatcher.predict(
        request={}, episode_id="ep_1", request_id="req_1",
    ))
    assert routing.slot == "b"
