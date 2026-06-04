"""Tests for src/tether/runtime/two_policy_dispatcher.py — closes the
locally-testable policy-versioning Phase 1 chunk.

Per ADR 2026-04-25-policy-versioning-architecture: the dispatcher
composes PolicyRouter + Policy + PolicyCrashTracker into a single
predict() entry point. End-to-end /act -> route -> dispatch -> crash
update flow tested without needing the real TetherServer.
"""
from __future__ import annotations

import asyncio

import pytest

from tether.runtime.policy import Policy
from tether.runtime.two_policy_dispatcher import (
    TwoPolicyDecision,
    TwoPolicyDispatcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(slot: str) -> Policy:
    return Policy(
        slot=slot,
        model_id=f"pi0-{slot}",
        model_hash=f"{slot * 8}",
        export_dir=f"/exports/{slot}",
        runtime=None, action_guard=None, rtc_adapter=None,
    )


def _make_dispatcher(
    *,
    split_a: int = 50,
    crash_threshold: int = 5,
    predict_a=None,
    predict_b=None,
) -> TwoPolicyDispatcher:
    async def _ok_a(req):
        return {"slot": "a", "actions": [[0.1]]}

    async def _ok_b(req):
        return {"slot": "b", "actions": [[0.2]]}

    return TwoPolicyDispatcher(
        policy_a=_make_policy("a"), policy_b=_make_policy("b"),
        predict_a=predict_a or _ok_a, predict_b=predict_b or _ok_b,
        split_a_percent=split_a, crash_threshold=crash_threshold,
    )


def _run(coro):
    """Helper to run an async coroutine in a sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_dispatcher_rejects_wrong_slot_for_a():
    p_a = _make_policy("b")  # wrong slot for the "a" arg
    p_b = _make_policy("b")
    with pytest.raises(ValueError, match="policy_a.slot"):
        TwoPolicyDispatcher(
            policy_a=p_a, policy_b=p_b,
            predict_a=lambda r: r, predict_b=lambda r: r,
        )


def test_dispatcher_rejects_wrong_slot_for_b():
    p_a = _make_policy("a")
    p_b = _make_policy("a")  # wrong slot for the "b" arg
    with pytest.raises(ValueError, match="policy_b.slot"):
        TwoPolicyDispatcher(
            policy_a=p_a, policy_b=p_b,
            predict_a=lambda r: r, predict_b=lambda r: r,
        )


def test_dispatcher_rejects_invalid_split():
    p_a = _make_policy("a")
    p_b = _make_policy("b")
    with pytest.raises(ValueError, match="split_a_percent"):
        TwoPolicyDispatcher(
            policy_a=p_a, policy_b=p_b,
            predict_a=lambda r: r, predict_b=lambda r: r,
            split_a_percent=150,
        )


def test_dispatcher_rejects_zero_threshold():
    p_a = _make_policy("a")
    p_b = _make_policy("b")
    with pytest.raises(ValueError, match="crash_threshold"):
        TwoPolicyDispatcher(
            policy_a=p_a, policy_b=p_b,
            predict_a=lambda r: r, predict_b=lambda r: r,
            crash_threshold=0,
        )


# ---------------------------------------------------------------------------
# Happy-path predict
# ---------------------------------------------------------------------------


def test_dispatcher_routes_via_episode_id():
    """Same episode -> same slot; different episodes potentially differ."""
    dispatcher = _make_dispatcher()
    # Pick episodes that hash to known slots (router test verifies hash dist).
    decisions_one_ep = [
        _run(dispatcher.predict(
            request={}, episode_id="ep_xyz", request_id=f"req_{i}",
        ))[1]
        for i in range(5)
    ]
    slots = {d.slot for d in decisions_one_ep}
    assert len(slots) == 1  # all routed to same slot
    # First call is fresh, rest cached
    assert not decisions_one_ep[0].cached
    for d in decisions_one_ep[1:]:
        assert d.cached


def test_dispatcher_returns_predict_result():
    dispatcher = _make_dispatcher()
    result, decision = _run(dispatcher.predict(
        request={"foo": "bar"}, episode_id="ep_a", request_id="req_1",
    ))
    assert "actions" in result
    # Result.slot matches decision.slot
    assert result["slot"] == decision.slot


def test_dispatcher_decision_carries_routing_key():
    dispatcher = _make_dispatcher()
    _, decision = _run(dispatcher.predict(
        request={}, episode_id="ep_xyz", request_id="req_1",
    ))
    assert decision.routing_key == "ep_xyz"
    assert not decision.degraded_routing


def test_dispatcher_degraded_routing_when_no_episode_id():
    dispatcher = _make_dispatcher()
    _, decision = _run(dispatcher.predict(
        request={}, episode_id=None, request_id="req_abc",
    ))
    assert decision.degraded_routing
    assert decision.routing_key == "req_abc"


# ---------------------------------------------------------------------------
# Crash tracker integration
# ---------------------------------------------------------------------------


def test_dispatcher_increments_counter_on_error_response():
    """Predict returns dict with 'error' -> tracker counts the crash."""
    async def _err_a(req):
        return {"error": "boom"}

    async def _ok_b(req):
        return {"actions": [[0.0]]}

    dispatcher = _make_dispatcher(
        split_a=100, predict_a=_err_a, predict_b=_ok_b,
        crash_threshold=5,
    )
    # All requests go to A (split=100). Each returns error -> count up.
    for i in range(3):
        _run(dispatcher.predict(
            request={}, episode_id=f"ep_{i}", request_id=f"req_{i}",
        ))
    assert dispatcher.crash_counts()["a"] == 3
    assert dispatcher.crash_counts()["b"] == 0
    assert dispatcher.crash_verdict().verdict == "healthy"  # 3 < 5


def test_dispatcher_clean_response_resets_counter():
    """Error then clean -> counter resets."""
    call_count = {"n": 0}

    async def _flaky_a(req):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return {"error": "boom"}
        return {"actions": [[0.0]]}

    async def _ok_b(req):
        return {"actions": [[0.0]]}

    dispatcher = _make_dispatcher(
        split_a=100, predict_a=_flaky_a, predict_b=_ok_b,
    )
    for i in range(4):
        _run(dispatcher.predict(
            request={}, episode_id=f"ep_{i}", request_id=f"req_{i}",
        ))
    # 2 errors then 2 cleans -> last reset clears
    assert dispatcher.crash_counts()["a"] == 0


def test_dispatcher_drains_a_when_a_exceeds_threshold():
    """A fails 5x -> tracker says drain-a -> next requests forced to B."""
    async def _err_a(req):
        return {"error": "boom"}

    async def _ok_b(req):
        return {"actions": [[0.0]]}

    dispatcher = _make_dispatcher(
        split_a=100, predict_a=_err_a, predict_b=_ok_b,
        crash_threshold=5,
    )
    for i in range(5):
        _run(dispatcher.predict(
            request={}, episode_id=f"ep_{i}", request_id=f"req_{i}",
        ))
    assert dispatcher.crash_verdict().verdict == "drain-a"

    # Next request: even though split_a=100 (would route to A), the drain
    # forces it to B
    _, decision = _run(dispatcher.predict(
        request={}, episode_id="ep_after_drain", request_id="req_drain",
    ))
    assert decision.slot == "b"


def test_dispatcher_predict_raise_increments_counter():
    """If predict raises, the counter still ticks + the exception propagates."""
    async def _crash_a(req):
        raise RuntimeError("inference crashed")

    async def _ok_b(req):
        return {"actions": [[0.0]]}

    dispatcher = _make_dispatcher(
        split_a=100, predict_a=_crash_a, predict_b=_ok_b,
    )
    with pytest.raises(RuntimeError, match="inference crashed"):
        _run(dispatcher.predict(
            request={}, episode_id="ep_x", request_id="req_x",
        ))
    assert dispatcher.crash_counts()["a"] == 1


def test_dispatcher_reset_crash_counters():
    """Operator-driven reset clears counts."""
    async def _err_a(req):
        return {"error": "boom"}

    async def _ok_b(req):
        return {"actions": [[0.0]]}

    dispatcher = _make_dispatcher(
        split_a=100, predict_a=_err_a, predict_b=_ok_b,
    )
    for i in range(4):
        _run(dispatcher.predict(
            request={}, episode_id=f"ep_{i}", request_id=f"req_{i}",
        ))
    assert dispatcher.crash_counts()["a"] == 4
    dispatcher.reset_crash_counters(slot="a")
    assert dispatcher.crash_counts()["a"] == 0


def test_dispatcher_decision_includes_verdict_snapshot():
    async def _err_a(req):
        return {"error": "boom"}

    async def _ok_b(req):
        return {"actions": [[0.0]]}

    dispatcher = _make_dispatcher(
        split_a=100, predict_a=_err_a, predict_b=_ok_b,
        crash_threshold=3,
    )
    # First request: a fails -> verdict still healthy (1 < 3)
    _, d1 = _run(dispatcher.predict(
        request={}, episode_id="ep_1", request_id="req_1",
    ))
    assert d1.crash_verdict == "healthy"
    assert d1.crash_counts == {"a": 1, "b": 0}

    # Two more: count hits 3 -> drain-a
    for i in range(2, 4):
        _, d = _run(dispatcher.predict(
            request={}, episode_id=f"ep_{i}", request_id=f"req_{i}",
        ))
    assert d.crash_verdict == "drain-a"
    assert d.crash_counts == {"a": 3, "b": 0}


# ---------------------------------------------------------------------------
# Snapshot accessors
# ---------------------------------------------------------------------------


def test_dispatcher_split_a_percent_property():
    dispatcher = _make_dispatcher(split_a=80)
    assert dispatcher.split_a_percent == 80


def test_dispatcher_policies_dict_returns_copy():
    """Mutation of returned dict shouldn't affect internal state."""
    dispatcher = _make_dispatcher()
    snapshot = dispatcher.policies
    snapshot["a"] = None  # shouldn't propagate
    assert dispatcher.policies["a"] is not None
