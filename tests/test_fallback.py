"""Tests for src/tether/runtime/fallback.py — fallback chain primitive."""
from __future__ import annotations

import asyncio

import pytest

from tether.runtime.fallback import (
    AllTiersFailed,
    FallbackChain,
    FallbackTier,
)


# ---------------------------------------------------------------------------
# FallbackTier construction
# ---------------------------------------------------------------------------


def _noop():
    async def f(*args, **kwargs):
        return "ok"
    return f


def test_tier_rejects_empty_name():
    with pytest.raises(ValueError):
        FallbackTier(name="", predict=_noop())


def test_tier_rejects_zero_timeout():
    with pytest.raises(ValueError):
        FallbackTier(name="main", predict=_noop(), timeout_ms=0)


def test_tier_rejects_negative_timeout():
    with pytest.raises(ValueError):
        FallbackTier(name="main", predict=_noop(), timeout_ms=-5)


def test_tier_allows_none_timeout():
    t = FallbackTier(name="main", predict=_noop(), timeout_ms=None)
    assert t.timeout_ms is None


# ---------------------------------------------------------------------------
# FallbackChain construction
# ---------------------------------------------------------------------------


def test_chain_rejects_empty_tiers():
    with pytest.raises(ValueError):
        FallbackChain(embodiment="franka", tiers=[])


def test_chain_rejects_empty_embodiment():
    with pytest.raises(ValueError):
        FallbackChain(embodiment="", tiers=[FallbackTier(name="main", predict=_noop())])


# ---------------------------------------------------------------------------
# First-tier success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_tier_success_returns_immediately():
    async def main(x):
        return {"actions": x * 2}

    chain = FallbackChain(
        embodiment="franka",
        tiers=[
            FallbackTier(name="main", predict=main),
            FallbackTier(name="student", predict=_noop()),
        ],
    )
    result = await chain.predict(5)
    assert result.tier_used == "main"
    assert result.value == {"actions": 10}
    assert len(result.attempts) == 1
    assert result.attempts[0][0] == "main"
    assert result.attempts[0][1] == "success"


# ---------------------------------------------------------------------------
# Fall-through: timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_advances_to_next_tier():
    async def slow_main(**kwargs):
        await asyncio.sleep(0.5)
        return "main-result"

    async def fast_student(**kwargs):
        return "student-result"

    chain = FallbackChain(
        embodiment="franka",
        tiers=[
            FallbackTier(name="main", predict=slow_main, timeout_ms=30),
            FallbackTier(name="student", predict=fast_student, timeout_ms=30),
        ],
    )
    result = await chain.predict()
    assert result.tier_used == "student"
    assert result.value == "student-result"
    assert len(result.attempts) == 2
    assert result.attempts[0][:2] == ("main", "timeout")
    assert result.attempts[1][:2] == ("student", "success")


# ---------------------------------------------------------------------------
# Fall-through: exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_advances_to_next_tier():
    async def bad_main(**kwargs):
        raise RuntimeError("main crashed")

    async def good_student(**kwargs):
        return "student-result"

    chain = FallbackChain(
        embodiment="franka",
        tiers=[
            FallbackTier(name="main", predict=bad_main),
            FallbackTier(name="student", predict=good_student),
        ],
    )
    result = await chain.predict()
    assert result.tier_used == "student"
    assert result.attempts[0][:2] == ("main", "exception")
    assert result.attempts[1][:2] == ("student", "success")


# ---------------------------------------------------------------------------
# Multi-tier fall-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_through_all_the_way_to_classical():
    async def bad_main(**kwargs):
        raise RuntimeError("main crashed")

    async def slow_student(**kwargs):
        await asyncio.sleep(0.5)
        return "student"

    async def classical(**kwargs):
        return "classical-action"

    chain = FallbackChain(
        embodiment="franka",
        tiers=[
            FallbackTier(name="main", predict=bad_main),
            FallbackTier(name="student", predict=slow_student, timeout_ms=20),
            FallbackTier(name="classical", predict=classical),
        ],
    )
    result = await chain.predict()
    assert result.tier_used == "classical"
    assert result.value == "classical-action"
    assert len(result.attempts) == 3
    assert result.attempts[0][:2] == ("main", "exception")
    assert result.attempts[1][:2] == ("student", "timeout")
    assert result.attempts[2][:2] == ("classical", "success")


# ---------------------------------------------------------------------------
# All tiers fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_tiers_fail_raises_AllTiersFailed():
    async def bad(**kwargs):
        raise RuntimeError("boom")

    async def slow(**kwargs):
        await asyncio.sleep(0.5)
        return "nope"

    chain = FallbackChain(
        embodiment="franka",
        tiers=[
            FallbackTier(name="main", predict=bad),
            FallbackTier(name="student", predict=slow, timeout_ms=20),
            FallbackTier(name="classical", predict=bad),
        ],
    )
    with pytest.raises(AllTiersFailed) as excinfo:
        await chain.predict()

    attempts = excinfo.value.attempts
    assert len(attempts) == 3
    assert attempts[0][:2] == ("main", "exception")
    assert attempts[1][:2] == ("student", "timeout")
    assert attempts[2][:2] == ("classical", "exception")


# ---------------------------------------------------------------------------
# Args passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kwargs_passed_through_to_all_tiers():
    captured: list[dict] = []

    async def main(**kwargs):
        captured.append(("main", kwargs))
        raise RuntimeError("main fail")

    async def student(**kwargs):
        captured.append(("student", kwargs))
        return {"actions": [1, 2]}

    chain = FallbackChain(
        embodiment="franka",
        tiers=[
            FallbackTier(name="main", predict=main),
            FallbackTier(name="student", predict=student),
        ],
    )
    await chain.predict(image_b64="foo", instruction="bar", state=[0.0])

    assert captured[0][1] == {"image_b64": "foo", "instruction": "bar", "state": [0.0]}
    assert captured[1][1] == {"image_b64": "foo", "instruction": "bar", "state": [0.0]}


# ---------------------------------------------------------------------------
# No timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier_with_no_timeout_runs_without_limit():
    async def main():
        await asyncio.sleep(0.05)  # 50ms — would time out with a tight limit
        return "ok"

    chain = FallbackChain(
        embodiment="franka",
        tiers=[FallbackTier(name="main", predict=main, timeout_ms=None)],
    )
    result = await chain.predict()
    assert result.tier_used == "main"
    assert result.value == "ok"
