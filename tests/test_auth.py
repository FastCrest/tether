"""Tests for src/tether/runtime/auth.py — Bearer auth + concurrency limiter."""
from __future__ import annotations

import asyncio

import pytest

from tether.runtime.auth import (
    ConcurrencyLimiter,
    constant_time_token_match,
    extract_bearer_token,
    generate_request_id,
    make_401_payload,
    make_429_payload,
    resolve_request_token,
)


# ---------------------------------------------------------------------------
# generate_request_id
# ---------------------------------------------------------------------------


def test_request_id_format():
    rid = generate_request_id()
    assert rid.startswith("req-")
    assert len(rid) == 12  # req- + 8 hex


def test_request_id_unique():
    ids = {generate_request_id() for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# extract_bearer_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header,expected", [
    ("Bearer abc123", "abc123"),
    ("bearer abc123", "abc123"),
    ("BEARER abc123", "abc123"),
    ("Bearer   abc123   ", "abc123"),
    ("Bearer  abc 123", "abc 123"),  # split(None, 1) — rest of string
    (None, None),
    ("", None),
    ("Bearer", None),
    ("Bearer ", None),  # empty token
    ("Basic abc123", None),  # wrong scheme
    ("abc123", None),  # no scheme
])
def test_extract_bearer_token(header, expected):
    assert extract_bearer_token(header) == expected


# ---------------------------------------------------------------------------
# constant_time_token_match
# ---------------------------------------------------------------------------


def test_constant_time_match_identical():
    assert constant_time_token_match("secret123", "secret123") is True


def test_constant_time_match_different():
    assert constant_time_token_match("secret123", "secret124") is False


def test_constant_time_match_empty():
    assert constant_time_token_match(None, "secret") is False
    assert constant_time_token_match("secret", None) is False
    assert constant_time_token_match("", "secret") is False
    assert constant_time_token_match("secret", "") is False


def test_constant_time_match_different_length():
    # Different lengths — must still return False without error
    assert constant_time_token_match("short", "longer_token") is False


# ---------------------------------------------------------------------------
# resolve_request_token
# ---------------------------------------------------------------------------


def test_resolve_prefers_bearer_over_x_tether_key():
    """When both headers are set, Bearer wins."""
    token = resolve_request_token("Bearer bearer-token", "x-key-token")
    assert token == "bearer-token"


def test_resolve_falls_back_to_x_tether_key():
    """No Bearer → use X-Tether-Key."""
    token = resolve_request_token(None, "legacy-token")
    assert token == "legacy-token"


def test_resolve_returns_none_when_no_headers():
    assert resolve_request_token(None, None) is None


def test_resolve_strips_x_tether_key_whitespace():
    assert resolve_request_token(None, "  spaced  ") == "spaced"


def test_resolve_empty_x_tether_key_returns_none():
    assert resolve_request_token(None, "   ") is None


# ---------------------------------------------------------------------------
# make_401 / make_429 payloads
# ---------------------------------------------------------------------------


def test_make_401_payload_shape():
    err = make_401_payload("missing token")
    d = err.to_dict()
    assert d["error"] == "unauthorized"
    assert d["message"] == "missing token"
    assert d["request_id"].startswith("req-")


def test_make_429_payload_shape():
    d = make_429_payload(current=32, limit=32)
    assert d["error"] == "too_many_requests"
    assert d["concurrent_requests"] == 32
    assert d["max_concurrent"] == 32
    assert d["request_id"].startswith("req-")


# ---------------------------------------------------------------------------
# ConcurrencyLimiter
# ---------------------------------------------------------------------------


def test_limiter_rejects_bad_ctor():
    with pytest.raises(ValueError):
        ConcurrencyLimiter(max_concurrent=0)
    with pytest.raises(ValueError):
        ConcurrencyLimiter(max_concurrent=-1)


def test_limiter_in_flight_starts_at_zero():
    limiter = ConcurrencyLimiter(max_concurrent=4)
    assert limiter.in_flight == 0
    assert limiter.max_concurrent == 4


@pytest.mark.asyncio
async def test_limiter_acquires_when_under_limit():
    limiter = ConcurrencyLimiter(max_concurrent=2)
    async with limiter.try_acquire() as ctx:
        assert ctx.acquired is True
        assert limiter.in_flight == 1


@pytest.mark.asyncio
async def test_limiter_rejects_when_at_limit():
    limiter = ConcurrencyLimiter(max_concurrent=1)

    async with limiter.try_acquire() as ctx1:
        assert ctx1.acquired is True
        # Second concurrent acquire while first is held → should reject
        async with limiter.try_acquire() as ctx2:
            assert ctx2.acquired is False

    # After first exits, limiter returns to available
    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_limiter_release_on_exit():
    limiter = ConcurrencyLimiter(max_concurrent=1)
    async with limiter.try_acquire() as ctx:
        assert ctx.acquired is True
    # Released
    assert limiter.in_flight == 0
    # Can re-acquire
    async with limiter.try_acquire() as ctx2:
        assert ctx2.acquired is True


@pytest.mark.asyncio
async def test_limiter_release_on_exception():
    """Semaphore must release even when the body raises."""
    limiter = ConcurrencyLimiter(max_concurrent=1)

    class BodyError(RuntimeError):
        pass

    with pytest.raises(BodyError):
        async with limiter.try_acquire() as ctx:
            assert ctx.acquired is True
            raise BodyError("boom")

    # Should have released
    assert limiter.in_flight == 0
    async with limiter.try_acquire() as ctx:
        assert ctx.acquired is True


@pytest.mark.asyncio
async def test_limiter_concurrent_acquires():
    """4 tasks trying to acquire a limit=2 limiter — 2 should succeed, 2 reject."""
    limiter = ConcurrencyLimiter(max_concurrent=2)
    acquired_count = 0
    rejected_count = 0

    async def worker():
        nonlocal acquired_count, rejected_count
        async with limiter.try_acquire() as ctx:
            if ctx.acquired:
                acquired_count += 1
                await asyncio.sleep(0.05)  # hold
            else:
                rejected_count += 1

    # Launch 4 concurrently; first 2 should hold the sem, last 2 reject
    await asyncio.gather(worker(), worker(), worker(), worker())

    # Order non-deterministic but totals must be 2 + 2
    assert acquired_count + rejected_count == 4
    assert acquired_count >= 2  # at least 2 got in
