"""Tests for src/tether/observability/webhooks.py — webhook event dispatcher."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tether.observability.webhooks import (
    ALL_WEBHOOK_EVENTS,
    WebhookDispatcher,
    WebhookEvent,
    compute_hmac_signature,
    parse_event_list,
)


# ---------------------------------------------------------------------------
# compute_hmac_signature
# ---------------------------------------------------------------------------


def test_hmac_signature_format():
    sig = compute_hmac_signature("secret", b"{}")
    assert sig.startswith("sha256=")
    assert len(sig) == len("sha256=") + 64  # 64 hex chars for SHA-256


def test_hmac_signature_deterministic():
    sig1 = compute_hmac_signature("secret", b"hello")
    sig2 = compute_hmac_signature("secret", b"hello")
    assert sig1 == sig2


def test_hmac_signature_different_secrets_differ():
    sig1 = compute_hmac_signature("key1", b"hello")
    sig2 = compute_hmac_signature("key2", b"hello")
    assert sig1 != sig2


def test_hmac_signature_different_bodies_differ():
    sig1 = compute_hmac_signature("secret", b"hello")
    sig2 = compute_hmac_signature("secret", b"world")
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# parse_event_list
# ---------------------------------------------------------------------------


def test_parse_event_list_empty_means_all():
    assert parse_event_list("") == set(ALL_WEBHOOK_EVENTS)
    assert parse_event_list("   ") == set(ALL_WEBHOOK_EVENTS)


def test_parse_event_list_single_event():
    assert parse_event_list("boot") == {"boot"}


def test_parse_event_list_comma_separated():
    assert parse_event_list("boot,safety_violation") == {"boot", "safety_violation"}


def test_parse_event_list_strips_whitespace():
    assert parse_event_list("boot, safety_violation , crash") == {
        "boot", "safety_violation", "crash",
    }


def test_parse_event_list_rejects_unknown():
    with pytest.raises(ValueError, match="unknown webhook event"):
        parse_event_list("boot,NOT_REAL")


def test_parse_event_list_all_five():
    got = parse_event_list(
        "boot,safety_violation,slo_violation,model_swap,crash"
    )
    assert got == set(ALL_WEBHOOK_EVENTS)


# ---------------------------------------------------------------------------
# WebhookDispatcher — construction
# ---------------------------------------------------------------------------


def test_dispatcher_rejects_empty_url():
    with pytest.raises(ValueError):
        WebhookDispatcher(url="")


def test_dispatcher_defaults_all_events():
    d = WebhookDispatcher(url="https://example.com/hook")
    assert d.subscribed_events == set(ALL_WEBHOOK_EVENTS)


def test_dispatcher_respects_subscribed_events():
    d = WebhookDispatcher(url="https://example.com/hook",
                           subscribed_events={"boot", "crash"})
    assert d.is_subscribed("boot") is True
    assert d.is_subscribed("crash") is True
    assert d.is_subscribed("slo_violation") is False


# ---------------------------------------------------------------------------
# emit() behavior before start
# ---------------------------------------------------------------------------


def test_emit_returns_false_before_start():
    d = WebhookDispatcher(url="https://example.com/hook")
    ok = d.emit(WebhookEvent(event_type="boot", payload={}))
    assert ok is False


# ---------------------------------------------------------------------------
# emit() + start() + stop() happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_delivered_via_worker():
    """Happy path: start → emit → worker POSTs → stop."""
    # Mock httpx.AsyncClient
    posts = []

    async def mock_post(url, content=None, headers=None):
        posts.append((url, content, headers))
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        client_instance = MagicMock()
        client_instance.post = AsyncMock(side_effect=mock_post)
        client_instance.aclose = AsyncMock()
        MockClient.return_value = client_instance

        d = WebhookDispatcher(
            url="https://example.com/hook",
            secret="mysecret",
            subscribed_events={"boot", "safety_violation"},
        )
        await d.start()
        ok = d.emit(WebhookEvent(event_type="boot", payload={"v": "0.1"}))
        assert ok is True
        # Give the worker a tick to drain
        await asyncio.sleep(0.05)
        await d.stop()

    assert len(posts) == 1
    url, body, headers = posts[0]
    assert url == "https://example.com/hook"
    assert headers["Content-Type"] == "application/json"
    assert "X-Webhook-Signature" in headers
    assert headers["X-Webhook-Signature"].startswith("sha256=")
    # Body is JSON with envelope
    import json
    envelope = json.loads(body)
    assert envelope["event_type"] == "boot"
    assert envelope["payload"] == {"v": "0.1"}
    assert "timestamp" in envelope


@pytest.mark.asyncio
async def test_emit_without_secret_omits_signature():
    posts = []

    async def mock_post(url, content=None, headers=None):
        posts.append(headers)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        client_instance = MagicMock()
        client_instance.post = AsyncMock(side_effect=mock_post)
        client_instance.aclose = AsyncMock()
        MockClient.return_value = client_instance

        d = WebhookDispatcher(url="https://example.com/hook")  # no secret
        await d.start()
        d.emit(WebhookEvent(event_type="boot", payload={}))
        await asyncio.sleep(0.05)
        await d.stop()

    assert "X-Webhook-Signature" not in posts[0]


# ---------------------------------------------------------------------------
# emit() drop behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_drops_unsubscribed_event():
    with patch("httpx.AsyncClient") as MockClient:
        client_instance = MagicMock()
        client_instance.post = AsyncMock()
        client_instance.aclose = AsyncMock()
        MockClient.return_value = client_instance

        d = WebhookDispatcher(
            url="https://example.com/hook",
            subscribed_events={"boot"},
        )
        await d.start()
        ok = d.emit(WebhookEvent(event_type="safety_violation", payload={}))
        assert ok is False
        await asyncio.sleep(0.05)
        await d.stop()

    # No post happened
    client_instance.post.assert_not_called()


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_5xx():
    call_count = 0

    async def mock_post(url, content=None, headers=None):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            resp.status_code = 503  # first attempt fails
        else:
            resp.status_code = 200  # retry succeeds
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        client_instance = MagicMock()
        client_instance.post = AsyncMock(side_effect=mock_post)
        client_instance.aclose = AsyncMock()
        MockClient.return_value = client_instance

        d = WebhookDispatcher(url="https://example.com/hook")
        await d.start()
        d.emit(WebhookEvent(event_type="boot", payload={}))
        # Let the retry + backoff complete (100ms retry delay)
        await asyncio.sleep(0.3)
        await d.stop()

    assert call_count >= 2  # retried at least once


@pytest.mark.asyncio
async def test_no_retry_on_4xx():
    call_count = 0

    async def mock_post(url, content=None, headers=None):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.status_code = 401
        return resp

    with patch("httpx.AsyncClient") as MockClient:
        client_instance = MagicMock()
        client_instance.post = AsyncMock(side_effect=mock_post)
        client_instance.aclose = AsyncMock()
        MockClient.return_value = client_instance

        d = WebhookDispatcher(url="https://example.com/hook")
        await d.start()
        d.emit(WebhookEvent(event_type="boot", payload={}))
        await asyncio.sleep(0.2)
        await d.stop()

    # 4xx should not retry — exactly one call
    assert call_count == 1
