"""Webhook event dispatcher for `tether serve --webhook <url>`.

Fire-and-forget async delivery of structured JSON events to a customer
URL. HMAC-SHA256 signature in `X-Webhook-Signature` header. Retry with
exponential backoff (3 attempts). Bounded queue drops + metric when
overloaded (can't backpressure /act).

Phase 1 ships the dispatcher primitive + public emit helpers + CLI
wiring. Auto-emission from existing inc_*_violation helpers is Phase 1.5.

Feature spec: features/01_serve/subfeatures/_ecosystem/webhooks/
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from prometheus_client import Counter

from tether.observability.prometheus import REGISTRY

logger = logging.getLogger(__name__)


WebhookEventType = Literal[
    "boot", "safety_violation", "slo_violation", "model_swap", "crash",
]


ALL_WEBHOOK_EVENTS: frozenset[WebhookEventType] = frozenset({
    "boot", "safety_violation", "slo_violation", "model_swap", "crash",
})


# Delivery metrics (scoped to this module; keep the prometheus.py cardinality
# budget clean by registering them here).
tether_webhook_delivered_total = Counter(
    "tether_webhook_delivered_total",
    "Webhook deliveries (successful or failed-terminal) by event + status",
    labelnames=("event", "status"),  # event: boot|safety_violation|..., status: ok|failed
    registry=REGISTRY,
)
tether_webhook_dropped_total = Counter(
    "tether_webhook_dropped_total",
    "Webhook events dropped (queue full or misconfigured)",
    labelnames=("event", "reason"),  # reason: queue_full|no_dispatcher|event_not_subscribed
    registry=REGISTRY,
)
tether_webhook_retry_total = Counter(
    "tether_webhook_retry_total",
    "Webhook delivery retries",
    labelnames=("event", "attempt"),  # attempt: 1|2|3
    registry=REGISTRY,
)


@dataclass(frozen=True)
class WebhookEvent:
    """Structured event passed through the dispatcher.

    payload is the event-specific dict (shape depends on event_type). The
    dispatcher wraps it in an envelope with timestamp + signature before
    POSTing.
    """
    event_type: WebhookEventType
    payload: dict


def compute_hmac_signature(secret: str, body: bytes) -> str:
    """Return `sha256=<hex>` signature for `body` under `secret`.

    Mirrors the GitHub webhook signature format, which is the de-facto
    standard. Customers validate by recomputing HMAC-SHA256 on the raw
    body using their shared secret and comparing via constant-time compare.
    """
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def parse_event_list(raw: str) -> set[WebhookEventType]:
    """Parse `--webhook-events boot,safety_violation,...` into a validated set.

    Empty/None → all 5 events (the sensible default for a customer who
    opted in to webhooks at all).
    """
    if not raw or not raw.strip():
        return set(ALL_WEBHOOK_EVENTS)
    events = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in ALL_WEBHOOK_EVENTS:
            raise ValueError(
                f"unknown webhook event {tok!r}; valid: {sorted(ALL_WEBHOOK_EVENTS)}"
            )
        events.add(tok)
    if not events:
        raise ValueError(f"--webhook-events produced no valid events: {raw!r}")
    return events


class WebhookDispatcher:
    """Async worker dispatching webhook events to a customer URL.

    Design:
    - `emit(event)` is sync + non-blocking — drops the event into an
      asyncio.Queue with `put_nowait()`. Never blocks /act or any other hot
      path; if the queue is full (producer faster than consumer + slow
      receiver), the event is DROPPED + counter increments.
    - A background coroutine drains the queue and POSTs each event with
      retry (3 attempts: 100ms, 500ms, 2000ms backoff).
    - HMAC-SHA256 signature included in `X-Webhook-Signature` header per
      GitHub convention.
    - `start()` launches the worker; `stop()` drains gracefully.

    Usage inside create_app():

        dispatcher = WebhookDispatcher(
            url="https://customer/hook",
            secret="s3cr3t",
            subscribed_events={"safety_violation", "slo_violation", "boot"},
        )
        await dispatcher.start()
        dispatcher.emit(WebhookEvent(event_type="boot",
                                     payload={"version": "0.1.0"}))
        # ... on shutdown:
        await dispatcher.stop()
    """

    __slots__ = (
        "_url", "_secret", "_subscribed_events",
        "_queue", "_worker_task", "_stopping",
        "_httpx_client", "_max_queue", "_timeout_s",
    )

    def __init__(
        self,
        url: str,
        secret: Optional[str] = None,
        *,
        subscribed_events: Optional[set[WebhookEventType]] = None,
        max_queue: int = 1000,
        timeout_s: float = 5.0,
    ):
        if not url:
            raise ValueError("WebhookDispatcher.url must be non-empty")
        self._url = url
        self._secret = secret or ""
        self._subscribed_events = (
            set(subscribed_events) if subscribed_events is not None
            else set(ALL_WEBHOOK_EVENTS)
        )
        self._max_queue = max_queue
        self._timeout_s = timeout_s
        self._queue: Optional[asyncio.Queue[WebhookEvent]] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._stopping = False
        self._httpx_client: Any = None  # httpx.AsyncClient, lazy-imported

    @property
    def url(self) -> str:
        return self._url

    @property
    def subscribed_events(self) -> set[WebhookEventType]:
        return set(self._subscribed_events)

    def is_subscribed(self, event_type: WebhookEventType) -> bool:
        return event_type in self._subscribed_events

    def emit(self, event: WebhookEvent) -> bool:
        """Enqueue an event for async delivery.

        Returns True on successful enqueue, False if dropped (subscription
        mismatch, queue full, or dispatcher not started). Never raises.

        Non-blocking — safe to call from /act hot path.
        """
        if self._queue is None:
            tether_webhook_dropped_total.labels(
                event=event.event_type, reason="no_dispatcher",
            ).inc()
            return False
        if event.event_type not in self._subscribed_events:
            tether_webhook_dropped_total.labels(
                event=event.event_type, reason="event_not_subscribed",
            ).inc()
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            tether_webhook_dropped_total.labels(
                event=event.event_type, reason="queue_full",
            ).inc()
            return False

    async def start(self) -> None:
        """Create the queue + start the background worker. Idempotent."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "WebhookDispatcher requires httpx. It's in the base deps — "
                "this shouldn't happen."
            ) from exc
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._httpx_client = httpx.AsyncClient(timeout=self._timeout_s)
        self._stopping = False
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="webhook-dispatcher",
        )

    async def stop(self) -> None:
        """Drain the queue + shut down the worker. Idempotent."""
        if self._worker_task is None:
            return
        self._stopping = True
        # Wait for the worker to finish draining (or timeout)
        try:
            await asyncio.wait_for(self._worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("webhook worker did not drain in 5s; cancelling")
            self._worker_task.cancel()
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
        self._queue = None
        self._worker_task = None

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            if self._stopping and self._queue.empty():
                return
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue  # check _stopping again
            try:
                await self._deliver(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "webhook.worker_crash event=%s exc=%s: %s",
                    event.event_type, type(exc).__name__, exc,
                )
            finally:
                self._queue.task_done()

    async def _deliver(self, event: WebhookEvent) -> None:
        assert self._httpx_client is not None
        envelope = {
            "event_type": event.event_type,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "payload": event.payload,
        }
        body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["X-Webhook-Signature"] = compute_hmac_signature(
                self._secret, body
            )

        # Retry: 100ms, 500ms, 2000ms backoff
        delays = [0.1, 0.5, 2.0]
        for attempt, delay in enumerate(delays, start=1):
            try:
                response = await self._httpx_client.post(
                    self._url, content=body, headers=headers,
                )
                if 200 <= response.status_code < 300:
                    tether_webhook_delivered_total.labels(
                        event=event.event_type, status="ok",
                    ).inc()
                    return
                # Non-2xx — retry on 5xx, give up on 4xx
                if 400 <= response.status_code < 500:
                    logger.warning(
                        "webhook.delivery_4xx event=%s status=%d — not retrying",
                        event.event_type, response.status_code,
                    )
                    tether_webhook_delivered_total.labels(
                        event=event.event_type, status="failed",
                    ).inc()
                    return
                # 5xx falls through to retry
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "webhook.delivery_exception event=%s attempt=%d exc=%s: %s",
                    event.event_type, attempt, type(exc).__name__, exc,
                )

            if attempt < len(delays):
                tether_webhook_retry_total.labels(
                    event=event.event_type, attempt=str(attempt),
                ).inc()
                await asyncio.sleep(delay)

        # All retries exhausted
        tether_webhook_delivered_total.labels(
            event=event.event_type, status="failed",
        ).inc()
        logger.error(
            "webhook.delivery_failed event=%s attempts=%d — giving up",
            event.event_type, len(delays),
        )
