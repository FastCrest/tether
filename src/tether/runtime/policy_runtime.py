"""Per-policy queue + worker + scheduler + cost model.

Owned by the server (single-policy production: `{"prod": PolicyRuntime(...)}`;
multi-policy via policy-versioning: `{"a": PR, "b": PR}`). Replaces
`server._batch_queue` + `_batch_worker_task` + `_batches_run` from the
legacy single-server batching path.

Per ADR 2026-04-24-chunk-budget-batching-architecture decision #3:
this refactor lands in chunk-budget-batching Phase 1 and policy-versioning
inherits it (per its own ADR's per-policy state binding requirement).

Design:
- Backing store is a `collections.deque` (NOT `asyncio.Queue`) because
  the scheduler needs to PEEK at the oldest enqueue time + the full
  pending list to make its flush decision. asyncio.Queue exposes neither.
- Single-event-loop semantics (FastAPI is one loop) eliminates the need
  for locks around deque access.
- The worker is policy-agnostic — it gets a `run_batch_callback(requests)
  -> list[results]` from the policy. The runtime never imports
  Pi05DecomposedInference / TetherServer / etc.
- `submit()` returns an asyncio.Future the /act handler awaits. On
  queue-full, raises `QueueFull` so the handler can return 503 + Retry-
  After (TGI overload pattern, same as the auth-bearer concurrency
  limiter).

Test surface: see `tests/test_policy_runtime.py`.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, Awaitable, Callable

from tether.runtime.batching import CostBudgetScheduler, GpuMsCostModel

# Optional metric emission — gated on the [serve] extra. When prometheus_client
# isn't installed, the runtime still works; metrics just no-op.
try:
    from tether.observability import observe_batch_flush
    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _METRICS_AVAILABLE = False
    def observe_batch_flush(**kwargs): pass  # type: ignore

logger = logging.getLogger(__name__)


# Bounded queue ceiling — protects against unbounded backlog → memory blow-up
# under load. 1000 is generous (at 10 QPS that's 100 seconds of backlog;
# customers should set --max-concurrent lower than this anyway via the
# auth-bearer limiter for hot-path protection).
_DEFAULT_MAX_QUEUE = 1000

# Worker idle-poll period when the queue is empty. Longer than this delays
# shutdown; shorter wastes CPU on empty queues. 1s is the common asyncio
# pattern (cf. webhooks worker).
_IDLE_POLL_S = 1.0


class QueueFull(Exception):
    """Raised by `PolicyRuntime.submit()` when the bounded queue is at capacity.
    Callers should map to HTTP 503 + Retry-After per the standard backpressure
    contract (matches WebhookDispatcher + ConcurrencyLimiter conventions)."""


# Type alias for the policy-side run-batch entry point. Takes a list of
# requests + returns a list of results (or raises). Async because the
# decomposed dispatch is async.
RunBatchCallback = Callable[[list[Any]], Awaitable[list[Any]]]

# Type alias for the shape-key extractor. Maps each request to the shape
# string the cost model + scheduler use for cost lookup.
ShapeKeyFn = Callable[[Any], str]


class PolicyRuntime:
    """One queue + one worker + one scheduler per policy.

    Lifecycle:
      runtime = PolicyRuntime(...)
      await runtime.start()
      result = await runtime.submit(request)  # /act handler awaits this
      ...
      await runtime.stop()

    The runtime is started + stopped from the FastAPI lifespan handler; it
    owns no model, no auth, no metrics emission of its own — just the
    queue + flush-decision + result-fanout primitive.

    Args:
        policy_id: human-readable identifier (e.g. "prod" / "a" / "b").
            Surfaced in logs + (later) Prometheus labels.
        model_id: the loaded model identifier (used for cost-model lookups).
        embodiment: the loaded embodiment identifier (used for cost-model lookups).
        scheduler: a CostBudgetScheduler instance (the flush-decision primitive).
        cost_model: the GpuMsCostModel instance (for post-flush measurement
            updates). The scheduler also references this; pass the SAME
            instance to both.
        run_batch_callback: async function `(list[request]) -> list[result]`.
            Called once per flush. May raise — the runtime fans the exception
            out to all in-batch futures.
        shape_key_fn: function `(request) -> str` for cost-model lookups.
        max_queue: queue capacity. submit() raises QueueFull when at limit.
    """

    __slots__ = (
        "_policy_id", "_model_id", "_embodiment",
        "_scheduler", "_cost_model",
        "_run_batch_callback", "_shape_key_fn",
        "_pending", "_wake", "_worker_task", "_stopping",
        "_max_queue", "_batches_run", "_requests_processed",
    )

    def __init__(
        self,
        *,
        policy_id: str,
        model_id: str,
        embodiment: str,
        scheduler: CostBudgetScheduler,
        cost_model: GpuMsCostModel,
        run_batch_callback: RunBatchCallback,
        shape_key_fn: ShapeKeyFn,
        max_queue: int = _DEFAULT_MAX_QUEUE,
    ):
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}")
        self._policy_id = policy_id
        self._model_id = model_id
        self._embodiment = embodiment
        self._scheduler = scheduler
        self._cost_model = cost_model
        self._run_batch_callback = run_batch_callback
        self._shape_key_fn = shape_key_fn
        self._max_queue = int(max_queue)
        # Each entry: (request, future, enqueue_ts_perf_counter)
        self._pending: collections.deque = collections.deque()
        self._wake: asyncio.Event | None = None
        self._worker_task: asyncio.Task | None = None
        self._stopping = False
        self._batches_run = 0
        self._requests_processed = 0

    # --- accessors ---------------------------------------------------------

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embodiment(self) -> str:
        return self._embodiment

    @property
    def scheduler(self) -> CostBudgetScheduler:
        return self._scheduler

    @property
    def cost_model(self) -> GpuMsCostModel:
        return self._cost_model

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    def queue_depth(self) -> int:
        return len(self._pending)

    def batches_run(self) -> int:
        return self._batches_run

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe diagnostic snapshot. Used by /diagnostics + Prometheus."""
        return {
            "policy_id": self._policy_id,
            "model_id": self._model_id,
            "embodiment": self._embodiment,
            "queue_depth": self.queue_depth(),
            "max_queue": self._max_queue,
            "batches_run": self._batches_run,
            "requests_processed": self._requests_processed,
            "is_running": self.is_running,
            "scheduler": {
                "max_cost_ms": self._scheduler.max_cost_ms,
                "max_wait_ms": self._scheduler.max_wait_ms,
                "mode": self._scheduler.mode,
            },
        }

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Spawn the worker. Idempotent — second call is a no-op when already
        running."""
        if self.is_running:
            return
        self._stopping = False
        self._wake = asyncio.Event()
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name=f"policy-runtime-{self._policy_id}",
        )

    async def stop(self) -> None:
        """Drain pending (up to `max_wait_ms` × 5) then cancel the worker.
        Idempotent."""
        if self._worker_task is None:
            return
        self._stopping = True
        if self._wake is not None:
            self._wake.set()
        # Wait up to a generous window for the worker to drain.
        timeout_s = max(5.0, self._scheduler.max_wait_ms * 5 / 1000)
        try:
            await asyncio.wait_for(self._worker_task, timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "policy_runtime.stop_timeout policy_id=%s — cancelling worker",
                self._policy_id,
            )
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # Fail any in-flight futures.
        while self._pending:
            _, fut, _ = self._pending.popleft()
            if not fut.done():
                fut.set_exception(RuntimeError("policy_runtime stopped"))
        self._worker_task = None
        self._wake = None

    # --- request submission ------------------------------------------------

    async def submit(self, request: Any) -> Any:
        """Enqueue a request and await its result.

        Raises QueueFull when the queue is at capacity — callers should map
        to HTTP 503 + Retry-After (TGI overload pattern).
        Raises RuntimeError when the runtime is not started.
        """
        if not self.is_running:
            raise RuntimeError(
                f"policy_runtime not running (policy_id={self._policy_id})"
            )
        if len(self._pending) >= self._max_queue:
            raise QueueFull(
                f"policy_runtime queue full (max_queue={self._max_queue}, "
                f"policy_id={self._policy_id})"
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        enqueue_ts = time.perf_counter()
        self._pending.append((request, future, enqueue_ts))
        if self._wake is not None:
            self._wake.set()
        return await future

    # --- worker loop -------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Drain queue, ask scheduler "flush now?", on yes → run batch + fan
        out results. On no → sleep until next likely flush event."""
        assert self._wake is not None

        while True:
            if self._stopping and not self._pending:
                return

            # Idle: nothing to do — wait for a wake or timeout.
            if not self._pending:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=_IDLE_POLL_S)
                except asyncio.TimeoutError:
                    continue
                self._wake.clear()
                continue

            # Decide whether to flush.
            now = time.perf_counter()
            oldest_wait_ms = (now - self._pending[0][2]) * 1000.0
            requests = [item[0] for item in self._pending]
            decision = self._scheduler.should_flush(
                requests,
                model_id=self._model_id,
                embodiment=self._embodiment,
                oldest_wait_ms=oldest_wait_ms,
                shape_key_fn=self._shape_key_fn,
            )

            if not decision.flush:
                # Sleep until next likely flush event: max_wait remainder or
                # next wake (new request enqueued).
                remaining_wait_ms = max(0.0, self._scheduler.max_wait_ms - oldest_wait_ms)
                # If we're about to time out, poll fast; else wait for either
                # a wake or remaining_wait_ms.
                wait_s = max(0.001, remaining_wait_ms / 1000.0)
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=wait_s)
                except asyncio.TimeoutError:
                    pass
                if self._wake is not None:
                    self._wake.clear()
                continue

            # Flush — drain pending, run batch, fan out.
            batch_items = list(self._pending)
            self._pending.clear()
            batch_requests = [item[0] for item in batch_items]
            batch_futures = [item[1] for item in batch_items]
            t_run0 = time.perf_counter()
            try:
                results = await self._run_batch_callback(batch_requests)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "policy_runtime.run_batch_failed policy_id=%s size=%d exc=%s: %s",
                    self._policy_id, len(batch_requests),
                    type(exc).__name__, exc,
                )
                for fut in batch_futures:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            elapsed_ms = (time.perf_counter() - t_run0) * 1000.0

            # Validate result count + fan out. A length mismatch is a policy-
            # implementation bug, not a runtime bug; fail the futures with a
            # clear error rather than silently dropping requests.
            if len(results) != len(batch_requests):
                exc = RuntimeError(
                    f"policy {self._policy_id} run_batch returned "
                    f"{len(results)} results for {len(batch_requests)} requests"
                )
                for fut in batch_futures:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            for fut, result in zip(batch_futures, results):
                if not fut.done():
                    fut.set_result(result)

            # Update cost model with measured per-request cost (batch elapsed
            # divided by batch size — naive but cheap; refined post-Phase 1
            # if shape mix matters).
            per_request_ms = elapsed_ms / len(batch_requests)
            for req in batch_requests:
                self._cost_model.record_measurement(
                    model_id=self._model_id,
                    embodiment=self._embodiment,
                    shape_key=self._shape_key_fn(req),
                    gpu_ms=per_request_ms,
                )

            self._batches_run += 1
            self._requests_processed += len(batch_requests)

            # Emit per-flush diagnostics (chunk-budget-batching ADR decision #4).
            try:
                observe_batch_flush(
                    embodiment=self._embodiment,
                    policy_slot=self._policy_id,
                    reason=decision.reason,
                    batch_cost_ms=decision.batch_cost_ms,
                    batch_size=decision.size,
                    shape_homogeneous=decision.shape_homogeneous,
                    queue_depth_after=len(self._pending),
                )
            except Exception as exc:  # noqa: BLE001 — metrics never break the hot path
                logger.warning("policy_runtime.metric_emit_failed: %s", exc)

            logger.debug(
                "policy_runtime.flushed policy_id=%s reason=%s size=%d cost_est_ms=%.1f "
                "elapsed_ms=%.1f shape_homogeneous=%s",
                self._policy_id, decision.reason, decision.size,
                decision.batch_cost_ms, elapsed_ms, decision.shape_homogeneous,
            )


__all__ = [
    "PolicyRuntime",
    "QueueFull",
    "RunBatchCallback",
    "ShapeKeyFn",
]
