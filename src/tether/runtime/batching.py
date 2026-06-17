"""Cost-weighted batch scheduler primitives for Tether serve.

Per ADR 2026-04-24-chunk-budget-batching-architecture:
  - Cost budget in GPU-ms, not fixed request count
  - Profiled cost model (prewarm measurement + rolling-bucket updates)
  - Cold-start shapes fall back to a conservative constant
  - Shape-homogeneity preferred but not required — the scheduler flags
    mixed-shape batches so cuda-graphs code can decide whether to
    eager-fallback at dispatch time

Research sidecar:
  features/01_serve/subfeatures/_perf_compound/chunk-budget-batching/
  chunk-budget-batching_research.md

Composition notes:
  - `PolicyRuntime` (runtime/policy_runtime.py) owns one scheduler + one
    cost model per policy.
  - The decomposed and monolithic dispatch paths both populate the cost
    model via `record_measurement()` after each /act completes.
  - The scheduler is pure: `should_flush()` takes pending requests + a cost
    model and returns a boolean. No asyncio state inside.
"""
from __future__ import annotations

import collections
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Cold-start fallback — the cost we assume for a shape we've never measured.
# Calibrated against the 2026-04-14 trt-fp16-vs-torch-compile bench: A10G
# decomposed pi0.5 is ~45 ms cache-hit / ~400 ms cache-miss. 50 ms is a
# conservative upper bound on the cache-hit path (where most traffic lives)
# and a lower bound on miss — operators get predictable behavior while the
# profiling pass populates the real numbers.
_DEFAULT_COLD_START_MS = 50.0

# Rolling window for cost-model measurements. 100 samples gives ~10 seconds
# of steady-state history at 10 QPS; long enough to absorb single-outlier
# spikes, short enough to track real drift (thermal throttling, concurrent
# workload pressure).
_ROLLING_WINDOW_SIZE = 100

# Minimum budget a scheduler will accept. Below this, any single request
# would immediately flush the batch — no batching happens. 10 ms is below
# any realistic GPU inference, so the floor exists mostly to catch
# misconfiguration (operator typed `0.01` intending seconds).
_MIN_BUDGET_MS = 10.0

# Maximum budget. Above this, a queue can hold an unbounded backlog without
# flushing, starving clients. 500 ms = half the worst-case A10G decomposed
# cache-miss latency; anything slower is misconfiguration.
_MAX_BUDGET_MS = 500.0


class CostMode:
    """Bounded enum for the cost-model mode. String values are surfaced as
    Prometheus labels + CLI output, so they must stay stable across minor
    releases."""

    PROFILED = "profiled"
    FALLBACK = "fallback"  # cache-hit-weighted static constant

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.PROFILED, cls.FALLBACK)


@dataclass(frozen=True)
class CostKey:
    """Immutable key into the cost model.

    Single-embodiment-per-process today (Phase 1); Phase 2 per-embodiment
    routing composes with this — the key already carries `embodiment`.
    `shape_key` is a string representation (e.g. "b1_seq968" for
    batch=1, seq=968) rather than a tuple so it stays JSON-serializable
    for metric snapshots.
    """

    model_id: str
    embodiment: str
    shape_key: str


class GpuMsCostModel:
    """Per-policy rolling-window cost model in GPU milliseconds.

    Thread-safe for the typical pattern of worker-loop writes +
    `/act`-hot-path reads. Writes acquire a lock (rare — one per /act
    completion); reads are lock-free dict lookups (common — one per
    scheduler decision).

    The rolling window uses `collections.deque` + `statistics.median` —
    O(N) per estimate but N is bounded at 100 so in practice this is
    single-digit microseconds. No heap/order-statistic data structure
    needed at this scale.

    Mode selection (profiled vs fallback) is a policy decision, not a
    model property — the cost model returns the best estimate it has and
    the caller decides whether to trust it or fall back to the static
    constant.
    """

    __slots__ = ("_measurements", "_lock", "_default_cold_start_ms")

    def __init__(self, default_cold_start_ms: float = _DEFAULT_COLD_START_MS):
        if default_cold_start_ms <= 0:
            raise ValueError(
                f"default_cold_start_ms must be positive, got {default_cold_start_ms}"
            )
        self._measurements: dict[CostKey, collections.deque[float]] = {}
        self._lock = threading.Lock()
        self._default_cold_start_ms = float(default_cold_start_ms)

    def record_measurement(
        self,
        model_id: str,
        embodiment: str,
        shape_key: str,
        gpu_ms: float,
    ) -> None:
        """Append one measurement to the rolling window for this (model,
        embodiment, shape). Thread-safe.

        Negative / zero / NaN measurements are silently dropped — the caller
        is expected to pass wall-clock values from `time.perf_counter()` which
        are always positive, but /act hot-path reliability trumps strict
        validation here.
        """
        if not (gpu_ms > 0) or gpu_ms != gpu_ms:  # rejects NaN + non-positive
            return
        key = CostKey(model_id=model_id, embodiment=embodiment, shape_key=shape_key)
        with self._lock:
            window = self._measurements.get(key)
            if window is None:
                window = collections.deque(maxlen=_ROLLING_WINDOW_SIZE)
                self._measurements[key] = window
            window.append(float(gpu_ms))

    def estimate(
        self,
        model_id: str,
        embodiment: str,
        shape_key: str,
    ) -> float:
        """Return the current best cost estimate in ms.

        Returns the median of the rolling window when at least 3 measurements
        exist (3 is the minimum for a stable median); otherwise the cold-start
        default. Median-of-window absorbs outliers (GC pauses, noisy
        neighbor) better than mean.
        """
        key = CostKey(model_id=model_id, embodiment=embodiment, shape_key=shape_key)
        # Lock-free read — dict lookup is atomic under GIL, and we copy
        # the window to a list before computing median so a concurrent
        # append doesn't raise.
        window = self._measurements.get(key)
        if window is None:
            return self._default_cold_start_ms
        snapshot = list(window)
        if len(snapshot) < 3:
            return self._default_cold_start_ms
        return float(statistics.median(snapshot))

    def has_measurements(
        self,
        model_id: str,
        embodiment: str,
        shape_key: str,
    ) -> bool:
        """True once the cost model has at least one measurement for this
        shape. Used by metric labels to distinguish profiled vs cold-start
        reads."""
        key = CostKey(model_id=model_id, embodiment=embodiment, shape_key=shape_key)
        window = self._measurements.get(key)
        return window is not None and len(window) > 0

    def export_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the entire cost model. Used by
        `/diagnostics/cost-model` debug endpoint + Prometheus metric emission
        on a flush cadence."""
        out: dict[str, Any] = {
            "default_cold_start_ms": self._default_cold_start_ms,
            "rolling_window_size": _ROLLING_WINDOW_SIZE,
            "entries": [],
        }
        with self._lock:
            for key, window in self._measurements.items():
                samples = list(window)
                if not samples:
                    continue
                entry = {
                    "model_id": key.model_id,
                    "embodiment": key.embodiment,
                    "shape_key": key.shape_key,
                    "n": len(samples),
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                }
                out["entries"].append(entry)
        return out


@dataclass
class SchedulerDecision:
    """The output of `should_flush()`. Frozen so callers can log + inspect
    without worrying about mutation."""

    flush: bool
    reason: str  # bounded enum: "budget_reached" | "timeout" | "empty" | "single_request_over_budget"
    batch_cost_ms: float
    budget_ms: float
    size: int
    shape_homogeneous: bool  # True when all pending requests share shape_key


class CostBudgetScheduler:
    """Cost-weighted batch flush scheduler.

    Decides when to flush a pending batch based on summed GPU-ms cost (NOT
    fixed request count). Flushes when:
      - summed cost >= budget, OR
      - time-since-oldest >= max_wait_ms (timeout), OR
      - a single request's estimated cost already exceeds the budget
        (one-shot flush — no point waiting)

    The scheduler is PURE: it holds no asyncio state, no queue. The caller
    (PolicyRuntime) owns the queue and asks the scheduler "should I flush
    now?" on each tick.
    """

    __slots__ = ("_max_cost_ms", "_max_wait_ms", "_cost_model", "_mode")

    def __init__(
        self,
        max_cost_per_batch_ms: float,
        cost_model: GpuMsCostModel,
        *,
        max_wait_ms: float = 5.0,
        mode: str = CostMode.PROFILED,
    ):
        if not (_MIN_BUDGET_MS <= max_cost_per_batch_ms <= _MAX_BUDGET_MS):
            raise ValueError(
                f"max_cost_per_batch_ms must be in [{_MIN_BUDGET_MS}, "
                f"{_MAX_BUDGET_MS}], got {max_cost_per_batch_ms}"
            )
        if max_wait_ms <= 0:
            raise ValueError(f"max_wait_ms must be positive, got {max_wait_ms}")
        if mode not in CostMode.all():
            raise ValueError(
                f"mode must be one of {CostMode.all()}, got {mode!r}"
            )
        self._max_cost_ms = float(max_cost_per_batch_ms)
        self._max_wait_ms = float(max_wait_ms)
        self._cost_model = cost_model
        self._mode = mode

    @property
    def max_cost_ms(self) -> float:
        return self._max_cost_ms

    @property
    def max_wait_ms(self) -> float:
        return self._max_wait_ms

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cost_model(self) -> GpuMsCostModel:
        return self._cost_model

    def batch_cost_ms(
        self,
        pending: Iterable[Any],
        *,
        model_id: str,
        embodiment: str,
        shape_key_fn,
    ) -> float:
        """Estimate total GPU-ms cost of the pending batch.

        `shape_key_fn(request) -> str` extracts the shape key from each
        request. For the decomposed path this encodes (batch=1, seq_len,
        cache-hit-vs-miss) — the dimensions that make per-session cost
        vary. Monolithic path uses a simpler key.

        In `fallback` mode, returns `len(pending) * _DEFAULT_COLD_START_MS`
        regardless of the cost model — for customers who opt out of
        profiled scheduling under pathological measurement conditions.
        """
        pending_list = list(pending)
        if self._mode == CostMode.FALLBACK:
            return len(pending_list) * self._cost_model._default_cold_start_ms
        total = 0.0
        for req in pending_list:
            shape_key = shape_key_fn(req)
            total += self._cost_model.estimate(
                model_id=model_id, embodiment=embodiment, shape_key=shape_key
            )
        return total

    def should_flush(
        self,
        pending: Iterable[Any],
        *,
        model_id: str,
        embodiment: str,
        oldest_wait_ms: float,
        shape_key_fn,
    ) -> SchedulerDecision:
        """Decide whether to flush the pending batch.

        Args:
            pending: iterable of requests in the queue.
            oldest_wait_ms: how long the oldest request has been waiting
                (ms). Drives the timeout flush path.
            shape_key_fn: callable(request) -> str for cost lookup.
        """
        pending_list = list(pending)
        size = len(pending_list)
        if size == 0:
            return SchedulerDecision(
                flush=False, reason="empty",
                batch_cost_ms=0.0, budget_ms=self._max_cost_ms,
                size=0, shape_homogeneous=True,
            )

        shape_keys = {shape_key_fn(req) for req in pending_list}
        homogeneous = len(shape_keys) == 1
        batch_cost = self.batch_cost_ms(
            pending_list,
            model_id=model_id, embodiment=embodiment,
            shape_key_fn=shape_key_fn,
        )

        # Single request already exceeds budget — one-shot flush (waiting
        # for more requests only makes throughput worse).
        if size == 1 and batch_cost >= self._max_cost_ms:
            return SchedulerDecision(
                flush=True, reason="single_request_over_budget",
                batch_cost_ms=batch_cost, budget_ms=self._max_cost_ms,
                size=1, shape_homogeneous=True,
            )

        # Budget reached.
        if batch_cost >= self._max_cost_ms:
            return SchedulerDecision(
                flush=True, reason="budget_reached",
                batch_cost_ms=batch_cost, budget_ms=self._max_cost_ms,
                size=size, shape_homogeneous=homogeneous,
            )

        # Timeout — oldest request has waited too long.
        if oldest_wait_ms >= self._max_wait_ms:
            return SchedulerDecision(
                flush=True, reason="timeout",
                batch_cost_ms=batch_cost, budget_ms=self._max_cost_ms,
                size=size, shape_homogeneous=homogeneous,
            )

        # Hold — wait for more requests or budget / timeout trigger.
        return SchedulerDecision(
            flush=False, reason="under_budget",
            batch_cost_ms=batch_cost, budget_ms=self._max_cost_ms,
            size=size, shape_homogeneous=homogeneous,
        )


__all__ = [
    "CostKey",
    "CostMode",
    "CostBudgetScheduler",
    "GpuMsCostModel",
    "SchedulerDecision",
]
