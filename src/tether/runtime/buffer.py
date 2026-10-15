"""Action chunk buffer for async replan-while-execute serving.

Physical Intelligence's pattern: the robot actuator runs at ~100Hz but VLA
inference is ~50-500ms per call. Naive serving (run full chunk, then ask
for another) produces 500ms of open-loop motion per inference call —
stale whenever the environment changes faster than that.

Fix: decouple execution from replan. The server keeps a ring buffer of
the "current plan" — next N actions to execute. The execute loop pops
one action per tick at `execute_hz`. A replan loop refreshes the head of
the buffer at `replan_hz`, overwriting stale actions with fresh chunks.

Expiry is checked synchronously by pop_next, never by a background timer.
Consumers own polling and robot-side fallback. HTTP /act currently returns
full chunks and does not consume this buffer.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tether.runtime.adaptive_chunking import AdaptiveChunkDecision


@dataclass
class BufferStats:
    """Snapshot of the buffer state at a moment in time."""

    size: int
    capacity: int
    replans: int
    stale_overwrites: int
    last_replan_age_ms: float
    expired_actions: int = 0
    late_chunks: int = 0
    dropped_replans: int = 0
    max_observed_lateness_ms: float = 0.0


@dataclass
class _Plan:
    ref: float
    late: bool = False


@dataclass
class _Action:
    value: np.ndarray
    due: float
    plan: _Plan


class ActionChunkBuffer:
    """Thread-safe ring buffer holding the current plan's remaining actions.

    Semantics:
      - `push_chunk(chunk, overwrite_stale=True)` — replace the buffer
        contents with the leading `capacity` actions from `chunk`. When
        `overwrite_stale=True` (the production default), any still-pending
        actions in the buffer are discarded — the new chunk is fresher and
        should supersede the tail of the old one. Set False for append-only
        cases (replay, pre-computed plans).
      - `pop_next()` — returns and removes the leftmost action, or None if
        empty or expired. Expiry discards the remaining tail.
      - `should_replan(threshold_ratio)` — True when size <= capacity *
        threshold_ratio. Callers use this to decide when to kick off a
        fresh VLA inference.

    Thread-safety: an internal lock guards all state mutations. Expected
    usage is: an integration reads via pop_next() + should_replan() from
    any thread; a separate replan worker writes via push_chunk().
    """

    def __init__(
        self, capacity: int = 50, *, control_hz: float = 100.0,
        max_lateness_ms: float | None = None,
        max_age_ms: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        for name, bound in (("max_lateness_ms", max_lateness_ms), ("max_age_ms", max_age_ms)):
            if bound is not None and (not math.isfinite(bound) or bound < 0):
                raise ValueError(f"{name} must be finite and nonnegative, or None")
        self.control_hz = control_hz
        self.max_lateness_ms = max_lateness_ms
        self.max_age_ms = max_age_ms
        self._clock = clock
        self._clock_high_water = -math.inf
        self._expired_actions = 0
        self._late_chunks = 0
        self._dropped_replans = 0
        self._max_observed_lateness_ms = 0.0
        self._buf: deque[_Action] = deque(maxlen=capacity)
        self._capacity = capacity
        self._lock = threading.Lock()
        self._replans = 0
        self._stale_overwrites = 0
        self._last_replan_ts: float | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    def push_chunk(
        self, chunk: np.ndarray, overwrite_stale: bool = True, *, t_obs: float | None = None
    ) -> int:
        """Refill the buffer with up to `capacity` actions from `chunk`.

        Args:
            chunk: [chunk_size, action_dim] array. Only the first
                `capacity` actions are taken.
            overwrite_stale: if True (default), discard any actions
                still in the buffer before filling. Use this when the
                new chunk reflects the latest observation and stale
                actions would cause the robot to commit to an outdated
                plan.
            t_obs: optional observation time in seconds on the same monotonic
                clock. Future or non-finite values are rejected. Without it,
                deadlines start at push, so inference age is unknown.

        Returns: count of actions accepted with expiry enabled. With both
        bounds OFF, preserves the legacy capacity-limited requested count
        even when append-only mode has less room. Late leading actions are
        skipped before applying capacity, retaining their original indices.
        """
        if chunk.ndim != 2:
            raise ValueError(
                f"chunk must be 2D [T, action_dim]; got shape {chunk.shape}"
            )
        take = min(len(chunk), self._capacity)
        with self._lock:
            now = self._now()
            if t_obs is not None and (not math.isfinite(t_obs) or t_obs > now):
                raise ValueError("t_obs must be finite and no later than the monotonic clock")
            plan = _Plan(now if t_obs is None else t_obs)
            if overwrite_stale and self._buf:
                self._stale_overwrites += len(self._buf)
                self._buf.clear()
            first = 0
            # Keep original action indices: skipping never resets the due schedule.
            while first < len(chunk) and self._is_expired(
                now, plan.ref + first / self.control_hz, plan,
            ):
                first += 1
            self._expired_actions += first
            if first:
                self._mark_late(plan)
                if first == len(chunk):
                    self._dropped_replans += 1
            pushed = 0
            for i in range(first, min(len(chunk), first + self._capacity)):
                if len(self._buf) >= self._capacity:
                    break
                self._buf.append(_Action(chunk[i].copy(), plan.ref + i / self.control_hz, plan))
                pushed += 1
            self._replans += 1
            self._last_replan_ts = now
        if self.max_lateness_ms is None and self.max_age_ms is None:
            return take
        return pushed

    def _now(self) -> float:
        """Read under the lock; a regressing injected clock cannot rejuvenate a plan."""
        now = self._clock()
        if not math.isfinite(now):
            raise ValueError("clock must return finite monotonic seconds")
        self._clock_high_water = max(self._clock_high_water, now)
        return self._clock_high_water

    def _mark_late(self, plan: _Plan) -> None:
        if not plan.late:
            self._late_chunks += 1
            plan.late = True

    def _is_expired(self, now: float, due: float, plan: _Plan) -> bool:
        self._max_observed_lateness_ms = max(
            self._max_observed_lateness_ms, (now - due) * 1000.0,
        )
        # Compare absolute deadlines to preserve equality at the bound.
        return (
            self.max_lateness_ms is not None
            and now > due + self.max_lateness_ms / 1000.0
        ) or (
            self.max_age_ms is not None
            and now > plan.ref + self.max_age_ms / 1000.0
        )

    def pop_next(self) -> np.ndarray | None:
        """Pop an action, or discard the entire remaining tail on expiry.

        Action i is due at (t_obs or push time) + i/control_hz. Strictly
        greater than either enabled bound expires; equality is accepted.
        No hold, repeat, synthesized action or pose command is produced.
        """
        with self._lock:
            if not self._buf:
                return None
            head = self._buf[0]
            if self._is_expired(self._now(), head.due, head.plan):
                self._expired_actions += len(self._buf)
                for action in self._buf:
                    self._mark_late(action.plan)
                self._buf.clear()
                return None
            return self._buf.popleft().value

    def peek_next(self) -> np.ndarray | None:
        """Return the leftmost action without removing, or None if empty."""
        with self._lock:
            if not self._buf:
                return None
            return self._buf[0].value.copy()

    def peek_all(self) -> np.ndarray | None:
        """Return a stacked snapshot of all buffered actions, or None if empty.

        Read-only — does not check expiry or modify the buffer. Used by RtcAdapter to
        capture `prev_chunk_left_over` before `push_chunk(overwrite_stale=True)`
        wipes the buffer for the new chunk.

        Returns:
            np.ndarray of shape (size, action_dim), or None if empty.
            Always a copy — caller can mutate without affecting the buffer.
        """
        with self._lock:
            if not self._buf:
                return None
            return np.stack([action.value for action in self._buf], axis=0).copy()

    def should_replan(
        self,
        threshold_ratio: float = 0.5,
        *,
        adaptive_decision: "AdaptiveChunkDecision | None" = None,
    ) -> bool:
        """True when the buffer size has dropped to or below threshold.

        Default threshold_ratio=0.5 means: when half or fewer of the
        planned actions remain, it's time to replan. Caller decides
        whether to actually trigger predict() — the buffer just
        advises.

        When an AdaptiveChunkDecision is supplied, its threshold wins. This
        keeps the existing fixed-ratio behavior as the default while allowing
        AAC-style horizons to replan sooner under uncertainty and later when
        the scene is stable.
        """
        if adaptive_decision is not None:
            threshold_ratio = adaptive_decision.replan_threshold_ratio
        with self._lock:
            return len(self._buf) <= self._capacity * threshold_ratio

    def clear(self) -> None:
        """Drop all pending actions (e.g. on emergency stop)."""
        with self._lock:
            if self._buf:
                self._stale_overwrites += len(self._buf)
                self._buf.clear()

    def stats(self) -> BufferStats:
        """Snapshot for /act diagnostics."""
        with self._lock:
            last_age = (
                (self._now() - self._last_replan_ts) * 1000.0
                if self._last_replan_ts is not None
                else 0.0
            )
            return BufferStats(
                size=len(self._buf),
                capacity=self._capacity,
                replans=self._replans,
                stale_overwrites=self._stale_overwrites,
                last_replan_age_ms=round(last_age, 1),
                expired_actions=self._expired_actions,
                late_chunks=self._late_chunks,
                dropped_replans=self._dropped_replans,
                max_observed_lateness_ms=self._max_observed_lateness_ms,
            )


def compute_replan_window(
    execute_hz: float, replan_hz: float, chunk_size: int = 50
) -> dict[str, Any]:
    """Derive sane buffer capacity + replan threshold from the two Hz values.

    If the robot executes at 100Hz and we replan at 20Hz, we execute 5
    actions per replan cycle (100/20). The buffer needs enough head-room
    that it never drains before the next replan lands:

        capacity = 2 * ceil(execute_hz / replan_hz)

    Threshold: replan when half of one cycle remains, i.e. when the buffer
    has roughly `execute_hz / replan_hz` actions left (the amount the
    robot will consume in one replan cycle).
    """
    if execute_hz <= 0 or replan_hz <= 0:
        raise ValueError("execute_hz and replan_hz must be positive")
    if replan_hz > execute_hz:
        raise ValueError(
            f"replan_hz ({replan_hz}) > execute_hz ({execute_hz}) is backwards — "
            f"replanning faster than executing wastes inference."
        )
    per_cycle = max(1, int(-(-execute_hz // replan_hz)))  # ceil division
    capacity = min(chunk_size, 2 * per_cycle)
    threshold_ratio = per_cycle / max(1, 2 * per_cycle)
    return {
        "capacity": capacity,
        "threshold_ratio": threshold_ratio,
        "actions_per_replan_cycle": per_cycle,
    }


__all__ = ["ActionChunkBuffer", "BufferStats", "compute_replan_window"]
