"""Bounded async offload executor for synchronous inference work."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class InferenceExecutorFull(RuntimeError):
    """Raised when all inference worker and queue slots are occupied."""


@dataclass(frozen=True)
class InferenceExecutorSnapshot:
    """Point-in-time executor state for health and metrics surfaces."""

    max_workers: int
    max_queue: int
    capacity: int
    pending: int
    running: int
    queue_depth: int
    rejected: int


class BoundedInferenceExecutor:
    """Run sync inference functions in a bounded dedicated thread pool.

    The default asyncio executor has an unbounded submission queue. For robot
    serving, that hides overload and lets latency tails grow silently. This
    wrapper rejects fast when all worker + queue slots are occupied.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_queue: int = 8,
        thread_name_prefix: str = "tether-inference",
        on_state_change: Callable[[InferenceExecutorSnapshot], None] | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {max_workers}")
        if max_queue < 0:
            raise ValueError(f"max_queue must be >= 0, got {max_queue}")
        self._max_workers = int(max_workers)
        self._max_queue = int(max_queue)
        self._capacity = self._max_workers + self._max_queue
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._pending = 0
        self._running = 0
        self._rejected = 0
        self._closed = False
        self._on_state_change = on_state_change

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def max_queue(self) -> int:
        return self._max_queue

    @property
    def capacity(self) -> int:
        return self._capacity

    async def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Submit sync work or raise InferenceExecutorFull without waiting."""

        full_message = ""
        with self._lock:
            if self._closed:
                raise RuntimeError("inference executor is shut down")
            if self._pending >= self._capacity:
                self._rejected += 1
                snapshot = self._snapshot_locked()
                full_message = (
                    "inference executor is full "
                    f"(pending={self._pending}, capacity={self._capacity})"
                )
            else:
                self._pending += 1
                snapshot = self._snapshot_locked()
        self._notify_state(snapshot)
        if full_message:
            raise InferenceExecutorFull(full_message)

        loop = asyncio.get_running_loop()
        callback = functools.partial(self._invoke, fn, args, kwargs)
        try:
            return await loop.run_in_executor(self._executor, callback)
        finally:
            with self._lock:
                self._pending -= 1
                snapshot = self._snapshot_locked()
            self._notify_state(snapshot)

    def snapshot(self) -> InferenceExecutorSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _invoke(
        self,
        fn: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        with self._lock:
            self._running += 1
            snapshot = self._snapshot_locked()
        self._notify_state(snapshot)
        try:
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self._running -= 1
                snapshot = self._snapshot_locked()
            self._notify_state(snapshot)

    def _snapshot_locked(self) -> InferenceExecutorSnapshot:
        return InferenceExecutorSnapshot(
            max_workers=self._max_workers,
            max_queue=self._max_queue,
            capacity=self._capacity,
            pending=self._pending,
            running=self._running,
            queue_depth=max(0, self._pending - self._running),
            rejected=self._rejected,
        )

    def _notify_state(self, snapshot: InferenceExecutorSnapshot) -> None:
        if self._on_state_change is None:
            return
        try:
            self._on_state_change(snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("inference executor state callback failed: %s", exc)
