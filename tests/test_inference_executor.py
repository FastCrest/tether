"""Tests for the bounded inference offload executor."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tether.runtime.inference_executor import (
    BoundedInferenceExecutor,
    InferenceExecutorFull,
)


def test_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="max_workers"):
        BoundedInferenceExecutor(max_workers=0)
    with pytest.raises(ValueError, match="max_queue"):
        BoundedInferenceExecutor(max_queue=-1)


@pytest.mark.asyncio
async def test_rejects_when_worker_and_queue_are_full():
    executor = BoundedInferenceExecutor(max_workers=1, max_queue=0)
    started = threading.Event()
    release = threading.Event()

    def blocking_work():
        started.set()
        release.wait(timeout=2.0)
        return "done"

    first = asyncio.create_task(executor.submit(blocking_work))
    try:
        assert await asyncio.to_thread(started.wait, 1.0)

        with pytest.raises(InferenceExecutorFull):
            await executor.submit(lambda: "second")

        snapshot = executor.snapshot()
        assert snapshot.pending == 1
        assert snapshot.running == 1
        assert snapshot.queue_depth == 0
        assert snapshot.rejected == 1
    finally:
        release.set()
        assert await first == "done"
        executor.shutdown()


@pytest.mark.asyncio
async def test_reports_queued_and_running_work():
    states = []
    executor = BoundedInferenceExecutor(
        max_workers=1,
        max_queue=1,
        on_state_change=states.append,
    )
    first_started = threading.Event()
    release_first = threading.Event()

    def blocking_work():
        first_started.set()
        release_first.wait(timeout=2.0)
        return "first"

    first = asyncio.create_task(executor.submit(blocking_work))
    second = None
    try:
        assert await asyncio.to_thread(first_started.wait, 1.0)

        second = asyncio.create_task(executor.submit(lambda: "second"))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            snapshot = executor.snapshot()
            if snapshot.pending == 2 and snapshot.queue_depth == 1:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail(f"queued work was not observed: {executor.snapshot()}")

        assert any(state.running == 1 for state in states)
        assert any(state.queue_depth == 1 for state in states)
    finally:
        release_first.set()
        assert await first == "first"
        if second is not None:
            assert await second == "second"
        executor.shutdown()
