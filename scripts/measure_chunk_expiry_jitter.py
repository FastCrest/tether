"""Local software expiry-decision timing; no robot or real-time qualification.

Run with PYTHONPATH=src python scripts/measure_chunk_expiry_jitter.py.
One million real-clock, one-action pushes followed by expired pops. A zero
bound makes the push-time timestamp the deadline, avoiding timed sleeps.
"""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np

from tether.runtime.buffer import ActionChunkBuffer


class Clock:
    last = 0.0

    def __call__(self) -> float:
        self.last = time.monotonic()
        return self.last


def main() -> None:
    clock = Clock()
    buf = ActionChunkBuffer(capacity=1, max_lateness_ms=0, clock=clock)
    chunk = np.ones((1, 1), dtype=np.float32)
    samples = 1_000_000
    decision_ms = np.empty(samples)
    overshoot_ms = np.empty(samples)
    for i in range(samples + 10_000):
        buf.push_chunk(chunk)
        deadline = clock.last
        start = time.monotonic()
        action = buf.pop_next()
        end = time.monotonic()
        if action is not None:
            raise AssertionError("expected an expired action on a real monotonic clock")
        if i >= 10_000:
            decision_ms[i - 10_000] = (end - start) * 1000
            overshoot_ms[i - 10_000] = (end - deadline) * 1000
    def summary(values):
        p50, p99 = np.percentile(values, [50, 99])
        return {"p50_ms": float(p50), "p99_ms": float(p99),
                "max_ms": float(values.max()), "p99_minus_p50_ms": float(p99 - p50)}
    print(json.dumps({
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "samples": samples, "warmup_pops": 10_000,
        "bound_ms": 0, "control_hz": 100,
        "pop_duration": summary(decision_ms),
        "deadline_to_return": summary(overshoot_ms),
        "scope": "macOS dev-machine only; single thread, no model, no sleeps; "
                 "includes GIL/scheduler effects during this run, not a wakeup or Thor bound",
    }, indent=2))


if __name__ == "__main__":
    main()
