"""Bounded, optional host-memory observations during server readiness.

Host RSS/available RAM do not measure GPU allocation or prove an OOM kill.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


def _memory_sample(pid: int) -> dict[str, Any]:
    try:
        import psutil

        return {
            "source": "psutil",
            "rss_bytes": psutil.Process(pid).memory_info().rss,
            "host_available_bytes": psutil.virtual_memory().available,
        }
    except (ImportError, OSError, ValueError):
        pass
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        memory = Path("/proc/meminfo").read_text()

        def field(text, key):
            return next(
                int(line.split()[1]) * 1024
                for line in text.splitlines()
                if line.startswith(key + ":")
            )

        return {
            "source": "procfs",
            "rss_bytes": field(status, "VmRSS"),
            "host_available_bytes": field(memory, "MemAvailable"),
        }
    except (OSError, ValueError, StopIteration):
        return {"source": "unavailable", "rss_bytes": None, "host_available_bytes": None}


class StartupObservation:
    """At most 64 retained samples; fixed one-second sampling, no subprocesses."""

    def __init__(self, process: Any, timeout_s: float):
        self.process = process
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.report: dict[str, Any] = {
            "timeout_s": timeout_s,
            "status": "waiting",
            "diagnostic": None,
            "process_memory": {
                "scope": "server_process_host_memory",
                "peak_rss_bytes": None,
                "min_host_available_bytes": None,
                "samples": [],
                "sample_count": 0,
                "sampling_interval_s": 1,
                "retained_sample_limit": 64,
                "gpu_memory_measured": False,
                "oom_confirmed": False,
            },
        }
        self.thread = threading.Thread(target=self._loop, daemon=True, name="tether-proof-startup")

    def _loop(self):
        while not self.stop.is_set():
            try:
                row = _memory_sample(self.process.pid)
            except Exception:  # telemetry must not break or stall the readiness contract
                row = {"source": "unavailable", "rss_bytes": None, "host_available_bytes": None}
            row["elapsed_ms"] = round((time.monotonic() - self.started) * 1000, 1)
            memory = self.report["process_memory"]
            memory["sample_count"] += 1
            if len(memory["samples"]) < 64:
                memory["samples"].append(row)
            if row["rss_bytes"] is not None:
                memory["peak_rss_bytes"] = max(memory["peak_rss_bytes"] or 0, row["rss_bytes"])
            if row["host_available_bytes"] is not None:
                previous = memory["min_host_available_bytes"]
                memory["min_host_available_bytes"] = (
                    row["host_available_bytes"]
                    if previous is None
                    else min(previous, row["host_available_bytes"])
                )
            if self.stop.wait(1):
                break

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=0.25)
        self.report["duration_ms"] = round((time.monotonic() - self.started) * 1000, 1)
        self.report["status"] = "ready" if exc is None else "failed"
        if exc is not None:
            code = self.process.poll()
            self.report["exit_code_before_cleanup"] = code
            self.report["diagnostic"] = (
                "server_exit_before_ready"
                if code is not None
                else "startup_timeout_or_health_error"
            )
            if code == -9:
                self.report["termination_note"] = (
                    "SIGKILL observed; OOM or external termination cause is unconfirmed."
                )
