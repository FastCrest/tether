"""Rollout-time evidence capture for ``tether verify``.

The memory sampler records process RSS and process-attributed device memory at
10 Hz.  CPU execution has a proven zero device allocation; CUDA execution uses
``nvidia-smi``'s per-process accounting rather than framework allocator stats.
"""

from __future__ import annotations

import math
import os
import hashlib
import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tether.verification_evidence import (
    EvidenceValue,
    UnavailableReason,
    normalize_verification_device,
)


MEMORY_SAMPLE_INTERVAL_S = 0.1
MEMORY_CAPTURE_JITTER_S = 0.02
MEMORY_DEADLINE_EPSILON_S = 1e-6


@dataclass(frozen=True)
class MemorySample:
    scheduled_monotonic_s: float
    captured_monotonic_s: float
    process_rss_bytes: int
    device_allocated_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "scheduled_monotonic_s": self.scheduled_monotonic_s,
            "captured_monotonic_s": self.captured_monotonic_s,
            "process_rss_bytes": self.process_rss_bytes,
            "device_allocated_bytes": self.device_allocated_bytes,
        }


@dataclass(frozen=True)
class CanonicalSafetyLimits:
    """Immutable, hash-addressed seven-dimensional verification limits."""

    joint_names: tuple[str, ...]
    position_min: tuple[float, ...]
    position_max: tuple[float, ...]
    velocity_max: tuple[float, ...]
    effort_max: tuple[float, ...]
    workspace_indices: tuple[int, ...]
    workspace_min: tuple[float, ...]
    workspace_max: tuple[float, ...]

    def to_dict(self) -> dict[str, list[str] | list[float] | list[int]]:
        return {
            "joint_names": list(self.joint_names),
            "position_min": list(self.position_min),
            "position_max": list(self.position_max),
            "velocity_max": list(self.velocity_max),
            "effort_max": list(self.effort_max),
            "workspace_indices": list(self.workspace_indices),
            "workspace_min": list(self.workspace_min),
            "workspace_max": list(self.workspace_max),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_safety_limits(self) -> Any:
        from tether.safety import SafetyLimits

        return SafetyLimits(
            joint_names=list(self.joint_names),
            position_min=list(self.position_min),
            position_max=list(self.position_max),
            velocity_max=list(self.velocity_max),
            effort_max=list(self.effort_max),
            workspace_indices=list(self.workspace_indices),
            workspace_min=list(self.workspace_min),
            workspace_max=list(self.workspace_max),
        )


def canonicalize_safety_limits(limits: Any) -> CanonicalSafetyLimits:
    """Validate once and freeze the exact value sent to both rollout children."""

    validate_safety_limits(limits, action_dim=7)
    workspace_indices = tuple(int(value) for value in limits.workspace_indices)
    workspace_min = (
        tuple(float(value) for value in limits.workspace_min) if workspace_indices else ()
    )
    workspace_max = (
        tuple(float(value) for value in limits.workspace_max) if workspace_indices else ()
    )
    return CanonicalSafetyLimits(
        joint_names=tuple(str(value) for value in limits.joint_names),
        position_min=tuple(float(value) for value in limits.position_min),
        position_max=tuple(float(value) for value in limits.position_max),
        velocity_max=tuple(float(value) for value in limits.velocity_max),
        effort_max=tuple(float(value) for value in limits.effort_max),
        workspace_indices=workspace_indices,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
    )


def load_canonical_safety_limits(path: str | Path) -> CanonicalSafetyLimits:
    """Read a safety file once in the parent, then pass only its frozen value."""

    from tether.safety import SafetyLimits

    return canonicalize_safety_limits(SafetyLimits.from_json(path))


def nearest_rank_percentile(values: list[int] | list[float], percentile: float) -> float:
    """Nearest-rank percentile, as required by Verification Evidence v1."""

    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def validate_safety_limits(limits: Any, *, action_dim: int = 7) -> None:
    """Require complete, finite limits for every executed action dimension."""

    arrays = {
        "joint_names": getattr(limits, "joint_names", None),
        "position_min": getattr(limits, "position_min", None),
        "position_max": getattr(limits, "position_max", None),
        "velocity_max": getattr(limits, "velocity_max", None),
        "effort_max": getattr(limits, "effort_max", None),
    }
    for name, values in arrays.items():
        if not isinstance(values, list) or len(values) != action_dim:
            raise ValueError(f"SafetyLimits.{name} must contain exactly {action_dim} entries")
    position_min = arrays["position_min"]
    position_max = arrays["position_max"]
    velocity_max = arrays["velocity_max"]
    effort_max = arrays["effort_max"]
    assert isinstance(position_min, list)
    assert isinstance(position_max, list)
    assert isinstance(velocity_max, list)
    assert isinstance(effort_max, list)
    for index in range(action_dim):
        lower = float(position_min[index])
        upper = float(position_max[index])
        velocity = float(velocity_max[index])
        effort = float(effort_max[index])
        if not all(math.isfinite(value) for value in (lower, upper, velocity, effort)):
            raise ValueError("SafetyLimits values must all be finite")
        if lower >= upper:
            raise ValueError(f"SafetyLimits position range is invalid at index {index}")
        if velocity <= 0 or effort < 0:
            raise ValueError(
                f"SafetyLimits velocity must be positive and effort non-negative at index {index}"
            )

    workspace_indices = getattr(limits, "workspace_indices", [])
    workspace_min = getattr(limits, "workspace_min", [])
    workspace_max = getattr(limits, "workspace_max", [])
    if not all(
        isinstance(values, list) for values in (workspace_indices, workspace_min, workspace_max)
    ):
        raise ValueError("SafetyLimits workspace arrays must be lists")
    if workspace_indices and not (
        len(workspace_indices) == len(workspace_min) == len(workspace_max)
    ):
        raise ValueError("SafetyLimits workspace arrays must have equal lengths")
    for index, lower, upper in zip(workspace_indices, workspace_min, workspace_max):
        if not isinstance(index, int) or not 0 <= index < action_dim:
            raise ValueError("SafetyLimits workspace index is outside the action")
        if not (math.isfinite(float(lower)) and math.isfinite(float(upper))):
            raise ValueError("SafetyLimits workspace bounds must be finite")
        if float(lower) >= float(upper):
            raise ValueError("SafetyLimits workspace range is invalid")


def _read_process_rss_bytes(pid: int) -> int:
    """Read current RSS without introducing a mandatory psutil dependency."""

    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss)
    except ImportError:
        pass

    statm = Path(f"/proc/{pid}/statm")
    if statm.is_file():
        pages = int(statm.read_text().split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))

    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    return int(proc.stdout.strip()) * 1024


def _read_cuda_process_bytes(pid: int) -> int:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    used_mib = 0
    matched_pid = False
    for raw_line in proc.stdout.splitlines():
        fields = [part.strip() for part in raw_line.split(",")]
        if len(fields) != 2 or fields[0] != str(pid):
            continue
        matched_pid = True
        used_mib += int(fields[1])
    if not matched_pid:
        raise ProcessLookupError(
            f"nvidia-smi returned no compute-app row for verification pid {pid}"
        )
    return used_mib * 1024 * 1024


class ProcessDeviceMemorySampler:
    """Sample process/device allocation at a fixed 10 Hz cadence."""

    def __init__(
        self,
        *,
        device: str,
        sample_hz: float = 10.0,
        pid: int | None = None,
        rss_probe: Callable[[int], int] | None = None,
        device_probe: Callable[[int], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if sample_hz != 10.0:
            raise ValueError("verification memory capture must run at exactly 10 Hz")
        self.device = normalize_verification_device(device)
        self.sample_hz = sample_hz
        self.pid = pid or os.getpid()
        self._rss_probe = rss_probe or _read_process_rss_bytes
        self._clock = clock
        self._device_probe = device_probe
        self._samples: list[MemorySample] = []
        self._error: UnavailableReason | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._capture_id = uuid.uuid4().hex

        if self.device == "cpu":
            self._device_probe = self._device_probe or (lambda _pid: 0)
        elif self.device == "cuda":
            self._device_probe = self._device_probe or _read_cuda_process_bytes

    def sample(self, *, scheduled_at: float | None = None) -> None:
        """Take one sample; public so deterministic tests need no sleeping."""

        if self._error is not None:
            return
        assert self._device_probe is not None
        try:
            rss = int(self._rss_probe(self.pid))
            device = int(self._device_probe(self.pid))
            if rss < 0 or device < 0:
                raise ValueError("memory probes returned a negative value")
            captured_at = self._clock()
            self._samples.append(
                MemorySample(
                    scheduled_monotonic_s=(captured_at if scheduled_at is None else scheduled_at),
                    captured_monotonic_s=captured_at,
                    process_rss_bytes=rss,
                    device_allocated_bytes=device,
                )
            )
        except FileNotFoundError:
            self._error = UnavailableReason.BACKEND_UNSUPPORTED
            self._stop_event.set()
        except Exception:  # noqa: BLE001 - probe failures are typed evidence
            self._error = UnavailableReason.MEASUREMENT_FAILED
            self._stop_event.set()

    def start(self, *, background: bool = True) -> None:
        if self._started_at is not None:
            return
        self._started_at = self._clock()
        self.sample(scheduled_at=self._started_at)
        if not background or self._error is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="tether-verify-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        interval = 1.0 / self.sample_hz
        assert self._started_at is not None
        deadline = self._started_at + interval
        while True:
            delay = max(0.0, deadline - self._clock())
            if self._stop_event.wait(delay):
                return
            self.sample(scheduled_at=deadline)
            now = self._clock()
            deadline += interval
            while deadline <= now:
                deadline += interval

    def stop(self) -> EvidenceValue[dict[str, object]]:
        if self._started_at is None:
            return EvidenceValue.unavailable(UnavailableReason.CAPTURE_DISABLED)
        self._ended_at = self._clock()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._error is not None:
            return EvidenceValue.unavailable(self._error)
        if not self._samples:
            return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
        assert self._ended_at is not None

        interval = MEMORY_SAMPLE_INTERVAL_S
        duration = self._ended_at - self._started_at
        captured_times = [sample.captured_monotonic_s for sample in self._samples]
        scheduled_times = [sample.scheduled_monotonic_s for sample in self._samples]
        gaps = [current - previous for previous, current in zip(captured_times, captured_times[1:])]
        expected_samples = max(
            2,
            math.floor((duration + MEMORY_DEADLINE_EPSILON_S) / interval) + 1,
        )
        expected_deadlines = [
            self._started_at + (index * interval) for index in range(expected_samples)
        ]
        coverage_ok = (
            duration >= interval
            and len(self._samples) == expected_samples
            and len(scheduled_times) == len(expected_deadlines)
            and all(
                math.isclose(
                    scheduled,
                    expected,
                    rel_tol=0.0,
                    abs_tol=MEMORY_DEADLINE_EPSILON_S,
                )
                for scheduled, expected in zip(scheduled_times, expected_deadlines)
            )
            and self._ended_at - scheduled_times[-1] <= interval + MEMORY_CAPTURE_JITTER_S
            and all(
                interval - MEMORY_CAPTURE_JITTER_S <= gap <= interval + MEMORY_CAPTURE_JITTER_S
                for gap in gaps
            )
            and all(
                0.0 <= captured - scheduled <= MEMORY_CAPTURE_JITTER_S
                for captured, scheduled in zip(captured_times, scheduled_times)
            )
        )
        if not coverage_ok:
            return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)

        rss_values = [sample.process_rss_bytes for sample in self._samples]
        device_values = [sample.device_allocated_bytes for sample in self._samples]
        combined = [rss + device for rss, device in zip(rss_values, device_values)]
        return EvidenceValue.available(
            {
                "sample_hz": self.sample_hz,
                "backend": self.device,
                "process_identity": {
                    "pid": self.pid,
                    "capture_id": self._capture_id,
                },
                "window": {
                    "started_monotonic_s": self._started_at,
                    "ended_monotonic_s": self._ended_at,
                    "duration_s": duration,
                    "expected_samples": expected_samples,
                    "captured_samples": len(self._samples),
                    "max_gap_s": max(gaps) if gaps else 0.0,
                },
                "samples": [sample.to_dict() for sample in self._samples],
                "process_rss": {
                    "peak_bytes": max(rss_values),
                    "p95_bytes": int(nearest_rank_percentile(rss_values, 95.0)),
                },
                "device_allocated": {
                    "peak_bytes": max(device_values),
                    "p95_bytes": int(nearest_rank_percentile(device_values, 95.0)),
                },
                "combined": {
                    "peak_bytes": max(combined),
                    "p95_bytes": int(nearest_rank_percentile(combined, 95.0)),
                },
            }
        )


def apply_action_guard(
    action: Any,
    guard: Any | None,
    *,
    previous_action: Any | None = None,
) -> tuple[Any, int]:
    """Return the exact executed action and number of guard clamp events."""

    import numpy as np

    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if guard is None:
        return action_array, 0
    if not np.isfinite(action_array).all():
        guarded, results = guard.check(action_array.reshape(1, -1))
        safe_action = np.asarray(guarded[0], dtype=np.float32).reshape(-1)
        return safe_action, sum(1 for result in results if result.clamped)
    if hasattr(guard, "check_single"):
        previous = (
            np.asarray(previous_action, dtype=np.float32).reshape(-1)
            if previous_action is not None
            else None
        )
        result = guard.check_single(action_array, previous_action=previous)
        return np.asarray(result.safe_action, dtype=np.float32), int(result.clamped)
    guarded, results = guard.check(action_array.reshape(1, -1))
    safe_action = np.asarray(guarded[0], dtype=np.float32).reshape(-1)
    return safe_action, sum(1 for result in results if result.clamped)


__all__ = [
    "MemorySample",
    "ProcessDeviceMemorySampler",
    "apply_action_guard",
    "nearest_rank_percentile",
    "validate_safety_limits",
]
