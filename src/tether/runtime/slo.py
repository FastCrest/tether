"""Rolling-window latency SLO tracker for `tether serve --slo p99=Xms`.

Measures per-request /act latency, computes rolling percentiles over the
last N samples, and flags violation state when the configured percentile
exceeds the configured threshold. Three modes (`log_only`, `503`, `degrade`)
are interpreted by the caller (server middleware) — the tracker itself just
reports the state.

Recovery logic: once the tracker is in violation state, the rolling p99
must drop below `recover_ratio × threshold` for `recover_windows` consecutive
checks before the tracker clears the violation state. Prevents flapping.

Pattern source: EasyInference InferScope rolling-window tracker (sibling
project; same data structure, VLA-specific metric labels).

Feature spec: features/01_serve/subfeatures/_ecosystem/latency-slo-enforcement/
Execution plan: features/01_serve/subfeatures/_ecosystem/latency-slo-enforcement/latency-slo-enforcement_plan.md
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np


SloMode = Literal["log_only", "503", "degrade"]


VALID_SLO_MODES: frozenset[SloMode] = frozenset({"log_only", "503", "degrade"})


@dataclass(frozen=True)
class SloSpec:
    """Parsed `--slo` flag.

    percentile: 0-100 (e.g., 99.0 for p99, 99.9 for p99.9, 95.0 for p95).
    threshold_ms: latency ceiling in milliseconds.
    """

    percentile: float
    threshold_ms: float

    def __post_init__(self) -> None:
        if not (0.0 < self.percentile <= 100.0):
            raise ValueError(
                f"SloSpec.percentile must be in (0, 100], got {self.percentile}"
            )
        if self.threshold_ms <= 0:
            raise ValueError(
                f"SloSpec.threshold_ms must be > 0, got {self.threshold_ms}"
            )


_SLO_SPEC_RE = re.compile(
    r"^p(?P<pct>\d+(?:\.\d+)?)\s*=\s*(?P<ms>\d+(?:\.\d+)?)\s*ms$",
    re.IGNORECASE,
)


def parse_slo_spec(raw: str) -> SloSpec:
    """Parse 'p99=150ms' / 'p95=200ms' / 'p99.9=500ms' into SloSpec.

    Case-insensitive. Whitespace around '=' ignored. Only 'ms' unit supported
    in Phase 1.

    Raises:
        ValueError: if `raw` does not match the `p<N>=<X>ms` pattern or if
            parsed values are out of bounds.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"--slo must be non-empty string, got {raw!r}")
    m = _SLO_SPEC_RE.match(raw.strip())
    if m is None:
        raise ValueError(
            f"--slo must match 'p<N>=<X>ms' (e.g. 'p99=150ms'), got {raw!r}"
        )
    return SloSpec(
        percentile=float(m.group("pct")),
        threshold_ms=float(m.group("ms")),
    )


def validate_slo_mode(raw: str) -> SloMode:
    """Validate a --slo-mode string; raises ValueError if unknown.

    Valid values: 'log_only', '503', 'degrade'.
    """
    if raw not in VALID_SLO_MODES:
        raise ValueError(
            f"--slo-mode must be one of {sorted(VALID_SLO_MODES)}, got {raw!r}"
        )
    return raw  # type: ignore[return-value]


class SLOTracker:
    """Thread-safe rolling-window percentile tracker.

    Usage:
        spec = parse_slo_spec("p99=150ms")
        tracker = SLOTracker(spec)
        # per /act request:
        tracker.record_latency_ms(elapsed_ms)
        if tracker.is_violating():
            # mode-specific response (log, 503, degrade)

    Thread safety: `record_latency_ms()`, `is_violating()`, `current_p99()`,
    and `should_check()` are all safe to call concurrently from multiple
    FastAPI workers. State transitions (violating ↔ recovered) happen
    atomically under the internal lock.
    """

    __slots__ = (
        "_spec",
        "_window_size",
        "_check_every",
        "_recover_ratio",
        "_recover_windows",
        "_window",
        "_request_count",
        "_violating",
        "_consecutive_recovery_windows",
        "_last_computed_p99",
        "_lock",
    )

    def __init__(
        self,
        spec: SloSpec,
        *,
        window_size: int = 1000,
        check_every: int = 32,
        recover_ratio: float = 0.8,
        recover_windows: int = 3,
    ):
        """Initialize a tracker.

        Args:
            spec: the SLO specification (percentile + threshold_ms).
            window_size: rolling-window sample count (default 1000).
            check_every: recompute percentile every N requests (default 32).
                Smaller = more responsive + more CPU; larger = coarser.
            recover_ratio: violation clears when measured p99 drops below
                `recover_ratio × threshold_ms` (default 0.8).
            recover_windows: number of consecutive under-threshold checks
                before clearing violation state (default 3).
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if check_every <= 0:
            raise ValueError(f"check_every must be > 0, got {check_every}")
        if not (0.0 < recover_ratio < 1.0):
            raise ValueError(
                f"recover_ratio must be in (0, 1), got {recover_ratio}"
            )
        if recover_windows <= 0:
            raise ValueError(f"recover_windows must be > 0, got {recover_windows}")

        self._spec = spec
        self._window_size = window_size
        self._check_every = check_every
        self._recover_ratio = recover_ratio
        self._recover_windows = recover_windows

        self._window: deque[float] = deque(maxlen=window_size)
        self._request_count = 0
        self._violating = False
        self._consecutive_recovery_windows = 0
        self._last_computed_p99: float = float("nan")
        self._lock = threading.Lock()

    @property
    def spec(self) -> SloSpec:
        return self._spec

    def record_latency_ms(self, ms: float) -> None:
        """Append a latency sample; trigger violation-state recomputation
        every `check_every` requests.
        """
        if ms < 0:
            # Negative latency is nonsensical; clamp to 0 to avoid polluting
            # percentile math (vs. raising, which would break the hot path).
            ms = 0.0
        with self._lock:
            self._window.append(ms)
            self._request_count += 1
            if self._request_count % self._check_every == 0:
                self._recompute_locked()

    def current_p99(self) -> float:
        """Return the most recently computed percentile (NaN until the first
        check fires).

        Note: this returns the LAST COMPUTED value, not a fresh computation
        on every call. Cheap — ok to call from metric exporters.
        """
        with self._lock:
            return self._last_computed_p99

    def is_violating(self) -> bool:
        """True if the tracker is currently flagging an SLO violation."""
        with self._lock:
            return self._violating

    def should_check(self) -> bool:
        """True when the next `record_latency_ms()` call will trigger a
        percentile recomputation. Exposed so tests can assert timing; not
        load-bearing for the check itself.
        """
        with self._lock:
            # +1 because record_latency_ms increments BEFORE checking
            return (self._request_count + 1) % self._check_every == 0

    def reset(self) -> None:
        """Clear all state (window + violation flags). Intended for test
        fixtures; callers should not use this in production to "reset" a
        legitimate violation.
        """
        with self._lock:
            self._window.clear()
            self._request_count = 0
            self._violating = False
            self._consecutive_recovery_windows = 0
            self._last_computed_p99 = float("nan")

    def _recompute_locked(self) -> None:
        """Recompute percentile + update violation state. Caller must hold
        self._lock.
        """
        if len(self._window) < self._check_every:
            # Not enough samples yet; leave state alone
            return
        samples = np.asarray(self._window, dtype=np.float64)
        p = float(np.percentile(samples, self._spec.percentile))
        self._last_computed_p99 = p

        threshold = self._spec.threshold_ms
        recover_below = self._recover_ratio * threshold

        if not self._violating:
            if p > threshold:
                self._violating = True
                self._consecutive_recovery_windows = 0
        else:
            if p < recover_below:
                self._consecutive_recovery_windows += 1
                if self._consecutive_recovery_windows >= self._recover_windows:
                    self._violating = False
                    self._consecutive_recovery_windows = 0
            else:
                # p rebounded above recover threshold — reset counter
                self._consecutive_recovery_windows = 0
