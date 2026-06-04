"""Post-swap monitoring window — passive guard against mid-window regression.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #4: the
9-gate eval (eval_gate.py) gates the swap at promotion time, but
production traffic can still uncover regressions the held-out eval
missed. The PostSwapMonitor watches the first 24h OR 500 episodes
post-swap and triggers an auto-rollback when ANY of three trip signals
fires.

Three trip signals (rolling-window):
- T1: safety-clamp rolling p95 > 2× pre-swap baseline
- T2: action cos-similarity to previous-deployed-model < 0.85
- T3: webhook safety_violation count > 5 in any 5-minute window

Auto-rollback trigger: 2-in-a-row exceedance (configurable via
`--rollback-sensitivity aggressive=1 | normal=2 | tolerant=3`).
Single-strike triggers are too noisy; three-strike costs too much
trust.

Pure primitive — caller (Day 8+ wiring) drives observations + invokes
should_rollback() periodically. State is held inside the monitor; reset
on swap (call .reset() with new baseline).
"""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


# Window size for the post-swap watch — whichever ENDS first triggers
# end-of-window. After end-of-window the monitor stops tripping (model is
# considered stable).
WINDOW_DURATION_HOURS = 24
WINDOW_EPISODE_COUNT = 500

# Rolling p95 sample window for safety-clamp signal. Tracks the most-recent
# N episodes; matches webhooks + slo conventions.
ROLLING_WINDOW_SIZE = 100

# How many consecutive trip observations are needed to fire rollback.
# Bounded enum: aggressive=1 | normal=2 | tolerant=3.
SENSITIVITY_AGGRESSIVE = 1
SENSITIVITY_NORMAL = 2
SENSITIVITY_TOLERANT = 3

# Default trip thresholds (matches ADR; customers tune via tether.yaml).
DEFAULT_T1_CLAMP_RATE_RELATIVE = 2.0  # 2× baseline
DEFAULT_T2_COS_MIN = 0.85
DEFAULT_T3_VIOLATION_COUNT_5MIN = 5

# 5-min window for T3 (webhook safety_violation count).
T3_WINDOW_SECONDS = 300


SensitivityMode = Literal["aggressive", "normal", "tolerant"]


@dataclass(frozen=True)
class MonitorConfig:
    """Frozen monitor config. Customer tunes via tether.yaml; defaults
    locked at the conservative end (normal sensitivity, ADR thresholds)."""

    sensitivity: SensitivityMode = "normal"
    t1_clamp_rate_relative: float = DEFAULT_T1_CLAMP_RATE_RELATIVE
    t2_cos_min: float = DEFAULT_T2_COS_MIN
    t3_violation_count_5min: int = DEFAULT_T3_VIOLATION_COUNT_5MIN
    window_duration_hours: int = WINDOW_DURATION_HOURS
    window_episode_count: int = WINDOW_EPISODE_COUNT
    rolling_window_size: int = ROLLING_WINDOW_SIZE

    def __post_init__(self) -> None:
        if self.sensitivity not in ("aggressive", "normal", "tolerant"):
            raise ValueError(
                f"sensitivity must be aggressive|normal|tolerant, got "
                f"{self.sensitivity!r}"
            )
        if self.t1_clamp_rate_relative <= 1.0:
            raise ValueError(
                f"t1_clamp_rate_relative must be > 1.0, got "
                f"{self.t1_clamp_rate_relative}"
            )
        if not (0.0 < self.t2_cos_min < 1.0):
            raise ValueError(
                f"t2_cos_min must be in (0, 1), got {self.t2_cos_min}"
            )
        if self.t3_violation_count_5min < 1:
            raise ValueError(
                f"t3_violation_count_5min must be >= 1, got "
                f"{self.t3_violation_count_5min}"
            )
        if self.window_duration_hours < 1:
            raise ValueError(
                f"window_duration_hours must be >= 1, got "
                f"{self.window_duration_hours}"
            )
        if self.window_episode_count < 1:
            raise ValueError(
                f"window_episode_count must be >= 1, got "
                f"{self.window_episode_count}"
            )
        if self.rolling_window_size < 1:
            raise ValueError(
                f"rolling_window_size must be >= 1, got "
                f"{self.rolling_window_size}"
            )

    def required_consecutive_trips(self) -> int:
        return {
            "aggressive": SENSITIVITY_AGGRESSIVE,
            "normal": SENSITIVITY_NORMAL,
            "tolerant": SENSITIVITY_TOLERANT,
        }[self.sensitivity]


@dataclass(frozen=True)
class TripDecision:
    """Output of `should_rollback()` — bounded fields for log/metric emission."""

    should_rollback: bool
    reason: str  # bounded enum: T1|T2|T3|window-closed|in-window
    consecutive_trips: int
    required_trips: int
    measured: float
    threshold: float
    samples_in_window: int
    seconds_in_window: float


class PostSwapMonitor:
    """Stateful 24h / 500-episode rolling-window watcher.

    Lifecycle:
        monitor = PostSwapMonitor(config=MonitorConfig())
        monitor.start_window(baseline_clamp_rate=0.005, swap_at=datetime.now(utc))
        # in /act handler post-result:
        monitor.record_episode(
            safety_clamp_count=N,
            action_trajectory=[...],
            previous_action_trajectory=[...],  # for T2 cos
            webhook_violations_in_5min=K,
        )
        decision = monitor.should_rollback()
        if decision.should_rollback:
            # caller (Day 8+) fires the rollback handler

    PURE state — no I/O, no asyncio. Caller drives observations.
    """

    __slots__ = (
        "_config", "_baseline_clamp_rate", "_swap_at",
        "_clamp_window", "_cos_window", "_violation_window",
        "_consecutive_trips", "_episodes_seen",
    )

    def __init__(self, config: MonitorConfig | None = None):
        self._config = config or MonitorConfig()
        self._baseline_clamp_rate: float = 0.0
        self._swap_at: datetime | None = None
        self._clamp_window: collections.deque[float] = collections.deque(
            maxlen=self._config.rolling_window_size,
        )
        self._cos_window: collections.deque[float] = collections.deque(
            maxlen=self._config.rolling_window_size,
        )
        self._violation_window: collections.deque[tuple[datetime, int]] = (
            collections.deque()
        )
        self._consecutive_trips = 0
        self._episodes_seen = 0

    @property
    def config(self) -> MonitorConfig:
        return self._config

    @property
    def episodes_seen(self) -> int:
        return self._episodes_seen

    @property
    def is_window_open(self) -> bool:
        if self._swap_at is None:
            return False
        return not self._is_window_closed(now=datetime.now(timezone.utc))

    def start_window(
        self,
        *,
        baseline_clamp_rate: float,
        swap_at: datetime | None = None,
    ) -> None:
        """Reset state + open a fresh post-swap window. Call once per
        successful swap. Idempotent for repeat calls (resets state)."""
        if baseline_clamp_rate < 0:
            raise ValueError(
                f"baseline_clamp_rate must be >= 0, got {baseline_clamp_rate}"
            )
        self._baseline_clamp_rate = float(baseline_clamp_rate)
        self._swap_at = swap_at or datetime.now(timezone.utc)
        self._clamp_window.clear()
        self._cos_window.clear()
        self._violation_window.clear()
        self._consecutive_trips = 0
        self._episodes_seen = 0

    def record_episode(
        self,
        *,
        safety_clamp_count: int,
        cos_to_previous_model: float | None = None,
        webhook_violations_count: int = 0,
        now: datetime | None = None,
    ) -> None:
        """Record one /act episode. cos_to_previous_model can be None
        when the previous model's actions aren't available; in that case
        T2 isn't checked for this episode."""
        if self._swap_at is None:
            # Window not open — silently drop (caller forgot to start_window)
            return
        now = now or datetime.now(timezone.utc)
        self._episodes_seen += 1
        # Per-episode safety-clamp rate
        self._clamp_window.append(float(safety_clamp_count))
        if cos_to_previous_model is not None:
            self._cos_window.append(float(cos_to_previous_model))
        if webhook_violations_count > 0:
            self._violation_window.append((now, int(webhook_violations_count)))
        # Prune violation window to last 5 min
        self._prune_violation_window(now=now)

    def should_rollback(self, now: datetime | None = None) -> TripDecision:
        """Check whether any trip signal has fired enough times in a row
        to require rollback."""
        now = now or datetime.now(timezone.utc)
        if self._swap_at is None:
            return TripDecision(
                should_rollback=False, reason="in-window",
                consecutive_trips=0, required_trips=self._config.required_consecutive_trips(),
                measured=0.0, threshold=0.0,
                samples_in_window=0, seconds_in_window=0.0,
            )
        seconds_in_window = (now - self._swap_at).total_seconds()
        if self._is_window_closed(now=now):
            return TripDecision(
                should_rollback=False, reason="window-closed",
                consecutive_trips=self._consecutive_trips,
                required_trips=self._config.required_consecutive_trips(),
                measured=0.0, threshold=0.0,
                samples_in_window=self._episodes_seen,
                seconds_in_window=seconds_in_window,
            )

        # T1: safety-clamp rolling p95 > 2× baseline
        t1_decision = self._check_t1()
        if t1_decision is not None:
            self._consecutive_trips += 1
            return self._maybe_fire(
                reason="T1", measured=t1_decision[0], threshold=t1_decision[1],
                seconds_in_window=seconds_in_window,
            )

        # T2: action cos < 0.85
        t2_decision = self._check_t2()
        if t2_decision is not None:
            self._consecutive_trips += 1
            return self._maybe_fire(
                reason="T2", measured=t2_decision[0], threshold=t2_decision[1],
                seconds_in_window=seconds_in_window,
            )

        # T3: webhook violations > 5 in 5min
        t3_decision = self._check_t3(now=now)
        if t3_decision is not None:
            self._consecutive_trips += 1
            return self._maybe_fire(
                reason="T3", measured=float(t3_decision[0]),
                threshold=float(t3_decision[1]),
                seconds_in_window=seconds_in_window,
            )

        # No trip — reset consecutive counter
        self._consecutive_trips = 0
        return TripDecision(
            should_rollback=False, reason="in-window",
            consecutive_trips=0,
            required_trips=self._config.required_consecutive_trips(),
            measured=0.0, threshold=0.0,
            samples_in_window=self._episodes_seen,
            seconds_in_window=seconds_in_window,
        )

    # --- internals ---------------------------------------------------------

    def _is_window_closed(self, *, now: datetime) -> bool:
        if self._swap_at is None:
            return True
        elapsed_h = (now - self._swap_at).total_seconds() / 3600
        if elapsed_h >= self._config.window_duration_hours:
            return True
        if self._episodes_seen >= self._config.window_episode_count:
            return True
        return False

    def _check_t1(self) -> tuple[float, float] | None:
        if len(self._clamp_window) < 5:
            return None  # need a few samples
        threshold = (
            self._baseline_clamp_rate * self._config.t1_clamp_rate_relative
        )
        sorted_samples = sorted(self._clamp_window)
        idx_p95 = int(0.95 * (len(sorted_samples) - 1))
        p95 = sorted_samples[idx_p95]
        if p95 > threshold:
            return (p95, threshold)
        return None

    def _check_t2(self) -> tuple[float, float] | None:
        if len(self._cos_window) < 5:
            return None
        avg_cos = sum(self._cos_window) / len(self._cos_window)
        if avg_cos < self._config.t2_cos_min:
            return (avg_cos, self._config.t2_cos_min)
        return None

    def _check_t3(self, *, now: datetime) -> tuple[int, int] | None:
        self._prune_violation_window(now=now)
        total = sum(c for _, c in self._violation_window)
        if total > self._config.t3_violation_count_5min:
            return (total, self._config.t3_violation_count_5min)
        return None

    def _prune_violation_window(self, *, now: datetime) -> None:
        cutoff = now.timestamp() - T3_WINDOW_SECONDS
        while self._violation_window and self._violation_window[0][0].timestamp() < cutoff:
            self._violation_window.popleft()

    def _maybe_fire(
        self,
        *,
        reason: str,
        measured: float,
        threshold: float,
        seconds_in_window: float,
    ) -> TripDecision:
        required = self._config.required_consecutive_trips()
        should_fire = self._consecutive_trips >= required
        if should_fire:
            logger.warning(
                "post_swap_monitor.rollback_triggered reason=%s consecutive=%d "
                "required=%d measured=%s threshold=%s",
                reason, self._consecutive_trips, required, measured, threshold,
            )
        return TripDecision(
            should_rollback=should_fire, reason=reason,
            consecutive_trips=self._consecutive_trips,
            required_trips=required,
            measured=measured, threshold=threshold,
            samples_in_window=self._episodes_seen,
            seconds_in_window=seconds_in_window,
        )


__all__ = [
    "DEFAULT_T1_CLAMP_RATE_RELATIVE",
    "DEFAULT_T2_COS_MIN",
    "DEFAULT_T3_VIOLATION_COUNT_5MIN",
    "MonitorConfig",
    "PostSwapMonitor",
    "T3_WINDOW_SECONDS",
    "TripDecision",
    "WINDOW_DURATION_HOURS",
    "WINDOW_EPISODE_COUNT",
]
