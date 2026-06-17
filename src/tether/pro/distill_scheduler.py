"""Pro-tier distill trigger scheduler.

Per ADR 2026-04-25-self-distilling-serve-architecture: every N hours
(or when the customer's traffic distribution drifts) kick a distill
job against the customer's parquet data. Phase 1 ships the trigger
PRIMITIVE; the actual Modal kick + result-fanout wiring lands Day 4+.

Trigger modes (all map to a single `should_kick()` decision):
- `"nightly"` — fires once per UTC day at a chosen hour
- `"cron:<spec>"` — fires per a 5-field cron spec (e.g., "0 3 * * *")
- `"samples:N"` — fires when N new collected samples accumulated since
  last kick
- `"quality-drop"` — fires when a watched metric (e.g., task-success
  rate, safety-clamp rate) crosses a threshold
- `"manual"` — never fires automatically; only `kick_now()` triggers

The scheduler is PURE — pass it the current time + sample count + the
last-kick state, get back a `KickDecision`. State (last_kick_at,
samples_at_last_kick) is owned by the caller and persisted to disk
across restarts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Literal

logger = logging.getLogger(__name__)


# Bounded enum of trigger modes. Stable across minor releases — surfaced
# in CLI flags + tether.yaml + telemetry labels.
TriggerMode = Literal["nightly", "cron", "samples", "quality-drop", "manual"]
ALL_TRIGGER_MODES: tuple[str, ...] = (
    "nightly", "cron", "samples", "quality-drop", "manual",
)


# Default hour-of-day (UTC) for the nightly trigger. 03:00 UTC ≈ 23:00
# US-Eastern / 04:00 CET / 12:00 JST — minimal cross-timezone disruption.
DEFAULT_NIGHTLY_UTC_HOUR = 3

# Default sample-count threshold for the "samples:N" trigger when no
# explicit N is given.
DEFAULT_SAMPLES_THRESHOLD = 1000

# Minimum gap between kicks (seconds) — protects against accidental
# re-trigger storms (e.g., quality-drop oscillating around the
# threshold). Operator-tunable.
DEFAULT_MIN_KICK_GAP_S = 3600  # 1 hour


@dataclass(frozen=True)
class KickDecision:
    """The output of `should_kick()` — used for logging + metric labels.
    Frozen so caller can pass it around without worrying about mutation."""

    kick: bool
    reason: str  # bounded enum: "nightly" | "cron" | "samples" | "quality-drop" |
                 # "manual" | "min-gap" | "no-trigger"
    samples_since_last_kick: int
    seconds_since_last_kick: float
    next_kick_estimated_at: str | None  # ISO 8601 or None when not derivable


@dataclass(frozen=True)
class SchedulerConfig:
    """Frozen scheduler configuration. Built once at startup; passed
    verbatim to `should_kick()`."""

    mode: str  # one of ALL_TRIGGER_MODES
    nightly_utc_hour: int = DEFAULT_NIGHTLY_UTC_HOUR
    samples_threshold: int = DEFAULT_SAMPLES_THRESHOLD
    cron_spec: str = ""  # for mode="cron"; minimal subset (Phase 1)
    min_kick_gap_s: float = DEFAULT_MIN_KICK_GAP_S
    quality_drop_threshold: float = 0.05  # 5pp drop default (relative to last_kick value)

    def __post_init__(self) -> None:
        if self.mode not in ALL_TRIGGER_MODES:
            raise ValueError(
                f"mode must be one of {ALL_TRIGGER_MODES}, got {self.mode!r}"
            )
        if not (0 <= self.nightly_utc_hour <= 23):
            raise ValueError(
                f"nightly_utc_hour must be in [0, 23], got {self.nightly_utc_hour}"
            )
        if self.samples_threshold < 1:
            raise ValueError(
                f"samples_threshold must be >= 1, got {self.samples_threshold}"
            )
        if self.min_kick_gap_s < 0:
            raise ValueError(
                f"min_kick_gap_s must be >= 0, got {self.min_kick_gap_s}"
            )
        if not (0.0 < self.quality_drop_threshold < 1.0):
            raise ValueError(
                f"quality_drop_threshold must be in (0, 1), got "
                f"{self.quality_drop_threshold}"
            )
        if self.mode == "cron" and not self.cron_spec:
            raise ValueError("cron_spec must be non-empty when mode='cron'")


@dataclass(frozen=True)
class SchedulerState:
    """Persisted across restarts (Day 5+ wiring saves to ~/.tether/distill_state.json).
    Phase 1: in-memory only; the caller passes the prior state on each
    should_kick() call."""

    last_kick_at: str | None  # ISO 8601 UTC; None = never
    samples_at_last_kick: int
    quality_at_last_kick: float | None  # Used for "quality-drop" mode

    def __post_init__(self) -> None:
        if self.samples_at_last_kick < 0:
            raise ValueError(
                f"samples_at_last_kick must be >= 0, got "
                f"{self.samples_at_last_kick}"
            )


class DistillScheduler:
    """Pure trigger-decision primitive. Stateless — feed it the current
    config + state + observation; receive a KickDecision. No I/O, no
    persistence, no asyncio.

    Usage:
        sch = DistillScheduler(config=SchedulerConfig(mode="nightly"))
        decision = sch.should_kick(
            state=SchedulerState(last_kick_at="2026-04-24T03:00:00Z",
                                 samples_at_last_kick=500,
                                 quality_at_last_kick=None),
            current_samples=12345,
            current_quality=None,
            now=None,  # defaults to datetime.now(utc)
        )
        if decision.kick:
            # Caller fires the actual distill job + persists new state
            ...
    """

    __slots__ = ("_config",)

    def __init__(self, config: SchedulerConfig):
        self._config = config

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def should_kick(
        self,
        *,
        state: SchedulerState,
        current_samples: int,
        current_quality: float | None = None,
        now: datetime | None = None,
    ) -> KickDecision:
        """Decide whether to kick a distill job. Pure function: same
        inputs always produce same KickDecision."""
        if current_samples < 0:
            raise ValueError(
                f"current_samples must be >= 0, got {current_samples}"
            )

        now = now or datetime.now(timezone.utc)
        last_at = _parse_iso(state.last_kick_at)
        seconds_since = (
            (now - last_at).total_seconds() if last_at is not None
            else float("inf")
        )
        samples_since = max(0, current_samples - state.samples_at_last_kick)

        # Min-gap protection — never kick within the cooldown window
        # regardless of trigger.
        if last_at is not None and seconds_since < self._config.min_kick_gap_s:
            return KickDecision(
                kick=False, reason="min-gap",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=(
                    last_at + timedelta(seconds=self._config.min_kick_gap_s)
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )

        if self._config.mode == "manual":
            return KickDecision(
                kick=False, reason="manual",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )

        if self._config.mode == "nightly":
            return self._decide_nightly(
                now=now, last_at=last_at, samples_since=samples_since,
                seconds_since=seconds_since,
            )

        if self._config.mode == "samples":
            if samples_since >= self._config.samples_threshold:
                return KickDecision(
                    kick=True, reason="samples",
                    samples_since_last_kick=samples_since,
                    seconds_since_last_kick=seconds_since,
                    next_kick_estimated_at=None,
                )
            return KickDecision(
                kick=False, reason="no-trigger",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )

        if self._config.mode == "quality-drop":
            return self._decide_quality_drop(
                state=state, current_quality=current_quality,
                samples_since=samples_since, seconds_since=seconds_since,
            )

        if self._config.mode == "cron":
            return self._decide_cron(
                now=now, last_at=last_at,
                samples_since=samples_since, seconds_since=seconds_since,
            )

        # Unreachable — config validation rejects unknown modes
        return KickDecision(
            kick=False, reason="no-trigger",
            samples_since_last_kick=samples_since,
            seconds_since_last_kick=seconds_since,
            next_kick_estimated_at=None,
        )

    def _decide_nightly(
        self, *, now: datetime, last_at: datetime | None,
        samples_since: int, seconds_since: float,
    ) -> KickDecision:
        # Fire at the configured hour, exactly once per UTC day.
        nightly_h = self._config.nightly_utc_hour
        target = now.replace(hour=nightly_h, minute=0, second=0, microsecond=0)
        # If we're past today's target hour AND haven't kicked yet today, fire.
        if now >= target and (last_at is None or last_at < target):
            return KickDecision(
                kick=True, reason="nightly",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=(
                    target + timedelta(days=1)
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )
        # Otherwise compute the next nightly target.
        next_target = (
            target if now < target else target + timedelta(days=1)
        )
        return KickDecision(
            kick=False, reason="no-trigger",
            samples_since_last_kick=samples_since,
            seconds_since_last_kick=seconds_since,
            next_kick_estimated_at=next_target.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )

    def _decide_quality_drop(
        self, *, state: SchedulerState, current_quality: float | None,
        samples_since: int, seconds_since: float,
    ) -> KickDecision:
        # Fire when current quality has dropped by more than threshold
        # vs the last kick's recorded quality.
        if current_quality is None or state.quality_at_last_kick is None:
            return KickDecision(
                kick=False, reason="no-trigger",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )
        drop = state.quality_at_last_kick - current_quality
        if drop >= self._config.quality_drop_threshold:
            return KickDecision(
                kick=True, reason="quality-drop",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )
        return KickDecision(
            kick=False, reason="no-trigger",
            samples_since_last_kick=samples_since,
            seconds_since_last_kick=seconds_since,
            next_kick_estimated_at=None,
        )

    def _decide_cron(
        self, *, now: datetime, last_at: datetime | None,
        samples_since: int, seconds_since: float,
    ) -> KickDecision:
        # Phase 1: minimal cron support — only `H * * * *` (hour of day,
        # everything else wildcards). Phase 1.5 wires a real cron parser.
        spec = self._config.cron_spec.split()
        if len(spec) != 5:
            return KickDecision(
                kick=False, reason="no-trigger",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )
        # Only the hour field is interpreted; minute/day/month/dow must be
        # wildcards in Phase 1. Reject more complex specs loudly.
        try:
            target_hour = int(spec[1])
        except ValueError:
            return KickDecision(
                kick=False, reason="no-trigger",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )
        if not (0 <= target_hour <= 23):
            return KickDecision(
                kick=False, reason="no-trigger",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=None,
            )
        target = now.replace(
            hour=target_hour, minute=0, second=0, microsecond=0,
        )
        if now >= target and (last_at is None or last_at < target):
            return KickDecision(
                kick=True, reason="cron",
                samples_since_last_kick=samples_since,
                seconds_since_last_kick=seconds_since,
                next_kick_estimated_at=(
                    target + timedelta(days=1)
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )
        next_target = (
            target if now < target else target + timedelta(days=1)
        )
        return KickDecision(
            kick=False, reason="no-trigger",
            samples_since_last_kick=samples_since,
            seconds_since_last_kick=seconds_since,
            next_kick_estimated_at=next_target.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = [
    "ALL_TRIGGER_MODES",
    "DEFAULT_NIGHTLY_UTC_HOUR",
    "DEFAULT_SAMPLES_THRESHOLD",
    "DEFAULT_MIN_KICK_GAP_S",
    "DistillScheduler",
    "KickDecision",
    "SchedulerConfig",
    "SchedulerState",
]
