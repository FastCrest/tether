"""Per-policy circuit-breaker for 2-policy A/B serve mode.

Per ADR 2026-04-25-policy-versioning-architecture Day 8: tracks
consecutive crash counts per policy slot, decides when one slot
should be drained (route 100% to the other) vs full server-degraded
state.

Single-policy mode: behaves exactly like the legacy single
`server.consecutive_crash_count` counter (all increments hit the
"prod" slot; threshold flips full degraded).

2-policy mode (slots 'a' + 'b'):
- One slot exceeds threshold: emit `inc_model_swap(from=<bad>, to=<good>)`,
  log warning, signal `should_drain(<bad>)`. Caller flips the router
  split to 100% on the surviving slot.
- BOTH slots exceed threshold: full server `degraded` state (existing
  legacy behavior). Both policies are contributing errors -- the
  problem isn't slot-specific.

Pure state primitive -- no I/O, no asyncio. Caller drives observations
via record_crash() / record_clean() and queries via verdict().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


# Bounded enum of verdicts. Surfaced in caller telemetry + audit logs.
Verdict = Literal["healthy", "drain-a", "drain-b", "degraded"]
ALL_VERDICTS: tuple[str, ...] = ("healthy", "drain-a", "drain-b", "degraded")


@dataclass(frozen=True)
class CrashTrackerVerdict:
    """Frozen verdict from PolicyCrashTracker.verdict()."""

    verdict: str  # one of ALL_VERDICTS
    crash_counts: dict[str, int]  # per-slot counter snapshot
    threshold: int
    reason: str  # operator-readable explanation

    def __post_init__(self) -> None:
        if self.verdict not in ALL_VERDICTS:
            raise ValueError(
                f"verdict must be one of {ALL_VERDICTS}, got {self.verdict!r}"
            )
        if self.threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {self.threshold}")

    @property
    def should_degrade(self) -> bool:
        """True when the verdict requires full server-degraded state."""
        return self.verdict == "degraded"

    @property
    def slot_to_drain(self) -> str | None:
        """Which slot to drain (route 0% to), or None when no drain needed."""
        if self.verdict == "drain-a":
            return "a"
        if self.verdict == "drain-b":
            return "b"
        return None


class PolicyCrashTracker:
    """Per-slot consecutive-crash counter + verdict primitive.

    Lifecycle:
        tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
        # in /act handler post-result:
        if result_was_error:
            tracker.record_crash(slot="a")
        else:
            tracker.record_clean(slot="a")
        verdict = tracker.verdict()
        if verdict.should_degrade:
            server.health_state = "degraded"
        elif verdict.slot_to_drain:
            policy_router.set_split_to_zero_for(verdict.slot_to_drain)

    Single-policy mode usage:
        tracker = PolicyCrashTracker(slots=("prod",), threshold=5)
        # all crashes hit "prod"; verdict 'degraded' on threshold (matches
        # legacy single-counter behavior)
    """

    __slots__ = ("_slots", "_threshold", "_counts")

    def __init__(self, *, slots: tuple[str, ...], threshold: int):
        if not slots:
            raise ValueError("slots must be non-empty")
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        if len(set(slots)) != len(slots):
            raise ValueError(f"slots must be unique, got {slots}")
        self._slots: tuple[str, ...] = tuple(slots)
        self._threshold = int(threshold)
        self._counts: dict[str, int] = {s: 0 for s in slots}

    @property
    def slots(self) -> tuple[str, ...]:
        return self._slots

    @property
    def threshold(self) -> int:
        return self._threshold

    def crash_count(self, slot: str) -> int:
        """Per-slot counter. KeyError on unknown slot (loud-fail per
        CLAUDE.md no-band-aid principle -- caller passing wrong slot
        is a bug, not a runtime degraded state)."""
        if slot not in self._counts:
            raise KeyError(
                f"slot {slot!r} not in tracker; known slots: {self._slots}"
            )
        return self._counts[slot]

    def record_crash(self, *, slot: str) -> None:
        """Increment the per-slot consecutive-crash counter."""
        if slot not in self._counts:
            raise KeyError(
                f"slot {slot!r} not in tracker; known slots: {self._slots}"
            )
        self._counts[slot] += 1
        logger.debug(
            "policy_crash_tracker.crash slot=%s count=%d threshold=%d",
            slot, self._counts[slot], self._threshold,
        )

    def record_clean(self, *, slot: str) -> None:
        """Reset the per-slot counter on a clean response."""
        if slot not in self._counts:
            raise KeyError(
                f"slot {slot!r} not in tracker; known slots: {self._slots}"
            )
        self._counts[slot] = 0

    def reset(self, *, slot: str | None = None) -> None:
        """Reset one slot's counter, or ALL slots when slot=None.
        Used after a manual operator intervention or after a successful
        rollback completes."""
        if slot is None:
            for s in self._slots:
                self._counts[s] = 0
            return
        if slot not in self._counts:
            raise KeyError(
                f"slot {slot!r} not in tracker; known slots: {self._slots}"
            )
        self._counts[slot] = 0

    def verdict(self) -> CrashTrackerVerdict:
        """Compute the current verdict from per-slot counters.

        Single-policy mode (one slot): exceeds threshold -> degraded.
        2-policy mode (two slots a + b):
            - Both exceed -> degraded.
            - a exceeds, b healthy -> drain-a (route 100% to b).
            - b exceeds, a healthy -> drain-b (route 100% to a).
            - Neither exceeds -> healthy.
        """
        snapshot = dict(self._counts)
        exceeders = [s for s, c in snapshot.items() if c >= self._threshold]

        if not exceeders:
            return CrashTrackerVerdict(
                verdict="healthy",
                crash_counts=snapshot,
                threshold=self._threshold,
                reason="all slots below threshold",
            )

        # 2-policy single-slot drain logic. Only valid when slots == {a, b}.
        if (
            len(self._slots) == 2
            and set(self._slots) == {"a", "b"}
            and len(exceeders) == 1
        ):
            bad = exceeders[0]
            return CrashTrackerVerdict(
                verdict=f"drain-{bad}",  # type: ignore[arg-type]
                crash_counts=snapshot,
                threshold=self._threshold,
                reason=(
                    f"slot {bad!r} exceeded threshold {self._threshold} "
                    f"(count={snapshot[bad]}); route 100% to surviving slot"
                ),
            )

        # All other "at least one exceeded" cases -> degraded.
        # Single-policy: the one slot exceeded.
        # 2-policy: both slots exceeded (problem isn't slot-specific).
        return CrashTrackerVerdict(
            verdict="degraded",
            crash_counts=snapshot,
            threshold=self._threshold,
            reason=(
                f"slots {sorted(exceeders)!r} exceeded threshold "
                f"{self._threshold}; full server degraded"
            ),
        )


__all__ = [
    "ALL_VERDICTS",
    "CrashTrackerVerdict",
    "PolicyCrashTracker",
    "Verdict",
]
