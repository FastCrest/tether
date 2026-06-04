"""2-policy dispatcher -- composes policy_router + policy + crash_tracker.

Per ADR 2026-04-25-policy-versioning-architecture: ships a focused
2-policy dispatch primitive that wraps two TetherServer instances
(or any object exposing a `predict_async` callable) and routes /act
requests via PolicyRouter. Crash counts go through PolicyCrashTracker;
when one slot drains, the dispatcher temporarily routes 100% to the
surviving slot.

This is a LIBRARY-level primitive -- it doesn't itself bind FastAPI
or own the HTTP surface. Phase 1 follow-up wires it into
server.start() lifespan so `tether serve --policy-a ... --policy-b ...`
actually serves two policies. This commit ships the dispatcher +
tests; the FastAPI integration + /metrics + /health aggregation
follow.

Lifecycle:
    dispatcher = TwoPolicyDispatcher(
        policy_a=policy_a_inference,
        policy_b=policy_b_inference,
        split_a_percent=80,
        crash_threshold=5,
    )
    result, decision = await dispatcher.predict(
        request=req, episode_id="ep_xyz", request_id="req_42",
    )
    # result is whatever the per-policy predict_async returned.
    # decision is a TwoPolicyDecision capturing slot + crash-count snapshot.

Pure composition -- no I/O of its own; the per-policy predict_async
does the actual model call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from tether.runtime.policy import Policy
from tether.runtime.policy_crash_tracker import (
    CrashTrackerVerdict, PolicyCrashTracker,
)
from tether.runtime.policy_router import (
    ALL_SLOTS, PolicyRouter, RoutingDecision, SlotName,
)

logger = logging.getLogger(__name__)


# Type alias for the per-policy predict callable injected at construction.
# Production wires this to TetherServer.predict_from_base64_async or
# equivalent; tests stub it. Returns whatever the inference produces;
# dispatcher doesn't care about shape.
PredictCallable = Callable[[Any], Awaitable[Any]]


_PR = TypeVar("_PR")  # per-policy predict result type


@dataclass(frozen=True)
class TwoPolicyDecision:
    """Frozen decision record from one TwoPolicyDispatcher.predict() call.

    Surfaced to the caller for record-replay logging + Prometheus
    label emission.
    """

    slot: SlotName
    routing_key: str
    degraded_routing: bool  # True when episode_id was missing
    cached: bool
    crash_verdict: str  # one of ALL_VERDICTS
    crash_counts: dict[str, int]  # snapshot post-record


class TwoPolicyDispatcher(Generic[_PR]):
    """Episode-sticky 2-policy /act dispatcher.

    Args:
        policy_a / policy_b: either TetherServer instances OR test stubs
            implementing the Policy Protocol from policy_router.py
            (model_id + model_hash). The dispatcher uses these for
            routing meta only -- the actual call goes through the
            predict_a / predict_b callables.
        predict_a / predict_b: async callables that take the inbound
            /act request and return whatever the per-policy inference
            produces. Production binds to
            policy_a.predict_from_base64_async (or analogous);
            tests stub.
        split_a_percent: integer percent of episodes routed to slot A
            in [0, 100]. Default 50 for clean A/B.
        crash_threshold: per-slot crash counter threshold; when a slot
            exceeds, it's drained (route 100% to the survivor). Default
            5 matches the legacy single-policy default.
        is_error_response_fn: callable(result) -> bool that decides
            whether a returned result counts as a "crash" (drives the
            tracker). Default checks `isinstance(result, dict) and
            "error" in result` (matches the legacy server.py path).
    """

    def __init__(
        self,
        *,
        policy_a: Policy,
        policy_b: Policy,
        predict_a: PredictCallable,
        predict_b: PredictCallable,
        split_a_percent: int = 50,
        crash_threshold: int = 5,
        is_error_response_fn: Callable[[Any], bool] | None = None,
    ):
        if policy_a.slot != "a":
            raise ValueError(
                f"policy_a.slot must be 'a', got {policy_a.slot!r}"
            )
        if policy_b.slot != "b":
            raise ValueError(
                f"policy_b.slot must be 'b', got {policy_b.slot!r}"
            )
        if not (0 <= split_a_percent <= 100):
            raise ValueError(
                f"split_a_percent must be in [0, 100], got {split_a_percent}"
            )
        if crash_threshold < 1:
            raise ValueError(
                f"crash_threshold must be >= 1, got {crash_threshold}"
            )

        self._policies: dict[SlotName, Policy] = {"a": policy_a, "b": policy_b}
        self._predict: dict[SlotName, PredictCallable] = {
            "a": predict_a, "b": predict_b,
        }
        self._original_split = int(split_a_percent)
        self._effective_split = int(split_a_percent)
        self._router = PolicyRouter(
            policies=self._policies, split_a_percent=split_a_percent,
        )
        self._tracker = PolicyCrashTracker(
            slots=ALL_SLOTS, threshold=crash_threshold,
        )
        self._is_error_response_fn = (
            is_error_response_fn or _default_is_error_response
        )

    @property
    def split_a_percent(self) -> int:
        """Currently-effective split (may differ from original after a
        drain decision). Surfaced for observability."""
        return self._effective_split

    @property
    def policies(self) -> dict[SlotName, Policy]:
        return dict(self._policies)

    def crash_counts(self) -> dict[str, int]:
        """Snapshot of per-slot crash counters."""
        return {s: self._tracker.crash_count(s) for s in ALL_SLOTS}

    def crash_verdict(self) -> CrashTrackerVerdict:
        """Current verdict (healthy / drain-a / drain-b / degraded)."""
        return self._tracker.verdict()

    async def predict(
        self,
        *,
        request: Any,
        episode_id: str | None,
        request_id: str,
    ) -> tuple[_PR, TwoPolicyDecision]:
        """Route + dispatch one /act request.

        Returns (result, decision) where result is whatever the per-policy
        predict callable returned, and decision is a TwoPolicyDecision
        describing the routing + crash-tracker snapshot.

        Raises whatever the underlying predict callable raises (caller
        handles -- typically the FastAPI handler converts to 500).
        """
        # Honor any active drain decision (route 100% to surviving slot).
        verdict_pre = self._tracker.verdict()
        forced_slot = verdict_pre.slot_to_drain  # the slot to NOT use
        if forced_slot is not None:
            # Route to the OTHER slot.
            slot: SlotName = "b" if forced_slot == "a" else "a"
            routing = RoutingDecision(
                slot=slot, routing_key=episode_id or request_id,
                degraded=episode_id is None, cached=False,
            )
        else:
            routing = self._router.route(
                episode_id=episode_id, request_id=request_id,
            )

        slot = routing.slot
        try:
            result = await self._predict[slot](request)
        except Exception:
            self._tracker.record_crash(slot=slot)
            verdict_post = self._tracker.verdict()
            logger.error(
                "two_policy.predict_raise slot=%s crash_count=%d verdict=%s",
                slot, self._tracker.crash_count(slot), verdict_post.verdict,
            )
            raise

        # Successful return -- check if it's an error-response (counts as crash)
        if self._is_error_response_fn(result):
            self._tracker.record_crash(slot=slot)
        else:
            self._tracker.record_clean(slot=slot)

        verdict_post = self._tracker.verdict()

        decision = TwoPolicyDecision(
            slot=slot,
            routing_key=routing.routing_key,
            degraded_routing=routing.degraded,
            cached=routing.cached,
            crash_verdict=verdict_post.verdict,
            crash_counts=self.crash_counts(),
        )
        return result, decision  # type: ignore[return-value]

    def reset_crash_counters(self, *, slot: str | None = None) -> None:
        """Reset per-slot crash counters (operator intervention).
        Used after a manual investigation or when restarting the
        affected policy."""
        self._tracker.reset(slot=slot)


def _default_is_error_response(result: Any) -> bool:
    """Default error detector: result is a dict with 'error' key.
    Matches the legacy server.py crash-counter convention."""
    return isinstance(result, dict) and "error" in result


__all__ = [
    "PredictCallable",
    "TwoPolicyDecision",
    "TwoPolicyDispatcher",
]
