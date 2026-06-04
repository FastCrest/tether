"""Policy router for 2-policy A/B serve mode.

Ship gate: ADR 2026-04-25-policy-versioning-architecture.
Research:  features/01_serve/subfeatures/_ecosystem/policy-versioning/
           policy-versioning_research.md

Routes each /act request to one of two loaded policies based on a
deterministic hash of the request's `episode_id`. Decision is sticky
per episode — the first request of an episode decides the bucket, and
subsequent requests within the same episode use the same bucket. This
preserves the 9× episode-cache moat (which is per-policy: switching
mid-episode destroys the cached past_kv) and RTC carry-over state
(chunk N+1's denoise anchors to chunk N's trailing actions; cross-
policy carry-over produces out-of-distribution actions).

When `episode_id` is absent (a caller that hasn't adopted the episode
API), the router falls back to hashing `request_id`. This is a DEGRADED
mode: each request gets an independent decision, which means a client
that issues multiple requests without an episode_id during a real
episode will flip-flop between policies and trigger the same cache-
coherence + RTC discontinuity problems. Log loudly on the first occurrence
per process so operators notice before it bites them in production.

The router is pure: given (episode_id, request_id, split, policies), the
output is deterministic. State is held outside the router (LRU cache of
(episode_id → slot) decisions) so the router itself is test-friendly.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, Iterator, Literal, Mapping, Protocol, TypeVar

logger = logging.getLogger(__name__)

SlotName = Literal["a", "b"]
ALL_SLOTS: tuple[SlotName, ...] = ("a", "b")

# Max episodes to remember bucket decisions for. At 1 decision per episode
# and a typical customer doing 1 episode per minute, 10k slots gives ~7
# days of memory. LRU evicts the oldest when full — a 7-day-old episode
# almost certainly won't reappear, and if it does, it gets re-bucketed on
# the same deterministic hash (producing the same slot → no semantic drift).
_DEFAULT_CACHE_SIZE = 10_000


@dataclass(frozen=True)
class RoutingDecision:
    """The output of a single `route()` call.

    Fields:
        slot: which policy slot handles this request.
        routing_key: the actual string that was hashed (episode_id or request_id).
        degraded: True when fallen back to request_id because episode_id was
            missing. Operator signal — never affects the slot choice.
        cached: True when a prior request in the same episode already
            established this slot and we're reusing the decision.
    """
    slot: SlotName
    routing_key: str
    degraded: bool
    cached: bool


class Policy(Protocol):
    """Minimal interface the router requires from each policy.

    Production binds to a wrapper around `Pi05DecomposedInference` that
    holds the per-policy ActionGuard / EpisodeCache / RtcAdapter. Tests
    bind to a mock with just `model_id` + `model_hash`. The router itself
    never invokes the policy — dispatch happens one layer up (in the
    /act handler) after the router returns a slot.
    """

    @property
    def model_id(self) -> str:
        """Human-readable policy identifier (e.g. "pi0-libero-v1"); goes
        into `X-Tether-Model-Version` headers after combining with hash."""

    @property
    def model_hash(self) -> str:
        """16-hex SHA-256 prefix of the policy's model files; stable
        identity for record-replay JSONL trace + header response."""


_P = TypeVar("_P", bound=Policy)


def _hash_to_bucket(key: str) -> int:
    """SHA-256(key) → integer in [0, 100). Deterministic across platforms
    and Python restarts."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Take the first 8 bytes as a big-endian unsigned int and mod by 100.
    # The top 2^64 bits of SHA-256 have uniform distribution; the mod
    # bias for 2^64 / 100 is well below 1 in 10^18 and irrelevant here.
    n = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return n % 100


def _slot_for_split(bucket: int, split_a_percent: int) -> SlotName:
    """Given a 0-99 bucket and the % weight for slot A, return the chosen slot.

    split_a_percent = 80 means: buckets 0..79 → 'a', 80..99 → 'b'.
    Edge cases:
      split_a_percent = 0   → all traffic to 'b'
      split_a_percent = 100 → all traffic to 'a'
    Both are valid; they effectively turn the router into a single-policy
    passthrough while keeping the other policy loaded (shadow-staging mode).
    """
    return "a" if bucket < split_a_percent else "b"


class PolicyRouter(Generic[_P]):
    """Episode-sticky 2-policy router.

    Usage:

        router = PolicyRouter(
            policies={"a": policy_v1, "b": policy_v2},
            split_a_percent=80,
        )

        decision = router.route(episode_id="ep_abc", request_id="req_42")
        policy = router.get_policy(decision.slot)
        result = policy.inference.predict_action_chunk(...)

    The router is thread-safe for the typical FastAPI pattern: a single
    router instance shared across handlers, with `route()` called once per
    request. The LRU cache uses an `OrderedDict` under a mutex-free design
    — race conditions on concurrent first-time-for-episode inserts settle
    on the deterministic hash result, so even a lost update is correct.

    Args:
        policies: mapping of slot name → policy. Must contain BOTH "a" and
            "b" (partial routers are a bug — shipped-inert routers with
            only one policy are single-policy mode and should use the
            existing `server.policies = {"prod": ...}` code path).
        split_a_percent: integer percentage of traffic routed to slot 'a'
            in [0, 100]. Default 50 for a clean A/B split.
        cache_size: max sticky-episode entries held in memory. Defaults
            to 10k; bump for customers running long-running episodes or
            >10k episodes/day.
    """

    __slots__ = (
        "_policies",
        "_split_a_percent",
        "_cache",
        "_cache_size",
        "_degraded_warned",
    )

    def __init__(
        self,
        policies: Mapping[SlotName, _P],
        split_a_percent: int,
        *,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ):
        if set(policies.keys()) != set(ALL_SLOTS):
            raise ValueError(
                f"policies must have exactly slots {sorted(ALL_SLOTS)}; got "
                f"{sorted(policies.keys())}"
            )
        if not (0 <= split_a_percent <= 100):
            raise ValueError(
                f"split_a_percent must be in [0, 100], got {split_a_percent}"
            )
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        self._policies: dict[SlotName, _P] = dict(policies)
        self._split_a_percent = int(split_a_percent)
        self._cache: OrderedDict[str, SlotName] = OrderedDict()
        self._cache_size = int(cache_size)
        self._degraded_warned = False

    @property
    def split_a_percent(self) -> int:
        return self._split_a_percent

    @property
    def slots(self) -> tuple[SlotName, ...]:
        return ALL_SLOTS

    def get_policy(self, slot: SlotName) -> _P:
        """Return the policy object bound to `slot`. Raises KeyError on bad slot."""
        return self._policies[slot]

    def policies(self) -> Iterator[tuple[SlotName, _P]]:
        """Iterate over (slot, policy) pairs. Useful for prewarm, metric
        emission, and per-policy diagnostic endpoints."""
        yield from self._policies.items()

    def route(
        self,
        *,
        episode_id: str | None,
        request_id: str,
    ) -> RoutingDecision:
        """Decide which slot handles the request. Sticky per episode.

        Args:
            episode_id: the request's episode identifier, when present.
                Sticky-hash routing key.
            request_id: the request's unique identifier. Used as fallback
                when episode_id is None, and always returned in the
                RoutingDecision for audit log correlation.

        Returns:
            RoutingDecision describing the slot, the actual routing key
            that was hashed, and whether the decision came from the
            sticky cache or is a fresh hash.
        """
        if episode_id:
            cached_slot = self._cache.get(episode_id)
            if cached_slot is not None:
                # Touch to mark recently-used (LRU).
                self._cache.move_to_end(episode_id)
                return RoutingDecision(
                    slot=cached_slot,
                    routing_key=episode_id,
                    degraded=False,
                    cached=True,
                )
            # First time we see this episode — hash, remember, return.
            bucket = _hash_to_bucket(episode_id)
            slot = _slot_for_split(bucket, self._split_a_percent)
            self._cache[episode_id] = slot
            # Evict oldest when over capacity.
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return RoutingDecision(
                slot=slot,
                routing_key=episode_id,
                degraded=False,
                cached=False,
            )

        # Degraded path: no episode_id. Hash on request_id, log once per process.
        if not self._degraded_warned:
            self._degraded_warned = True
            logger.warning(
                "policy_router.degraded_mode request_id=%s — no episode_id "
                "provided. Per-request routing destroys episode-cache locality "
                "and causes RTC carry-over discontinuities. Callers should pass "
                "episode_id on every /act request in 2-policy mode.",
                request_id[:64],
            )
        bucket = _hash_to_bucket(request_id)
        slot = _slot_for_split(bucket, self._split_a_percent)
        return RoutingDecision(
            slot=slot,
            routing_key=request_id,
            degraded=True,
            cached=False,
        )

    # --- introspection -----------------------------------------------------

    def cache_size(self) -> int:
        """Number of sticky-episode decisions currently cached."""
        return len(self._cache)

    def get_cached_slot(self, episode_id: str) -> SlotName | None:
        """Return the cached slot for an episode_id without updating LRU order.
        None when the episode isn't cached. Useful for testing + debugging."""
        return self._cache.get(episode_id)


__all__ = [
    "ALL_SLOTS",
    "Policy",
    "PolicyRouter",
    "RoutingDecision",
    "SlotName",
]
