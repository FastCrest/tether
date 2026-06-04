"""Tests for src/tether/runtime/policy_router.py — episode-sticky A/B router.

Covers: deterministic hashing, stickiness, fallback-to-request-id degraded
mode, split boundaries (0/100, 50/50, 80/20, 99/1), LRU eviction,
construction invariants, decision audit fields.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import pytest

from tether.runtime.policy_router import (
    ALL_SLOTS,
    PolicyRouter,
    RoutingDecision,
    _hash_to_bucket,
    _slot_for_split,
)


@dataclass(frozen=True)
class FakePolicy:
    model_id: str
    model_hash: str


def _mk_router(split: int = 50, cache_size: int = 10_000) -> PolicyRouter:
    return PolicyRouter(
        policies={
            "a": FakePolicy(model_id="pi0-v1", model_hash="abc123def456"),
            "b": FakePolicy(model_id="pi0-v2", model_hash="fed654cba321"),
        },
        split_a_percent=split,
        cache_size=cache_size,
    )


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_rejects_missing_slot_a():
    with pytest.raises(ValueError, match="slots"):
        PolicyRouter(
            policies={"b": FakePolicy("x", "y")},  # type: ignore[dict-item]
            split_a_percent=50,
        )


def test_rejects_extra_slot():
    with pytest.raises(ValueError, match="slots"):
        PolicyRouter(
            policies={"a": FakePolicy("x", "y"), "b": FakePolicy("p", "q"), "c": FakePolicy("r", "s")},  # type: ignore[dict-item]
            split_a_percent=50,
        )


def test_rejects_negative_split():
    with pytest.raises(ValueError, match="split_a_percent"):
        _mk_router(split=-1)


def test_rejects_split_above_100():
    with pytest.raises(ValueError, match="split_a_percent"):
        _mk_router(split=101)


def test_accepts_boundary_splits():
    _mk_router(split=0)
    _mk_router(split=100)


def test_rejects_zero_cache_size():
    with pytest.raises(ValueError, match="cache_size"):
        _mk_router(cache_size=0)


def test_accepts_cache_size_one():
    _mk_router(cache_size=1)


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------


def test_hash_is_deterministic():
    assert _hash_to_bucket("episode_42") == _hash_to_bucket("episode_42")


def test_hash_differs_for_different_keys():
    assert _hash_to_bucket("ep_a") != _hash_to_bucket("ep_b")


def test_hash_in_bucket_range():
    for i in range(1000):
        b = _hash_to_bucket(f"episode_{i}")
        assert 0 <= b < 100


def test_slot_for_split_at_boundary_zero_all_b():
    # split_a_percent = 0 → all traffic to 'b' regardless of bucket
    for bucket in range(100):
        assert _slot_for_split(bucket, 0) == "b"


def test_slot_for_split_at_boundary_hundred_all_a():
    # split_a_percent = 100 → all traffic to 'a'
    for bucket in range(100):
        assert _slot_for_split(bucket, 100) == "a"


def test_slot_for_split_eighty_twenty():
    # split_a_percent = 80 → buckets 0..79 → a, 80..99 → b
    for b in range(80):
        assert _slot_for_split(b, 80) == "a"
    for b in range(80, 100):
        assert _slot_for_split(b, 80) == "b"


# ---------------------------------------------------------------------------
# Routing behavior — happy path
# ---------------------------------------------------------------------------


def test_route_with_episode_id_is_sticky():
    router = _mk_router(split=50)
    first = router.route(episode_id="ep_42", request_id="req_1")
    second = router.route(episode_id="ep_42", request_id="req_2")
    third = router.route(episode_id="ep_42", request_id="req_3")
    assert first.slot == second.slot == third.slot
    assert first.cached is False
    assert second.cached is True
    assert third.cached is True


def test_route_different_episodes_may_differ():
    """With split=50, we expect roughly half the episodes to go to each slot."""
    router = _mk_router(split=50)
    slots = set()
    for i in range(100):
        d = router.route(episode_id=f"ep_{i}", request_id=f"req_{i}")
        slots.add(d.slot)
    # At least one of each slot should appear across 100 episodes.
    assert slots == {"a", "b"}


def test_route_uses_episode_id_as_routing_key():
    router = _mk_router()
    d = router.route(episode_id="ep_x", request_id="req_y")
    assert d.routing_key == "ep_x"
    assert d.degraded is False


def test_route_returns_cached_flag_true_on_repeat():
    router = _mk_router()
    d1 = router.route(episode_id="ep_1", request_id="r1")
    d2 = router.route(episode_id="ep_1", request_id="r2")
    assert d1.cached is False
    assert d2.cached is True


def test_route_is_deterministic_across_router_instances():
    """Same episode_id + same split must produce the same slot in a fresh router."""
    r1 = _mk_router(split=80)
    r2 = _mk_router(split=80)
    for i in range(50):
        ep = f"ep_{i}"
        assert r1.route(episode_id=ep, request_id="x").slot == \
               r2.route(episode_id=ep, request_id="y").slot


# ---------------------------------------------------------------------------
# Degraded mode (no episode_id)
# ---------------------------------------------------------------------------


def test_route_falls_back_to_request_id_when_episode_missing():
    router = _mk_router()
    d = router.route(episode_id=None, request_id="req_abc")
    assert d.degraded is True
    assert d.routing_key == "req_abc"
    assert d.slot in ALL_SLOTS


def test_route_falls_back_when_episode_empty_string():
    router = _mk_router()
    d = router.route(episode_id="", request_id="req_z")
    assert d.degraded is True
    assert d.routing_key == "req_z"


def test_degraded_mode_logs_once_per_process(caplog):
    caplog.set_level(logging.WARNING)
    router = _mk_router()
    router.route(episode_id=None, request_id="r1")
    router.route(episode_id=None, request_id="r2")
    router.route(episode_id=None, request_id="r3")
    # Only one warning should be emitted, matching the substring.
    matching = [r for r in caplog.records if "policy_router.degraded_mode" in r.message]
    assert len(matching) == 1


def test_degraded_mode_never_caches():
    router = _mk_router()
    router.route(episode_id=None, request_id="r1")
    router.route(episode_id=None, request_id="r2")
    assert router.cache_size() == 0


# ---------------------------------------------------------------------------
# Split ratios at scale
# ---------------------------------------------------------------------------


def test_split_50_yields_approximately_half_to_each_slot():
    router = _mk_router(split=50)
    counts = {"a": 0, "b": 0}
    for i in range(2000):
        d = router.route(episode_id=f"ep_{i}", request_id=f"r_{i}")
        counts[d.slot] += 1
    ratio_a = counts["a"] / 2000
    # With SHA-256, 50/50 split on 2000 samples should be within ~3% of 0.5.
    assert 0.47 < ratio_a < 0.53


def test_split_80_20_yields_approximately_80_20():
    router = _mk_router(split=80)
    counts = {"a": 0, "b": 0}
    for i in range(2000):
        d = router.route(episode_id=f"ep_{i}", request_id=f"r_{i}")
        counts[d.slot] += 1
    ratio_a = counts["a"] / 2000
    assert 0.77 < ratio_a < 0.83


def test_split_0_all_to_b():
    router = _mk_router(split=0)
    for i in range(500):
        d = router.route(episode_id=f"ep_{i}", request_id=f"r_{i}")
        assert d.slot == "b"


def test_split_100_all_to_a():
    router = _mk_router(split=100)
    for i in range(500):
        d = router.route(episode_id=f"ep_{i}", request_id=f"r_{i}")
        assert d.slot == "a"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_cache_size_bounded():
    router = _mk_router(cache_size=5)
    for i in range(20):
        router.route(episode_id=f"ep_{i}", request_id=f"r_{i}")
    assert router.cache_size() == 5


def test_oldest_episode_evicted_when_full():
    router = _mk_router(cache_size=3)
    router.route(episode_id="ep_oldest", request_id="r")
    router.route(episode_id="ep_2", request_id="r")
    router.route(episode_id="ep_3", request_id="r")
    router.route(episode_id="ep_newest", request_id="r")  # evicts ep_oldest
    assert router.get_cached_slot("ep_oldest") is None
    assert router.get_cached_slot("ep_newest") is not None


def test_recently_used_episode_moves_to_end_and_avoids_eviction():
    router = _mk_router(cache_size=3)
    router.route(episode_id="ep_1", request_id="r")
    router.route(episode_id="ep_2", request_id="r")
    router.route(episode_id="ep_3", request_id="r")
    # Touch ep_1 to mark it recently used.
    router.route(episode_id="ep_1", request_id="r_again")
    router.route(episode_id="ep_4", request_id="r")  # evicts ep_2 (oldest-not-touched)
    assert router.get_cached_slot("ep_1") is not None
    assert router.get_cached_slot("ep_2") is None
    assert router.get_cached_slot("ep_4") is not None


# ---------------------------------------------------------------------------
# Introspection + accessors
# ---------------------------------------------------------------------------


def test_get_policy_returns_bound_policy():
    router = _mk_router()
    p_a = router.get_policy("a")
    p_b = router.get_policy("b")
    assert p_a.model_id == "pi0-v1"
    assert p_b.model_id == "pi0-v2"
    assert p_a is not p_b


def test_get_policy_raises_on_bad_slot():
    router = _mk_router()
    with pytest.raises(KeyError):
        router.get_policy("c")  # type: ignore[arg-type]


def test_policies_iterator_yields_both():
    router = _mk_router()
    got = dict(router.policies())
    assert set(got.keys()) == {"a", "b"}


def test_split_percent_exposed():
    router = _mk_router(split=73)
    assert router.split_a_percent == 73


def test_slots_property_returns_canonical_tuple():
    router = _mk_router()
    assert router.slots == ("a", "b")


# ---------------------------------------------------------------------------
# Routing decision payload
# ---------------------------------------------------------------------------


def test_decision_is_frozen_dataclass():
    router = _mk_router()
    d = router.route(episode_id="ep_x", request_id="r")
    with pytest.raises(AttributeError):
        d.slot = "b"  # type: ignore[misc]


def test_decision_fields_populated():
    router = _mk_router()
    d = router.route(episode_id="ep_y", request_id="req_z")
    assert isinstance(d, RoutingDecision)
    assert d.slot in ALL_SLOTS
    assert d.routing_key == "ep_y"
    assert d.degraded is False
    assert d.cached is False
