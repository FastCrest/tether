"""Day 9-10 integration tests: end-to-end composition of policy-versioning
substrate (router + Policy bundle + crash tracker + record-replay).

Per ADR 2026-04-25-policy-versioning-architecture: validates the
PRIMITIVES compose without needing the actual server.py 2-instance
load path (which lands in a follow-up). Tests use stub policies +
in-memory state so they run fast.

Coverage:
- Router routes deterministically by episode_id
- Sticky-per-episode behavior preserved across requests
- 2-policy split ratio matches --split flag
- Crash tracker drains the bad slot when one exceeds threshold
- Record-replay captures per-request routing decisions
- Headers carry per-policy meta when 2-policy mode active
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tether.runtime.policy import (
    DEFAULT_SINGLE_POLICY_SLOT,
    Policy,
    make_single_policy,
    validate_split_and_no_rtc,
)
from tether.runtime.policy_crash_tracker import PolicyCrashTracker
from tether.runtime.policy_router import PolicyRouter
from tether.runtime.record import RecordWriter


# ---------------------------------------------------------------------------
# Stub policy that satisfies the Policy Protocol from policy_router.py
# ---------------------------------------------------------------------------


@dataclass
class _StubPolicy:
    """Minimal Policy Protocol implementation for the router."""
    model_id: str
    model_hash: str
    crashes: int = 0
    cleans: int = 0


def _make_stub_policies() -> dict[str, _StubPolicy]:
    return {
        "a": _StubPolicy(model_id="pi0-libero-v1", model_hash="aaaaaaaaaaaaaaaa"),
        "b": _StubPolicy(model_id="pi0-libero-v2", model_hash="bbbbbbbbbbbbbbbb"),
    }


# ---------------------------------------------------------------------------
# Router + sticky-per-episode behavior
# ---------------------------------------------------------------------------


def test_router_sticky_per_episode():
    """Multiple requests within one episode -> same slot."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=50)
    decisions = [
        router.route(episode_id="ep_xyz", request_id=f"req_{i}")
        for i in range(20)
    ]
    slots = {d.slot for d in decisions}
    # All 20 requests in the same episode -> exactly one slot
    assert len(slots) == 1
    # First call is fresh, rest are cached
    assert not decisions[0].cached
    for d in decisions[1:]:
        assert d.cached


def test_router_split_50_50_distributes_episodes():
    """100 distinct episodes at 50/50 split -> roughly even distribution."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=50)
    slots = [
        router.route(episode_id=f"ep_{i}", request_id=f"req_{i}").slot
        for i in range(100)
    ]
    a_count = slots.count("a")
    b_count = slots.count("b")
    # Within tolerance for n=100 random hash distribution
    assert 35 <= a_count <= 65
    assert 35 <= b_count <= 65
    assert a_count + b_count == 100


def test_router_split_100_routes_all_to_a():
    """split=100 -> shadow-staging mode (B is loaded but inactive)."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=100)
    for i in range(50):
        decision = router.route(episode_id=f"ep_{i}", request_id=f"req_{i}")
        assert decision.slot == "a"


def test_router_split_zero_routes_all_to_b():
    """split=0 -> all to B (mirror of shadow-staging on B)."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=0)
    for i in range(50):
        decision = router.route(episode_id=f"ep_{i}", request_id=f"req_{i}")
        assert decision.slot == "b"


def test_router_degraded_mode_when_no_episode_id():
    """No episode_id -> falls back to request_id hashing + degraded flag."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=50)
    decision = router.route(episode_id=None, request_id="req_abc")
    assert decision.degraded
    assert decision.routing_key == "req_abc"


# ---------------------------------------------------------------------------
# Crash tracker + drain decision
# ---------------------------------------------------------------------------


def test_crash_tracker_drains_b_when_b_fails():
    """End-to-end: simulate 5 crashes on B + clean on A -> drain-b verdict."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    # B fails 5x in a row; A is clean throughout
    for _ in range(5):
        tracker.record_crash(slot="b")
    tracker.record_clean(slot="a")
    verdict = tracker.verdict()
    assert verdict.verdict == "drain-b"
    assert verdict.slot_to_drain == "b"


def test_crash_tracker_drain_persists_across_clean_b_attempts():
    """B keeps crashing -> verdict stays drain-b. Single clean on B
    resets and verdict goes back to healthy."""
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)
    for _ in range(7):
        tracker.record_crash(slot="b")
    assert tracker.verdict().verdict == "drain-b"
    # One clean on B: counter resets to 0 -> healthy
    tracker.record_clean(slot="b")
    assert tracker.verdict().verdict == "healthy"


# ---------------------------------------------------------------------------
# Record-replay carries routing decisions per request
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_record_captures_per_request_routing(tmp_path):
    """Compose router + record: per-request routing decision lands in
    the JSONL trace."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=50)
    writer = RecordWriter(
        record_dir=tmp_path,
        model_hash="abc123",
        config_hash="def456",
        export_dir=tmp_path,
        model_type="pi05_decomposed",
        export_kind="decomposed",
        providers=["CUDAExecutionProvider"],
        gzip_output=True,
        policies=[
            {"slot": "a", "model_id": "pi0-libero-v1", "model_hash": "aaaaaaaaaaaaaaaa"},
            {"slot": "b", "model_id": "pi0-libero-v2", "model_hash": "bbbbbbbbbbbbbbbb"},
        ],
    )
    # Simulate 5 episodes; record each
    for i in range(5):
        ep = f"ep_{i}"
        decision = router.route(episode_id=ep, request_id=f"req_{i}")
        writer.write_request(
            chunk_id=i, image_b64=None, instruction="x", state=None,
            actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
            routing={
                "slot": decision.slot,
                "routing_key": decision.routing_key,
                "degraded": decision.degraded,
                "cached": decision.cached,
            },
        )
    writer.close()

    records = _read_jsonl(writer.filepath)
    header = records[0]
    requests = [r for r in records if r["kind"] == "request"]

    # Header carries the policies block
    assert "policies" in header
    assert len(header["policies"]) == 2

    # Each request carries its routing decision
    assert len(requests) == 5
    for req in requests:
        assert "routing" in req
        assert req["routing"]["slot"] in ("a", "b")
        assert req["routing"]["routing_key"].startswith("ep_")


# ---------------------------------------------------------------------------
# Full composition: router + tracker + record (end-to-end happy path)
# ---------------------------------------------------------------------------


def test_e2e_2_policy_serving_loop_happy_path(tmp_path):
    """Compose all primitives. 50/50 split, no crashes, clean record."""
    # 1. Validate flag combo (Day 5)
    validate_split_and_no_rtc(split_a_percent=50, no_rtc=True)

    # 2. Build router (Days 1-2 substrate)
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=50)

    # 3. Build crash tracker (Day 8)
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)

    # 4. Build recorder (Day 7 schema)
    writer = RecordWriter(
        record_dir=tmp_path,
        model_hash="abc", config_hash="def",
        export_dir=tmp_path,
        model_type="pi05_decomposed", export_kind="decomposed",
        providers=["CUDAExecutionProvider"], gzip_output=True,
        policies=[
            {"slot": "a", "model_id": "v1", "model_hash": "aaaaaaaaaaaaaaaa"},
            {"slot": "b", "model_id": "v2", "model_hash": "bbbbbbbbbbbbbbbb"},
        ],
    )

    # 5. Simulate 10 episodes -- all clean
    for i in range(10):
        decision = router.route(episode_id=f"ep_{i}", request_id=f"req_{i}")
        # Stub /act handler succeeds; record clean
        tracker.record_clean(slot=decision.slot)
        writer.write_request(
            chunk_id=i, image_b64=None, instruction="x", state=None,
            actions=[[0.1, 0.2]], action_dim=2, latency_total_ms=5.0,
            routing={"slot": decision.slot, "routing_key": decision.routing_key},
        )

    writer.close()

    # 6. Assertions
    verdict = tracker.verdict()
    assert verdict.verdict == "healthy"
    assert verdict.slot_to_drain is None

    records = _read_jsonl(writer.filepath)
    requests = [r for r in records if r["kind"] == "request"]
    assert len(requests) == 10
    # Each request was routed to a valid slot
    for req in requests:
        assert req["routing"]["slot"] in ("a", "b")


def test_e2e_2_policy_serving_loop_b_crashes_drains_b(tmp_path):
    """Same as above but B crashes 5x -> tracker says drain-b. Caller
    would call router.set_split(100) to route 100% to A."""
    router = PolicyRouter(policies=_make_stub_policies(), split_a_percent=50)
    tracker = PolicyCrashTracker(slots=("a", "b"), threshold=5)

    # Simulate enough episodes that ~half hit each slot
    crash_count_b = 0
    for i in range(50):
        decision = router.route(episode_id=f"ep_{i}", request_id=f"req_{i}")
        if decision.slot == "b":
            tracker.record_crash(slot="b")
            crash_count_b += 1
            if crash_count_b >= 5:
                break
        else:
            tracker.record_clean(slot="a")

    assert tracker.verdict().verdict == "drain-b"


# ---------------------------------------------------------------------------
# Single-policy back-compat: existing single-policy callers unaffected
# ---------------------------------------------------------------------------


def test_single_policy_back_compat():
    """Single-policy callers continue to use make_single_policy() + the
    'prod' slot. PolicyCrashTracker(slots=('prod',)) preserves the
    legacy single-counter behavior."""
    p = make_single_policy(
        model_id="pi0-libero", model_hash="abc12345",
        export_dir="/exports/pi0",
    )
    assert p.slot == DEFAULT_SINGLE_POLICY_SLOT
    assert p.slot == "prod"

    tracker = PolicyCrashTracker(slots=(p.slot,), threshold=5)
    for _ in range(4):
        tracker.record_crash(slot="prod")
    assert tracker.verdict().verdict == "healthy"  # 4 < 5
    tracker.record_crash(slot="prod")
    assert tracker.verdict().verdict == "degraded"  # 5 >= 5
