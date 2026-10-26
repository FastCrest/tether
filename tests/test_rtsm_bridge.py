"""RTSM bridge gating against fake responses shaped like rtsm 0.3.0 (85b94e6).

Shapes follow rtsm/api/server.py (semantic_search, label_search, sessions_list)
and rtsm/stores/working_memory.py (_robot_pose_payload). The last tests replay
responses recorded from a real rtsm 0.3.0 run (see the fixture's source).
"""

from __future__ import annotations

import email.utils
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import requests

from tether.integrations import rtsm_bridge
from tether.integrations.rtsm_bridge import RtsmBridge

NOW = 1_800_000_000.0
LIVE = "20261025-120000_e2_abcd"


def pose(epoch: Any = 2, **overrides: Any) -> dict[str, Any]:
    # pose_clock "server": timestamp is RTSM's time.time() at the write, so
    # timestamp + age_s is RTSM's clock at response time (here: NOW).
    body = {
        "xyz": [0.1, 0.2, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "timestamp": NOW - 0.05,
        "frame_epoch": epoch,
        "sensor_ts_ns": 123,
        "pose_clock": "server",
        "age_s": 0.05,
        "stale": False,
        "stale_after_s": 2.0,
        "writes_accepted": 10,
        "sensor_ts_regressions": 0,
        "rejected_writes": 0,
        "rejected_by_reason": {},
    }
    body.update(overrides)
    return body


def sender_pose(**overrides: Any) -> dict[str, Any]:
    # A sender's own clock (a phone, or bag time): says nothing about RTSM's clock.
    return pose(pose_clock="sender", timestamp=1305031456.7, **overrides)


def hit(oid: str = "obj_1", **overrides: Any) -> dict[str, Any]:
    body = {
        "id": oid,
        "score": 0.12,
        "confirmed": True,
        "stability": 0.8,
        "xyz_world": [1.0, 2.0, 0.5],
        "last_seen_wall_utc": NOW - 1.0,
    }
    body.update(overrides)
    return body


def sessions(key: str | None = LIVE, epoch: int = 2, enabled: bool = True) -> dict[str, Any]:
    listed = [{"key": "20261025-110000_e1_abcd", "epoch": 1, "live": False}]
    if key is not None:
        listed.insert(0, {"key": key, "epoch": epoch, "live": True})
    return {"enabled": enabled, "live": key, "sessions": listed}


def date(t: float) -> str:
    return email.utils.formatdate(t, usegmt=True)


class FakeRtsm:
    """Stands in for requests.Session against one RTSM server."""

    def __init__(self) -> None:
        self.search: Any = {"query": "mug", "robot_pose": pose(), "results": [hit()]}
        self.sessions: list[Any] = [sessions(), sessions()]
        self.date: str | None = date(NOW)
        self.status = 200
        self.raw_search: bytes | None = None
        self.calls: list[tuple[str, Any]] = []

    def get(
        self, url: str, params: Any = None, timeout: float | None = None, stream: bool = False
    ) -> requests.Response:
        path = url.removeprefix("http://rtsm.invalid")
        self.calls.append((path, params))
        resp = requests.Response()
        resp.url = url
        if path == "/sessions":
            body = self.sessions.pop(0) if len(self.sessions) > 1 else self.sessions[0]
            resp.status_code = 200
            resp._content = json.dumps(body).encode()
        else:
            resp.status_code = self.status
            resp._content = (
                self.raw_search if self.raw_search is not None else json.dumps(self.search).encode()
            )
        setattr(resp, "_content_consumed", True)
        if self.date is not None:
            resp.headers["Date"] = self.date
        return resp


@pytest.fixture
def rtsm(monkeypatch: pytest.MonkeyPatch) -> FakeRtsm:
    fake = FakeRtsm()
    monkeypatch.setattr(rtsm_bridge.requests, "Session", lambda: fake)
    monkeypatch.setattr(rtsm_bridge.time, "time", lambda: NOW)
    return fake


def bridge(**kwargs: Any) -> RtsmBridge:
    kwargs.setdefault("max_age_s", 5.0)
    return RtsmBridge("http://rtsm.invalid/", **kwargs)


def test_fresh_confirmed_hit_becomes_target_in_pose_frame(rtsm: FakeRtsm) -> None:
    out = bridge().locate("mug", top_k=3)
    assert out.status == "ok"
    assert out.target is not None and out.robot_pose is not None
    assert out.target.xyz_world == (1.0, 2.0, 0.5)
    assert out.target.frame_epoch == out.robot_pose.frame_epoch == 2
    assert out.target.session_key == out.robot_pose.session_key == LIVE
    assert out.frame == (LIVE, 2)
    assert out.target.clock_source == "pose"
    assert out.target.last_seen_age_s == pytest.approx(1.0, abs=0.002)
    assert out.target.endpoint == "/search/semantic"
    assert out.target.base_url == "http://rtsm.invalid"
    assert out.robot_pose.xyz == (0.1, 0.2, 0.3)
    assert out.robot_pose.age_s >= 0.05
    assert not out.frame_changed
    # Bracket order: live session, search, live session.
    assert [c[0] for c in rtsm.calls] == ["/sessions", "/search/semantic", "/sessions"]
    assert rtsm.calls[1][1] == {"query": "mug", "top_k": 3}


def test_target_age_advances_on_our_monotonic_clock(rtsm: FakeRtsm) -> None:
    target = bridge().locate("mug").target
    assert target is not None
    assert target.age_s(target.fetched_at_mono + 4.0) == pytest.approx(5.0, abs=0.002)


def test_label_mode_uses_label_endpoint_and_keeps_matched_label(rtsm: FakeRtsm) -> None:
    rtsm.search["results"] = [hit(matched_label="tissue box")]
    out = bridge(mode="label").locate("tissue box")
    assert rtsm.calls[1][0] == "/search/label"
    assert out.target is not None and out.target.matched_label == "tissue box"


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"confirmed": False}, "unconfirmed"),
        ({"xyz_world": None}, "no_position"),
        ({"last_seen_wall_utc": None}, "no_last_seen"),
        ({"last_seen_wall_utc": 0.0}, "no_last_seen"),
        ({"last_seen_wall_utc": NOW - 5.01}, "stale"),
        ({"last_seen_wall_utc": NOW + 0.5}, "future_last_seen"),
    ],
)
def test_unqualified_hits_are_rejected_with_reason(
    rtsm: FakeRtsm, overrides: dict[str, Any], reason: str
) -> None:
    rtsm.search["results"] = [hit("bad", **overrides)]
    out = bridge().locate("mug")
    assert out.status == "none_qualify"
    assert out.target is None and out.candidates == ()
    assert [(r.object_id, r.reason) for r in out.rejected] == [("bad", reason)]
    assert out.robot_pose is not None


def test_best_qualifying_hit_wins_and_order_is_preserved(rtsm: FakeRtsm) -> None:
    rtsm.search["results"] = [
        hit("stale", score=0.3, last_seen_wall_utc=NOW - 60),
        hit("a", score=0.2),
        hit("proto", score=0.15, confirmed=False),
        hit("b", score=0.1),
    ]
    out = bridge().locate("mug")
    assert out.status == "ok"
    assert out.target is not None and out.target.object_id == "a"
    assert [c.object_id for c in out.candidates] == ["a", "b"]
    assert [(r.object_id, r.reason) for r in out.rejected] == [
        ("stale", "stale"),
        ("proto", "unconfirmed"),
    ]


def test_empty_results_are_explicit(rtsm: FakeRtsm) -> None:
    rtsm.search["results"] = []
    out = bridge().locate("mug")
    assert out.status == "empty"
    assert out.target is None and out.rejected == ()
    assert out.robot_pose is not None  # the frame was still verified


# ---- clocks ----


def test_sender_clock_gates_on_the_oldest_age_date_allows(rtsm: FakeRtsm) -> None:
    # Only Date bounds RTSM's clock: up to 1 s truncation + 1 s staleness ahead.
    rtsm.search["robot_pose"] = sender_pose()
    out = bridge().locate("mug")
    assert out.target is not None and out.target.clock_source == "date"
    assert out.target.last_seen_age_s == pytest.approx(3.0)
    rtsm.search["results"] = [hit(last_seen_wall_utc=NOW - 3.01)]
    assert bridge().locate("mug").status == "none_qualify"


def test_rtsm_ahead_within_tolerance_cannot_make_old_hits_fresh(rtsm: FakeRtsm) -> None:
    # Review finding: RTSM 2.9 s ahead (accepted), object seen 5.5 s ago on
    # RTSM's clock. Ages measured on our clock alone would read 2.6 s.
    ahead = 2.9
    rtsm.date = date(NOW + ahead)
    rtsm.search["robot_pose"] = sender_pose()
    rtsm.search["results"] = [hit(last_seen_wall_utc=NOW + ahead - 5.5)]
    out = bridge().locate("mug")
    assert out.status == "none_qualify" and out.rejected[0].reason == "stale"


def test_server_pose_clock_measures_skew_to_the_millisecond(rtsm: FakeRtsm) -> None:
    ahead = 1.9  # within max_clock_skew_s; the pose clock pins it to 1 ms
    rtsm.date = date(NOW + ahead)
    rtsm.search["robot_pose"] = pose(timestamp=NOW + ahead - 0.05)
    rtsm.search["results"] = [
        hit("old", last_seen_wall_utc=NOW + ahead - 5.01),
        hit("ok", last_seen_wall_utc=NOW + ahead - 4.9),
    ]
    out = bridge().locate("mug")
    assert out.target is not None and out.target.object_id == "ok"
    assert out.target.clock_source == "pose"
    assert out.target.last_seen_age_s == pytest.approx(4.9, abs=0.002)
    assert [(r.object_id, r.reason) for r in out.rejected] == [("old", "stale")]


def test_server_tagged_stamp_that_contradicts_date_is_ignored(rtsm: FakeRtsm) -> None:
    # RTSM's ZeroMQ path tags a sender's wall stamp "server"; Date wins.
    rtsm.search["robot_pose"] = pose(timestamp=NOW - 100.0)
    out = bridge().locate("mug")
    assert out.target is not None and out.target.clock_source == "date"


@pytest.mark.parametrize("offset", [-4.5, 3.0])
def test_clock_skew_refuses(rtsm: FakeRtsm, offset: float) -> None:
    rtsm.date = date(NOW + offset)
    assert bridge().locate("mug").status == "clock_skew"


def test_date_header_truncation_is_not_skew(rtsm: FakeRtsm) -> None:
    # Date drops the fraction: 2.5 s behind reads as 3 s behind, and RTSM may
    # still be within 2 s of us.
    rtsm.date = date(NOW - 2.5)
    assert bridge().locate("mug").status == "ok"


def test_missing_date_header_refuses(rtsm: FakeRtsm) -> None:
    rtsm.date = None
    assert bridge().locate("mug").status == "clock_skew"


@pytest.mark.parametrize(
    "value", ["not a date", "Mon, 26 Oct 2026 05:03:12 -0000", "Mon, 26 Oct 2026 05:03:12 +0200"]
)
def test_malformed_date_header_raises(rtsm: FakeRtsm, value: str) -> None:
    rtsm.date = value
    with pytest.raises(ValueError, match="Date"):
        bridge().locate("mug")


# ---- frames ----


def test_pose_epoch_ahead_of_live_session_is_a_frame_transition(rtsm: FakeRtsm) -> None:
    # RTSM writes the new-epoch pose at receive time, before the pipeline
    # freezes the old epoch's objects; the hits are still epoch 2.
    rtsm.search["robot_pose"] = pose(epoch=3)
    out = bridge().locate("mug")
    assert out.status == "frame_transition"
    assert out.target is None and out.robot_pose is None


@pytest.mark.parametrize(
    "after",
    [sessions(epoch=3, key="20261025-120500_e3_abcd"), sessions(key="20261025-120500_e2_abcd")],
    ids=["epoch-bump", "reset-same-epoch"],
)
def test_live_session_change_during_query_is_a_frame_transition(
    rtsm: FakeRtsm, after: dict[str, Any]
) -> None:
    rtsm.sessions = [sessions(), after]
    out = bridge().locate("mug")
    assert out.status == "frame_transition"
    assert out.frame == (after["live"], after["sessions"][0]["epoch"])


def test_frame_change_between_lookups_is_flagged(rtsm: FakeRtsm) -> None:
    b = bridge()
    assert b.locate("mug").frame_changed is False
    new_key = "20261025-121000_e3_abcd"
    rtsm.sessions = [sessions(key=new_key, epoch=3)]
    rtsm.search["robot_pose"] = pose(epoch=3)
    out = b.locate("mug")
    assert out.status == "ok" and out.frame_changed is True
    assert out.target is not None and out.target.frame_epoch == 3
    assert b.locate("mug").frame_changed is False


@pytest.mark.parametrize(
    "robot_pose,status",
    [
        (pose(epoch=3, stale=True), "pose_stale"),
        (pose(epoch=4), "frame_transition"),
        (None, "no_pose"),
    ],
)
def test_refusals_report_a_frame_change(rtsm: FakeRtsm, robot_pose: Any, status: str) -> None:
    # Review finding: a consumer holding an epoch-2 target must be told to
    # drop it even when the lookup that saw the new frame is refused.
    b = bridge()
    assert b.locate("mug").status == "ok"
    rtsm.sessions = [sessions(key="k3", epoch=3)]
    rtsm.search["robot_pose"] = robot_pose
    out = b.locate("mug")
    assert out.status == status
    assert out.frame_changed is True and out.frame == ("k3", 3)
    if out.robot_pose is not None:
        assert out.robot_pose.frame_epoch == 3 and out.robot_pose.session_key == "k3"
    rtsm.search["robot_pose"] = pose(epoch=3)
    after = b.locate("mug")
    assert after.status == "ok" and after.frame_changed is False


def test_epoch_going_backwards_is_a_new_frame(rtsm: FakeRtsm) -> None:
    # Epochs only rise within one RTSM process; lower means RTSM restarted.
    b = bridge()
    b.locate("mug")
    rtsm.sessions = [sessions(key="restarted_e1", epoch=1)]
    rtsm.search["robot_pose"] = pose(epoch=1)
    out = b.locate("mug")
    assert out.status == "ok" and out.frame_changed is True


@pytest.mark.parametrize(
    "robot_pose,status",
    [
        (None, "no_pose"),
        (pose(epoch=None), "epochless"),
        (pose(stale=True), "pose_stale"),
        (pose(age_s=1.5, timestamp=NOW - 1.5), "pose_stale"),
    ],
)
def test_pose_gates(rtsm: FakeRtsm, robot_pose: Any, status: str) -> None:
    rtsm.search["robot_pose"] = robot_pose
    out = bridge().locate("mug")
    assert out.status == status and out.target is None


@pytest.mark.parametrize(
    "payload", [sessions(enabled=False), sessions(key=None)], ids=["disabled", "no-live"]
)
def test_sessions_unavailable_refuses(rtsm: FakeRtsm, payload: dict[str, Any]) -> None:
    rtsm.sessions = [payload]
    out = bridge().locate("mug")
    assert out.status == "sessions_unavailable" and out.frame is None


# ---- malformed input and transport ----


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("robot_pose"),
        lambda b: b["robot_pose"].update(xyz=[0.0, 0.0]),
        lambda b: b["robot_pose"].update(xyz=[math.nan, 0.0, 0.0]),
        lambda b: b["robot_pose"].update(quaternion_xyzw=[0.0, 0.0, 0.0, 0.0]),
        lambda b: b["robot_pose"].update(quaternion_xyzw=[0.0, 0.0, 0.0, 7.0]),
        lambda b: b["robot_pose"].update(frame_epoch=True),
        lambda b: b["robot_pose"].update(frame_epoch="2"),
        lambda b: b["robot_pose"].update(age_s=-1.0),
        lambda b: b["results"][0].update(score="0.1"),
        lambda b: b["results"][0].update(confirmed="true"),
        lambda b: b["results"][0].update(xyz_world=[1.0, math.inf, 0.0]),
        lambda b: b["results"][0].update(xyz_world=[1.0, 2.0, 3.0, 4.0]),
        lambda b: b["results"][0].pop("id"),
        lambda b: b["results"][0].update(id=""),
        lambda b: b["results"][0].pop("confirmed"),
    ],
)
def test_malformed_search_body_raises(rtsm: FakeRtsm, mutate: Any) -> None:
    mutate(rtsm.search)
    with pytest.raises(ValueError, match="malformed RTSM"):
        bridge().locate("mug")


@pytest.mark.parametrize(
    "body", [None, [], {"error": "x"}, {"results": None}, {"results": ["obj_1"]}]
)
def test_non_envelope_search_body_raises(rtsm: FakeRtsm, body: Any) -> None:
    rtsm.search = body
    with pytest.raises(ValueError, match="malformed RTSM"):
        bridge().locate("mug")


def test_invalid_json_raises(rtsm: FakeRtsm) -> None:
    rtsm.raw_search = b"{not json"
    with pytest.raises(ValueError):
        bridge().locate("mug")


def test_oversized_body_raises(rtsm: FakeRtsm) -> None:
    rtsm.raw_search = b" " * (rtsm_bridge._MAX_BODY_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        bridge().locate("mug")


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": True, "live": "missing", "sessions": []},
        {"enabled": True, "live": LIVE, "sessions": [{"key": LIVE, "epoch": None}]},
        {"live": LIVE, "sessions": []},
        [],
    ],
)
def test_malformed_sessions_body_raises(rtsm: FakeRtsm, payload: Any) -> None:
    rtsm.sessions = [payload]
    with pytest.raises(ValueError, match="malformed RTSM /sessions"):
        bridge().locate("mug")


def test_http_failure_propagates(rtsm: FakeRtsm) -> None:
    rtsm.status = 503
    with pytest.raises(requests.HTTPError):
        bridge().locate("mug")


def test_whole_lookup_has_one_deadline(rtsm: FakeRtsm, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count(0.0, 1.0)  # each clock read advances 1 s
    monkeypatch.setattr(rtsm_bridge.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="exceeded 2.5 s"):
        bridge(timeout_s=2.5).locate("mug")


@pytest.mark.parametrize(
    "kwargs",
    [{"max_age_s": 0}, {"max_age_s": math.nan}, {"max_pose_age_s": -1}, {"mode": "spatial"}],
)
def test_invalid_configuration_is_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        bridge(**kwargs)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_refused(rtsm: FakeRtsm, top_k: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        bridge().locate("mug", top_k=top_k)
    assert rtsm.calls == []


def test_upstream_client_parses_the_same_fields(rtsm: FakeRtsm) -> None:
    """Differential oracle: rtsm.client.RtsmClient (0.3.0) reads our fixture identically."""
    client_mod = pytest.importorskip("rtsm.client")
    rtsm.search["results"] = [hit("a"), hit("b", score=0.05, last_seen_wall_utc=NOW - 2)]
    theirs = client_mod.RtsmClient("http://rtsm.invalid")._parse_search(
        deepcopy(rtsm.search), "mug"
    )
    ours = bridge().locate("mug")
    assert ours.robot_pose is not None
    assert theirs.robot_pose.frame_epoch == ours.robot_pose.frame_epoch
    assert list(ours.robot_pose.xyz) == theirs.robot_pose.xyz
    assert [
        (c.object_id, c.score, list(c.xyz_world), c.last_seen_wall_utc) for c in ours.candidates
    ] == [(h.id, h.score, h.xyz_world, h.last_seen_wall_utc) for h in theirs.results]


RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "rtsm" / "fr1_desk_0.3.0.json").read_text()
)["records"]


@pytest.mark.parametrize(
    "name,status,target",
    [
        ("before_first_pose", "no_pose", None),
        ("during_bag", "ok", "4d0488958be64fcb"),
        ("after_bag", "pose_stale", None),
    ],
)
def test_recorded_rtsm_responses(
    rtsm: FakeRtsm, monkeypatch: pytest.MonkeyPatch, name: str, status: str, target: str | None
) -> None:
    """Real rtsm 0.3.0 bodies from the TUM fr1/desk bag (see the fixture's source)."""
    rec = RECORDED[name]
    rtsm.sessions = [rec["sessions"]]
    rtsm.search = rec["semantic"]["body"]
    rtsm.date = rec["semantic"]["date"]
    monkeypatch.setattr(rtsm_bridge.time, "time", lambda: rec["t_wall"])
    out = bridge().locate("keyboard", top_k=3)
    assert out.status == status == rec["bridge_live"]["status"]
    assert (out.target and out.target.object_id) == target
    if target is not None:
        assert out.target is not None and out.robot_pose is not None
        # A bag pose carries bag time ("sender"), so only Date bounds the clock.
        assert out.target.clock_source == "date"
        assert out.target.frame_epoch == out.robot_pose.frame_epoch == 1
        assert [(r.object_id, r.reason) for r in out.rejected] == [("08d4405b0c4b4184", "stale")]


def test_server_pose_clock_refuses_skew_date_alone_would_accept(rtsm: FakeRtsm) -> None:
    # 2.9 s ahead: Date alone allows [2, 4] (could be 2 s), the pose clock says 2.9.
    rtsm.date = date(NOW + 2.9)
    rtsm.search["robot_pose"] = pose(timestamp=NOW + 2.9 - 0.05)
    assert bridge().locate("mug").status == "clock_skew"
