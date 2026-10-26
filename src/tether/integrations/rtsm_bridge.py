"""RTSM bridge: turn RTSM spatial-memory hits into a target a policy can consume.

Contract: calabi-inc/rtsm 0.3.0 (commit 85b94e6). RTSM is a memory you query;
it emits nothing shaped for ``/act``. Freshness and frame consistency are the
reader's job, so this module does both, pull-side, and refuses rather than
guesses. See docs/rtsm-bridge.md for the rules and the evidence behind them.

Frame rule: RTSM writes the robot pose (with its ``frame_epoch``) when a frame
is received, but clears the previous epoch's objects later, when the pipeline
dequeues that frame. One search response can therefore pair a new-epoch pose
with old-epoch object positions. Each lookup is bracketed by two reads of the
live session (``GET /sessions``); a hit is in the pose's frame only when both
reads name the same live session and its epoch equals the pose's epoch.
"""

from __future__ import annotations

import email.utils
import math
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tether.integrations.connector import _query_envelope

Status = Literal[
    "ok",  # target is set
    "empty",  # RTSM returned no results
    "none_qualify",  # results exist, every one was rejected (see ``rejected``)
    "frame_transition",  # a frame_epoch boundary or reset overlapped the query; retry
    "no_pose",  # RTSM has no robot pose yet
    "epochless",  # pose has no frame_epoch, so its frame cannot be checked
    "pose_stale",  # pose flagged stale by RTSM or older than max_pose_age_s
    "sessions_unavailable",  # sessions disabled: old-epoch objects are never cleared
    "clock_skew",  # RTSM wall clock disagrees with ours; last_seen ages are meaningless
]
RejectReason = Literal["unconfirmed", "no_position", "no_last_seen", "stale", "future_last_seen"]
Mode = Literal["semantic", "label"]

_Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
_Quat = Annotated[list[float], Field(min_length=4, max_length=4)]


class _Wire(BaseModel):
    # strict: no "1.5" -> 1.5 or "true" -> True coercion; finite numbers only.
    model_config = ConfigDict(strict=True, allow_inf_nan=False, frozen=True, extra="ignore")


class _Pose(_Wire):
    xyz: _Vec3
    quaternion_xyzw: _Quat
    timestamp: float
    frame_epoch: int | None
    age_s: float = Field(ge=0)
    stale: bool


class _Hit(_Wire):
    id: str = Field(min_length=1)
    score: float
    confirmed: bool
    stability: float
    xyz_world: _Vec3 | None
    last_seen_wall_utc: float | None
    matched_label: str | None = None


class _Search(_Wire):
    robot_pose: _Pose | None
    results: list[_Hit]


class _SessionSummary(_Wire):
    key: str
    epoch: int


class _Sessions(_Wire):
    enabled: bool
    live: str | None
    sessions: list[_SessionSummary]


_M = TypeVar("_M", bound=_Wire)


def _parse(model: type[_M], body: Any, where: str) -> _M:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise ValueError(f"malformed RTSM {where} response: {exc}") from exc


@dataclass(frozen=True)
class RobotPose:
    xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame_epoch: int
    session_key: str
    # RTSM's own age of the pose at response time plus our request round trip.
    age_s: float
    fetched_at_mono: float


@dataclass(frozen=True)
class TargetFix:
    """One qualified object: a world-frame position in the same frame as ``RobotPose``."""

    object_id: str
    xyz_world: tuple[float, float, float]
    frame_epoch: int
    session_key: str
    score: float
    stability: float
    matched_label: str | None
    last_seen_wall_utc: float
    # Our wall clock at response minus RTSM's last-seen stamp.
    last_seen_age_s: float
    fetched_at_mono: float
    query: str
    endpoint: str
    base_url: str

    def age_s(self, now_mono: float | None = None) -> float:
        """Last-seen age now, advanced on our monotonic clock since the fetch."""
        now = time.monotonic() if now_mono is None else now_mono
        return self.last_seen_age_s + (now - self.fetched_at_mono)


@dataclass(frozen=True)
class RejectedHit:
    object_id: str
    reason: RejectReason


@dataclass(frozen=True)
class RtsmLookup:
    status: Status
    query: str
    target: TargetFix | None
    candidates: tuple[TargetFix, ...]
    rejected: tuple[RejectedHit, ...]
    robot_pose: RobotPose | None
    # True when this frame differs from the last frame this bridge accepted:
    # drop any target or pose held from earlier lookups.
    frame_changed: bool
    detail: str = ""


class RtsmBridge:
    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        *,
        max_age_s: float,
        max_pose_age_s: float = 1.0,
        max_clock_skew_s: float = 2.0,
        mode: Mode = "semantic",
        timeout_s: float = 3.0,
    ) -> None:
        for name, value in (
            ("max_age_s", max_age_s),
            ("max_pose_age_s", max_pose_age_s),
            ("max_clock_skew_s", max_clock_skew_s),
            ("timeout_s", timeout_s),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if mode not in ("semantic", "label"):
            raise ValueError(f"mode must be 'semantic' or 'label', got {mode!r}")
        self.base_url = base_url.rstrip("/")
        self.max_age_s = max_age_s
        self.max_pose_age_s = max_pose_age_s
        self.max_clock_skew_s = max_clock_skew_s
        self.mode: Mode = mode
        self.timeout_s = timeout_s
        self._last_frame: tuple[str, int] | None = None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> requests.Response:
        return requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout_s)

    def _live_session(self) -> tuple[str, int] | None:
        resp = self._get("/sessions")
        resp.raise_for_status()
        sessions = _parse(_Sessions, resp.json(), "/sessions")
        if not sessions.enabled or sessions.live is None:
            return None
        for rec in sessions.sessions:
            if rec.key == sessions.live:
                return rec.key, rec.epoch
        raise ValueError(f"malformed RTSM /sessions response: live {sessions.live!r} is not listed")

    def locate(self, query: str, *, top_k: int = 5) -> RtsmLookup:
        """Query RTSM once and return the best fresh, confirmed, frame-consistent hit."""
        endpoint = f"/search/{self.mode}"
        before = self._live_session()
        sent_mono = time.monotonic()
        resp = self._get(endpoint, {"query": query, "top_k": top_k})
        fetched_mono = time.monotonic()
        fetched_wall = time.time()
        after = self._live_session()
        search = _parse(_Search, _query_envelope(resp, "results"), endpoint)

        def refuse(status: Status, detail: str, pose: RobotPose | None = None) -> RtsmLookup:
            return RtsmLookup(status, query, None, (), (), pose, False, detail)

        date = resp.headers.get("Date")
        if date is None:
            return refuse("clock_skew", "RTSM response has no Date header; cannot bound clock skew")
        try:
            server_wall = email.utils.parsedate_to_datetime(date).timestamp()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed RTSM Date header {date!r}") from exc
        # Date is truncated to whole seconds and stamped while we waited, so
        # RTSM's clock minus ours lies in [lo, hi]. Refuse only when all of it
        # is outside the tolerance (same host, a Date of 12:00:00 at our
        # 12:00:00.9 is not skew).
        lo = server_wall - fetched_wall - (fetched_mono - sent_mono)
        hi = server_wall + 1.0 - fetched_wall
        if lo > self.max_clock_skew_s or hi < -self.max_clock_skew_s:
            return refuse("clock_skew", f"RTSM clock is {lo:+.1f} to {hi:+.1f} s from ours")

        raw_pose = search.robot_pose
        if raw_pose is None:
            return refuse("no_pose", "RTSM has not received a robot pose")
        if raw_pose.frame_epoch is None:
            return refuse("epochless", "robot_pose.frame_epoch is null")
        if math.fsum(v * v for v in raw_pose.quaternion_xyzw) < 1e-12:
            raise ValueError(f"malformed RTSM {endpoint} response: zero-norm robot_pose quaternion")
        if before is None or after is None:
            return refuse(
                "sessions_unavailable", "RTSM sessions are disabled or have no live session"
            )
        if before != after or before[1] != raw_pose.frame_epoch:
            return refuse(
                "frame_transition",
                f"live session {before} -> {after}, pose frame_epoch {raw_pose.frame_epoch}",
            )

        session_key, epoch = before
        x, y, z = raw_pose.xyz
        qx, qy, qz, qw = raw_pose.quaternion_xyzw
        pose = RobotPose(
            xyz=(x, y, z),
            quaternion_xyzw=(qx, qy, qz, qw),
            frame_epoch=epoch,
            session_key=session_key,
            age_s=raw_pose.age_s + (fetched_mono - sent_mono),
            fetched_at_mono=fetched_mono,
        )
        if raw_pose.stale or pose.age_s > self.max_pose_age_s:
            return refuse(
                "pose_stale", f"pose age {pose.age_s:.3f} s, stale={raw_pose.stale}", pose
            )

        frame_changed = self._last_frame is not None and self._last_frame != before
        self._last_frame = before

        candidates: list[TargetFix] = []
        rejected: list[RejectedHit] = []
        for hit in search.results:
            seen = hit.last_seen_wall_utc
            reason: RejectReason | None = None
            if not hit.confirmed:
                reason = "unconfirmed"
            elif hit.xyz_world is None:
                reason = "no_position"
            # RTSM reports 0.0 for an object that has no observation stamp.
            elif seen is None or seen <= 0:
                reason = "no_last_seen"
            elif fetched_wall - seen < -self.max_clock_skew_s:
                reason = "future_last_seen"
            elif fetched_wall - seen > self.max_age_s:
                reason = "stale"
            if reason is not None:
                rejected.append(RejectedHit(hit.id, reason))
                continue
            assert hit.xyz_world is not None and seen is not None
            age = fetched_wall - seen
            hx, hy, hz = hit.xyz_world
            candidates.append(
                TargetFix(
                    object_id=hit.id,
                    xyz_world=(hx, hy, hz),
                    frame_epoch=epoch,
                    session_key=session_key,
                    score=hit.score,
                    stability=hit.stability,
                    matched_label=hit.matched_label,
                    last_seen_wall_utc=seen,
                    last_seen_age_s=max(0.0, age),
                    fetched_at_mono=fetched_mono,
                    query=query,
                    endpoint=endpoint,
                    base_url=self.base_url,
                )
            )

        if not search.results:
            status: Status = "empty"
        elif not candidates:
            status = "none_qualify"
        else:
            status = "ok"
        return RtsmLookup(
            status=status,
            query=query,
            target=candidates[0] if candidates else None,
            candidates=tuple(candidates),
            rejected=tuple(rejected),
            robot_pose=pose,
            frame_changed=frame_changed,
        )
