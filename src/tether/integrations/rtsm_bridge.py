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
reads name the same live session and its epoch equals the pose's epoch. This
covers boundaries that change ``frame_epoch``. RTSM boundaries that keep the
epoch (a SLAM map reset, an inferred restart) are not visible until they have
happened; see docs/rtsm-bridge.md.

Clock rule: ``last_seen_wall_utc`` is RTSM's wall clock. The bridge bounds
RTSM's clock minus ours to an interval and gates on the oldest age that
interval allows, so an accepted skew can make a hit look older, never younger.
"""

from __future__ import annotations

import datetime
import email.utils
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar

import requests  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
ClockSource = Literal["pose", "date"]

# uvicorn (RTSM's server) rebuilds the Date header every 10 ticks of 0.1 s
# (uvicorn 0.54 server.py:243-255), and Date drops the fraction of a second.
# ponytail: assumes RTSM's event loop is not stalled; a stalled loop serves an
# older Date and this bound is too small. The pose clock avoids it when present.
_DATE_TRUNCATION_S = 1.0
_DATE_STALENESS_S = 1.0
# RTSM rounds pose age_s to milliseconds.
_POSE_AGE_ROUNDING_S = 0.001
_MAX_BODY_BYTES = 8 * 1024 * 1024

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
    pose_clock: str | None = None


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
    # Upper bound on the age at fetch: RTSM's last-seen stamp against the
    # latest RTSM clock the skew interval allows.
    last_seen_age_s: float
    fetched_at_mono: float
    clock_source: ClockSource
    query: str
    endpoint: str
    base_url: str

    def age_s(self, now_mono: float | None = None) -> float:
        """Last-seen age bound now, advanced on our monotonic clock since the fetch."""
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
    # The live (session_key, frame_epoch) RTSM reported after the search, on
    # every status where sessions answered; None otherwise.
    frame: tuple[str, int] | None
    # True when ``frame`` differs from the frame of the previous lookup:
    # drop any target or pose held from earlier lookups. Set on refusals too.
    frame_changed: bool
    detail: str = ""


class RtsmBridge:
    """Pull-side gate over RTSM's REST API. One lookup at a time per bridge (locked)."""

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
        # Budget for one whole locate() (three requests). A server that
        # trickles bytes can hold one socket read for the remaining budget,
        # so the worst case is about twice this.
        self.timeout_s = timeout_s
        self._http = requests.Session()
        self._lock = threading.Lock()
        self._last_frame: tuple[str, int] | None = None

    def _get(self, path: str, params: dict[str, Any] | None, deadline: float) -> tuple[Any, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"RTSM lookup exceeded {self.timeout_s} s before GET {path}")
        with self._http.get(
            f"{self.base_url}{path}", params=params, timeout=remaining, stream=True
        ) as resp:
            resp.raise_for_status()
            body = bytearray()
            for chunk in resp.iter_content(65536):
                body += chunk
                if len(body) > _MAX_BODY_BYTES:
                    raise ValueError(f"RTSM {path} response exceeds {_MAX_BODY_BYTES} bytes")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"RTSM lookup exceeded {self.timeout_s} s reading {path}")
            return resp.headers, json.loads(body)

    def _live_session(self, deadline: float) -> tuple[str, int] | None:
        _, body = self._get("/sessions", None, deadline)
        sessions = _parse(_Sessions, body, "/sessions")
        if not sessions.enabled or sessions.live is None:
            return None
        for rec in sessions.sessions:
            if rec.key == sessions.live:
                return rec.key, rec.epoch
        raise ValueError(f"malformed RTSM /sessions response: live {sessions.live!r} is not listed")

    def _skew_bounds(
        self, headers: Any, pose: _Pose | None, sent_wall: float, fetched_wall: float
    ) -> tuple[float, float, ClockSource] | None:
        """Interval holding RTSM's wall clock minus ours, or None without a Date header."""
        date = headers.get("Date")
        if date is None:
            return None
        try:
            parsed = email.utils.parsedate_to_datetime(date)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed RTSM Date header {date!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
            raise ValueError(f"malformed RTSM Date header {date!r}: not GMT")
        server_date = parsed.timestamp()
        lo = server_date - fetched_wall
        hi = server_date + _DATE_TRUNCATION_S + _DATE_STALENESS_S - sent_wall
        if pose is not None and pose.pose_clock == "server":
            # timestamp is RTSM's time.time() at the pose write and age_s its
            # monotonic time since, both read while building this response.
            # RTSM also tags sender stamps "server" on ZeroMQ, so use this
            # only when it agrees with the Date bound.
            server_now = pose.timestamp + pose.age_s
            p_lo = server_now - fetched_wall - _POSE_AGE_ROUNDING_S
            p_hi = server_now - sent_wall + _POSE_AGE_ROUNDING_S
            if max(lo, p_lo) <= min(hi, p_hi):
                return max(lo, p_lo), min(hi, p_hi), "pose"
        return lo, hi, "date"

    def locate(self, query: str, *, top_k: int = 5) -> RtsmLookup:
        """Query RTSM once and return the best fresh, confirmed, frame-consistent hit."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}")
        with self._lock:
            return self._locate(query, top_k)

    def _locate(self, query: str, top_k: int) -> RtsmLookup:
        endpoint = f"/search/{self.mode}"
        deadline = time.monotonic() + self.timeout_s
        before = self._live_session(deadline)
        sent_mono = time.monotonic()
        sent_wall = time.time()
        headers, body = self._get(endpoint, {"query": query, "top_k": top_k}, deadline)
        fetched_mono = time.monotonic()
        fetched_wall = time.time()
        after = self._live_session(deadline)
        search = _parse(_Search, body, endpoint)
        raw_pose = search.robot_pose
        if raw_pose is not None:
            norm = math.sqrt(math.fsum(v * v for v in raw_pose.quaternion_xyzw))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"malformed RTSM {endpoint} response: robot_pose quaternion norm {norm}"
                )

        frame_changed = (
            after is not None and self._last_frame is not None and self._last_frame != after
        )
        if after is not None:
            self._last_frame = after

        def refuse(status: Status, detail: str, pose: RobotPose | None = None) -> RtsmLookup:
            return RtsmLookup(status, query, None, (), (), pose, after, frame_changed, detail)

        bounds = self._skew_bounds(headers, raw_pose, sent_wall, fetched_wall)
        if bounds is None:
            return refuse("clock_skew", "RTSM response has no Date header; cannot bound clock skew")
        skew_lo, skew_hi, clock_source = bounds
        if skew_lo > self.max_clock_skew_s or skew_hi < -self.max_clock_skew_s:
            return refuse(
                "clock_skew", f"RTSM clock is {skew_lo:+.2f} to {skew_hi:+.2f} s from ours"
            )

        if raw_pose is None:
            return refuse("no_pose", "RTSM has not received a robot pose")
        if raw_pose.frame_epoch is None:
            return refuse("epochless", "robot_pose.frame_epoch is null")
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

        # The latest RTSM clock reading consistent with the skew bounds.
        server_now_max = fetched_wall + skew_hi
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
            # Stamped by RTSM's own clock, so it cannot be later than RTSM's now.
            elif seen > server_now_max:
                reason = "future_last_seen"
            elif server_now_max - seen > self.max_age_s:
                reason = "stale"
            if reason is not None:
                rejected.append(RejectedHit(hit.id, reason))
                continue
            assert hit.xyz_world is not None and seen is not None
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
                    last_seen_age_s=server_now_max - seen,
                    fetched_at_mono=fetched_mono,
                    clock_source=clock_source,
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
            frame=after,
            frame_changed=frame_changed,
        )
