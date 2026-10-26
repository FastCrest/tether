# RTSM bridge

`tether.integrations.rtsm_bridge` is Tether's side of the contract with
[RTSM](https://github.com/calabi-inc/rtsm), a queryable spatial memory. RTSM
emits nothing shaped for `/act`; it answers queries. The bridge turns one
answer into a target a policy or instruction layer can use, or says why it
cannot.

```python
from tether.integrations.rtsm_bridge import RtsmBridge

bridge = RtsmBridge("http://localhost:8002", max_age_s=5.0)  # you pick max_age_s
found = bridge.locate("keyboard")
if found.frame_changed:
    ...  # drop any target or pose held from an earlier lookup (set on refusals too)
if found.status == "ok":
    found.target.xyz_world   # metres, SLAM world frame
    found.robot_pose.xyz     # same frame: same session_key and frame_epoch
```

Contract: `rtsm==0.3.0` (commit `85b94e6`). Shapes were read from that wheel's
`rtsm/api/server.py`, `rtsm/stores/working_memory.py` and `rtsm/core/sessions.py`.

## Rules

| Check | Outcome |
|---|---|
| no results | `empty` |
| results, none pass the hit gates | `none_qualify`; `rejected` lists each id and reason |
| hit not `confirmed` | rejected `unconfirmed` |
| hit has no `xyz_world` | rejected `no_position` |
| hit `last_seen_wall_utc` null or 0 | rejected `no_last_seen` |
| oldest age the clock bounds allow (see Clocks) above `max_age_s` | rejected `stale` |
| `last_seen_wall_utc` later than the latest RTSM clock the bounds allow | rejected `future_last_seen` |
| no `robot_pose` | `no_pose` |
| `robot_pose.frame_epoch` null | `epochless` |
| pose `stale`, or its `age_s` plus our round trip above `max_pose_age_s` | `pose_stale` |
| sessions disabled, or no live session | `sessions_unavailable` |
| live session changed during the query, or its epoch differs from the pose's | `frame_transition` (retry) |
| no `Date` header, or RTSM's clock is certainly more than `max_clock_skew_s` from ours | `clock_skew` |
| body not matching the 0.3.0 shapes (wrong types, NaN, wrong vector length, quaternion norm not 1 ± 1e-3, `Date` not GMT, body over 8 MiB) | `ValueError` |
| `top_k` < 1 | `ValueError`, before any request |
| HTTP error | `requests.HTTPError` |
| the three requests together take longer than `timeout_s` | `TimeoutError` |

`TargetFix.age_s(now_mono)` re-ages a held target on our monotonic clock.
`max_age_s` has no default: how old an observation may be depends on the task.
`RtsmLookup.frame` is the live `(session_key, frame_epoch)` whenever sessions
answered, so a consumer can compare it with what it holds.

`timeout_s` is a budget for the whole lookup, checked before each request and
after each chunk read. One socket read can still wait for the remaining budget,
so a server that trickles bytes can hold a lookup for up to about twice
`timeout_s`. Lookups on one bridge are serialised by a lock, so `frame_changed`
compares lookups in the order they ran.

## Why each lookup reads `/sessions` twice

RTSM writes the robot pose, with its new `frame_epoch`, when a frame is
received (`WorkingMemory.update_robot_pose`). The old epoch's objects are frozen
and cleared later, when the pipeline thread dequeues that frame
(`Pipeline` -> `SessionRegistry.observe_frame`; `on_session_change` says
boundaries are decided on the pipeline thread). A search builds its results
before it reads the pose. So one response can carry an epoch-N+1 pose next to
epoch-N object positions, and the pose epoch alone does not prove the hits are
in its frame.

The live session's epoch is the epoch of the objects in working memory. The
bridge reads it before and after the search. If both reads name the same live
session and its epoch equals the pose's epoch, no epoch-changing boundary
completed while the results were built, so hits and pose share a frame.
Otherwise the lookup returns `frame_transition`. RTSM holds its session lock
for the whole of a freeze and while listing sessions (`sessions.py:620-626,
805-858`), so `/sessions` never shows a half-done freeze. This costs two extra
local requests.

**Open gap: boundaries that keep the epoch.** A SLAM `reset` correction only
sets a pending flag (`sessions.py:461-462`). The `map_reset` boundary fires when
the pipeline dequeues the next frame and opens the new session with the **same**
`frame_epoch` (`sessions.py:358-370`); a clock-rebase `restart_inferred` does
too. In the window between the correction and that dequeue, the receiver can
write poses in the new map frame while working memory still holds the old map's
objects, and neither `/sessions` nor the search response shows the pending
reset. The bridge would return `ok` with a pose and hits from different maps.
Whether a given SLAM's poses change frame at a reset is not verified. This needs
a change on RTSM's side: bump `frame_epoch` at a map reset, or expose the
pending boundary. It is an open question for the RTSM author.

With `sessions.enable: false` RTSM does not clear old-epoch objects at a
boundary at all, so nothing in a response says which frame a hit is in. The
bridge refuses with `sessions_unavailable` instead of guessing.

## Clocks

`last_seen_wall_utc` is RTSM's wall clock, so its age has to be measured on
RTSM's clock. The bridge bounds RTSM's clock minus ours to an interval
`[lo, hi]`, and gates on the oldest age that interval allows:
`our_now + hi - last_seen`. An accepted skew can make a hit look older, never
younger. It refuses with `clock_skew` only when the whole interval is outside
`±max_clock_skew_s`.

- **From `Date`** (always): `[Date - fetched, Date + 2 - sent]`. `Date` drops
  the fraction of a second, and uvicorn rebuilds it only every 10 ticks of
  0.1 s (uvicorn 0.54 `server.py:243-255`), so it can be up to about 1 s old.
  The 1 s staleness allowance assumes RTSM's event loop is not stalled. On the
  same host this makes ages read up to about 2 s older than they are.
- **From the pose** (when `pose_clock` is `"server"`): `timestamp + age_s` is
  RTSM's `time.time()` at response time to within a millisecond, giving
  `[S - fetched, S - sent]`. RTSM's ZeroMQ path also tags a sender's own wall
  stamp `"server"` (`ingest_frontend.py:133, 503-508`), so this interval is used
  only when it overlaps the `Date` interval; then the overlap is used. A bag or
  phone pose (`"sender"`) leaves only `Date`. `TargetFix.clock_source` says which
  one applied.

Two things the clock cannot see. RTSM stamps `last_seen_wall_utc` when its
pipeline processes the frame (`working_memory.py:337, 433`), not when the
camera captured it, so pipeline queue lag makes every age read younger than the
observation. And loop-closure corrections move objects within a session
(`sessions.py:474-513`), so a held `TargetFix` can drift without
`frame_changed`; re-query rather than holding targets for long.

Pose age comes from RTSM's own monotonic clock (`age_s`) plus our measured
round trip.

Correction (2026-10-25): the first version of this section said `Date`
truncation explained a same-host reading of -1.8 s. Truncation alone cannot go
below -1 s minus the round trip on one host; the uvicorn refresh interval
accounts for the rest. The first version also measured ages on our clock only,
so RTSM running ahead within tolerance made old hits pass as fresh (review of
#385).

## Verified against a real RTSM

`rtsm==0.3.0` on a Modal A10G, `rtsm --bag rgbd_dataset_freiburg1_desk.bag
--bag-speed 1.0`, polled every 2 s for 71 polls (2026-10-26 05:02-05:05 UTC,
2026-10-25 PDT). The bridge ran
live against it and parsed every response without error. It returned `no_pose`
until the first frame, then `empty`, then `ok` / `none_qualify` as keyboard
hits aged past `max_age_s=5`, and `pose_stale` once the bag ended. Three of the
responses are kept verbatim in `tests/fixtures/rtsm/fr1_desk_0.3.0.json` and
replayed by `test_recorded_rtsm_responses`; they still produce the same statuses
under the corrected clock rule. The bag's poses carry bag time (`"sender"`),
so that run exercised only the `Date` bound.

The bag is a single session (`frame_epoch` 1 throughout), so the
`frame_transition` path is covered by fake responses only. It has not been
seen against a live epoch bump.

## Not covered

- `/search/spatial`: its hits carry no `last_seen_wall_utc`, so they cannot be
  freshness-gated.
- MCP (`/mcp/sse`): the tools return no `robot_pose` or `frame_epoch`.
- Converting the target into the robot base frame. The bridge returns both in
  the SLAM world frame; the consumer applies its own transform.

## Alternatives considered

| Option | Why not |
|---|---|
| `rtsm.client.RtsmClient` (Apache-2.0, needs only `requests`) | Replaces malformed fields with defaults (`score` 0.0, `id` "None"), treats a malformed pose as no pose, has no session bracket, and the wheel is 22 MB. The tests use it as a differential oracle when it is installed. |
| MCP `/mcp/sse` | Needs an MCP client and an SSE session; tool output has no pose or epoch. |
| Hand-written dict checks | Pydantic 2 is already a Tether dependency; strict models check types, finiteness and vector lengths. |
