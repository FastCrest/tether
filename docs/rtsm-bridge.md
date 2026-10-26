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
    ...  # drop any target or pose held from an earlier lookup
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
| our `time.time()` minus `last_seen_wall_utc` above `max_age_s` | rejected `stale` |
| `last_seen_wall_utc` ahead of our clock by more than `max_clock_skew_s` | rejected `future_last_seen` |
| no `robot_pose` | `no_pose` |
| `robot_pose.frame_epoch` null | `epochless` |
| pose `stale`, or its `age_s` plus our round trip above `max_pose_age_s` | `pose_stale` |
| sessions disabled, or no live session | `sessions_unavailable` |
| live session changed during the query, or its epoch differs from the pose's | `frame_transition` (retry) |
| no `Date` header, or it differs from our clock by more than `max_clock_skew_s` | `clock_skew` |
| body not matching the 0.3.0 shapes (wrong types, NaN, wrong vector length, zero quaternion) | `ValueError` |
| HTTP error | `requests.HTTPError` |

`TargetFix.age_s(now_mono)` re-ages a held target on our monotonic clock.
`max_age_s` has no default: how old an observation may be depends on the task.

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
session and its epoch equals the pose's epoch, no boundary or reset happened
while the results were built, so hits and pose share a frame. Otherwise the
lookup returns `frame_transition`. This costs two extra local requests.

With `sessions.enable: false` RTSM does not clear old-epoch objects at a
boundary at all, so nothing in a response says which frame a hit is in. The
bridge refuses with `sessions_unavailable` instead of guessing.

## Clocks

`last_seen_wall_utc` is RTSM's wall clock. The bridge measures age on its own
clock and bounds the difference between the two clocks with the response's
HTTP `Date` header. `Date` drops the fraction of a second, so RTSM's clock
minus ours lies between `Date - now - round_trip` and `Date + 1 - now`; the
bridge refuses only when that whole interval is outside `max_clock_skew_s`.
In the recorded same-host run the naive `Date - now` read as low as -1.8 s,
close enough to the 2 s default to refuse without this correction. Pose age
comes from RTSM's own monotonic clock (`age_s`) plus our measured round trip.

## Verified against a real RTSM

`rtsm==0.3.0` on a Modal A10G, `rtsm --bag rgbd_dataset_freiburg1_desk.bag
--bag-speed 1.0`, polled every 2 s for 71 polls (2026-10-26). The bridge ran
live against it and parsed every response without error. It returned `no_pose`
until the first frame, then `empty`, then `ok` / `none_qualify` as keyboard
hits aged past `max_age_s=5`, and `pose_stale` once the bag ended. Three of the
responses are kept verbatim in `tests/fixtures/rtsm/fr1_desk_0.3.0.json` and
replayed by `test_recorded_rtsm_responses`.

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
