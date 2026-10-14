# Chunk lateness test plan for WIM Robotics

WIM runs GR00T TensorRT on Jetson Thor beside a 1 kHz EtherCAT servo loop
and a ros2_control joint trajectory controller. WIM's policy layer owns the
hold/halt response when the target stream goes stale. Its external 100 Hz
chunk-feed mode has a 6 ms stale deadline. The purpose of this change is to
stop the chunk adapter from returning a late target to that layer.

Nothing in this document is verified on Thor. The local tests use arrays,
fake clocks and unloaded or stubbed server backends. They do not qualify
GR00T inference, TensorRT, EtherCAT, ros2_control or robot stopping behavior.

## Exact contract

All timestamps are seconds on the same monotonic clock. For a chunk pushed
at `t_push`, let `t_ref = t_obs` if the caller supplies an observation timestamp,
otherwise `t_ref = t_push`. Action indices start at zero and
`due(i) = t_ref + i / control_hz`. Lateness in milliseconds is
`1000 * (now - due(i))`. Total age is `1000 * (now - t_ref)`.

`pop_next()` acquires the buffer lock, samples the clock and checks the head.
If lateness is **strictly greater** than `max_lateness_ms`, or total age is
strictly greater than `max_age_ms`, it discards the entire remaining buffer,
counts those actions and returns `None`. Equality is allowed. Empty pops also
return `None`. Subsequent pops return `None` until a fresh push. A valid fresh
push permits emission again; expiry is not an emergency-stop latch.

On arrival, the adapter skips and counts the leading actions already past an
enabled bound. It retains the original indices and due times of the surviving
actions, then applies capacity. If every action is late, it rejects the chunk.
With overwrite enabled, even a rejected replacement discards the old plan,
counting that replacement discard as `stale_overwrites`. Append mode preserves
each chunk's own reference time; head expiry discards the entire queued tail,
including appended chunks. Capacity truncation is not age expiry.

Both bounds default to `None`, which disables expiry and preserves FIFO,
overwrite, append, clear and peek behavior. The legacy return value of an
unbounded append is the capacity-limited requested count, even when there is
less space; bounded pushes return the number actually accepted. No new
dependency is needed.

Peeks do not expire or consume actions. `peek_next()` and `peek_all()` still
return copies. RTC `merge_and_update()` snapshots `peek_all()` before replacing
the chunk, and keeps that snapshot as `prev_chunk_left_over`. Advancing time
alone does not alter that carry snapshot. Once `pop_next()` discards an expired
tail, a subsequent peek returns `None`, and the next RTC merge captures `None`.
Previously captured RTC guidance remains unchanged. A peek/carry snapshot is
not permission to send an action to the robot; execution must call `pop_next()`.

A backwards injected clock is clamped to its last observed value under the
lock, so it cannot rejuvenate a plan. A forward jump can expire it immediately.
Non-finite clock values raise `ValueError`. `t_obs` must be finite and no later
than the sampled clock. Wall-clock, ROS-time and remote-host timestamps must
be converted to this clock domain before use; no synchronization is inferred.
Without `t_obs`, this check cannot detect inference or transport age before push.

## Configuration and integration

| Interface | Knobs |
| --- | --- |
| `ActionChunkBuffer(...)` | `max_lateness_ms=None`, `max_age_ms=None`, `control_hz=100.0`, `clock=time.monotonic`, existing `capacity` |
| `push_chunk(chunk, ..., t_obs=...)` | Optional observation timestamp in monotonic seconds |
| `RtcAdapter.merge_and_update(actions, elapsed_time, t_obs=...)` | Forwards observation time to its supplied buffer |
| `TetherServer(...)` | Optional `max_lateness_ms`, `max_age_ms`, used when `configure_replan()` builds the buffer |
| `TetherServer.configure_replan(...)` | `replan_hz`, `execute_hz`, optional bounds and test clock; omitted bounds use constructor settings |
| `configure_action_buffer(server, ...)` | Common backend configuration using `replan_hz`, `execute_hz`, both bounds and an optional test clock |
| `create_app(...)` | `replan_hz`, `execute_hz`, `max_lateness_ms`, `max_age_ms` |
| `tether serve` | `--replan-hz`, `--execute-hz`, `--max-lateness-ms`, `--max-age-ms` |

Bounds must be finite and nonnegative, or `None`; zero is supported. The control
rate must be finite and positive. Server expiry configuration requires both
replan and execute rates. Configure bounds before sharing a buffer between
threads. Reconfiguration creates a new buffer and a new counter lifetime; it
also updates the RTC adapter's buffer reference.

For WIM's external adapter, an example is:

```python
import time
from tether.runtime.buffer import ActionChunkBuffer

buffer = ActionChunkBuffer(
    capacity=50, control_hz=100, max_lateness_ms=4.0,
    # max_age_ms=100.0,  # optional, independent total-age cap
)
observation_time = time.monotonic()  # capture alongside the observation
# ... infer and transfer a chunk using that observation ...
buffer.push_chunk(actions, t_obs=observation_time)
# On each execution attempt, hand a target to the policy layer only if present.
action = buffer.pop_next()
if action is not None:
    policy_layer.accept_target(action)
# None means no new target. The policy layer owns stale-stream fallback.
```

A 4 ms lateness limit allows action 1 at 10 ms in a 100 Hz schedule. A 4 ms
*total-age* cap expires it at that time even though its lateness is zero.
Do not translate "chunk age under 6 ms" into a total-age cap without accounting
for the intended chunk horizon. Set the two bounds separately.

For an in-process server integration, the corresponding configuration is
`create_app(..., replan_hz=20, execute_hz=100, max_lateness_ms=4.0)` or
`--replan-hz 20 --execute-hz 100 --max-lateness-ms 4.0`. The server configuration
uses `execute_hz` as `control_hz`. The common buffer setup covers legacy/GR00T,
monolithic pi0/pi0.5/SmolVLA, native SmolVLA and decomposed pi0.5 backends.

**Current serving limitation:** HTTP `/act` returns full inference chunks and
never calls `pop_next()`. These flags configure the server's execution buffer
for an in-process consumer; they do not expire HTTP response actions or run a
100 Hz feed. An external HTTP client must apply the buffer locally before
forwarding targets. HTTP RTC serving remains rejected at startup because its
denoise guidance path is not connected. Unit-tested RTC carry-forward and
buffer configuration do not establish an available HTTP RTC mode. ZMQ's policy
server returns chunks and does not use `ActionChunkBuffer` either. The
`Pi05DecomposedInference` BID/action caches are separate caches, not execution
buffers. Record/replay writes traces and is not an executor.

`GET /config` includes buffer settings, a stats snapshot and
`http_act_consumes_buffer: false` when a buffer is configured. `GET /metrics`
exports these snapshots as `tether_action_buffer_<field>` gauges. Gauges reflect
cumulative values in the current buffer lifetime, sampled on scrape. JSONL
request records include an additive `action_buffer` snapshot when a buffer
exists. `RtcAdapter.get_stats()` includes the same snapshot. No response or
metric read triggers expiry.

| Field | Meaning |
| --- | --- |
| `expired_actions` | Leading arrival skips plus actions discarded from an expired tail |
| `late_chunks` | Chunks that lost an action to expiry, counted once per chunk, including discarded appended tails |
| `dropped_replans` | Submitted nonempty chunks rejected because all their actions were late on arrival |
| `max_observed_lateness_ms` | Maximum positive lateness sampled at arrival checks or nonempty pops; zero until a positive sample |
| `stale_overwrites` | Existing counter for replacement or explicit-clear discards, separate from expiry |

A missing replan has no push event and cannot increment `dropped_replans`.
Log expected replan IDs and inference starts in the caller to detect that case.
With lateness alone, an old chunk can continue emitting on-time actions until
its tail runs out: missing a replan does not itself violate an action's due time.
Use the separate total-age cap if the integration also needs a maximum plan
lifetime, and distinguish that cap from a target-stream stale watchdog.
Counters never count capacity truncation as expiry. Stats and peeks are snapshots,
not a background watchdog. There is no new paid service or runtime cost beyond
local bookkeeping, copies and clock reads.

## Deadline ordering and limits

Tether guarantees a synchronous decision about the head at the clock sample
inside each `pop_next()` call. It returns no action after that sampled deadline
has been exceeded. It never holds, repeats, synthesizes an action or commands
a pose on expiry. WIM's policy layer is the only stale-stream fallback in this
adapter contract. The separate, existing ActionGuard and inference-deadline
features have their own behavior; buffer expiry does not invoke them.

Tether gives **no real-time guarantee**. Python's GIL, lock contention, scheduler
jitter, garbage collection, inference load, copies, transport and downstream
queues can delay decisions and delivery. An action can become late between the
clock sample and delivery. There is no timer that runs at the expiry instant.
Popping earlier than an action's due time is still allowed, as before; consumers
must pace their schedule and must not prefetch past the policy's freshness check.
Already handed-off targets cannot be recalled from ros2_control or another queue.

For each action, the software threshold is `due + bound`; a test poll at
`due + 4.001 ms` expires before `due + 6 ms`. That is the fake-clock ordering
assertion across control rates and backend configurations. It does **not** prove
that wall-clock expiry precedes the policy's 6 ms stale timeout. With only one
poll per 10 ms at 100 Hz, the next decision can occur after 6 ms even with a
4 ms bound. Moreover, a stale timeout measured from the last target's receipt
has a different origin from lateness measured from the next action's due time.
At 100 Hz, a 10 ms target interval itself exceeds a 6 ms receipt-based timeout.
WIM must define the actual deadline origin and target refresh cadence, and test
both timestamps. Do not claim strict pre-6 ms physical ordering based solely on
setting a number below 6. The policy's watchdog must remain independent.

## Local automated cases

`tests/test_chunk_lateness.py` exercises the following with a fake clock:

- Late inference with observation time: skip a late prefix, preserve its original
  indices, skip before capacity truncation, reject an entirely late chunk and
  check all counters. Missing observation time deliberately starts age at push.
- Dropped replan: submit no replacement, advance past the pending head's deadline,
  get `None`, discard/count the tail, request replan and remain empty.
- Due-time ordering at 50, 100, 200 and 1000 Hz, with bounds 0, 2 and 5.9 ms.
  Equality emits; a poll just over the bound and before due + 6 ms expires.
  Shared buffer setup is exercised on every create_app backend class, and actual
  app lifespan/config/metrics paths use stub model loads. These are configuration
  tests, not loaded-model or robot tests.
- Total-age expiry while the next action is not yet due; exact equality and
  the next representable clock value; invalid bounds/rates/timestamps; forwards
  and backwards clock jumps; recovery on a fresh chunk.
- Default-off FIFO, append, overwrite, clear and copied peeks across rates.
- Concurrent fresh/late pushes, pops, peeks and stats; account for every submitted
  action as emitted, still pending, overwritten or expired.
- RTC read-only peeks, carry snapshot identity before and after expiry, stats,
  Prometheus publication and JSONL round-trip.

Existing buffer, RTC, server, recorder and full-suite tests also run. Local
execution evidence and the measurement below apply only to this checkout and
this development environment. No test in this change requires a robot.

## Local run evidence

Final full-suite run on 2026-10-14, based on public main
`7efcaa464d6a38fbdea9b0ee427cbfff70f33cf5`, with the uncommitted implementation:

| Checkout | Passed | Failed | Errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Unmodified main, same interpreter/environment | 4177 | 22 | 10 | 140 |
| This working tree | 4249 | 22 | 10 | 140 |

All 72 added tests passed. Failure/error test identities and all 140 skip
identities/reasons match main, after normalizing checkout and sibling receipt
paths. The zero-failure target was **not met**: 21 failures and 10 fixture errors
come from sandbox-denied socket binds; the remaining failure is main's existing
`test_2policy_valid_combo_surfaces_active_banner` assertion. No tests were disabled
or newly skipped to hide these results. Thor and loaded-model timing remain
unverified. The final run took 207.28 seconds.

The full run used the requested interpreter and
`PYTHONPATH=<cwd>/src:/Users/romirjain/dev/.sol-out/tether-mcpdeps:/Users/romirjain/dev/.sol-out/tether-asyncdeps`,
with `-m pytest -o tmp_path_retention_policy=failed --basetemp=/tmp/chunk-bt`,
plus `-ra` and a JUnit output file for baseline comparison. The baseline used a
`git archive 7efcaa4` copy under `/tmp` with equivalent paths. Temporary test
directories were deleted after comparison. Ruff passed for the changed runtime,
observability, test and benchmark files. The CLI's 37 pre-existing lint findings
match main, with no added finding. No dependencies were added and no commit was
created.

## macOS dev-machine jitter measurement

Run `PYTHONPATH=src python scripts/measure_chunk_expiry_jitter.py` with the same
interpreter as the tests. The script warms up 10,000 pops, then measures exactly
1,000,000 real-clock expired pops with a one-action buffer, `control_hz=100` and
`max_lateness_ms=0`. Each push is accepted at equality, and the following pop
expires. There are no sleeps or model inference. This makes every sample an
expiry decision without pretending that Python can schedule a 4 ms timer.

It records both pop-call duration and deadline-to-return overshoot, using
`time.monotonic()` around the call. Deadline-to-return includes push copying,
loop/instrumentation overhead and the pop itself. Distribution tails include
GIL/scheduler interruptions that happened during the run. The p99 minus p50
spread describes observed jitter; neither measurement establishes a worst-case
bound or a timer wakeup distribution. Active competing suite processes must
finish before the reported run.

Measured on 2026-10-14, macOS-26.3.1-arm64-arm-64bit, Python 3.12.12, arm64,
using `/Users/romirjain/dev/studio-wt-s11/.sol-scratch/venv/bin/python`.
These are **macOS dev-machine numbers only**, with no Thor or inference load.

| Measured interval, 1,000,000 pops | p50, ms | p99, ms | max, ms | p99 minus p50, ms |
| --- | ---: | ---: | ---: | ---: |
| Expiring pop duration | 0.001000 | 0.001166 | 2.063209 | 0.000166 |
| Deadline to return overshoot | 0.002334 | 0.002625 | 3.186584 | 0.000291 |

The millisecond-scale maxima despite microsecond-scale medians are why these
samples cannot establish a real-time guarantee. This benchmark does not test
1,000,000 scheduled 100 Hz cycles; `control_hz` defines action spacing, while
one-action chunks and the zero bound exercise the expiry branch in a tight loop.

## Required Thor runs and logs

Run on the actual WIM software/hardware configuration, initially with outputs
disconnected or in WIM's established test setup. Record the Tether revision,
Python/OS/JetPack/TensorRT versions, GR00T export/config hashes, affinity and
isolation settings, power mode, thermal state, control/replan rates, both bounds,
clock source and observation-to-monotonic conversion. Log without blocking the
servo loop, using preallocated storage or WIM's existing tracing path.

For every execution attempt, log monotonic `t_obs`, inference start/end,
`t_push`, chunk ID, original action index, `t_due`, configured expiry deadline,
pop entry/clock-sample/return times, result present/None, all four counters,
target send and receipt times, and the policy watchdog's stale origin,
deadline and hold/halt transition. Log rejected/skipped chunks even if no action
is emitted. Correlate the policy and controller records by sequence number and
clock domain. Instrument the sampled clock with the injectable `clock` callback
if its exact timestamp is needed; do not infer it from HTTP latency.

1. **Co-residency.** Keep GR00T inference on non-isolated cores as in WIM's
   deployment. Run the 1 kHz EtherCAT servo and ros2_control concurrently, both
   idle inference and sustained/replanning inference. Log planned and actual
   servo start time, cycle-to-cycle interval and deviation from 1 ms **per cycle**,
   execution time, missed cycles, CPU affinity/load and inference overlap.
   Report p50/p99/max, all outliers and test duration, plus the expiry timing
   distribution. Measure co-residency effects; do not substitute model latency
   for servo cycle deviation.
2. **Late inference.** Delay GR00T completion and transport independently so
   some and then all actions are late at arrival. Supply the real observation
   timestamp. Confirm skipped indices/counters, no old tail emission after
   expiry, no synthesized target, and fresh-plan recovery. Repeat without
   observation time to demonstrate the documented visibility limit.
3. **Dropped replan.** Suppress a scheduled inference result entirely. Confirm
   the caller's missing-replan log, expiry of the old tail on the first eligible
   pop, `None` thereafter, and the policy's hold/halt based on its own stale
   watchdog. Repeat with an already empty buffer and with concurrent pushes.
   Also keep popping the old plan on time: lateness alone permits that tail,
   whereas the optional total-age cap stops it at the first pop past the cap.
4. **Deadline ordering across modes.** Exercise WIM's external 100 Hz feed,
   each in-process buffer backend actually deployed, and standalone RTC only if
   WIM has a qualified policy contract. Test 50/100/200/1000 Hz and bounds 0/2/4/5.9
   ms, exact boundary, missed polls and inference contention. Run a sufficiently
   fast decision poll as a separate case from a once-per-action poll. Compare
   threshold, actual decision and policy stale-transition times using both
   due-based and receipt-based origins. Record positive and negative margin to
   the 6 ms deadline for every event. Fail any claimed strict-ordering setup if
   one measured decision crosses that deadline. HTTP RTC is unavailable, and
   ordinary HTTP/ZMQ full-chunk serving needs the external adapter; record those
   limits rather than reporting them as passed execution modes.
5. **Regression and fault handling.** Disable both bounds and compare the same
   inputs against main. Exercise total-age caps, reconnection/new episodes,
   queue backlog, clock-domain errors and policy recovery rules. Validate
   that ros2_control cannot keep executing prefetched stale targets after the
   policy says stale. WIM sets the acceptance limits and owns the fallback.

Keep raw per-cycle and per-pop logs alongside summarized distributions. Report
misses and unavailable cases, including any configuration that cannot beat
6 ms. Thor qualification remains outstanding until WIM runs these cases.
