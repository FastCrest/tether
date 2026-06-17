# Policy versioning (2-policy A/B serve mode)

`tether serve --policy-a ./v1/ --policy-b ./v2/ --split 80 --no-rtc` loads two policies side-by-side and routes /act traffic deterministically per-episode. 80% of episodes go to A, 20% to B. Sticky-per-episode (router uses SHA-256 hash of `episode_id`), so cache locality + RTC carry-over (when applicable) is preserved within an episode.

Per ADR `2026-04-25-policy-versioning-architecture`. Phase 1 ships the substrate (router + crash tracker + record-replay schema + Prometheus labels). Full 2-instance load wires in a follow-up; this doc shows what's available today + what lands when.

## Quick start

```bash
# 1. Verify both policies pass `tether doctor` first
tether doctor ./v1/
tether doctor ./v2/

# 2. Start 2-policy serve (80/20 split, RTC off)
tether serve ./v1/ \
    --policy-a ./v1/ \
    --policy-b ./v2/ \
    --split 80 \
    --no-rtc

# 3. /act requests carry routing decisions in the response headers + record-replay trace
curl -X POST http://localhost:8000/act \
    -H "Content-Type: application/json" \
    -d '{"episode_id": "ep_xyz", "image": "...", "instruction": "pick up the cup"}'
# Response headers:
#   X-Tether-Policy-Slot: a
#   X-Tether-Model-Version: pi0-libero-v1@<hash>
```

## Why this exists

Three load-bearing customer signals:
1. **Production rollout** — ship a new policy to 5% of traffic, watch metrics for an hour, ramp to 100%. The classic A/B framework, applied to robot policies.
2. **Risk-free comparison** — load the next-gen policy alongside the current one + compare per-episode metrics in your dashboard, no production traffic risk (set `--split 100` to keep all traffic on A while B is loaded inert).
3. **Self-distilling-serve safety** — the auto-distill loop (`docs/self_distilling_serve.md`) needs a warm secondary slot for ≤60s rollback when the post-swap monitor trips.

## The 5 flags

| Flag | Default | Notes |
|---|---|---|
| `--policy-a <path>` | (unset) | 2-policy mode: path to policy A export. Must be set together with `--policy-b`. |
| `--policy-b <path>` | (unset) | 2-policy mode: path to policy B export. Mutually exclusive with `--shadow-policy`. |
| `--split <int>` | `50` | Percent of episodes routed to A. `0` = all to B; `100` = all to A (shadow-staging). |
| `--shadow-policy <path>` | (unset) | Shadow inference: mirror sampled traffic to a candidate export and append `shadow_result` evidence. |
| `--shadow-sample <float>` | `1.0` | Fraction of `/act` requests mirrored to `--shadow-policy` in `[0, 1]`. |
| `--shadow-queue-size <int>` | `32` | Bounded background shadow queue. `0` disables queueing; overload records `shadow_queue_full`. |
| `--no-rtc` | `false` | **REQUIRED** in 2-policy mode. RTC carry-over is per-policy; cross-policy carry-over produces OOD actions. |

## Sticky-per-episode routing

Routing decision is hashed on `episode_id` (or `request_id` when `episode_id` is missing — the **degraded** path). The first request of an episode picks the slot; all subsequent requests within the same `episode_id` get the same slot. This preserves:

- **9× episode-cache moat** — the `Pi05DecomposedInference` `EpisodeCache` is per-policy. Switching mid-episode destroys the cached `past_kv` and falls back to full denoise.
- **RTC carry-over** — chunk N+1's denoise anchors to chunk N's trailing actions. Cross-policy carry-over produces out-of-distribution actions (this is why `--no-rtc` is enforced when 2-policy mode is active).

Hash distribution is deterministic across processes + Python restarts (uses SHA-256 of the routing key). Same `episode_id` → same slot, every time.

### Degraded mode (no `episode_id`)

When the caller doesn't pass `episode_id`, the router falls back to hashing `request_id`. Each request gets an independent decision → flip-flopping between policies → cache + RTC discontinuities. The router logs a **one-time warning per process** when this happens so operators notice before it bites.

```
WARN policy_router.degraded_mode request_id=req_abc — no episode_id provided.
Per-request routing destroys episode-cache locality and causes RTC carry-over
discontinuities. Callers should pass episode_id on every /act request in
2-policy mode.
```

Fix: every client-side `/act` call should set `episode_id` to a stable identifier for the current task (e.g., `ep_<robot_id>_<task_start_ts>`).

## Per-policy circuit breaker

Each policy slot has its own consecutive-crash counter. When one slot exceeds `--max-consecutive-crashes` (default 5):

| Scenario | Verdict | Action |
|---|---|---|
| Slot A crashes ≥5x; B is clean | `drain-a` | Caller routes 100% to B (set `--split 0` runtime, OR auto via the distill-serve rollback handler). |
| Slot B crashes ≥5x; A is clean | `drain-b` | Mirror: route 100% to A. |
| Both slots crash ≥5x | `degraded` | Full server `degraded` state — both policies are contributing errors; the problem isn't slot-specific. `/health` returns 503. |

A clean response on a slot **resets** that slot's counter to 0. The drain decision is sticky until you reset (operator intervention OR a clean response). Single-policy mode behaves exactly like the legacy single counter (any 5 consecutive crashes → degraded).

## Record-replay schema additions

The JSONL trace produced by `--record` gains two **optional, additive** fields in 2-policy mode (no `schema_version` bump — v1 readers ignore them):

**Header gains a `policies` block** listing each loaded policy:
```json
{
  "kind": "header",
  "schema_version": 1,
  "policies": [
    {"slot": "a", "model_id": "pi0-libero-v1", "model_hash": "aaaa..."},
    {"slot": "b", "model_id": "pi0-libero-v2", "model_hash": "bbbb..."}
  ]
}
```

**Per-request gains a `routing` block** with the decision:
```json
{
  "kind": "request",
  "seq": 0,
  "routing": {
    "slot": "a",
    "routing_key": "ep_xyz",
    "degraded": false,
    "cached": false
  }
}
```

Replay tools that parse the trace can split per-slot statistics by grouping on `routing.slot`.

## Prometheus metrics

5 metrics gain a `policy_slot` bounded-enum label (`prod` | `a` | `b`):
- `reflex_act_latency_seconds`
- `reflex_cache_hit_total`
- `reflex_cache_miss_total`
- `reflex_denoise_steps_total`
- `reflex_in_flight_requests`

Default value `policy_slot="prod"` preserves series meaning under single-policy deployments (existing dashboards continue to work without changes). Cardinality audit: 90 existing series × 4 slot values = 360 series (well within 10K budget).

Example PromQL:
```promql
# A vs B p99 latency
histogram_quantile(0.99,
  sum(rate(reflex_act_latency_seconds_bucket{policy_slot="a"}[5m])) by (le)
)
histogram_quantile(0.99,
  sum(rate(reflex_act_latency_seconds_bucket{policy_slot="b"}[5m])) by (le)
)

# Cache hit rate per slot
rate(reflex_cache_hit_total{policy_slot="a"}[5m])
  / (rate(reflex_cache_hit_total{policy_slot="a"}[5m])
   + rate(reflex_cache_miss_total{policy_slot="a"}[5m]))
```

## Memory check (refuse-to-load)

2-policy mode requires roughly 2× model_size_bytes of GPU VRAM. Before loading the second policy, `tether serve` checks:
```
2 × model_size_bytes > 0.7 × total_gpu_bytes
```

If true, the server **refuses to start** with a clear error message (no silent OOM at first inference). The 0.7 safety factor leaves 30% of VRAM for cuDNN workspace, IO buffers, OS, etc.

```
[red]2-policy mode requires 16.0GB VRAM but only 11.2GB (70% of 16.0GB)
is available. Either pick smaller models, run on a larger GPU, OR drop
to single-policy mode.[/red]
```

## What's NOT shipped Phase 1

- **Canary auto-promotion** — manual operator control over `--split` for now; Phase 2 wires automated ramp-up + rollback based on Prometheus signals.
- **Cross-policy memory pooling** — each policy holds its own ONNX session + buffers; no shared workspace. Phase 2 explores `onnxruntime` IO-binding sharing.

## Shipped 2026-04-25

- ✅ `setup_two_policy_serving` helper composes the substrate (`src/tether/runtime/two_policy_setup.py`)
- ✅ `create_app` lifespan loads 2 ReflexServers + builds dispatcher when `policy_b_export_dir` is set
- ✅ `/act` handler dispatches via `TwoPolicyDispatcher.predict()` when `server.two_policy_state` is set
- ✅ `X-Tether-Policy-Slot` + `X-Tether-Model-Version` + `X-Tether-Routing-Key` + `X-Tether-Routing-Degraded` response headers
- ✅ Per-request `routing` block in record-replay JSONL trace
- ✅ Shadow policy execution records candidate actions as append-only `shadow_result` rows without returning them to the robot client
- ✅ Per-slot `policy_slot` label on Prometheus `reflex_act_latency_seconds`
- ✅ **Per-slot `PolicyRuntime` queue + cost-budget scheduler** (chunk-budget-batching benefit in 2-policy mode)
- ✅ Refuse-to-load memory check fires before either ReflexServer loads
- ✅ Setup failure in lifespan falls back to single-policy serve (logs error; never breaks `/health`)

## Reference: ADR + research

- ADR: `reflex_context/01_decisions/2026-04-25-policy-versioning-architecture.md`
- Research: `reflex_context/features/01_serve/subfeatures/_ecosystem/policy-versioning/policy-versioning_research.md`
- Plan: `reflex_context/features/01_serve/subfeatures/_ecosystem/policy-versioning/policy-versioning_plan.md`
