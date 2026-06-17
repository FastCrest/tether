# Batching

`tether serve` queues `/act` requests through a per-policy `PolicyRuntime` and flushes batches based on **estimated GPU-ms cost**, not fixed request count. Replaces the legacy `--max-batch=N` semantics in Phase 1.

## Quick start

```bash
# Default — 100ms cost budget, 5ms timeout
tether serve ./my-export/

# Tune for throughput (deeper batches, longer per-request latency)
tether serve ./my-export/ --max-batch-cost-ms 250

# Tune for tail latency (smaller batches, faster flush)
tether serve ./my-export/ --max-batch-cost-ms 50 --batch-timeout-ms 2
```

That's the customer surface. The runtime + scheduler are always on; there's no opt-in flag because there's nothing to opt into — it's just better default behavior.

## How it decides to flush

The scheduler flushes a pending batch when ANY of these is true:

1. **Budget reached.** Summed estimated GPU-ms ≥ `--max-batch-cost-ms`. The good case — the scheduler is doing its job.
2. **Timeout.** The oldest queued request has waited ≥ `--batch-timeout-ms`. Fires under low traffic so a single request doesn't sit forever.
3. **Single request over budget.** A single request whose estimated cost already exceeds the budget — no point waiting for more.

Cost estimates come from a per-policy rolling-window cost model (`GpuMsCostModel`). The model starts with a 50 ms cold-start default per shape; after 3 measurements it switches to the rolling median. Updates happen post-flush with the actual measured wall-clock divided by batch size.

## Why cost-weighted, not count-weighted

A naïve `--max-batch=8` ignores per-request cost variance:
- 8 cache-hit requests (~50 ms each) = 400 ms / batch — fine
- 8 cache-miss requests (~400 ms each) = 3200 ms / batch — blows the SLO

`--max-batch-cost-ms 100` adapts: 2 cache-hits + 1 cache-miss is the same scheduler decision as 8 cache-hits, even though one is "size 3" and the other is "size 8".

## Migration from `--max-batch`

The legacy `--max-batch=N` flag still parses but is **ignored at the runtime layer** — `PolicyRuntime` always uses `--max-batch-cost-ms`. Setting `--max-batch > 1` triggers a one-time deprecation warning at startup. Migration formula:

```
--max-batch-cost-ms ≈ max_batch × per_request_cost_estimate_ms
```

For the 50 ms cold-start default: `--max-batch 8` → `--max-batch-cost-ms 400`. Adjust based on your measured per-request latency in production.

## Metrics

Every flush emits five Prometheus series. Labels: `embodiment` × `policy_slot` (`prod` for single-policy; `a`/`b`/`shadow` after policy-versioning lands).

| Metric | Type | Description |
|---|---|---|
| `reflex_batch_cost_per_flush_ms` | Histogram | Estimated GPU-ms cost of each flushed batch. Use `histogram_quantile(0.99, ...)` to track tail. |
| `reflex_batch_size_per_flush` | Histogram | Request count per flush. p50 < `--max-batch-cost-ms / 50` is normal under steady state. |
| `reflex_batch_flush_total` | Counter | Cumulative flushes by reason (`budget_reached` / `timeout` / `single_request_over_budget`). High `timeout` ratio → load too low to benefit from batching; consider lowering `--batch-timeout-ms`. |
| `reflex_captured_graph_hit_rate` | Gauge | Fraction of recent flushes whose batch landed on a captured-graph shape. Phase 1 single-shape: always 1.0. Phase 2 mixed-shape: < 1.0 reveals where eager-fallback bites. |
| `reflex_policy_runtime_queue_depth` | Gauge | Current pending requests. Sustained > 10 with `--max-concurrent` unset → throughput is bottlenecked. |

The `captured_graph_hit_rate` gauge composes with `--cuda-graphs` (per `docs/cuda_graphs.md`) — when both are set, the hit rate tells you how often the scheduler actually exercised the captured graph path vs eager fallback.

## Backpressure

When the queue hits its capacity (default 1000 requests), `/act` returns `HTTP 503` with:

```json
{
  "error": "queue_full",
  "message": "policy runtime queue at capacity",
  "policy_id": "prod",
  "max_queue": 1000
}
```

and a `Retry-After: 1` header. Matches `WebhookDispatcher` and `ConcurrencyLimiter` overload conventions. Customers should retry on receipt — well-behaved clients with backoff are fine.

This protects against unbounded backlog → memory blow-up under load. If you're hitting `queue_full` regularly, you likely also want `--max-concurrent N` (per `docs/auth.md`) at the request entry layer to bound concurrency upstream of the runtime.

## Tuning

| Symptom | Likely cause | Fix |
|---|---|---|
| p99 latency spikes | Budget too high — long batches | Lower `--max-batch-cost-ms` |
| Throughput plateau | Budget too low — small batches dominate | Raise `--max-batch-cost-ms` (in 50ms steps) |
| `timeout` flush ratio > 80% | Load too low to benefit from batching | Lower `--batch-timeout-ms` (e.g. 1ms) — accept slightly less batching for snappier response |
| `queue_full` 503s | Sustained burst exceeds runtime drain rate | Raise `--max-concurrent` ceiling? Or scale horizontally — one process per GPU. |
| `captured_graph_hit_rate` < 1.0 | Phase 2 mixed-shape batches | Check `gen_ai.action.embodiment` distribution; consider per-embodiment routing. |

## What's not in Phase 1

- **True dynamic-batch ORT dispatch.** Decomposed exports are static-shape (per ADR 2026-04-21); batching today fans out sequentially under the queue. Customers feel the queue / scheduler / per-policy isolation benefits; per-request compute cost is unchanged. Future feature `dynamic-batch-shapes` re-exports with dynamic batch dim.
- **Sub-queue-by-shape separation.** ADR flagged as a Phase 1.5 follow-up if `captured_graph_hit_rate` shows the single-queue design leaves throughput on the table.
- **Cross-policy batching.** Requires policy-versioning's per-embodiment routing (Phase 2).
- **Online bandit cost updates.** Phase 2 refinement if profiled GPU-ms proves insufficient.

## Architectural commitments (ADR)

Per `01_decisions/2026-04-24-chunk-budget-batching-architecture.md`:
- Cost-weighted (GPU-ms) scheduler, not fixed count
- Profiled cost model from Day 1, not deferred-static stub
- Per-policy queue refactor lands here (shared with `policy-versioning`)
- Diagnostic metrics ship with Phase 1, not in a follow-up release
- Single Phase 1 feature bundles the scheduler + decomposed-dispatch fix

Validation experiment (per CLAUDE.md): no Modal run yet — Phase 1 ship validates correctness via the integration test suite (`tests/test_chunk_budget_integration.py`). Throughput-vs-budget Modal A/B runs in Phase 1.5 once customer telemetry tells us where to optimize.
