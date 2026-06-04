# Latency SLO enforcement

`tether serve --slo p99=150ms` makes Tether measure every `/act` response, track a rolling p99, and react when it exceeds your threshold. Three modes:

| Mode | Behavior when p99 > threshold |
|---|---|
| `log_only` | Emit `reflex_slo_violations_total` Prometheus metric only |
| `503` | Metric + return HTTP 503 with `{"p99_measured_ms", "p99_slo_ms", "retry_after_s"}` body + `Retry-After: 1` header |
| `degrade` (default) | Phase 1: same as `log_only`. Phase 1.5 (once `adaptive-denoise-pi0` + `chunk-budget-batching` land): drops NFE 4→2, skips RTC eval, etc. |

## Quick start

```bash
# 503 mode — failover-capable clients get structured 503 when SLO is at risk
tether serve ./my-export/ --slo p99=150ms --slo-mode 503

# log_only — monitor SLO without changing response behavior
tether serve ./my-export/ --slo p99=200ms --slo-mode log_only

# default: --slo-mode degrade (telemetry only until P1.5 degradation knobs land)
tether serve ./my-export/ --slo p99=150ms
```

## Spec format

`--slo p<N>=<X>ms` where `N` is percentile (0-100, fractional allowed) and `X` is threshold in ms:

- `p99=150ms`
- `p95=200ms`
- `p99.9=500ms`
- `p50=50ms`

Only the `ms` unit is supported in Phase 1. Case-insensitive; whitespace around `=` is ignored.

## Recovery semantics

Once the tracker flags a violation, it clears only after:

1. The measured percentile drops below `0.8 × threshold` (default ratio)
2. For `3` consecutive recomputations (default recover_windows)

This prevents flapping on bursty workloads. Defaults chosen to match InferScope's pattern.

## Metric

`reflex_slo_violations_total{embodiment, kind="p{N}_exceeded"}` increments on every request where the tracker is in violation state. Scrape via `/metrics` or via MCP resource `metrics://prometheus`.

## Response shape (503 mode)

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 1
Content-Type: application/json

{
  "error": "slo_violation",
  "p99_measured_ms": 187.43,
  "p99_slo_ms": 150.0,
  "retry_after_s": 1
}
```

Clients that implement failover should retry on a different instance (e.g., cloud backup) without polling the same server. Clients without failover logic will see 503 until the SLO recovers.

## Monitoring

Recommended Grafana panels (not shipped by default; customer-side setup):

- `rate(reflex_slo_violations_total[5m])` — violation rate per minute
- `histogram_quantile(0.99, rate(reflex_act_latency_seconds_bucket[5m]))` — actual rolling p99 (compare against your `--slo`)
- `rate(reflex_act_latency_seconds_count{status="503"}[5m])` — 503 rate (only meaningful with `--slo-mode 503`)

## Phase 1 scope

- Single global SLO on `/act` (per-endpoint SLO deferred to Phase 1.5)
- Three modes above (`degrade` functional in Phase 1.5)
- Rolling-window p99 via numpy (last 1000 samples, recompute every 32 requests)
- Thread-safe across FastAPI workers

## Feature spec

- `features/01_serve/subfeatures/_ecosystem/latency-slo-enforcement/latency-slo-enforcement.md`
- `features/01_serve/subfeatures/_ecosystem/latency-slo-enforcement/latency-slo-enforcement_plan.md`

Pattern source: [InferScope](https://github.com/rylinjames/easyinference) rolling-window tracker (sibling project at `EasyInference-main/products/inferscope/`). Tether's tracker uses the same percentile/recovery semantics with VLA-specific Prometheus labels.
