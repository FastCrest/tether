"""Prometheus metrics for `reflex serve`.

12 metrics scoped to the cardinality budget — every label key is a
bounded enum (embodiment, model_id, cache_type, violation_kind,
slo_kind, fallback_target). Free-form per-request labels (instruction,
request_id, user_id, timestamp) are FORBIDDEN — they explode cardinality.

Cardinality budget: 3 embodiments × 6 models × ~5 sub-labels max ≈ 90
series. Within Prometheus single-instance comfort zone (target < 10K).

Uses a dedicated CollectorRegistry (not the global default) for test
isolation + clean cross-process export.

Spec: features/01_serve/subfeatures/_ecosystem/prometheus-grafana.md
Plan: features/01_serve/subfeatures/_ecosystem/prometheus-grafana_plan.md
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Dedicated registry — downstream features import this, NOT the global default.
REGISTRY = CollectorRegistry()

# Prometheus text-format media type (operators: serve /metrics with this header).
METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST  # "text/plain; version=0.0.4; charset=utf-8"


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

# /act request latency. Buckets span Jetson 5ms through cloud-A100 5s tail.
_LATENCY_BUCKETS = (
    0.005, 0.010, 0.020, 0.050, 0.100,
    0.200, 0.500, 1.0, 2.0, 5.0,
)
reflex_act_latency_seconds = Histogram(
    "reflex_act_latency_seconds",
    "End-to-end /act handler wall-clock latency",
    # policy_slot bounded enum: "prod" (single-policy) | "a" | "b" (2-policy mode).
    # Default "prod" preserves series meaning under single-policy deployments.
    # Cardinality: 3 embodiments × 6 models × 3 slots = 54 series (within 10K budget).
    labelnames=("embodiment", "model_id", "policy_slot"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

# ONNX session load time (cold start). Buckets span small CPU loads through
# 60s monolithic-pi0 builds.
_LOAD_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
reflex_onnx_load_time_seconds = Histogram(
    "reflex_onnx_load_time_seconds",
    "ONNX session creation + warmup wall-clock",
    labelnames=("model_id",),
    buckets=_LOAD_BUCKETS,
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

reflex_cache_hit_total = Counter(
    "reflex_cache_hit_total",
    "Cache hits, partitioned by cache type",
    labelnames=("embodiment", "cache_type", "policy_slot"),  # action_chunk | vlm_prefix
    registry=REGISTRY,
)

reflex_cache_miss_total = Counter(
    "reflex_cache_miss_total",
    "Cache misses, partitioned by cache type",
    labelnames=("embodiment", "cache_type", "policy_slot"),
    registry=REGISTRY,
)

reflex_denoise_steps_total = Counter(
    "reflex_denoise_steps_total",
    "Total denoise iterations executed (sum across all /act calls)",
    labelnames=("embodiment", "policy_slot"),
    registry=REGISTRY,
)

reflex_safety_violations_total = Counter(
    "reflex_safety_violations_total",
    "Safety/guard violations partitioned by kind",
    labelnames=("embodiment", "violation_kind"),  # nan | velocity_clamp | torque_clamp | workspace_breach
    registry=REGISTRY,
)

reflex_slo_violations_total = Counter(
    "reflex_slo_violations_total",
    "SLO threshold violations (per-call, observed at /act)",
    labelnames=("embodiment", "slo_kind"),  # p95_latency | p99_latency
    registry=REGISTRY,
)

reflex_fallback_invocations_total = Counter(
    "reflex_fallback_invocations_total",
    "Fallback path invocations (deadline miss, error recovery)",
    labelnames=("embodiment", "fallback_target"),  # previous_chunk | hold_position | abort
    registry=REGISTRY,
)

# Action-similarity fast-path skip counter (action-similarity-fast-path
# Phase 1.5 — FlashVLA). Increments when the inference path returns a
# cached action chunk instead of running the expert. Operator visibility
# on how often the fast path actually triggers (low rate = no real
# benefit; high rate near 100% = threshold may be too lax → drift risk).
reflex_action_skip_total = Counter(
    "reflex_action_skip_total",
    "Action-similarity fast path: cached-action returns instead of expert calls",
    registry=REGISTRY,
)

reflex_model_swaps_total = Counter(
    "reflex_model_swaps_total",
    "Hot-swap events (recorded at swap-complete)",
    labelnames=("embodiment", "from_model", "to_model"),
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

reflex_in_flight_requests = Gauge(
    "reflex_in_flight_requests",
    "/act requests currently being processed",
    labelnames=("embodiment", "policy_slot"),
    registry=REGISTRY,
)

reflex_episodes_active = Gauge(
    "reflex_episodes_active",
    "Distinct episode_ids seen in the last rolling window",
    labelnames=("embodiment",),
    registry=REGISTRY,
)

reflex_server_up = Gauge(
    "reflex_server_up",
    "Server liveness signal — 1 when serving /metrics, 0 on shutdown",
    registry=REGISTRY,
)

# Info-style metric for fleet aggregation (Phase 1 fleet-telemetry feature).
# Static per-process; labels are the only non-constant content. Grafana joins
# other metrics to this one via `instance` to surface human-readable robot_id
# (label cardinality stays flat — one series per process, not per request).
reflex_robot_info = Gauge(
    "reflex_robot_info",
    "Static per-process robot identity. Value always 1. Join via `instance`.",
    labelnames=("robot_id", "embodiment", "model_id"),
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers — typed call-sites keep the surface searchable
# ---------------------------------------------------------------------------


def record_act_latency(
    seconds: float, embodiment: str, model_id: str,
    policy_slot: str = "prod",
) -> None:
    reflex_act_latency_seconds.labels(
        embodiment=embodiment, model_id=model_id, policy_slot=policy_slot,
    ).observe(seconds)


def observe_onnx_load_time(seconds: float, model_id: str) -> None:
    reflex_onnx_load_time_seconds.labels(model_id=model_id).observe(seconds)


def inc_cache_hit(
    embodiment: str, cache_type: str, policy_slot: str = "prod",
) -> None:
    reflex_cache_hit_total.labels(
        embodiment=embodiment, cache_type=cache_type, policy_slot=policy_slot,
    ).inc()


def inc_cache_miss(
    embodiment: str, cache_type: str, policy_slot: str = "prod",
) -> None:
    reflex_cache_miss_total.labels(
        embodiment=embodiment, cache_type=cache_type, policy_slot=policy_slot,
    ).inc()


def inc_denoise_steps(
    embodiment: str, n_steps: int = 1, policy_slot: str = "prod",
) -> None:
    reflex_denoise_steps_total.labels(
        embodiment=embodiment, policy_slot=policy_slot,
    ).inc(n_steps)


def inc_safety_violation(embodiment: str, kind: str) -> None:
    reflex_safety_violations_total.labels(
        embodiment=embodiment, violation_kind=kind
    ).inc()


def inc_slo_violation(embodiment: str, kind: str) -> None:
    reflex_slo_violations_total.labels(
        embodiment=embodiment, slo_kind=kind
    ).inc()


def inc_fallback_invocation(embodiment: str, target: str) -> None:
    reflex_fallback_invocations_total.labels(
        embodiment=embodiment, fallback_target=target
    ).inc()


def inc_action_skip() -> None:
    reflex_action_skip_total.inc()


def inc_model_swap(embodiment: str, from_model: str, to_model: str) -> None:
    reflex_model_swaps_total.labels(
        embodiment=embodiment, from_model=from_model, to_model=to_model
    ).inc()


def set_server_up(value: int) -> None:
    """1 when serving, 0 on graceful shutdown."""
    reflex_server_up.set(value)


def set_robot_info(robot_id: str, embodiment: str, model_id: str) -> None:
    """Publish the identity of this process for fleet-scope Grafana queries.

    Call once at lifespan startup. Safe to call repeatedly; gauge overwrites.
    Empty `robot_id` is treated as unset — caller should skip when so.
    """
    reflex_robot_info.labels(
        robot_id=robot_id, embodiment=embodiment, model_id=model_id,
    ).set(1)


def set_episodes_active(embodiment: str, value: int) -> None:
    reflex_episodes_active.labels(embodiment=embodiment).set(value)


@contextmanager
def track_in_flight(embodiment: str, policy_slot: str = "prod") -> Iterator[None]:
    """Context manager increments/decrements in-flight gauge for safe
    try/finally semantics. Use:

        with track_in_flight(embodiment="franka"):
            result = await predict(...)

    policy_slot defaults to "prod" for single-policy mode; 2-policy mode
    callers pass the slot the request was routed to.
    """
    reflex_in_flight_requests.labels(
        embodiment=embodiment, policy_slot=policy_slot,
    ).inc()
    try:
        yield
    finally:
        reflex_in_flight_requests.labels(
            embodiment=embodiment, policy_slot=policy_slot,
        ).dec()


# ---------------------------------------------------------------------------
# CUDA graphs metrics (Phase 1 cuda-graphs feature)
#
# Per ADR 2026-04-24-cuda-graphs-architecture: two captured graphs per model
# (vlm_prefix + expert_denoise session), one shape per (model × embodiment)
# pair. Labels scoped to bounded enums — session ∈ {vlm_prefix, expert_denoise},
# reason ∈ {capture_failed, replay_failed, explicit_disable}.
#
# Cardinality: ~(3 embodiments × 6 models × 2 sessions) = 36 series per
# counter × 3 counters = 108 series. Within budget.
# ---------------------------------------------------------------------------

reflex_cuda_graph_captured_total = Counter(
    "reflex_cuda_graph_captured_total",
    "Cumulative CUDA graph captures (first successful run) per session",
    labelnames=("embodiment", "model_id", "session"),  # session: vlm_prefix | expert_denoise
    registry=REGISTRY,
)

reflex_cuda_graph_replayed_total = Counter(
    "reflex_cuda_graph_replayed_total",
    "Cumulative CUDA graph replays per session",
    labelnames=("embodiment", "model_id", "session"),
    registry=REGISTRY,
)

reflex_cuda_graph_eager_fallback_total = Counter(
    "reflex_cuda_graph_eager_fallback_total",
    "Cumulative eager fallbacks due to CUDA graph capture/replay failure",
    labelnames=("embodiment", "model_id", "reason"),  # reason: capture_failed | replay_failed | explicit_disable
    registry=REGISTRY,
)

# Distinct from eager_fallback_total: this fires ONCE per session when init-time
# capture fails (e.g., OOM on A10G's limited memory) and we fall back to an
# eager-only session for the rest of the process. Separated so operators can
# distinguish "this hardware can't capture at all" from "in-flight replay
# failed on a request."
reflex_cuda_graph_capture_failed_at_init_total = Counter(
    "reflex_cuda_graph_capture_failed_at_init_total",
    "Capture failures at session-init time (hardware can't capture this session; "
    "falls back to eager for the process lifetime)",
    labelnames=("embodiment", "model_id", "session", "reason"),
    registry=REGISTRY,
)

# Capture is a first-time cost: ~50-200ms on small sessions, up to multi-second
# on a full decomposed vlm_prefix. Buckets span that range.
_CUDA_GRAPH_CAPTURE_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
reflex_cuda_graph_capture_seconds = Histogram(
    "reflex_cuda_graph_capture_seconds",
    "Time spent capturing CUDA graph (first run of a session)",
    labelnames=("embodiment", "session"),
    buckets=_CUDA_GRAPH_CAPTURE_BUCKETS,
    registry=REGISTRY,
)

# Replay buckets match the /act latency budget — replay must stay well under
# the request-level p99 SLO.
_CUDA_GRAPH_REPLAY_BUCKETS = (0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250)
reflex_cuda_graph_replay_seconds = Histogram(
    "reflex_cuda_graph_replay_seconds",
    "Time spent in CUDA graph replay (subsequent runs)",
    labelnames=("embodiment", "session"),
    buckets=_CUDA_GRAPH_REPLAY_BUCKETS,
    registry=REGISTRY,
)


def inc_cuda_graph_captured(embodiment: str, model_id: str, session: str) -> None:
    reflex_cuda_graph_captured_total.labels(
        embodiment=embodiment, model_id=model_id, session=session
    ).inc()


def inc_cuda_graph_replayed(embodiment: str, model_id: str, session: str) -> None:
    reflex_cuda_graph_replayed_total.labels(
        embodiment=embodiment, model_id=model_id, session=session
    ).inc()


def inc_cuda_graph_eager_fallback(embodiment: str, model_id: str, reason: str) -> None:
    reflex_cuda_graph_eager_fallback_total.labels(
        embodiment=embodiment, model_id=model_id, reason=reason
    ).inc()


def inc_cuda_graph_capture_failed_at_init(
    embodiment: str, model_id: str, session: str, reason: str
) -> None:
    reflex_cuda_graph_capture_failed_at_init_total.labels(
        embodiment=embodiment, model_id=model_id, session=session, reason=reason
    ).inc()


def observe_cuda_graph_capture_seconds(embodiment: str, session: str, seconds: float) -> None:
    reflex_cuda_graph_capture_seconds.labels(
        embodiment=embodiment, session=session
    ).observe(seconds)


def observe_cuda_graph_replay_seconds(embodiment: str, session: str, seconds: float) -> None:
    reflex_cuda_graph_replay_seconds.labels(
        embodiment=embodiment, session=session
    ).observe(seconds)


# ---------------------------------------------------------------------------
# Chunk-budget batching metrics (Phase 1 chunk-budget-batching feature)
#
# Per ADR 2026-04-24-chunk-budget-batching-architecture decision #4:
# ship `captured_graph_hit_rate` + `batch_cost_per_flush` diagnostic
# metrics with Phase 1, NOT in a follow-up release. They answer the
# riskiest-assumption gate ("does the scheduler land batches at
# captured-graph batch sizes?") and unblock the Phase 2 compile-cache
# feature's telemetry surface.
#
# Cardinality: bounded enums on (embodiment, policy_slot). Phase 1
# single-policy collapses policy_slot to "prod"; policy-versioning
# adds {a, b, prod, shadow}. ~3 embodiments × 4 slots = 12 series per
# metric; well within budget.
# ---------------------------------------------------------------------------

# Bucket spans the realistic GPU-ms cost range: 10ms (tiny captured-graph
# replay) → 5000ms (worst-case A10G decomposed cache-miss + multi-NFE).
_BATCH_COST_BUCKETS = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
reflex_batch_cost_per_flush_ms = Histogram(
    "reflex_batch_cost_per_flush_ms",
    "Estimated GPU-ms cost of each scheduler-flushed batch",
    labelnames=("embodiment", "policy_slot"),
    buckets=_BATCH_COST_BUCKETS,
    registry=REGISTRY,
)

# Bucket spans batch sizes 1..32. Phase 1 single-shape decomposed dispatch
# has size = queue depth at flush; rarely exceeds 4 in practice today
# (workers drain fast). Future dynamic-shape exports may push this higher.
_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32)
reflex_batch_size_per_flush = Histogram(
    "reflex_batch_size_per_flush",
    "Number of requests in each scheduler-flushed batch",
    labelnames=("embodiment", "policy_slot"),
    buckets=_BATCH_SIZE_BUCKETS,
    registry=REGISTRY,
)

# Counter for flush reasons — operator wants to see whether the budget
# (good — scheduler doing its job) or the timeout (bad — load too low to
# benefit from batching) drives flushes most.
reflex_batch_flush_total = Counter(
    "reflex_batch_flush_total",
    "Cumulative scheduler flushes by reason",
    labelnames=("embodiment", "policy_slot", "reason"),  # reason: budget_reached | timeout | single_request_over_budget
    registry=REGISTRY,
)

# Gauge tracking the rolling captured-graph hit rate (fraction of recent
# flushes whose batch landed at a shape size the cuda-graphs ADR captures).
# Phase 1 single-shape: this is effectively shape_homogeneous == True for
# every flush (everyone has the same shape). Surfaced anyway as a forward
# affordance — Phase 2 mixed-shape batches make this load-bearing.
reflex_captured_graph_hit_rate = Gauge(
    "reflex_captured_graph_hit_rate",
    "Rolling fraction of flushed batches that hit a captured-graph shape",
    labelnames=("embodiment", "policy_slot"),
    registry=REGISTRY,
)

# Gauge for the runtime queue depth — exposed continuously (set per flush)
# so operators can graph backlog vs throughput.
reflex_policy_runtime_queue_depth = Gauge(
    "reflex_policy_runtime_queue_depth",
    "Current PolicyRuntime queue depth (pending requests)",
    labelnames=("embodiment", "policy_slot"),
    registry=REGISTRY,
)


def observe_batch_flush(
    embodiment: str,
    policy_slot: str,
    reason: str,
    batch_cost_ms: float,
    batch_size: int,
    shape_homogeneous: bool,
    queue_depth_after: int,
) -> None:
    """Record one scheduler-flushed batch. Called from PolicyRuntime worker.

    `shape_homogeneous` drives the captured-graph-hit-rate gauge — Phase 1
    single-shape always True; Phase 2 mixed-shape detection here.
    """
    reflex_batch_cost_per_flush_ms.labels(
        embodiment=embodiment, policy_slot=policy_slot,
    ).observe(batch_cost_ms)
    reflex_batch_size_per_flush.labels(
        embodiment=embodiment, policy_slot=policy_slot,
    ).observe(batch_size)
    reflex_batch_flush_total.labels(
        embodiment=embodiment, policy_slot=policy_slot, reason=reason,
    ).inc()
    # Hit rate is a per-flush 0/1 signal — set the gauge to the latest
    # value. Customers smooth via Prometheus rate() over a window in
    # their own dashboards; we don't try to keep a rolling buffer here.
    reflex_captured_graph_hit_rate.labels(
        embodiment=embodiment, policy_slot=policy_slot,
    ).set(1.0 if shape_homogeneous else 0.0)
    reflex_policy_runtime_queue_depth.labels(
        embodiment=embodiment, policy_slot=policy_slot,
    ).set(queue_depth_after)


# ---------------------------------------------------------------------------
# A2C2 correction-head metrics (Phase 1 a2c2-correction feature)
#
# Per a2c2-correction execution plan B.5 Day 3: emit applied vs skipped
# counters with a `reason` label so operators can graph the auto-skip
# behavior. Bounded reasons: applied | cold_start | low_latency |
# high_success.
# ---------------------------------------------------------------------------

reflex_a2c2_applied_total = Counter(
    "reflex_a2c2_applied_total",
    "Cumulative A2C2 correction applications",
    labelnames=("reason",),  # "applied"
    registry=REGISTRY,
)
reflex_a2c2_skipped_total = Counter(
    "reflex_a2c2_skipped_total",
    "Cumulative A2C2 skips by reason",
    labelnames=("reason",),  # cold_start | low_latency | high_success
    registry=REGISTRY,
)


def inc_a2c2_applied(reason: str) -> None:
    reflex_a2c2_applied_total.labels(reason=reason).inc()


def inc_a2c2_skipped(reason: str) -> None:
    reflex_a2c2_skipped_total.labels(reason=reason).inc()


# ---------------------------------------------------------------------------
# Episode cache memory tracking
#
# Note on naming: by Prometheus convention `_total` is reserved for monotonic
# Counters. `reflex_episode_cache_bytes_total` is implemented here as a Gauge
# because the resident byte size goes up on insert AND down on eviction/reset.
# Name preserved verbatim from the GFI spec; flagged for maintainer review.
#
# Cardinality: bounded enums on (embodiment, model_id, policy_slot) — same
# label set as reflex_act_latency_seconds. ~3 × 6 × 3 = 54 series.
# ---------------------------------------------------------------------------

reflex_episode_cache_bytes_total = Gauge(
    "reflex_episode_cache_bytes_total",
    "Resident byte size of the EpisodeCache (sum of past_kv + prefix_pad_masks "
    "across all retained entries)",
    labelnames=("embodiment", "model_id", "policy_slot"),
    registry=REGISTRY,
)


def set_episode_cache_bytes(
    value: int,
    embodiment: str,
    model_id: str,
    policy_slot: str = "prod",
) -> None:
    reflex_episode_cache_bytes_total.labels(
        embodiment=embodiment, model_id=model_id, policy_slot=policy_slot,
    ).set(value)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_metrics() -> bytes:
    """Generate Prometheus text-format payload for the /metrics endpoint."""
    return generate_latest(REGISTRY)
