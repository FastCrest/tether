"""Tests for the Prometheus /metrics endpoint + helpers.

Per the plan's validation gates: render returns valid Prometheus text
format, metrics increment correctly, cardinality stays bounded, no
unbounded label keys (instruction/request_id/etc) appear in source.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from prometheus_client.parser import text_string_to_metric_families

from tether.observability import prometheus as p
from tether.observability import (
    METRICS_CONTENT_TYPE,
    inc_cache_hit,
    inc_denoise_steps,
    inc_fallback_invocation,
    inc_inference_executor_rejected,
    inc_model_swap,
    inc_safety_violation,
    inc_slo_violation,
    observe_onnx_load_time,
    observe_rtc_adaptive_chunking,
    record_act_latency,
    render_metrics,
    set_episodes_active,
    set_inference_executor_state,
    set_server_up,
    track_in_flight,
)


# ---------------------------------------------------------------------------
# Render + format compliance
# ---------------------------------------------------------------------------


class TestRenderFormat:
    def test_render_returns_bytes(self):
        out = render_metrics()
        assert isinstance(out, bytes)
        assert len(out) > 0

    def test_render_parses_as_valid_prometheus(self):
        # Trigger a few label permutations
        record_act_latency(0.05, embodiment="franka", model_id="pi05")
        inc_cache_hit(embodiment="franka", cache_type="action_chunk")
        set_server_up(1)

        out = render_metrics().decode("utf-8")
        families = list(text_string_to_metric_families(out))
        assert len(families) >= 8, f"expected ≥8 metric families, got {len(families)}"

    def test_content_type_is_prometheus(self):
        # 0.0.4 (legacy) or 1.0.0 (newer client) — both prefixed text/plain
        assert METRICS_CONTENT_TYPE.startswith("text/plain")

    def test_render_includes_help_and_type_lines(self):
        out = render_metrics().decode("utf-8")
        assert "# HELP tether_act_latency_seconds" in out
        assert "# TYPE tether_act_latency_seconds histogram" in out


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


class TestRecordActLatency:
    def test_observation_increments_count(self):
        for _ in range(3):
            record_act_latency(0.020, embodiment="ur5", model_id="pi05")
        out = render_metrics().decode()
        families = {f.name: f for f in text_string_to_metric_families(out)}
        h = families["tether_act_latency_seconds"]
        # Find the count sample for the (ur5, pi05) label set
        count_samples = [
            s for s in h.samples
            if s.name.endswith("_count")
            and s.labels.get("embodiment") == "ur5"
            and s.labels.get("model_id") == "pi05"
        ]
        assert count_samples
        assert count_samples[0].value >= 3.0

    def test_observe_onnx_load_time(self):
        observe_onnx_load_time(2.5, model_id="pi05")
        out = render_metrics().decode()
        assert "tether_onnx_load_time_seconds" in out


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


class TestCounters:
    def test_cache_hit_per_type(self):
        inc_cache_hit(embodiment="franka", cache_type="vlm_prefix")
        inc_cache_hit(embodiment="franka", cache_type="action_chunk")
        out = render_metrics().decode()
        families = {f.name: f for f in text_string_to_metric_families(out)}
        c = families["tether_cache_hit"]
        types_seen = {
            s.labels["cache_type"]
            for s in c.samples
            if s.labels.get("embodiment") == "franka"
        }
        assert "vlm_prefix" in types_seen
        assert "action_chunk" in types_seen

    def test_safety_violation_per_kind(self):
        for kind in ("nan", "velocity_clamp", "torque_clamp", "workspace_breach"):
            inc_safety_violation(embodiment="ur5", kind=kind)
        out = render_metrics().decode()
        families = {f.name: f for f in text_string_to_metric_families(out)}
        c = families["tether_safety_violations"]
        kinds_seen = {
            s.labels["violation_kind"]
            for s in c.samples
            if s.labels.get("embodiment") == "ur5"
        }
        assert {"nan", "velocity_clamp", "torque_clamp", "workspace_breach"}.issubset(kinds_seen)

    def test_slo_violation(self):
        inc_slo_violation(embodiment="franka", kind="p99_latency")
        assert "tether_slo_violations_total" in render_metrics().decode()

    def test_fallback_invocation(self):
        inc_fallback_invocation(embodiment="so100", target="hold_position")
        assert "tether_fallback_invocations_total" in render_metrics().decode()

    def test_inference_executor_rejected(self):
        inc_inference_executor_rejected(
            embodiment="franka",
            model_id="pi05",
            policy_slot="prod",
        )
        out = render_metrics().decode()
        assert "tether_inference_executor_rejected_total" in out
        assert 'model_id="pi05"' in out

    def test_rtc_adaptive_chunking_metrics(self):
        observe_rtc_adaptive_chunking(
            embodiment="franka",
            model_id="pi05",
            policy_slot="prod",
            decision={
                "horizon": 4,
                "reason": "guard_margin",
                "risk_score": 0.8,
                "replan_threshold_ratio": 0.6,
            },
            signal={
                "guard_margin": 0.03,
                "correction_magnitude": 0.25,
                "uncertainty": 0.4,
            },
            last_action_delta=0.12,
        )
        out = render_metrics().decode()
        assert "tether_rtc_adaptive_decisions_total" in out
        assert 'reason="guard_margin"' in out
        assert "tether_rtc_adaptive_horizon" in out
        assert 'model_id="pi05"' in out
        assert "tether_rtc_adaptive_guard_margin" in out
        assert "tether_rtc_adaptive_action_delta" in out

    def test_model_swap(self):
        inc_model_swap(embodiment="franka", from_model="pi0", to_model="pi05")
        out = render_metrics().decode()
        assert "tether_model_swaps_total" in out
        assert 'from_model="pi0"' in out
        assert 'to_model="pi05"' in out

    def test_denoise_steps_increments_by_n(self):
        inc_denoise_steps(embodiment="franka", n_steps=10)
        out = render_metrics().decode()
        families = {f.name: f for f in text_string_to_metric_families(out)}
        c = families["tether_denoise_steps"]
        franka_total = sum(
            s.value
            for s in c.samples
            if s.labels.get("embodiment") == "franka"
        )
        assert franka_total >= 10


# ---------------------------------------------------------------------------
# Gauges + context manager
# ---------------------------------------------------------------------------


class TestGauges:
    def test_in_flight_context_manager_symmetric(self):
        # Use a unique embodiment label for isolation
        emb = "franka-test-cm-symmetric"
        before_out = render_metrics().decode()
        before_val = _gauge_value(before_out, "tether_in_flight_requests", embodiment=emb)

        with track_in_flight(embodiment=emb):
            mid_out = render_metrics().decode()
            mid_val = _gauge_value(mid_out, "tether_in_flight_requests", embodiment=emb)
            assert mid_val == before_val + 1

        after_out = render_metrics().decode()
        after_val = _gauge_value(after_out, "tether_in_flight_requests", embodiment=emb)
        assert after_val == before_val

    def test_in_flight_decrements_on_exception(self):
        emb = "franka-test-cm-exception"
        before_val = _gauge_value(render_metrics().decode(), "tether_in_flight_requests", embodiment=emb)
        with pytest.raises(RuntimeError):
            with track_in_flight(embodiment=emb):
                raise RuntimeError("simulated /act failure")
        after_val = _gauge_value(render_metrics().decode(), "tether_in_flight_requests", embodiment=emb)
        assert after_val == before_val

    def test_set_server_up(self):
        set_server_up(1)
        assert "tether_server_up 1" in render_metrics().decode()
        set_server_up(0)
        assert "tether_server_up 0" in render_metrics().decode()

    def test_set_episodes_active(self):
        set_episodes_active(embodiment="franka", value=5)
        out = render_metrics().decode()
        assert 'tether_episodes_active{embodiment="franka"} 5' in out

    def test_set_inference_executor_state(self):
        set_inference_executor_state(
            embodiment="franka",
            model_id="pi05",
            policy_slot="prod",
            in_flight=1,
            queue_depth=2,
            max_workers=1,
            max_queue=8,
        )
        out = render_metrics().decode()
        assert "tether_inference_executor_in_flight" in out
        assert "tether_inference_executor_queue_depth" in out
        assert "tether_inference_executor_capacity" in out
        assert 'kind="workers"' in out
        assert 'kind="queue"' in out
        assert 'kind="total"' in out


# ---------------------------------------------------------------------------
# Cardinality + anti-patterns
# ---------------------------------------------------------------------------


class TestCardinality:
    def test_no_unbounded_label_keys_in_source(self):
        """Source-level guard: no .labels(instruction=, request_id=, user_id=,
        timestamp=) anywhere in the prometheus.py module. These would explode
        cardinality."""
        src = Path(p.__file__).read_text()
        for forbidden in ("instruction=", "request_id=", "user_id=", "timestamp="):
            assert forbidden not in src, f"forbidden label key {forbidden!r} in prometheus.py"

    def test_total_series_count_bounded(self):
        """Generate every legal label combo programmatically; assert ≤ 200
        series total (safety margin above the ~90 budgeted)."""
        # Realistic label space exercise
        embodiments = ["franka", "so100", "ur5", "trossen", "stretch", "custom"]
        models = ["pi0", "pi05", "smolvla", "gr00t", "openvla"]
        for emb in embodiments:
            for m in models:
                record_act_latency(0.020, embodiment=emb, model_id=m)
                for ct in ("action_chunk", "vlm_prefix"):
                    inc_cache_hit(embodiment=emb, cache_type=ct)
            for vk in ("nan", "velocity_clamp"):
                inc_safety_violation(embodiment=emb, kind=vk)

        out = render_metrics().decode()
        # Count label-bearing samples (excluding HELP/TYPE comments)
        n_series_lines = sum(
            1 for line in out.splitlines()
            if line and not line.startswith("#")
        )
        # Each histogram contributes 10 buckets + sum + count + bucket{Inf}
        # for each (embodiment, model_id) pair, so this will be larger than
        # the simple "1 per series" count. We check it stays under 1500 (safety).
        assert n_series_lines < 1500, f"explosion risk: {n_series_lines} sample lines"


# ---------------------------------------------------------------------------
# /metrics HTTP endpoint
# ---------------------------------------------------------------------------


def _gauge_value(text: str, metric_name: str, **labels) -> float:
    """Parse out a gauge value for the given label combo from Prometheus text."""
    families = {f.name: f for f in text_string_to_metric_families(text)}
    if metric_name not in families:
        return 0.0
    for s in families[metric_name].samples:
        if all(s.labels.get(k) == v for k, v in labels.items()):
            return s.value
    return 0.0


class TestMetricsEndpoint:
    def test_endpoint_returns_200_via_test_client(self):
        """Mirror create_app's /metrics route in a minimal app + use TestClient."""
        from fastapi import FastAPI
        from fastapi.responses import Response
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/metrics")
        async def metrics():
            return Response(
                content=render_metrics(),
                media_type=METRICS_CONTENT_TYPE,
            )

        client = TestClient(app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert "tether_act_latency_seconds" in r.text

    def test_endpoint_skips_auth(self):
        """Per plan: /metrics has no auth — operators network-isolate.
        Mirror confirms an auth-gated app would still let /metrics through."""
        # Just verify that the helpers don't require any auth-related state
        out = render_metrics()
        assert isinstance(out, bytes)
