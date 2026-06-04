"""B.4 transfer-validation gate logic tests.

Pure Python (no torch / no GPU). Verifies the decision rule honors the plan's
PROCEED / PAUSE / ABORT thresholds across the matrix of MSE × delta × latency
edge cases.
"""
from __future__ import annotations

import json

import pytest

from tether.correction.transfer_gate import (
    GateDecision,
    GateThresholds,
    compute_gate_report,
)


def _proceed_inputs():
    return dict(
        in_dist_mse=0.001,
        held_out_mses=[0.0011, 0.0012, 0.001],
        task_success_on=0.94,
        task_success_off=0.87,
        eval_latency_ms=80.0,
    )


class TestGateProceed:
    def test_proceed_when_all_thresholds_met(self):
        r = compute_gate_report(**_proceed_inputs())
        assert r.decision == GateDecision.PROCEED
        assert r.failure_reasons == []

    def test_proceed_at_exact_thresholds(self):
        # ratio exactly 1.2x and delta exactly +5pp at exactly the latency floor
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.0012, 0.0012],
            task_success_on=0.92,
            task_success_off=0.87,
            eval_latency_ms=40.0,
        )
        assert r.decision == GateDecision.PROCEED


class TestGateAbort:
    def test_abort_when_max_ratio_exceeds_abort_threshold(self):
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.001, 0.0021, 0.001],
            task_success_on=0.94,
            task_success_off=0.87,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.ABORT
        assert any("MSE ratio" in r for r in r.failure_reasons)

    def test_abort_when_delta_negative(self):
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.0011],
            task_success_on=0.85,
            task_success_off=0.88,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.ABORT
        assert any("WORSE" in r for r in r.failure_reasons)

    def test_abort_when_in_dist_mse_zero(self):
        r = compute_gate_report(
            in_dist_mse=0.0,
            held_out_mses=[0.001],
            task_success_on=0.94,
            task_success_off=0.87,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.ABORT
        assert any("in_dist_mse" in r for r in r.failure_reasons)

    def test_abort_when_in_dist_mse_negative(self):
        r = compute_gate_report(
            in_dist_mse=-0.5,
            held_out_mses=[0.001],
            task_success_on=0.94,
            task_success_off=0.87,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.ABORT

    def test_abort_when_no_held_out_traces(self):
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[],
            task_success_on=0.94,
            task_success_off=0.87,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.ABORT
        assert any("no held-out" in r for r in r.failure_reasons)


class TestGatePause:
    def test_pause_when_ratio_in_warning_band(self):
        # ratio > proceed-max but ≤ abort-min → PAUSE
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.001, 0.0015, 0.001],  # max ratio 1.5
            task_success_on=0.94,
            task_success_off=0.87,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.PAUSE
        assert any("transfer is weak" in r for r in r.failure_reasons)

    def test_pause_when_delta_below_proceed_min_but_nonneg(self):
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.0011],
            task_success_on=0.89,
            task_success_off=0.87,
            eval_latency_ms=80.0,
        )
        assert r.decision == GateDecision.PAUSE
        assert any("delta" in r for r in r.failure_reasons)

    def test_pause_when_eval_latency_below_floor(self):
        # All thresholds passed but delta measured at 30ms < 40ms floor
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.0011],
            task_success_on=0.94,
            task_success_off=0.87,
            eval_latency_ms=30.0,
        )
        assert r.decision == GateDecision.PAUSE
        assert any("not measured at intended regime" in r for r in r.failure_reasons)


class TestThresholds:
    def test_custom_thresholds_propagate(self):
        # Stricter delta threshold
        th = GateThresholds(success_delta_proceed_min=0.10)
        r = compute_gate_report(
            in_dist_mse=0.001,
            held_out_mses=[0.0011],
            task_success_on=0.94,
            task_success_off=0.87,  # delta = +7pp, but threshold now +10pp
            eval_latency_ms=80.0,
            thresholds=th,
        )
        assert r.decision == GateDecision.PAUSE


class TestReportSerialization:
    def test_report_to_dict_is_json_serializable(self):
        r = compute_gate_report(**_proceed_inputs())
        d = r.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["decision"] == "PROCEED"

    def test_report_markdown_contains_decision(self):
        r = compute_gate_report(**_proceed_inputs())
        md = r.to_markdown()
        assert "Decision: PROCEED" in md
        assert "MSE ratio" in md

    def test_report_markdown_contains_per_trace_rows(self):
        r = compute_gate_report(**_proceed_inputs())
        md = r.to_markdown()
        for i in range(3):
            assert f"| {i} |" in md

    def test_report_write_creates_parent_dir(self, tmp_path):
        r = compute_gate_report(**_proceed_inputs())
        out = tmp_path / "nested" / "deep" / "report.md"
        r.write(out)
        assert out.exists()
        assert "Decision: PROCEED" in out.read_text()

    def test_report_write_json_extension(self, tmp_path):
        r = compute_gate_report(**_proceed_inputs())
        out = tmp_path / "report.json"
        r.write(out)
        loaded = json.loads(out.read_text())
        assert loaded["decision"] == "PROCEED"
        assert loaded["max_mse_ratio"] == pytest.approx(1.2, abs=0.001)


class TestNotes:
    def test_notes_propagate_to_report(self):
        r = compute_gate_report(
            **_proceed_inputs(),
            notes=["Modal A10G synthetic 100ms", "seed 42"],
        )
        md = r.to_markdown()
        assert "Modal A10G synthetic 100ms" in md
        assert "seed 42" in md
