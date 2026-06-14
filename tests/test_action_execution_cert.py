"""Tests for action-execution certificates."""

from __future__ import annotations

from tether.action_execution_cert import build_action_execution_certificate


def _receipt(*, roundtrip_ms: float = 90.0):
    return {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "act_samples": [
            {
                "sample": 1,
                "roundtrip_ms": roundtrip_ms,
                "actions": [[0.0, 0.0], [0.04, 0.02], [0.08, 0.04]],
                "action_execution": {
                    "executed_horizon": 3,
                    "adaptive_reason": "low_speed_transition",
                    "phase_transition_indices": [2],
                    "cache_status": "hit",
                },
            },
            {
                "sample": 2,
                "roundtrip_ms": roundtrip_ms,
                "actions": [[0.09, 0.04], [0.13, 0.06], [0.17, 0.08]],
                "action_execution": {
                    "executed_horizon": 3,
                    "adaptive_reason": "low_speed_transition",
                    "phase_transition_indices": [2],
                    "cache_status": "hit",
                },
            },
        ],
    }


def test_action_execution_certificate_passes_smooth_chunks() -> None:
    report = build_action_execution_certificate(
        _receipt(),
        control_hz=20.0,
        require_phase_aware_horizon=True,
    )

    assert report["kind"] == "tether.action_execution_certificate"
    assert report["decision"] == "PASS"
    assert report["metrics"]["stale_action_window_ms"]["max_ms"] == 0.0
    assert report["metrics"]["chunk_boundary_delta"]["max_abs"] == 0.01
    assert report["metrics"]["velocity_discontinuity"]["max_abs"] == 0.0
    assert report["summary"]["fail"] == 0


def test_action_execution_certificate_fails_stale_and_jumpy_chunks() -> None:
    receipt = _receipt(roundtrip_ms=250.0)
    receipt["act_samples"][1]["actions"] = [[1.0, 1.0], [1.8, 1.8], [2.6, 2.6]]

    report = build_action_execution_certificate(
        receipt,
        control_hz=20.0,
        max_stale_action_window_ms=50.0,
        max_chunk_boundary_delta=0.15,
        max_velocity_discontinuity=0.2,
    )

    assert report["decision"] == "FAIL"
    assert "stale_action_window_within_budget" in report["summary"]["failed_checks"]
    assert "chunk_boundary_delta_within_budget" in report["summary"]["failed_checks"]
    assert "velocity_discontinuity_within_budget" in report["summary"]["failed_checks"]


def test_action_execution_certificate_fails_missing_chunks() -> None:
    report = build_action_execution_certificate(
        {"kind": "tether.deployment_proof", "act_samples": [{"roundtrip_ms": 10.0}]},
        control_hz=20.0,
    )

    assert report["decision"] == "FAIL"
    assert "action_chunks_present" in report["summary"]["failed_checks"]
    assert "execution_horizon_present" in report["summary"]["failed_checks"]
