"""Tests for realtime serving certificates built from deployment proof packets."""

from __future__ import annotations

import json

from tether.realtime_cert import (
    build_realtime_certificate,
    format_realtime_certificate_human,
    format_realtime_certificate_markdown,
    load_deploy_proof,
    write_realtime_certificate,
)


def _receipt(tmp_path, *, roundtrip_p95=40.0, missed_samples=0):
    return {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "passed": True,
        "export_dir": str(tmp_path / "export"),
        "profile": {
            "name": "warehouse-safe",
            "thresholds": {
                "control_hz": 20,
                "max_jitter_p95_minus_p50_ms": 10,
            },
        },
        "act_samples": [
            {"roundtrip_ms": 20.0},
            {"roundtrip_ms": 35.0},
            {"roundtrip_ms": roundtrip_p95},
        ],
        "latency": {
            "samples": 3,
            "roundtrip_ms": {
                "p50_ms": 35.0,
                "p95_ms": roundtrip_p95,
                "p99_ms": roundtrip_p95,
                "max_ms": roundtrip_p95,
            },
            "jitter": {"p95_minus_p50_ms": 5.0},
            "control_budget": {
                "control_hz": 20.0,
                "period_ms": 50.0,
                "missed_samples": missed_samples,
            },
            "deadline_misses": 0,
            "act_errors": 0,
        },
    }


def test_build_realtime_certificate_passes_control_budget(tmp_path):
    report = build_realtime_certificate(_receipt(tmp_path), target="agx-orin")

    assert report["kind"] == "tether.realtime_serving_certificate"
    assert report["decision"] == "PASS"
    assert report["control_budget"]["period_ms"] == 50.0
    assert report["control_budget"]["roundtrip_p95_budget_ms"] == 50.0
    assert report["summary"]["fail"] == 0
    assert "PASS" in format_realtime_certificate_human(report)
    assert "Tether Realtime Serving Certificate" in format_realtime_certificate_markdown(report)


def test_build_realtime_certificate_fails_budget_and_misses(tmp_path):
    report = build_realtime_certificate(
        _receipt(tmp_path, roundtrip_p95=80.0, missed_samples=1),
        control_hz=20.0,
        max_control_budget_misses=0,
    )

    assert report["decision"] == "FAIL"
    assert "roundtrip_p95_within_budget" in report["summary"]["failed_checks"]
    assert "control_budget_misses_within_budget" in report["summary"]["failed_checks"]


def test_build_realtime_certificate_recomputes_misses_from_act_samples(tmp_path):
    receipt = _receipt(tmp_path, roundtrip_p95=60.0)
    receipt["latency"]["control_budget"] = {}

    report = build_realtime_certificate(receipt, control_hz=20.0)

    assert report["decision"] == "FAIL"
    assert report["control_budget"]["missed_samples"] == 1
    assert report["control_budget"]["missed_samples_source"] == "act_samples"


def test_build_realtime_certificate_embeds_execution_certificate(tmp_path):
    receipt = _receipt(tmp_path)
    receipt["act_samples"] = [
        {
            "roundtrip_ms": 40.0,
            "actions": [[0.0, 0.0], [0.04, 0.02], [0.08, 0.04]],
            "action_execution": {
                "executed_horizon": 3,
                "adaptive_reason": "low_speed_transition",
                "phase_transition_indices": [2],
                "cache_status": "hit",
            },
        },
        {
            "roundtrip_ms": 40.0,
            "actions": [[0.09, 0.04], [0.13, 0.06], [0.17, 0.08]],
            "action_execution": {
                "executed_horizon": 3,
                "adaptive_reason": "low_speed_transition",
                "phase_transition_indices": [2],
                "cache_status": "hit",
            },
        },
    ]

    report = build_realtime_certificate(
        receipt,
        control_hz=20.0,
        execution_cert=True,
        require_phase_aware_horizon=True,
    )

    assert report["decision"] == "PASS"
    assert report["execution_certificate"]["decision"] == "PASS"
    assert "Action Execution" in format_realtime_certificate_markdown(report)
    assert "execution: PASS" in format_realtime_certificate_human(report)


def test_build_realtime_certificate_fails_when_execution_certificate_fails(tmp_path):
    report = build_realtime_certificate(
        _receipt(tmp_path),
        control_hz=20.0,
        execution_cert=True,
    )

    assert report["decision"] == "FAIL"
    assert report["execution_certificate"]["decision"] == "FAIL"
    assert "action_execution_certificate" in report["summary"]["failed_checks"]


def test_load_and_write_realtime_certificate_packet(tmp_path):
    proof_dir = tmp_path / "proof"
    proof_dir.mkdir()
    (proof_dir / "deployment-proof.json").write_text(
        json.dumps(_receipt(tmp_path), indent=2) + "\n"
    )

    receipt = load_deploy_proof(proof_dir)
    report = build_realtime_certificate(receipt)
    manifest = write_realtime_certificate(report, tmp_path / "cert")

    assert receipt["_source_proof_path"].endswith("deployment-proof.json")
    assert (tmp_path / "cert" / "realtime-serving-cert.json").exists()
    assert (tmp_path / "cert" / "realtime-serving-cert.md").exists()
    assert (tmp_path / "cert" / "MANIFEST.json").exists()
    assert {"realtime-serving-cert.json", "realtime-serving-cert.md"} <= {
        item["name"] for item in manifest["files"]
    }
