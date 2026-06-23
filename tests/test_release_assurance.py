"""Tests for the composed release assurance packet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tether.deploy_proof import write_deploy_proof_packet
from tether.release_assurance import (
    ReleaseAssuranceError,
    build_release_assurance,
    format_release_assurance_markdown,
    write_release_assurance_packet,
)
from tether.runtime.record import RecordWriter


def _receipt(tmp_path: Path, *, roundtrip_p95: float = 40.0) -> dict:
    return {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "timestamp": "2026-06-20T00:00:00.000Z",
        "passed": True,
        "export_dir": str(tmp_path / "export"),
        "output_dir": str(tmp_path / "proof"),
        "profile": {"name": "ci", "thresholds": {}},
        "server": {"log_tail": []},
        "security": {"enabled": True},
        "safety_stress": {
            "enabled": True,
            "source": "embodiment",
            "checks": [{"name": "guard_importable", "status": "pass"}],
        },
        "trace": {
            "record_dir": str(tmp_path / "traces"),
            "files": [{"path": str(tmp_path / "traces" / "trace.jsonl"), "size_bytes": 10}],
        },
        "checks": [{"name": "server_health_ready", "status": "pass"}],
        "act_samples": [
            {
                "sample": 1,
                "roundtrip_ms": 40.0,
                "actions": [[0.0, 0.0], [0.04, 0.02], [0.08, 0.04]],
                "action_execution": {
                    "executed_horizon": 3,
                    "adaptive_reason": "low_speed_transition",
                    "phase_transition_indices": [2],
                    "cache_status": "rtc_carry_hit",
                },
            },
            {
                "sample": 2,
                "roundtrip_ms": 40.0,
                "actions": [[0.09, 0.04], [0.13, 0.06], [0.17, 0.08]],
                "action_execution": {
                    "executed_horizon": 3,
                    "adaptive_reason": "low_speed_transition",
                    "phase_transition_indices": [2],
                    "cache_status": "rtc_carry_hit",
                },
            },
        ],
        "latency": {
            "samples": 2,
            "roundtrip_ms": {
                "p50_ms": 40.0,
                "p95_ms": roundtrip_p95,
                "p99_ms": roundtrip_p95,
                "max_ms": roundtrip_p95,
            },
            "warm_roundtrip_ms": {
                "p50_ms": 40.0,
                "p95_ms": roundtrip_p95,
                "p99_ms": roundtrip_p95,
                "max_ms": roundtrip_p95,
            },
            "jitter": {"p95_minus_p50_ms": 0.0},
            "control_budget": {"control_hz": 20.0, "period_ms": 50.0, "missed_samples": 0},
            "deadline_misses": 0,
            "act_errors": 0,
            "guard_violations": 0,
        },
        "policy_diff": {
            "enabled": True,
            "report": {
                "kind": "tether.policy_diff",
                "summary": {
                    "verdict": "pass",
                    "action_failures": 0,
                    "latency_regressions": 0,
                    "guard_regressions": 0,
                    "shape_failures": 0,
                    "missing_candidate": 0,
                    "shadow_pending": 0,
                    "shadow_errors": 0,
                },
            },
        },
        "export_manifest": {"files": [{"name": "model.onnx", "sha256": "abc"}]},
    }


def _packet(tmp_path: Path, *, roundtrip_p95: float = 40.0) -> Path:
    packet = tmp_path / "proof"
    write_deploy_proof_packet(_receipt(tmp_path, roundtrip_p95=roundtrip_p95), packet)
    return packet


def _shadow_trace(tmp_path: Path) -> Path:
    writer = RecordWriter(
        record_dir=tmp_path / "shadow_trace",
        model_hash="deadbeefcafe0000",
        config_hash="0011223344556677",
        export_dir=str(tmp_path / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CPUExecutionProvider"],
        gzip_output=False,
    )
    seq = writer.write_request(
        chunk_id=0,
        image_b64="aGVsbG8=",
        instruction="pick",
        state=[0.1, 0.2],
        actions=[[0.1, 0.2]],
        action_dim=2,
        latency_total_ms=100.0,
        routing={
            "shadow_sampled": True,
            "shadow_mode": "background",
            "shadow_pending": False,
        },
    )
    writer.write_shadow_result(
        seq=seq,
        actions=[[0.11, 0.21]],
        action_dim=2,
        latency_total_ms=12.0,
        routing={
            "shadow_sampled": True,
            "shadow_mode": "background",
            "shadow_actions": [[0.11, 0.21]],
            "shadow_latency_ms": 12.0,
        },
    )
    writer.write_footer({"total_requests": 1})
    writer.close()
    return writer.filepath


def test_release_assurance_promotes_with_realtime_execution_cert(tmp_path: Path) -> None:
    packet = _packet(tmp_path)

    report = build_release_assurance(
        packet=packet,
        realtime=True,
        control_hz=20.0,
        execution_cert=True,
        require_phase_aware_horizon=True,
    )

    assert report["kind"] == "tether.release_assurance"
    assert report["decision"] == "PROMOTE"
    assert report["realtime_certificate"]["decision"] == "PASS"
    assert report["blocked_by"] == {"components": [], "signals": []}
    assert any(signal["name"] == "chunk_boundary_delta" for signal in report["risk_signals"])


def test_release_assurance_holds_on_realtime_failure(tmp_path: Path) -> None:
    packet = _packet(tmp_path, roundtrip_p95=80.0)

    report = build_release_assurance(
        packet=packet,
        realtime=True,
        control_hz=20.0,
    )

    assert report["decision"] == "HOLD"
    assert "realtime_certificate" in report["blocked_by"]["components"]
    assert "roundtrip_p95_within_budget" in report["blocked_by"]["signals"]


def test_release_assurance_rolls_back_active_candidate_on_failure(tmp_path: Path) -> None:
    packet = _packet(tmp_path, roundtrip_p95=80.0)

    report = build_release_assurance(
        packet=packet,
        realtime=True,
        control_hz=20.0,
        candidate_active=True,
    )

    assert report["decision"] == "ROLLBACK"


def test_release_assurance_shadow_trace_defaults_to_lab_profile(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    (packet / "promotion-decision.json").write_text('{"stale": true}\n', encoding="utf-8")
    trace = _shadow_trace(tmp_path)

    report = build_release_assurance(packet=packet, shadow_trace=trace)

    assert report["decision"] == "PROMOTE"
    assert report["profile"]["name"] == "lab-shadow"
    assert report["shadow_rollout"]["profile"] == "lab-shadow"


def test_write_release_assurance_packet(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    report = build_release_assurance(packet=packet)
    out = tmp_path / "release"

    manifest = write_release_assurance_packet(report, out)

    assert (out / "release-assurance.json").exists()
    assert (out / "release-assurance.md").exists()
    assert (out / "MANIFEST.json").exists()
    assert {row["name"] for row in manifest["files"]} == {
        "release-assurance.json",
        "release-assurance.md",
    }
    body = json.loads((out / "release-assurance.json").read_text())
    assert body["kind"] == "tether.release_assurance"
    assert "# Tether Release Assurance" in format_release_assurance_markdown(report)


def test_release_assurance_packet_refuses_to_overwrite_proof_manifest(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    report = build_release_assurance(packet=packet)

    with pytest.raises(ReleaseAssuranceError, match="separate from the input proof"):
        write_release_assurance_packet(report, packet)


def test_guard_violations_block_promotion() -> None:
    # Regression: a runtime ActionGuard clamp (guard_violations > 0) is a hard-safety
    # signal that must prevent PROMOTE even when the promotion gate passed — instead of
    # being listed in blocked_by while the verdict stays PROMOTE (the prior bug).
    from tether.release_assurance import _blocking_safety_signals, _final_decision

    clamping = [{"name": "guard_violations", "status": "fail"}]
    clean = [{"name": "guard_violations", "status": "pass"}]
    promote = {"decision": "PROMOTE"}

    assert _blocking_safety_signals(clamping) == ["guard_violations"]
    assert _blocking_safety_signals(clean) == []
    # Passing promotion gate + clamping -> HOLD; active candidate clamping -> ROLLBACK.
    assert _final_decision(
        promotion=promote, realtime=None, shadow=None,
        candidate_active=False, safety_blocked=True,
    ) == "HOLD"
    assert _final_decision(
        promotion=promote, realtime=None, shadow=None,
        candidate_active=True, safety_blocked=True,
    ) == "ROLLBACK"
    # No hard-safety signal -> the promotion gate decides (PROMOTE).
    assert _final_decision(
        promotion=promote, realtime=None, shadow=None,
        candidate_active=False, safety_blocked=False,
    ) == "PROMOTE"
