"""Tests for the first-class deployment proof receipt."""

from __future__ import annotations

import json

from tether.deploy_proof import (
    _run_policy_diff_evidence,
    _redact_command,
    format_deploy_proof_human,
    format_deploy_proof_markdown,
    load_deploy_profile,
    summarize_deploy_latency,
    write_deploy_proof_packet,
)


def _receipt(tmp_path):
    return {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "passed": True,
        "tether_version": "0.0.test",
        "python": "3.12.0",
        "export_dir": str(tmp_path / "export"),
        "output_dir": str(tmp_path / "proof"),
        "duration_ms": 123.4,
        "profile": {"name": "ci", "thresholds": {}},
        "server": {
            "url": "http://127.0.0.1:18080",
            "exit_code": 0,
            "log_tail": ["server ready"],
        },
        "doctor": {"summary": {"pass": 4, "fail": 0, "warn": 1, "skip": 2}},
        "latency": {
            "samples": 3,
            "ttfa_ms": 10.0,
            "first_sample": {"inference_ms": 4.0, "roundtrip_ms": 10.0},
            "inference_ms": {"p50_ms": 2.5, "p95_ms": 3.9, "p99_ms": 4.0, "max_ms": 4.0},
            "roundtrip_ms": {"p50_ms": 8.1, "p95_ms": 9.9, "p99_ms": 10.0, "max_ms": 10.0},
            "warm_inference_ms": {"p50_ms": 2.0, "p95_ms": 2.5, "p99_ms": 2.5, "max_ms": 2.5},
            "warm_roundtrip_ms": {"p50_ms": 7.0, "p95_ms": 8.1, "p99_ms": 8.1, "max_ms": 8.1},
            "jitter": {"p95_minus_p50_ms": 1.8},
            "control_budget": {"control_hz": 50.0, "period_ms": 20.0, "missed_samples": 0},
            "deadline_misses": 0,
        },
        "security": {"enabled": True, "checks": [{"status": "pass"}], "probes": [{}]},
        "metrics": {"status_code": 200, "metric_names": ["tether_act_latency_seconds"]},
        "trace": {"record_dir": str(tmp_path / "traces"), "files": [{"path": "trace.jsonl.gz"}]},
        "policy_diff": {
            "enabled": True,
            "baseline_trace": str(tmp_path / "traces" / "current.jsonl.gz"),
            "candidate_trace": str(tmp_path / "traces" / "candidate.jsonl.gz"),
            "shadow": False,
            "fail_on": "any",
            "report_artifact": "policy-diff.json",
            "error": None,
            "report": {
                "kind": "tether.policy_diff",
                "summary": {
                    "verdict": "pass",
                    "compared": 3,
                    "baseline_requests": 3,
                },
            },
            "checks": [{"name": "policy_diff_gate", "status": "pass"}],
        },
        "export_manifest": {"root": str(tmp_path / "export"), "files": [{"path": "model.onnx"}]},
        "checks": [
            {"name": "server_health_ready", "status": "pass"},
            {"name": "latency_roundtrip_p95", "status": "pass"},
            {"name": "policy_diff_gate", "status": "pass"},
        ],
    }


def test_load_deploy_profile_merges_yaml_defaults(tmp_path):
    profile = tmp_path / "production.yml"
    profile.write_text(
        """
schema_version: 1
name: production
thresholds:
  max_roundtrip_p95_ms: 200
  require_auth: true
  require_record_trace: true
"""
    )

    loaded = load_deploy_profile(profile)

    assert loaded["name"] == "production"
    assert loaded["thresholds"]["max_roundtrip_p95_ms"] == 200
    assert loaded["thresholds"]["require_auth"] is True
    assert loaded["thresholds"]["max_doctor_failures"] == 0
    assert loaded["profile_path"].endswith("production.yml")


def test_redact_command_removes_api_key_from_receipt():
    cmd = ["python", "-m", "tether.cli", "serve", "export", "--api-key", "secret"]

    assert _redact_command(cmd) == [
        "python",
        "-m",
        "tether.cli",
        "serve",
        "export",
        "--api-key",
        "<redacted>",
    ]
    assert cmd[-1] == "secret"


def test_summarize_deploy_latency_reports_realtime_fields():
    summary = summarize_deploy_latency(
        [
            {"latency_ms": 10.0, "roundtrip_ms": 15.0},
            {"latency_ms": 20.0, "roundtrip_ms": 25.0, "deadline_exceeded": True},
            {"latency_ms": 100.0, "roundtrip_ms": 110.0, "error": "timeout"},
        ],
        control_hz=20.0,
    )

    assert summary["samples"] == 3
    assert summary["ttfa_ms"] == 15.0
    assert summary["roundtrip_ms"]["p95_ms"] == 101.5
    assert summary["roundtrip_ms"]["p99_ms"] == 108.3
    assert summary["warm_roundtrip_ms"]["p95_ms"] == 105.8
    assert summary["jitter"]["p95_minus_p50_ms"] == 76.5
    assert summary["control_budget"]["period_ms"] == 50.0
    assert summary["control_budget"]["missed_samples"] == 1
    assert summary["deadline_misses"] == 1
    assert summary["act_errors"] == 1


def test_deploy_proof_formatters_and_packet_manifest(tmp_path):
    receipt = _receipt(tmp_path)

    human = format_deploy_proof_human(receipt)
    markdown = format_deploy_proof_markdown(receipt)

    assert "tether deploy-proof - PASS" in human
    assert "roundtrip p50/p95/p99=8.1/9.9/10.0ms" in human
    assert "policy diff: PASS" in human
    assert "- Status: PASS" in markdown
    assert "- Roundtrip p99: 10.0 ms" in markdown
    assert "- API-key checks enabled: True" in markdown
    assert "- Verdict: `pass`" in markdown

    proof_dir = tmp_path / "proof"
    manifest = write_deploy_proof_packet(receipt, proof_dir)

    assert (proof_dir / "deployment-proof.json").exists()
    assert (proof_dir / "deployment-proof.md").exists()
    assert (proof_dir / "policy-diff.json").exists()
    assert (proof_dir / "server.log").read_text() == "server ready\n"
    assert (proof_dir / "MANIFEST.json").exists()
    names = {item["name"] for item in manifest["files"]}
    assert {
        "deployment-proof.json",
        "deployment-proof.md",
        "profile.json",
        "export-manifest.json",
        "policy-diff.json",
    } <= names
    body = json.loads((proof_dir / "deployment-proof.json").read_text())
    assert body["kind"] == "tether.deployment_proof"
    policy_diff = json.loads((proof_dir / "policy-diff.json").read_text())
    assert policy_diff["kind"] == "tether.policy_diff"


def test_policy_diff_evidence_gates_failures(monkeypatch, tmp_path):
    report = {
        "kind": "tether.policy_diff",
        "mode": "trace_pair",
        "summary": {
            "verdict": "fail",
            "compared": 1,
            "action_failures": 1,
        },
    }

    def fake_diff_policy_traces(**kwargs):
        assert kwargs["baseline_trace"] == tmp_path / "current.jsonl.gz"
        assert kwargs["candidate_trace"] == tmp_path / "candidate.jsonl.gz"
        return report

    monkeypatch.setattr("tether.policy_diff.diff_policy_traces", fake_diff_policy_traces)

    evidence = _run_policy_diff_evidence(
        baseline_trace=tmp_path / "current.jsonl.gz",
        candidate_trace=tmp_path / "candidate.jsonl.gz",
        shadow=False,
        fail_on="any",
        min_action_cos=0.995,
        max_action_delta=0.10,
        max_latency_regression_pct=0.10,
    )

    assert evidence["enabled"] is True
    assert evidence["report"] == report
    gate = [check for check in evidence["checks"] if check["name"] == "policy_diff_gate"][0]
    assert gate["status"] == "fail"
