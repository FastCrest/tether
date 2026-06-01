"""Tests for Reflex Comply evidence-bundle export."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _seed_b64() -> str:
    return base64.b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")


def _write_verify_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "PARITY.md").write_text("# Reflex Action-Parity Verification\n\n**Verdict: PASS**\n")
    (path / "parity.cert.json").write_text(
        json.dumps(
            {
                "schema_version": "reflex.parity_cert.v1",
                "generated_at": "2026-06-01T00:00:00Z",
                "optimized_ref": "robot-policy/export",
                "original_ref": "robot-policy/native",
                "target": "orin",
                "verdict": "PASS",
                "passed": True,
                "artifacts": {},
            },
            indent=2,
        )
    )
    return path


def _write_audit_log(path: Path) -> Path:
    records = [
        {
            "kind": "header",
            "schema_version": 1,
            "session_id": "session-1",
            "started_at": "2026-06-01T00:00:00.000Z",
            "model_hash": "modelabc",
            "config_hash": "configabc",
            "redaction": {"image": "hash_only", "instruction": "hash_only"},
        },
        {
            "kind": "request",
            "schema_version": 1,
            "seq": 0,
            "timestamp": "2026-06-01T00:00:01.000Z",
            "request": {
                "instruction_hash": "insthash",
                "state": [0.0, 1.0],
                "image_sha256": "imagesha",
            },
            "response": {"actions": [[0.1, 0.2]], "num_actions": 1, "action_dim": 2},
            "latency": {"total_ms": 12.5},
            "guard": {"violations": ["joint_0 above max"], "clamped": True, "clamp_count": 1},
        },
        {
            "kind": "request",
            "schema_version": 1,
            "seq": 1,
            "timestamp": "2026-06-01T00:00:02.000Z",
            "request": {
                "instruction_hash": "insthash2",
                "state": [0.0, 1.0],
                "image_sha256": "imagesha2",
            },
            "response": {"actions": [[0.0, 0.1]], "num_actions": 1, "action_dim": 2},
            "latency": {"total_ms": 10.0},
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _write_actionguard(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "joint_names": ["joint_0", "joint_1"],
                "position_min": [-1.0, -1.0],
                "position_max": [1.0, 1.0],
                "velocity_max": [0.5, 0.5],
                "effort_max": [10.0, 10.0],
                "workspace_min": [-1.0, -1.0, 0.0],
                "workspace_max": [1.0, 1.0, 1.0],
            }
        )
    )
    return path


def test_comply_export_writes_signed_verifiable_bundle(tmp_path):
    from reflex.comply.export import export_conformity_bundle, verify_conformity_bundle
    from reflex.comply.schemas import DeploymentMetadata

    verify_dir = _write_verify_dir(tmp_path / "verify")
    audit = _write_audit_log(tmp_path / "audit.jsonl")
    guard = _write_actionguard(tmp_path / "safety_config.json")
    out = tmp_path / "bundle"

    result = export_conformity_bundle(
        verify_dir=verify_dir,
        audit_log=audit,
        actionguard=guard,
        out_dir=out,
        deployment=DeploymentMetadata(
            product_name="Test Robot",
            robot_id="robot-1",
            manufacturer="Acme Robotics",
            vulnerability_contact="security@acme.example",
        ),
        signing_key=_seed_b64(),
        key_id="test-key",
        include_environment_sbom=False,
    )

    assert result["signed"] is True
    assert (out / "TECHNICAL_FILE.md").exists()
    assert (out / "TECHNICAL_FILE.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "SBOM.cyclonedx.json").exists()
    assert (out / "GAP_REPORT.md").exists()
    assert (out / "TRUST_PAGE.md").exists()
    assert (out / "SECURITY_WHITEPAPER.md").exists()
    assert (out / "VULNERABILITY_HANDLING.md").exists()
    assert (out / "artifacts" / "audit_summary.json").exists()
    assert (out / "artifacts" / "safety_violations.jsonl").read_text().strip()

    conformity = json.loads((out / "conformity.json").read_text())
    assert conformity["schema_version"] == "reflex.comply_bundle.v1"
    assert conformity["signature"]["key_id"] == "test-key"
    assert conformity["evidence"]["audit_summary"]["request_count"] == 2
    assert conformity["evidence"]["audit_summary"]["safety_violation_count"] == 1
    assert conformity["evidence"]["actionguard"]["joint_count"] == 2
    assert conformity["artifact_manifest"]

    verification = verify_conformity_bundle(out, require_signature=True)
    assert verification["passed"], verification


def test_comply_cli_export_and_verify(tmp_path, monkeypatch):
    typer_testing = __import__("typer.testing").testing
    from reflex.cli import app

    verify_dir = _write_verify_dir(tmp_path / "verify")
    audit = _write_audit_log(tmp_path / "audit.jsonl")
    guard = _write_actionguard(tmp_path / "safety_config.json")
    out = tmp_path / "bundle"
    monkeypatch.setenv("REFLEX_COMPLY_TEST_KEY", _seed_b64())

    runner = typer_testing.CliRunner()
    export_result = runner.invoke(
        app,
        [
            "comply",
            "export",
            "--verify-dir",
            str(verify_dir),
            "--audit-log",
            str(audit),
            "--actionguard",
            str(guard),
            "--out",
            str(out),
            "--signing-key",
            "env:REFLEX_COMPLY_TEST_KEY",
            "--key-id",
            "cli-key",
            "--no-env-sbom",
        ],
    )
    assert export_result.exit_code == 0, export_result.output
    assert "evidence bundle exported" in export_result.output

    verify_result = runner.invoke(app, ["comply", "verify-bundle", str(out), "--require-signature"])
    assert verify_result.exit_code == 0, verify_result.output
    assert "PASS" in verify_result.output
