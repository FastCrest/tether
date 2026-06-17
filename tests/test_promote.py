from __future__ import annotations

from pathlib import Path

from tether.deploy_proof import write_deploy_proof_packet
from tether.promote import (
    decide_promotion,
    format_promotion_human,
    get_builtin_promotion_profile,
    list_promotion_profiles,
    load_promotion_profile,
)


def _receipt(
    tmp_path: Path,
    *,
    passed: bool = True,
    policy_summary: dict | None = None,
    failed_proof_checks: int = 0,
) -> dict:
    checks = [{"name": "server_health_ready", "status": "pass"}]
    for idx in range(failed_proof_checks):
        checks.append({"name": f"failed_{idx}", "status": "fail"})
    receipt = {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "passed": passed,
        "tether_version": "0.0.test",
        "python": "3.12.0",
        "export_dir": str(tmp_path / "export"),
        "output_dir": str(tmp_path / "proof"),
        "duration_ms": 100.0,
        "profile": {"name": "ci", "thresholds": {}},
        "server": {"url": "http://127.0.0.1:18080", "log_tail": ["ready"]},
        "doctor": {"summary": {"pass": 1, "fail": 0, "warn": 0, "skip": 0}},
        "latency": {
            "samples": 3,
            "ttfa_ms": 10.0,
            "first_sample": {"roundtrip_ms": 10.0},
            "roundtrip_ms": {"p50_ms": 8.0, "p95_ms": 9.0, "p99_ms": 10.0},
            "warm_roundtrip_ms": {"p95_ms": 8.0},
            "jitter": {"p95_minus_p50_ms": 1.0},
            "deadline_misses": 0,
            "control_budget": {"missed_samples": 0},
        },
        "security": {"enabled": False, "checks": []},
        "metrics": {"status_code": 200, "metric_names": ["tether_act_latency_seconds"]},
        "trace": {"record_dir": "", "files": []},
        "export_manifest": {"root": str(tmp_path / "export"), "files": [{"path": "model.onnx"}]},
        "checks": checks,
    }
    if policy_summary is not None:
        receipt["policy_diff"] = {
            "enabled": True,
            "fail_on": "any",
            "report_artifact": "policy-diff.json",
            "report": {
                "kind": "tether.policy_diff",
                "mode": "trace_pair",
                "summary": policy_summary,
            },
            "checks": [{"name": "policy_diff_gate", "status": "pass"}],
        }
    return receipt


def _write_packet(tmp_path: Path, receipt: dict) -> Path:
    packet = tmp_path / "proof"
    write_deploy_proof_packet(receipt, packet)
    return packet


def test_decide_promotion_promotes_clean_packet(tmp_path: Path) -> None:
    packet = _write_packet(
        tmp_path,
        _receipt(
            tmp_path,
            policy_summary={
                "verdict": "pass",
                "compared": 3,
                "action_failures": 0,
                "latency_regressions": 0,
                "guard_regressions": 0,
                "shape_failures": 0,
                "missing_candidate": 0,
            },
        ),
    )

    report = decide_promotion(packet)

    assert report["decision"] == "PROMOTE"
    assert report["policy_diff"]["present"] is True
    assert report["summary"]["fail"] == 0
    assert "tether promote - PROMOTE" in format_promotion_human(report)


def test_decide_promotion_blocks_policy_regression(tmp_path: Path) -> None:
    packet = _write_packet(
        tmp_path,
        _receipt(
            tmp_path,
            policy_summary={
                "verdict": "fail",
                "compared": 1,
                "action_failures": 1,
                "latency_regressions": 0,
                "guard_regressions": 0,
                "shape_failures": 0,
                "missing_candidate": 0,
            },
        ),
    )

    report = decide_promotion(packet)

    assert report["decision"] == "BLOCK"
    assert "policy_diff_verdict" in report["summary"]["failed_checks"]
    assert "policy_action_failures" in report["summary"]["failed_checks"]


def test_decide_promotion_rolls_back_active_candidate(tmp_path: Path) -> None:
    packet = _write_packet(
        tmp_path,
        _receipt(tmp_path, passed=False, failed_proof_checks=1),
    )

    report = decide_promotion(packet, candidate_active=True)

    assert report["decision"] == "ROLLBACK"
    assert report["candidate_active"] is True


def test_profile_can_require_policy_diff(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path, _receipt(tmp_path))
    profile = tmp_path / "warehouse-safe.yml"
    profile.write_text(
        """
name: warehouse-safe
thresholds:
  require_policy_diff: true
""",
        encoding="utf-8",
    )

    report = decide_promotion(packet, profile_path=profile)

    assert load_promotion_profile(profile)["name"] == "warehouse-safe"
    assert report["decision"] == "BLOCK"
    assert "policy_diff_present" in report["summary"]["failed_checks"]


def test_builtin_profiles_are_loadable() -> None:
    names = {profile["name"] for profile in list_promotion_profiles()}

    assert {"ci-default", "lab-shadow", "warehouse-safe", "contact-strict"} <= names
    assert load_promotion_profile("warehouse-safe")["thresholds"]["require_auth"] is True
    assert get_builtin_promotion_profile("contact_strict")["name"] == "contact-strict"


def test_warehouse_safe_requires_production_evidence(tmp_path: Path) -> None:
    packet = _write_packet(
        tmp_path,
        _receipt(
            tmp_path,
            policy_summary={
                "verdict": "pass",
                "compared": 3,
                "action_failures": 0,
                "latency_regressions": 0,
                "guard_regressions": 0,
                "shape_failures": 0,
                "missing_candidate": 0,
            },
        ),
    )

    report = decide_promotion(packet, profile_path="warehouse-safe")

    assert report["decision"] == "BLOCK"
    assert "proof_auth_required" in report["summary"]["failed_checks"]
    assert "proof_trace_required" in report["summary"]["failed_checks"]
    assert "proof_guard_required" in report["summary"]["failed_checks"]


def test_lab_shadow_allows_warn_policy_verdict(tmp_path: Path) -> None:
    packet = _write_packet(
        tmp_path,
        _receipt(
            tmp_path,
            policy_summary={
                "verdict": "warn",
                "compared": 3,
                "action_failures": 0,
                "latency_regressions": 1,
                "guard_regressions": 0,
                "shape_failures": 0,
                "missing_candidate": 0,
            },
        ),
    )

    report = decide_promotion(packet, profile_path="lab-shadow")

    assert report["decision"] == "PROMOTE"
    assert report["profile"]["name"] == "lab-shadow"


def test_manifest_hash_mismatch_blocks_promotion(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path, _receipt(tmp_path))
    proof_md = packet / "deployment-proof.md"
    proof_md.write_text(proof_md.read_text(encoding="utf-8") + "\nmodified\n", encoding="utf-8")

    report = decide_promotion(packet)

    assert report["decision"] == "BLOCK"
    assert "packet_manifest_hashes" in report["summary"]["failed_checks"]


def test_decide_accepts_deployment_proof_json_path(tmp_path: Path) -> None:
    packet = _write_packet(tmp_path, _receipt(tmp_path))

    report = decide_promotion(packet / "deployment-proof.json")

    assert report["decision"] == "PROMOTE"
    assert report["packet_dir"] == str(packet.resolve())
