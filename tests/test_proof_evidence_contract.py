"""Serving receipts never manufacture executed actions from model chunk lengths."""

import json
from types import SimpleNamespace

from tether.action_execution_cert import build_action_execution_certificate
from tether.deploy_proof import run_deploy_proof
from tether.proof_startup import StartupObservation
from tether.realtime_cert import build_realtime_certificate


def chunks(execution=None):
    return {
        "kind": "tether.deployment_proof",
        "act_samples": [
            {"actions": [[0.0], [0.8]], "roundtrip_ms": 70, "action_execution": execution or {}},
            {"actions": [[0.0], [0.8]], "roundtrip_ms": 70, "action_execution": execution or {}},
        ],
    }


def test_missing_execution_does_not_use_chunk_tail_as_executed_boundary():
    report = build_action_execution_certificate(chunks(), control_hz=20)
    assert report["decision"] == "FAIL"
    assert report["metrics"]["chunk_boundary_delta"]["max_abs"] is None
    assert report["metrics"]["velocity_discontinuity"]["max_abs"] is None
    assert report["diagnostics"]["returned_chunk_tail_to_next_head_max_abs"] == 0.8
    check = next(c for c in report["checks"] if c["name"] == "chunk_boundary_delta_within_budget")
    assert check["classification"] == "missing_evidence"


def test_synthetic_proof_cannot_promote_even_an_observed_marker_to_robot_execution():
    receipt = chunks({"evidence_scope": "consumer_observed", "executed_horizon": 2})
    receipt["evidence_contract"] = {"scope": "serving_only"}
    report = build_action_execution_certificate(receipt, control_hz=20)
    assert report["decision"] == "FAIL"
    assert report["metrics"]["execution_horizon_max"] is None
    assert report["physical_execution_verified"] is False


def test_configured_fractional_and_legacy_horizons_are_not_execution():
    for execution in (
        {"execution_horizon": 2},
        {"executed_horizon": 2},
        {"evidence_scope": "consumer_observed", "executed_horizon": 1.9},
    ):
        report = build_action_execution_certificate(chunks(execution), control_hz=20)
        assert report["decision"] == "FAIL"
        assert report["metrics"]["execution_horizon_max"] is None


def test_serving_metadata_does_not_amortize_50ms_budget_by_chunk():
    receipt = chunks()
    receipt.update(
        passed=True,
        evidence_contract={"scope": "serving_only"},
        export_metadata={"fields": {"model_type": "smolvla", "chunk_size": 50}},
        config={"body": {"execution_providers": {"requested": ["TensorrtExecutionProvider"]}}},
        latency={
            "samples": 2,
            "roundtrip_ms": {"p95_ms": 70},
            "control_budget": {},
            "deadline_misses": 0,
            "act_errors": 0,
        },
    )
    report = build_realtime_certificate(receipt, control_hz=20)
    assert report["decision"] == "FAIL"
    assert report["control_budget"]["roundtrip_p95_budget_ms"] == 50
    assert report["control_budget"]["chunk_amortization_applied"] is False
    assert report["model"] == {"model_type": "smolvla", "chunk_size": 50}
    assert report["runtime"]["execution_providers"]["requested"] == ["TensorrtExecutionProvider"]


def test_killed_startup_is_not_declared_oom(monkeypatch):
    monkeypatch.setattr(
        "tether.proof_startup._memory_sample",
        lambda _: {"source": "fake", "rss_bytes": 10, "host_available_bytes": 20},
    )
    observation = StartupObservation(SimpleNamespace(pid=1, poll=lambda: -9), 2)
    try:
        with observation:
            raise RuntimeError("server exited")
    except RuntimeError:
        pass
    assert observation.report["diagnostic"] == "server_exit_before_ready"
    assert observation.report["process_memory"]["oom_confirmed"] is False
    assert observation.report["process_memory"]["gpu_memory_measured"] is False


def test_startup_memory_samples_are_bounded_and_preserve_zero_available(monkeypatch):
    calls = []

    def sample(_):
        calls.append(1)
        return {
            "source": "fake",
            "rss_bytes": len(calls),
            "host_available_bytes": 0 if len(calls) == 1 else 20,
        }

    monkeypatch.setattr("tether.proof_startup._memory_sample", sample)
    observer = StartupObservation(SimpleNamespace(pid=1), 70)
    observer.stop = SimpleNamespace(is_set=lambda: False, wait=lambda _: len(calls) == 70)
    observer._loop()
    memory = observer.report["process_memory"]
    assert memory["sample_count"] == 70 and len(memory["samples"]) == 64
    assert memory["peak_rss_bytes"] == 70
    assert memory["min_host_available_bytes"] == 0


def test_legacy_killed_proof_with_zero_actions_remains_fail_without_invented_metrics():
    receipt = {
        "kind": "tether.deployment_proof",
        "passed": False,
        "act_samples": [],
        "health": None,
        "server": {"exit_code": -9},
        "error": "SmokeError: server exited before /health was ready (exit_code=-9)",
    }
    report = build_realtime_certificate(receipt, control_hz=20, execution_cert=True)
    execution = report["execution_certificate"]
    assert report["decision"] == "FAIL"
    assert execution["metrics"]["execution_horizon_max"] is None
    assert execution["metrics"]["chunk_boundary_delta"]["max_abs"] is None
    assert report["provenance"]["startup"]["diagnostic"] == "server_exit_before_ready"
    assert report["provenance"]["startup"]["process_memory"]["oom_confirmed"] is False


def test_deploy_proof_records_synthetic_input_and_runtime_without_execution_inference(
    tmp_path, monkeypatch
):
    import tether.deploy_proof as dp

    export = tmp_path / "export"
    export.mkdir()
    (export / "model.onnx").write_bytes(b"test")
    (export / "tether_config.json").write_text(
        json.dumps({"model_type": "smolvla", "chunk_size": 50})
    )
    monkeypatch.setattr(dp, "_require_serve_runtime_deps", lambda: None)
    monkeypatch.setattr(dp, "_run_deploy_doctor", lambda *a: {"summary": {"fail": 0}})
    monkeypatch.setattr(
        dp.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(dp.platform, "platform", lambda: "test-platform")
    limits = []
    monkeypatch.setattr(
        dp, "_wait_for_health", lambda *a, **k: limits.append(k["timeout_s"]) or {"status": "ok"}
    )
    monkeypatch.setattr(dp, "_stop_process", lambda *a: 0)
    monkeypatch.setattr(dp, "_run_security_checks", lambda *a, **k: {"checks": []})
    monkeypatch.setattr(dp, "_run_metrics_check", lambda *a, **k: {"checks": []})
    monkeypatch.setattr(dp, "_run_guard_stress", lambda *a, **k: {"enabled": False, "checks": []})
    monkeypatch.setattr(
        "tether.proof_startup._memory_sample",
        lambda _: {"source": "unavailable", "rss_bytes": None, "host_available_bytes": None},
    )
    requests = []

    def request(method, url, **kw):
        if method == "POST":
            requests.append(kw["body"])
            return {
                "status_code": 200,
                "body": {
                    "actions": [[0], [1]],
                    "active_providers": ["CPUExecutionProvider"],
                    "deadline_cause": "none",
                },
            }
        return {"status_code": 200, "body": {"model_type": "smolvla"}}

    monkeypatch.setattr(dp, "_request", request)
    receipt = run_deploy_proof(
        export_dir=export,
        output_dir=tmp_path / "proof",
        act_samples=2,
        timeout_s=3,
        startup_timeout_s=9,
        port=12345,
    )
    assert receipt["passed"] is True
    assert limits == [9]
    assert receipt["evidence_contract"]["scope"] == "serving_only"
    assert requests[0] == requests[1]
    assert requests[0]["state"] == [0.0] * 6
    assert receipt["export_metadata"]["fields"]["chunk_size"] == 50
    assert "action_execution" not in receipt["act_samples"][0]
    assert receipt["act_samples"][0]["deadline_cause"] == "none"
    cert = build_realtime_certificate(receipt, control_hz=20, execution_cert=True)
    assert cert["decision"] == "FAIL"
    assert cert["execution_certificate"]["metrics"]["execution_horizon_max"] is None
