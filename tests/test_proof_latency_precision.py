"""Decision thresholds use measured latency, before presentation rounding."""

import json
from types import SimpleNamespace

import pytest

from tether.deploy_proof import _evaluate_thresholds, run_deploy_proof, summarize_deploy_latency
from tether.realtime_cert import (
    RealtimeCertificateError,
    build_realtime_certificate,
    write_realtime_certificate,
)


def _passing_receipt(values, *, control_hz=20.0):
    samples = [{"roundtrip_ms": value} for value in values]
    return {
        "kind": "tether.deployment_proof",
        "passed": True,
        "act_samples": samples,
        "latency": summarize_deploy_latency(samples, control_hz=control_hz),
    }


def test_latency_summary_retains_sub_tenth_millisecond_measurements():
    latency = summarize_deploy_latency(
        [{"roundtrip_ms": 50.04, "latency_ms": 40.04}] * 200, control_hz=20.0
    )
    assert latency["roundtrip_ms"] == dict.fromkeys(("p50_ms", "p95_ms", "p99_ms", "max_ms"), 50.04)
    assert latency["first_sample"] == {"roundtrip_ms": 50.04, "inference_ms": 40.04}
    assert latency["ttfa_ms"] == 50.04
    assert latency["control_budget"]["missed_samples"] == 200


@pytest.mark.parametrize("value, decision", [(49.96, "PASS"), (50.0, "PASS"), (50.04, "FAIL")])
def test_realtime_p95_boundary_with_all_other_checks_passing(tmp_path, value, decision):
    receipt = _passing_receipt([value] * 200)
    # Permit the independently counted misses so this isolates the unchanged p95 gate.
    report = build_realtime_certificate(receipt, max_control_budget_misses=200)
    assert report["decision"] == decision
    assert report["summary"]["failed_checks"] == (
        ["roundtrip_p95_within_budget"] if decision == "FAIL" else []
    )
    assert report["control_budget"]["roundtrip_p95_budget_ms"] == 50.0
    assert report["latency"]["roundtrip_ms"]["p95_ms"] == value
    write_realtime_certificate(report, tmp_path)
    saved = json.loads((tmp_path / "realtime-serving-cert.json").read_text())
    assert saved["latency"]["roundtrip_ms"]["p95_ms"] == value
    assert saved["decision"] == decision


@pytest.mark.parametrize(
    "threshold, name, values, limit",
    [
        ("max_first_roundtrip_ms", "latency_first_roundtrip", [50.04] * 200, 50.0),
        ("max_roundtrip_p95_ms", "latency_roundtrip_p95", [50.04] * 200, 50.0),
        ("max_warm_roundtrip_p95_ms", "latency_warm_roundtrip_p95", [50.04] * 200, 50.0),
        ("max_jitter_p95_minus_p50_ms", "latency_jitter", [0.0] * 100 + [50.04] * 100, 25.0),
    ],
)
def test_deploy_profile_thresholds_do_not_round_down(threshold, name, values, limit):
    samples = [{"roundtrip_ms": value} for value in values]
    checks = []
    _evaluate_thresholds(
        checks=checks,
        profile={"thresholds": {threshold: limit}},
        doctor=None,
        latency=summarize_deploy_latency(samples),
        act_samples=samples,
        record_trace_files=[],
        record_dir=None,
        guard_stress={},
        policy_diff=None,
    )
    assert len(checks) == 1
    assert checks[0]["name"] == name
    assert checks[0]["status"] == "fail"
    assert checks[0]["actual"] > limit


def test_realtime_budget_preserves_fractional_control_period():
    period = 50.0004
    report = build_realtime_certificate(
        _passing_receipt([50.0002] * 200, control_hz=1000.0 / period)
    )
    assert report["decision"] == "PASS"
    assert report["summary"]["failed_checks"] == []
    assert report["control_budget"]["roundtrip_p95_budget_ms"] == pytest.approx(period)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, None])
def test_nonfinite_or_nonnumeric_certificate_p95_still_refuses(invalid):
    receipt = _passing_receipt([40.0] * 200)
    receipt["latency"]["roundtrip_ms"]["p95_ms"] = invalid
    report = build_realtime_certificate(receipt)
    assert report["decision"] == "FAIL"
    assert report["summary"]["failed_checks"] == ["roundtrip_p95_within_budget"]


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), True, False, 0.0, -1.0, 5e-324])
def test_explicit_control_rate_must_be_finite_positive_and_not_boolean(invalid):
    receipt = _passing_receipt([0.0] * 200)
    baseline = build_realtime_certificate(receipt)
    assert baseline["decision"] == "PASS" and baseline["summary"]["failed_checks"] == []
    with pytest.raises(RealtimeCertificateError, match="control_hz"):
        build_realtime_certificate(receipt, control_hz=invalid)


def test_receipt_control_rate_cannot_create_an_infinite_budget():
    receipt = _passing_receipt([0.0] * 200, control_hz=5e-324)
    report = build_realtime_certificate(receipt)
    assert report["decision"] == "FAIL"
    assert "control_hz_defined" in report["summary"]["failed_checks"]


@pytest.mark.parametrize(
    "name",
    [
        "max_roundtrip_p95_ms",
        "max_jitter_p95_minus_p50_ms",
        "max_deadline_misses",
        "max_control_budget_misses",
        "max_act_errors",
        "max_stale_action_window_ms",
        "max_chunk_boundary_delta",
        "max_velocity_discontinuity",
    ],
)
@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), True, -1.0])
def test_explicit_certificate_thresholds_refuse_invalid_numbers(name, invalid):
    receipt = _passing_receipt([0.0] * 200)
    with pytest.raises(RealtimeCertificateError, match=name):
        build_realtime_certificate(receipt, **{name: invalid})


def test_zero_thresholds_are_valid_and_do_not_relax_the_zero_latency_gate():
    report = build_realtime_certificate(
        _passing_receipt([0.0] * 200),
        max_roundtrip_p95_ms=0.0,
        max_jitter_p95_minus_p50_ms=0.0,
        max_deadline_misses=0,
        max_control_budget_misses=0,
        max_act_errors=0,
        max_stale_action_window_ms=0.0,
        max_chunk_boundary_delta=0.0,
        max_velocity_discontinuity=0.0,
    )
    assert report["decision"] == "PASS"
    assert report["summary"]["failed_checks"] == []
    assert report["control_budget"]["roundtrip_p95_budget_ms"] == 0.0


def test_actual_proof_packet_retains_measured_roundtrip(tmp_path, monkeypatch):
    import tether.deploy_proof as dp

    export = tmp_path / "export"
    export.mkdir()
    (export / "model.onnx").write_bytes(b"test")
    monkeypatch.setattr(dp, "_require_serve_runtime_deps", lambda: None)
    monkeypatch.setattr(dp, "_run_deploy_doctor", lambda *a: {"summary": {"fail": 0}})
    monkeypatch.setattr(dp.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(
        dp.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(dp, "_wait_for_health", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(dp, "_stop_process", lambda *a: 0)
    monkeypatch.setattr(dp, "_run_security_checks", lambda *a, **k: {"checks": []})
    monkeypatch.setattr(dp, "_run_metrics_check", lambda *a, **k: {"checks": []})
    monkeypatch.setattr(dp, "_run_guard_stress", lambda *a, **k: {"checks": []})
    monkeypatch.setattr(
        "tether.proof_startup._memory_sample",
        lambda _: {"source": "unavailable", "rss_bytes": None, "host_available_bytes": None},
    )
    now = [0.0]
    monkeypatch.setattr(dp, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def request(method, url, **kwargs):
        if method == "POST":
            now[0] += 0.05004
        return {"status_code": 200, "body": {"active_providers": ["CPUExecutionProvider"]}}

    monkeypatch.setattr(dp, "_request", request)
    receipt = run_deploy_proof(
        export_dir=export, output_dir=tmp_path / "proof", act_samples=2, port=12345
    )
    assert receipt["passed"] is True
    assert receipt["act_samples"][0]["roundtrip_ms"] == pytest.approx(50.04)
    saved = json.loads((tmp_path / "proof" / "deployment-proof.json").read_text())
    assert saved["act_samples"][0]["roundtrip_ms"] == pytest.approx(50.04)
    assert saved["latency"]["roundtrip_ms"]["p95_ms"] == pytest.approx(50.04)
