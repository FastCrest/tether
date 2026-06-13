"""Tests for CLI smoke tests."""

import builtins
import json
from types import SimpleNamespace

from typer.testing import CliRunner

from tether import __version__
from tether.cli import app, _skip_blocking_onboarding

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Deploy any VLA" in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_targets():
    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    assert "orin-nano" in result.output
    assert "Jetson Thor" in result.output


def test_export_help():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "HuggingFace model ID" in result.output
    assert "--export-mode" in result.output


def test_export_mode_rejected_for_monolithic():
    result = runner.invoke(
        app,
        ["export", "lerobot/pi05_libero_finetuned_v044", "--export-mode", "parallel"],
    )
    assert result.exit_code == 2
    assert "only applies to --decomposed" in result.output


def test_export_mode_rejected_for_legacy_decomposed_non_pi05():
    result = runner.invoke(
        app,
        ["export", "lerobot/smolvla_base", "--decomposed", "--export-mode", "parallel"],
    )
    assert result.exit_code == 2
    assert "only implemented for pi0.5 decomposed exports" in result.output


def test_export_mode_plumbed_to_pi05_decomposed(monkeypatch):
    seen = {}

    def fake_export_pi05_decomposed(**kwargs):
        seen.update(kwargs)
        return {
            "export_mode": kwargs["export_mode"].value,
            "vlm_prefix_onnx": "/tmp/vlm_prefix.onnx",
            "expert_denoise_onnx": "/tmp/expert_denoise.onnx",
            "vlm_prefix_mb": 1.0,
            "expert_denoise_mb": 2.0,
        }

    import tether.exporters.decomposed as decomposed

    monkeypatch.setattr(decomposed, "export_pi05_decomposed", fake_export_pi05_decomposed)

    result = runner.invoke(
        app,
        [
            "export",
            "lerobot/pi05_libero_finetuned_v044",
            "--decomposed",
            "--export-mode",
            "sequential",
            "--num-steps",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["export_mode"].value == "sequential"
    assert seen["num_steps"] == 3
    assert seen["student_checkpoint"] is None


def test_pi05_parallel_insufficient_vram_is_usage_error(monkeypatch):
    import tether.exporters._export_mode as export_mode

    monkeypatch.setattr(export_mode, "probe_free_vram", lambda: None)

    result = runner.invoke(
        app,
        [
            "export",
            "lerobot/pi05_libero_finetuned_v044",
            "--decomposed",
            "--export-mode",
            "parallel",
        ],
    )

    assert result.exit_code == 2
    assert "--export-mode parallel requires" in result.output


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "inference server" in result.output.lower() or "POST /act" in result.output


def test_smoke_help():
    result = runner.invoke(app, ["smoke", "--help"])
    assert result.exit_code == 0
    assert "/act" in result.output


def test_deploy_proof_help():
    result = runner.invoke(app, ["deploy-proof", "--help"])
    assert result.exit_code == 0
    assert "deployment proof" in result.output.lower()
    assert "--profile" in result.output
    assert "--policy-diff-baseline" in result.output


def test_prove_help_alias():
    result = runner.invoke(app, ["prove", "--help"])
    assert result.exit_code == 0
    assert "ready to deploy" in result.output.lower()
    assert "--profile" in result.output


def test_policy_diff_help():
    result = runner.invoke(app, ["policy", "diff", "--help"])
    assert result.exit_code == 0
    assert "shadow" in result.output.lower()
    assert "--fail-on" in result.output


def test_policy_shadow_gate_help():
    result = runner.invoke(app, ["policy", "shadow-gate", "--help"])
    assert result.exit_code == 0
    assert "PROMOTE, HOLD, or ROLLBACK" in result.output
    assert "--packet-dir" in result.output
    assert "--min-compared" in result.output


def test_policy_diff_fail_on_any_exits_three(monkeypatch):
    import tether.policy_diff as policy_diff_mod

    def fake_diff_policy_traces(**kwargs):
        assert kwargs["baseline_trace"] == "base.jsonl"
        assert kwargs["candidate_trace"] == "cand.jsonl"
        return {
            "kind": "tether.policy_diff",
            "mode": "trace_pair",
            "summary": {
                "verdict": "fail",
                "baseline_requests": 1,
                "compared": 1,
                "missing_candidate": 0,
                "request_mismatches": 0,
                "action_failures": 1,
                "latency_regressions": 0,
                "shape_failures": 0,
                "guard_regressions": 0,
                "max_action_delta": 0.2,
                "mean_action_delta": 0.2,
                "p95_action_delta": 0.2,
                "min_action_cosine": 0.9,
                "metadata_warnings": [],
            },
        }

    monkeypatch.setattr(policy_diff_mod, "diff_policy_traces", fake_diff_policy_traces)

    result = runner.invoke(
        app,
        ["policy", "diff", "base.jsonl", "cand.jsonl", "--fail-on", "any"],
    )

    assert result.exit_code == 3
    assert "FAIL" in result.output


def test_promote_help():
    result = runner.invoke(app, ["promote", "--help"])
    assert result.exit_code == 0
    assert "PROMOTE" in result.output
    assert "--candidate-active" in result.output
    assert "--profile" in result.output


def test_profiles_list_json():
    result = runner.invoke(app, ["profiles", "list", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    names = {profile["name"] for profile in body["profiles"]}
    assert {"ci-default", "lab-shadow", "warehouse-safe", "contact-strict"} <= names


def test_profiles_show_json():
    result = runner.invoke(app, ["profiles", "show", "warehouse-safe", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["name"] == "warehouse-safe"
    assert body["thresholds"]["require_policy_diff"] is True
    assert body["thresholds"]["require_auth"] is True


def test_profiles_init_writes_editable_profile(tmp_path):
    output = tmp_path / "lab-shadow.yml"

    result = runner.invoke(
        app,
        ["profiles", "init", "lab-shadow", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    text = output.read_text(encoding="utf-8")
    assert "name: lab-shadow" in text
    assert "require_policy_diff: true" in text


def test_promote_json_uses_decision_runner(monkeypatch, tmp_path):
    import tether.promote as promote_mod

    seen = {}

    def fake_decide_promotion(packet, **kwargs):
        seen["packet"] = packet
        seen.update(kwargs)
        return {
            "kind": "tether.promotion_decision",
            "decision": "PROMOTE",
            "summary": {"pass": 1, "fail": 0},
            "checks": [{"name": "ok", "status": "pass"}],
        }

    monkeypatch.setattr(promote_mod, "decide_promotion", fake_decide_promotion)

    result = runner.invoke(
        app,
        [
            "promote",
            str(tmp_path / "proof"),
            "--profile",
            str(tmp_path / "warehouse-safe.yml"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["decision"] == "PROMOTE"
    assert seen["packet"] == str(tmp_path / "proof")
    assert seen["profile_path"] == str(tmp_path / "warehouse-safe.yml")


def test_promote_candidate_active_failure_exits_rollback(monkeypatch, tmp_path):
    import tether.promote as promote_mod

    monkeypatch.setattr(
        promote_mod,
        "decide_promotion",
        lambda *_args, **_kwargs: {
            "kind": "tether.promotion_decision",
            "decision": "ROLLBACK",
            "summary": {"pass": 1, "fail": 1},
            "proof": {"passed": False, "check_failures": 1},
            "policy_diff": {"present": False},
            "checks": [{"name": "deployment_proof_passed", "status": "fail"}],
            "packet_dir": str(tmp_path / "proof"),
            "profile": {"name": "default"},
        },
    )

    result = runner.invoke(app, ["promote", str(tmp_path / "proof"), "--candidate-active"])

    assert result.exit_code == 4
    assert "ROLLBACK" in result.output


def test_smoke_json_uses_receipt_runner(tmp_path, monkeypatch):
    import tether.smoke as smoke_mod

    seen = {}
    markdown_path = tmp_path / "receipt.md"

    def fake_run_smoke(**kwargs):
        seen.update(kwargs)
        return {
            "schema_version": 1,
            "passed": True,
            "tether_version": "0.0.test",
            "python": "3.12.0",
            "offline": kwargs["offline"],
            "duration_ms": 12.3,
            "export_dir": str(tmp_path / "export"),
            "server": {"url": "http://127.0.0.1:12345"},
            "doctor": {"summary": {"pass": 1, "fail": 0, "warn": 0, "skip": 0}},
            "latency": {
                "samples": kwargs["act_samples"],
                "first_sample": {"inference_ms": 1.0, "roundtrip_ms": 2.0},
                "inference_ms": {"p50_ms": 1.0, "p95_ms": 1.0, "max_ms": 1.0},
                "roundtrip_ms": {"p50_ms": 2.0, "p95_ms": 2.0, "max_ms": 2.0},
                "warm_inference_ms": {"p50_ms": 1.0, "p95_ms": 1.0, "max_ms": 1.0},
                "warm_roundtrip_ms": {"p50_ms": 2.0, "p95_ms": 2.0, "max_ms": 2.0},
            },
            "act": {
                "num_actions": 50,
                "action_dim": 32,
                "provider_mode": "onnx_cpu",
                "active_providers": ["CPUExecutionProvider"],
            },
        }

    monkeypatch.setattr(smoke_mod, "run_smoke", fake_run_smoke)

    result = runner.invoke(
        app,
        [
            "smoke",
            "--json",
            "--export-dir",
            str(tmp_path / "custom-export"),
            "--port",
            "12345",
            "--timeout-s",
            "2",
            "--act-samples",
            "5",
            "--markdown-output",
            str(markdown_path),
            "--online",
            "--tmp-export",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["passed"] is True
    assert seen["export_dir"] == str(tmp_path / "custom-export")
    assert seen["port"] == 12345
    assert seen["timeout_s"] == 2.0
    assert seen["act_samples"] == 5
    assert seen["offline"] is False
    assert seen["keep_export"] is False
    assert markdown_path.exists()
    assert "- Status: PASS" in markdown_path.read_text()
    assert "- Samples: 5" in markdown_path.read_text()
    assert "- Warm roundtrip p95: 2.0 ms" in markdown_path.read_text()


def test_deploy_proof_json_uses_receipt_runner(tmp_path, monkeypatch):
    import tether.deploy_proof as proof_mod

    seen = {}

    def fake_run_deploy_proof(**kwargs):
        seen.update(kwargs)
        return {
            "schema_version": 1,
            "kind": "tether.deployment_proof",
            "passed": True,
            "output_dir": str(tmp_path / "proof"),
            "export_dir": kwargs["export_dir"],
            "checks": [{"status": "pass"}],
            "doctor": {"summary": {"pass": 1, "fail": 0, "warn": 0, "skip": 0}},
            "latency": {
                "samples": kwargs["act_samples"],
                "ttfa_ms": 1.0,
                "roundtrip_ms": {"p50_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0},
                "warm_roundtrip_ms": {"p95_ms": 1.0},
                "jitter": {"p95_minus_p50_ms": 0.0},
            },
            "server": {"url": "http://127.0.0.1:12345"},
            "security": {"enabled": True, "checks": []},
            "metrics": {"status_code": 200, "metric_names": ["tether_act_latency_seconds"]},
            "trace": {"record_dir": "", "files": []},
        }

    monkeypatch.setattr(proof_mod, "run_deploy_proof", fake_run_deploy_proof)

    result = runner.invoke(
        app,
        [
            "deploy-proof",
            str(tmp_path / "export"),
            "--json",
            "--output-dir",
            str(tmp_path / "proof"),
            "--profile",
            str(tmp_path / "profile.yml"),
            "--port",
            "12345",
            "--timeout-s",
            "2",
            "--samples",
            "7",
            "--device",
            "cuda",
            "--providers",
            "CUDAExecutionProvider,CPUExecutionProvider",
            "--no-strict-providers",
            "--embodiment",
            "franka",
            "--api-key",
            "secret",
            "--record-dir",
            str(tmp_path / "traces"),
            "--record-images",
            "none",
            "--policy-diff-baseline",
            str(tmp_path / "current.jsonl.gz"),
            "--policy-diff-candidate",
            str(tmp_path / "candidate.jsonl.gz"),
            "--policy-diff-fail-on",
            "guard",
            "--policy-diff-max-action-delta",
            "0.05",
            "--no-prewarm",
            "--instruction",
            "pick",
            "--state-dim",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["passed"] is True
    assert seen["export_dir"] == str(tmp_path / "export")
    assert seen["output_dir"] == str(tmp_path / "proof")
    assert seen["profile_path"] == str(tmp_path / "profile.yml")
    assert seen["port"] == 12345
    assert seen["timeout_s"] == 2.0
    assert seen["act_samples"] == 7
    assert seen["device"] == "cuda"
    assert seen["providers"] == "CUDAExecutionProvider,CPUExecutionProvider"
    assert seen["no_strict_providers"] is True
    assert seen["embodiment"] == "franka"
    assert seen["api_key"] == "secret"
    assert seen["record_dir"] == str(tmp_path / "traces")
    assert seen["record_images"] == "none"
    assert seen["policy_diff_baseline_trace"] == str(tmp_path / "current.jsonl.gz")
    assert seen["policy_diff_candidate_trace"] == str(tmp_path / "candidate.jsonl.gz")
    assert seen["policy_diff_fail_on"] == "guard"
    assert seen["policy_diff_max_action_delta"] == 0.05
    assert seen["prewarm"] is False
    assert seen["instruction"] == "pick"
    assert seen["state_dim"] == 9


def test_prove_alias_uses_deploy_proof_runner(tmp_path, monkeypatch):
    import tether.deploy_proof as proof_mod

    seen = {}

    def fake_run_deploy_proof(**kwargs):
        seen.update(kwargs)
        return {
            "schema_version": 1,
            "kind": "tether.deployment_proof",
            "passed": True,
            "output_dir": str(tmp_path / "proof"),
            "export_dir": kwargs["export_dir"],
            "checks": [{"status": "pass"}],
            "doctor": {"summary": {"pass": 1, "fail": 0, "warn": 0, "skip": 0}},
            "latency": {
                "samples": kwargs["act_samples"],
                "ttfa_ms": 1.0,
                "roundtrip_ms": {"p50_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0},
                "warm_roundtrip_ms": {"p95_ms": 1.0},
                "jitter": {"p95_minus_p50_ms": 0.0},
            },
            "server": {"url": "http://127.0.0.1:12345"},
            "security": {"enabled": False, "checks": []},
            "metrics": {"status_code": 200, "metric_names": ["tether_act_latency_seconds"]},
            "trace": {"record_dir": "", "files": []},
        }

    monkeypatch.setattr(proof_mod, "run_deploy_proof", fake_run_deploy_proof)

    result = runner.invoke(
        app,
        [
            "prove",
            str(tmp_path / "export"),
            "--json",
            "--samples",
            "3",
            "--embodiment",
            "franka",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["passed"] is True
    assert seen["export_dir"] == str(tmp_path / "export")
    assert seen["act_samples"] == 3
    assert seen["embodiment"] == "franka"


def test_smoke_failure_exits_nonzero(monkeypatch):
    import tether.smoke as smoke_mod

    monkeypatch.setattr(
        smoke_mod,
        "run_smoke",
        lambda **_: {
            "schema_version": 1,
            "passed": False,
            "error": "SmokeError: server did not start",
            "server": {"url": "http://127.0.0.1:12345"},
        },
    )

    result = runner.invoke(app, ["smoke", "--json"])

    assert result.exit_code == 1
    body = json.loads(result.output)
    assert body["passed"] is False
    assert "server did not start" in body["error"]


def test_deploy_proof_failure_exits_nonzero(monkeypatch, tmp_path):
    import tether.deploy_proof as proof_mod

    monkeypatch.setattr(
        proof_mod,
        "run_deploy_proof",
        lambda **_: {
            "schema_version": 1,
            "kind": "tether.deployment_proof",
            "passed": False,
            "error": "DeployProofError: p95 over budget",
            "checks": [{"status": "fail"}],
        },
    )

    result = runner.invoke(app, ["deploy-proof", str(tmp_path / "export"), "--json"])

    assert result.exit_code == 1
    body = json.loads(result.output)
    assert body["passed"] is False
    assert "p95 over budget" in body["error"]


def test_serve_missing_dir():
    result = runner.invoke(app, ["serve", "/nonexistent/path"])
    assert result.exit_code == 1


def test_serve_missing_onnxruntime_hint_preserves_extras(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"fake")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing ort")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = runner.invoke(app, ["serve", str(export_dir), "--device", "cpu"])

    assert result.exit_code == 1
    assert "fastcrest-tether[serve]" in result.output
    assert "fastcrest-tether[serve,gpu]" in result.output


def test_doctor_json_system_probe_is_machine_readable():
    result = runner.invoke(app, ["doctor", "--format", "json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["schema_version"] == 1
    assert "system_probe" in body
    assert isinstance(body["system_probe"]["checks"], list)
    assert body["system_probe"]["summary"]["pass"] >= 1
    assert "Tether Doctor" not in result.output


def test_deploy_commands_skip_blocking_onboarding():
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="serve")) is True
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="go")) is True
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="smoke")) is True
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="deploy-proof")) is True
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="prove")) is True
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="doctor")) is False
