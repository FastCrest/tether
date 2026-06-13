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
    assert _skip_blocking_onboarding(SimpleNamespace(invoked_subcommand="doctor")) is False
