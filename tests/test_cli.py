"""Tests for CLI smoke tests."""

from typer.testing import CliRunner

from tether import __version__
from tether.cli import app

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


def test_serve_missing_dir():
    result = runner.invoke(app, ["serve", "/nonexistent/path"])
    assert result.exit_code == 1
