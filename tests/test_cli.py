"""Tests for CLI smoke tests."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from tether import __version__
from tether.cli import app

runner = CliRunner()


def _fake_runtime_modules(torch_version="2.7.1", ort_version="1.25.1"):
    torch = types.ModuleType("torch")
    torch.__version__ = torch_version
    ort = types.ModuleType("onnxruntime")
    ort.__version__ = ort_version
    return {"torch": torch, "onnxruntime": ort}


def _seed_go_model_cache(tmp_path):
    target = tmp_path / "model_cache"
    target.mkdir()
    (target / "weights.bin").write_text("stub")
    return target


def _seed_go_export_cache(tmp_path, meta):
    export_dir = tmp_path / "tether_cache" / "exports" / "smolvla-base"
    export_dir.mkdir(parents=True)
    (export_dir / "VERIFICATION.md").write_text("# stub")
    (export_dir / "_tether_meta.json").write_text(json.dumps(meta))
    return export_dir


def _fake_export(model_path, output_dir, num_steps=10, target=None):
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "VERIFICATION.md").write_text("# stub")
    return {"onnx_path": str(export_dir / "model.onnx"), "size_mb": 100.0}


def _invoke_go_with_export_cache(tmp_path, monkeypatch, export_mock):
    target = _seed_go_model_cache(tmp_path)
    monkeypatch.setenv("TETHER_HOME", str(tmp_path / "tether_cache"))
    server = types.ModuleType("tether.runtime.server")
    server.create_app = Mock(side_effect=RuntimeError("serve-stub"))
    server.TetherServer = object

    with (
        patch("tether.exporters.monolithic.export_monolithic", export_mock),
        patch.dict("sys.modules", {"tether.runtime.server": server}),
    ):
        return runner.invoke(
            app,
            [
                "go",
                "--model",
                "smolvla-base",
                "--device-class",
                "a10g",
                "--target-dir",
                str(target),
            ],
        )


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


def test_go_export_cache_accepts_current_torch_and_ort_versions(tmp_path, monkeypatch):
    meta = {
        "tether_version": __version__,
        "torch_version": "2.7.1",
        "ort_version": "1.25.1",
        "model_id": "smolvla-base",
        "export_target": "desktop",
        "export_mode": "monolithic",
    }
    _seed_go_export_cache(tmp_path, meta)
    export_mock = Mock(side_effect=_fake_export)

    with patch.dict("sys.modules", _fake_runtime_modules()):
        result = _invoke_go_with_export_cache(tmp_path, monkeypatch, export_mock)

    assert result.exit_code == 1
    assert "export hit:" in result.output
    export_mock.assert_not_called()


def test_go_export_cache_rebuilds_when_torch_version_changes(tmp_path, monkeypatch):
    export_dir = _seed_go_export_cache(
        tmp_path,
        {
            "tether_version": __version__,
            "torch_version": "2.0.0",
            "ort_version": "1.25.1",
            "model_id": "smolvla-base",
            "export_target": "desktop",
            "export_mode": "monolithic",
        },
    )
    export_mock = Mock(side_effect=_fake_export)

    with patch.dict("sys.modules", _fake_runtime_modules()):
        result = _invoke_go_with_export_cache(tmp_path, monkeypatch, export_mock)

    assert result.exit_code == 1
    assert "Cache torch version mismatch" in result.output
    export_mock.assert_called_once()
    meta = json.loads((export_dir / "_tether_meta.json").read_text())
    assert meta["torch_version"] == "2.7.1"
    assert meta["ort_version"] == "1.25.1"


def test_go_export_cache_rebuilds_when_ort_version_changes(tmp_path, monkeypatch):
    _seed_go_export_cache(
        tmp_path,
        {
            "tether_version": __version__,
            "torch_version": "2.7.1",
            "ort_version": "1.20.0",
            "model_id": "smolvla-base",
            "export_target": "desktop",
            "export_mode": "monolithic",
        },
    )
    export_mock = Mock(side_effect=_fake_export)

    with patch.dict("sys.modules", _fake_runtime_modules()):
        result = _invoke_go_with_export_cache(tmp_path, monkeypatch, export_mock)

    assert result.exit_code == 1
    assert "Cache ORT version mismatch" in result.output
    export_mock.assert_called_once()


def test_go_export_cache_rebuilds_legacy_meta_without_runtime_versions(tmp_path, monkeypatch):
    _seed_go_export_cache(
        tmp_path,
        {
            "tether_version": __version__,
            "model_id": "smolvla-base",
            "export_target": "desktop",
            "export_mode": "monolithic",
        },
    )
    export_mock = Mock(side_effect=_fake_export)

    with patch.dict("sys.modules", _fake_runtime_modules()):
        result = _invoke_go_with_export_cache(tmp_path, monkeypatch, export_mock)

    assert result.exit_code == 1
    assert "Cache torch version mismatch" in result.output
    assert "torch unknown" in result.output
    export_mock.assert_called_once()


def test_go_export_meta_write_records_unknown_when_runtime_imports_fail(tmp_path, monkeypatch):
    def fake_export_and_hide_runtime_versions(model_path, output_dir, num_steps=10, target=None):
        result = _fake_export(model_path, output_dir, num_steps=num_steps, target=target)
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        return result

    export_mock = Mock(side_effect=fake_export_and_hide_runtime_versions)

    result = _invoke_go_with_export_cache(tmp_path, monkeypatch, export_mock)

    assert result.exit_code == 1
    export_mock.assert_called_once()
    meta_path = tmp_path / "tether_cache" / "exports" / "smolvla-base" / "_tether_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["torch_version"] == "unknown"
    assert meta["ort_version"] == "unknown"
