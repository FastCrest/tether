"""Requested precision must either be implemented or refused before work."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from tether.cli import app
from tether.config import ExportConfig, HARDWARE_PROFILES, get_hardware_profile
from tether.exporters import monolithic, trt_build


@pytest.mark.parametrize("precision", ["fp16", "fp8", "int8", "invalid"])
@pytest.mark.parametrize("student", [False, True])
def test_cli_rejects_unsupported_monolithic_precision(tmp_path, monkeypatch, precision, student):
    exporter = Mock(side_effect=AssertionError("must reject before loading weights"))
    monkeypatch.setattr(monolithic, "export_monolithic", exporter)
    monkeypatch.setattr(monolithic, "export_snapflow_student_monolithic", exporter)
    output = tmp_path / "out"
    args = ["export", "lerobot/pi0_base", "--precision", precision, "--output", str(output)]
    if student:
        args.append("--from-distilled")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert precision in result.output
    assert "Supported precisions: fp32" in result.output
    assert "Supported precisions: fp32" in result.stderr
    exporter.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("args", [[], ["--precision", "fp32"]])
@pytest.mark.parametrize("student", [False, True])
def test_cli_passes_actual_fp32_precision(tmp_path, monkeypatch, args, student):
    exporter = Mock(return_value={"onnx_path": str(tmp_path / "model.onnx"), "size_mb": 1.0})
    monkeypatch.setattr(monolithic, "export_monolithic", exporter)
    monkeypatch.setattr(monolithic, "export_snapflow_student_monolithic", exporter)
    result = CliRunner().invoke(
        app,
        [
            "export",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            *args,
            *(["--from-distilled"] if student else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert exporter.call_args.kwargs["precision"] == "fp32"


@pytest.mark.parametrize("precision", ["fp16", "int8", "fp8"])
@pytest.mark.parametrize(
    "exporter", [monolithic.export_monolithic, monolithic.export_snapflow_student_monolithic]
)
def test_python_api_rejects_unsupported_precision(tmp_path, precision, exporter):
    with pytest.raises(ValueError, match="Supported precisions: fp32"):
        exporter("unrelated", tmp_path / "out", precision=precision)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("builder", [trt_build.build_engine, trt_build.build_all])
@pytest.mark.parametrize("precision,support", [("int8", False), ("fp8", False), ("bogus", True)])
def test_trt_builder_refuses_unsupported_precision_before_work(
    tmp_path, monkeypatch, builder, precision, support
):
    hardware = replace(
        get_hardware_profile("desktop"), trt_precision=precision, fp8_support=support
    )
    run = Mock(side_effect=AssertionError("must reject before spawning trtexec"))
    monkeypatch.setattr(trt_build.subprocess, "run", run)
    with pytest.raises(ValueError, match="calibration|Supported precision") as refused:
        builder(tmp_path / "model.onnx", tmp_path / "out", hardware)
    if precision == "int8":
        assert "INT8 calibration is not implemented" in str(refused.value)
    run.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("precision", ["fp16", "fp8"])
def test_trt_builder_honours_supported_precision(tmp_path, monkeypatch, precision):
    hardware = replace(get_hardware_profile("desktop"), trt_precision=precision, fp8_support=True)
    output = tmp_path / "model.trt"
    seen = []
    monkeypatch.setattr(trt_build, "check_trtexec", lambda: True)

    def run(cmd, **kwargs):
        seen.extend(cmd)
        output.write_bytes(b"engine")
        return Mock(returncode=0)

    monkeypatch.setattr(trt_build.subprocess, "run", run)
    assert trt_build.build_engine(tmp_path / "model.onnx", output, hardware) == output
    assert f"--{precision}" in seen
    assert f"--{'fp8' if precision == 'fp16' else 'fp16'}" not in seen


@pytest.mark.parametrize("config", [{"type": "pi0fast"}, {"type": "pi0_fast"}, {"type": "unknown"}])
def test_cli_reports_unsupported_family_before_export(tmp_path, config):
    import json

    checkpoint = tmp_path / "pi0-base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))
    output = tmp_path / "out"
    result = CliRunner().invoke(app, ["export", str(checkpoint), "--output", str(output)])
    assert result.exit_code == 2, result.output
    assert ("pi0_fast" if config["type"] == "pi0fast" else config["type"]) in result.stderr
    assert "no monolithic exporter" in result.stderr or "Unsupported model family" in result.stderr
    assert not output.exists()


def test_decomposed_int8_refused_before_export(tmp_path):
    """review-5: decomposed path wrote precision=int8 metadata over fp32 ONNX / fp16 engines."""
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        ["export", "org/pi0_base", "--decomposed", "--precision", "int8", "--output", str(output)],
    )
    assert result.exit_code == 2, result.output
    assert "INT8 calibration is not implemented" in result.output
    assert not output.exists()


def test_chat_export_precisions_match_cli_support():
    from tether.chat.schema import _PRECISIONS

    assert "fp32" in _PRECISIONS and "int8" not in _PRECISIONS


@pytest.mark.parametrize("target", HARDWARE_PROFILES)
@pytest.mark.parametrize("precision", ["fp32", "fp16", "fp8", "int8", "fp4", "bogus"])
@pytest.mark.parametrize("all_models", [False, True])
def test_trt_profile_precision_matrix(tmp_path, monkeypatch, target, precision, all_models):
    hardware = get_hardware_profile(target)
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    onnx = onnx_dir / "model.onnx"
    onnx.write_bytes(b"fixture")
    output = tmp_path / "engines"
    commands = []
    monkeypatch.setattr(trt_build, "check_trtexec", lambda: True)

    def run(cmd, **kwargs):
        commands.append(cmd)
        saved = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--saveEngine="))
        Path(saved).write_bytes(b"engine")
        return Mock(returncode=0)

    monkeypatch.setattr(trt_build.subprocess, "run", run)
    builder = trt_build.build_all if all_models else trt_build.build_engine
    source = onnx_dir if all_models else onnx
    destination = output if all_models else output / "model.trt"
    if precision == "fp16" or (precision == "fp8" and hardware.fp8_support):
        builder(source, destination, hardware, precision=precision)
        assert len(commands) == 1
        flags = {arg for arg in commands[0] if arg in {"--fp16", "--fp8", "--int8"}}
        assert flags == {f"--{precision}"}
    else:
        with pytest.raises(ValueError, match="calibration|Supported precision"):
            builder(source, destination, hardware, precision=precision)
        assert not commands
        assert not output.exists()


@pytest.mark.parametrize(
    "exporter_name",
    ["export_pi0", "export_pi05", "export_smolvla", "export_gr00t", "export_gr00t_full"],
)
@pytest.mark.parametrize("target,precision", [("thor", "fp16"), ("desktop", "fp8")])
def test_decomposed_export_builds_requested_precision(
    tmp_path, monkeypatch, exporter_name, target, precision
):
    """Exercise each exporter through the real builder and saved config with tiny stubs."""
    import json
    import torch
    from tether.exporters import gr00t, pi0, smolvla
    from tether.models.vlas.gr00t import GR00TVLA
    from tether.models.vlas.smolvla import SmolVLA

    stack = SimpleNamespace(
        expert_hidden=2,
        action_in_proj=SimpleNamespace(in_features=2),
        layers=[None],
        cross_indices=[],
        vlm_kv_dim=2,
        parameters=lambda: [],
    )
    meta = {
        "action_dim": 2,
        "num_layers": 1,
        "total_params_m": 0,
        "chunk_size": 2,
        "hidden": 2,
        "output_dim": 2,
        "raw_action_dim": 2,
    }
    vla = SimpleNamespace(
        vla_head=SimpleNamespace(
            expert_stack=stack,
            dit_stack=stack,
            full_stack=stack,
            metadata=meta,
        )
    )
    monkeypatch.setattr(SmolVLA, "from_pretrained", Mock(return_value=vla))
    monkeypatch.setattr(GR00TVLA, "from_pretrained", Mock(return_value=vla))
    monkeypatch.setattr(pi0, "build_pi0_expert_stack", Mock(return_value=(stack, meta)))
    monkeypatch.setattr(pi0, "build_pi05_expert_stack", Mock(return_value=(stack, meta)))
    monkeypatch.setattr(gr00t, "_raw_action_dim_from_checkpoint", lambda *a: 2)

    def export_onnx(module, inputs, path, **kwargs):
        from tests.export_config_factory import write_test_onnx

        write_test_onnx(path, action_dim=2)

    for module in [pi0, smolvla, gr00t]:
        monkeypatch.setattr(module, "export_module_to_onnx", export_onnx)
        monkeypatch.setattr(module, "optimize_onnx", lambda *a: None)
        monkeypatch.setattr(module, "check_trtexec", lambda: True)
    monkeypatch.setattr(trt_build, "check_trtexec", lambda: True)
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        saved = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--saveEngine="))
        Path(saved).write_bytes(b"engine")
        return Mock(returncode=0)

    monkeypatch.setattr(trt_build.subprocess, "run", run)
    module = (
        pi0
        if exporter_name in {"export_pi0", "export_pi05"}
        else (smolvla if exporter_name == "export_smolvla" else gr00t)
    )
    config = ExportConfig("fixture", target, str(tmp_path), precision=precision, validate=False)
    result = getattr(module, exporter_name)(config, state_dict={"weight": torch.zeros(1)})
    assert Path(result["files"]["expert_trt"]).read_bytes() == b"engine"
    assert len(commands) == 1
    assert {arg for arg in commands[0] if arg in {"--fp16", "--fp8"}} == {f"--{precision}"}
    saved_config = json.loads(Path(result["files"]["config"]).read_text())
    assert saved_config["precision"] == precision
    assert saved_config["hardware"]["precision"] == precision


@pytest.mark.parametrize(
    "target,precision", [("orin", "fp8"), ("thor", "int8"), ("desktop", "fp32"), ("thor", "bogus")]
)
def test_cli_refuses_invalid_decomposed_precision_before_download(
    tmp_path, monkeypatch, target, precision
):
    download = Mock(side_effect=AssertionError("must refuse before downloading"))
    monkeypatch.setattr("tether.checkpoint.load_checkpoint", download)
    output = tmp_path / "export"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "org/pi0_base",
            "--decomposed",
            "--target",
            target,
            "--precision",
            precision,
            "--output",
            str(output),
        ],
        terminal_width=300,
    )
    assert result.exit_code == 2, result.output
    assert "Supported precision" in result.stderr
    download.assert_not_called()
    assert not output.exists()
