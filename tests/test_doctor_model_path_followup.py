"""Keep the user's export path intact when the TRT system probe builds ONNX."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import typer

from tether.cli import doctor


@pytest.mark.parametrize("model_path", ["", "/exports/customer-policy"])
@pytest.mark.parametrize("output_format", ["json", "human"])
def test_trt_system_probe_preserves_optional_model_path(
    monkeypatch, tmp_path, capsys, model_path, output_format
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TETHER_HOME", str(tmp_path / "tether"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    fake_onnx = MagicMock(__version__="test")
    fake_onnx.helper.make_model.return_value.SerializeToString.return_value = b"probe"
    fake_ort = MagicMock(__version__="1.25.1")
    fake_ort.get_available_providers.return_value = [
        "TensorrtExecutionProvider",
        "CPUExecutionProvider",
    ]
    fake_ort.InferenceSession.return_value.get_providers.return_value = [
        "TensorrtExecutionProvider",
        "CPUExecutionProvider",
    ]
    fake_torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    for name, value in {
        "torch": fake_torch,
        "onnx": fake_onnx,
        "onnxruntime": fake_ort,
        **{
            name: SimpleNamespace(__version__="test")
            for name in (
                "transformers",
                "onnxscript",
                "huggingface_hub",
                "fastapi",
                "uvicorn",
                "safetensors",
            )
        },
        "tether.runtime.server": SimpleNamespace(_gpu_is_blackwell=lambda: False),
        "tether.curate.nudge_engine": SimpleNamespace(doctor_status=lambda: "disabled"),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr("tether.cli._check_trt_ep_load_chain", lambda add: None)
    run_checks = MagicMock(return_value=[])
    monkeypatch.setattr("tether.diagnostics.run_all_checks", run_checks)
    monkeypatch.setattr(
        "tether.diagnostics.format_json",
        lambda results, **kwargs: json.dumps({"model": kwargs["model_path"]}),
    )
    try:
        doctor(
            model=model_path,
            embodiment="custom",
            rtc=False,
            output_format=output_format,
            json_output=False,
            skip=[],
            show_calibration=False,
            calibration_cache=str(tmp_path / "calibration.json"),
        )
    except typer.Exit as exc:
        assert exc.exit_code == 0
    fake_onnx.helper.make_model.assert_called_once()
    if model_path:
        assert run_checks.call_args.kwargs["model_path"] == model_path
    else:
        run_checks.assert_not_called()
