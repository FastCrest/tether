"""Exercise the precision command's real archive and publication boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.test_fp16_validated_export import _export
from tether.cli import app


def _run(src, dst, case, *extra):
    return CliRunner().invoke(
        app,
        [
            "export-precision",
            str(src),
            "--output",
            str(dst),
            "--validation-inputs",
            str(case),
            "--max-abs-threshold",
            "0.01",
            "--cos-threshold",
            "0.999",
            *extra,
        ],
        env={"TETHER_SKIP_ONBOARDING": "1", "COLUMNS": "40"},
    )


def test_command_creates_real_validated_export(tmp_path):
    src = _export(tmp_path / "fp32")
    case = tmp_path / "case.npz"
    np.savez(case, x=np.ones((1, 64), dtype=np.float32))
    dst = tmp_path / "fp16"
    result = _run(src, dst, case)
    assert result.exit_code == 0, result.output
    receipt = json.loads((dst / "precision-conversion.json").read_text())
    assert receipt["parity_verdict"] == "PASS"
    assert "Target memory fit" in result.output
    assert (src / "model.onnx").exists()


@pytest.mark.parametrize(
    "invalid", ["object", "wrong-name", "wrong-dtype", "nan-threshold", "zero-tolerance"]
)
def test_command_refuses_invalid_or_failed_cases_without_publication(tmp_path, invalid):
    src = _export(tmp_path / "fp32")
    case = tmp_path / "case.npz"
    arrays = {"x": np.ones((1, 64), dtype=np.float32)}
    extra = []
    if invalid == "object":
        arrays = {"x": np.array([{"pickle": True}], dtype=object)}
    elif invalid == "wrong-name":
        arrays = {"unexpected": arrays["x"]}
    elif invalid == "wrong-dtype":
        arrays = {"x": arrays["x"].astype(np.float64)}
    elif invalid == "nan-threshold":
        extra = ["--max-abs-threshold", "nan"]
    else:
        extra = ["--max-abs-threshold", "0"]
    np.savez(case, **arrays)
    dst = tmp_path / "refused"
    result = _run(src, dst, case, *extra)
    assert result.exit_code in (1, 2), result.output
    assert not dst.exists()
    assert (src / "model.onnx").exists()
    if invalid == "zero-tolerance":
        failed = json.loads(result.output[result.output.index("{\n") :])
        assert failed["parity_verdict"] == "FAIL"
