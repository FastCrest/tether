"""SmolVLA monolithic export end to end on a tiny random model: the real
exporter (onnx-diagnostic patches + Tether's eager_mask repair, torch.export,
torch.onnx.export, weight fusion) and then `tether validate`'s monolithic
path (ONNX Runtime vs native LeRobot sample_actions). See
_smolvla_monolithic_tiny.py.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent / "_smolvla_monolithic_tiny.py"


def _run(*args, timeout=900):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, str(HELPER), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    return result.stdout


def test_tiny_smolvla_monolithic_export_matches_pytorch(tmp_path):
    try:
        if importlib.metadata.version("lerobot") != "0.6.1":
            pytest.skip("exercises LeRobot 0.6.1")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires the monolithic extra (lerobot)")
    torch = pytest.importorskip("torch")
    if tuple(int(p) for p in torch.__version__.split(".")[:2]) < (2, 8):
        pytest.skip("onnx-diagnostic export patches need torch>=2.8")
    for module in ("onnx_diagnostic", "onnxruntime", "onnxscript"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"requires the monolithic extra ({module})")

    src, out = tmp_path / "src", tmp_path / "out"
    _run("build", str(src))
    _run("export", str(src), str(out))
    stdout = _run("validate", str(out))
    result = json.loads(stdout.split("RESULT=", 1)[1].splitlines()[0])

    assert result["export_kind"] == "monolithic_onnx"
    assert result["summary"]["passed"], result
    # FP32 CPU on both sides: hold the float32 floor, far inside the release
    # thresholds. The compounded weight-fusion defect measured 1.9e-2 here.
    assert result["summary"]["max_abs_diff_across_all"] < 1e-5, result
    assert result["summary"]["min_cosine_across_all"] > 0.9999999, result
