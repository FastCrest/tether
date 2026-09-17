"""D-4: build_engine must never return a path it did not write.

Before this, a missing trtexec logged a warning and returned ``output_path``
for a file that was never created; five exporter call sites recorded that path
in ``result["files"]["expert_trt"]`` and ``build_all`` put it in its dict.
"""

from pathlib import Path

import pytest

from tether.config import HardwareProfile, get_hardware_profile
from tether.exporters import trt_build


def _hardware() -> HardwareProfile:
    return get_hardware_profile("orin-nano")


def test_build_engine_raises_when_trtexec_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(trt_build, "check_trtexec", lambda: False)
    out = tmp_path / "engines" / "expert.trt"

    with pytest.raises(RuntimeError, match="trtexec not found"):
        trt_build.build_engine(tmp_path / "expert.onnx", out, _hardware())

    assert not out.exists(), "no engine may be left behind"


def test_build_engine_raises_when_trtexec_writes_nothing(tmp_path, monkeypatch):
    """trtexec exiting 0 without producing the engine is still a failure."""
    monkeypatch.setattr(trt_build, "check_trtexec", lambda: True)

    class _Ok:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(trt_build.subprocess, "run", lambda *a, **k: _Ok())
    out = tmp_path / "expert.trt"

    with pytest.raises(RuntimeError, match="wrote no engine"):
        trt_build.build_engine(tmp_path / "expert.onnx", out, _hardware())


def test_build_all_returns_only_engines_that_exist(tmp_path, monkeypatch):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "expert_stack.onnx").write_bytes(b"not-really-onnx")
    monkeypatch.setattr(trt_build, "check_trtexec", lambda: False)

    engines = trt_build.build_all(onnx_dir, tmp_path / "trt", _hardware())

    assert engines == {}
    assert all(Path(p).exists() for p in engines.values())
