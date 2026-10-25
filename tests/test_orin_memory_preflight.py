"""Orin engine-build OOM (tester bundle 2026-10-21): doctor must not stay silent about memory."""

import signal
import types

import pytest

from tether.diagnostics import check_model_load as load
from tether.smoke import SmokeError, _exit_hint, _wait_for_health

GB = 1024**3


def export(tmp_path, weight_bytes):
    (tmp_path / "model.onnx").write_bytes(b"\x00" * 16)
    with (tmp_path / "model.onnx.data").open("wb") as stream:
        stream.truncate(weight_bytes)  # sparse: no disk is used for the test
    return str(tmp_path)


def test_meminfo_is_used_when_psutil_is_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)  # import raises ImportError
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000000 kB\nMemAvailable: 5242880 kB\n")
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(meminfo if path == "/proc/meminfo" else path, *a, **k),
    )
    assert load._available_bytes() == 5242880 * 1024


def test_unreadable_memory_still_warns_not_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(load, "_available_bytes", lambda: None)
    result = load._run(export(tmp_path, GB))
    assert result.status == "warn" and "cannot read available memory" in result.actual


def test_fp32_export_that_would_oom_a_jetson_engine_build_warns_with_fp16_advice(
    monkeypatch, tmp_path
):
    # The observed case: 1.6 GB of weights on an 8 GB board with about 5 GB free.
    monkeypatch.setattr(load, "_available_bytes", lambda: int(5 * GB))
    monkeypatch.setattr(load, "_is_jetson", lambda: True)
    result = load._run(export(tmp_path, int(1.6 * GB)))
    assert result.status == "warn"
    assert "engine build" in result.expected and "FP16" in result.remediation


def test_same_export_on_a_workstation_or_with_ample_memory_passes(monkeypatch, tmp_path):
    path = export(tmp_path, int(1.6 * GB))
    monkeypatch.setattr(load, "_available_bytes", lambda: int(5 * GB))
    monkeypatch.setattr(load, "_is_jetson", lambda: False)
    assert load._run(path).status == "pass"
    monkeypatch.setattr(load, "_is_jetson", lambda: True)
    monkeypatch.setattr(load, "_available_bytes", lambda: int(40 * GB))
    assert load._run(path).status == "pass"


def test_model_that_cannot_even_load_still_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(load, "_available_bytes", lambda: int(1 * GB))
    assert load._run(export(tmp_path, int(1.6 * GB))).status == "fail"


def test_sigkill_exit_is_explained_and_other_exits_are_unchanged():
    assert "out-of-memory" in _exit_hint(-signal.SIGKILL) and "FP16" in _exit_hint(-signal.SIGKILL)
    assert _exit_hint(1) == "" and _exit_hint(None) == ""


def test_health_wait_reports_a_killed_server_with_the_hint():
    process = types.SimpleNamespace(poll=lambda: -9, returncode=-9)
    with pytest.raises(SmokeError, match=r"exit_code=-9\): SIGKILL .*out-of-memory"):
        _wait_for_health(process, "http://127.0.0.1:1", timeout_s=5)
