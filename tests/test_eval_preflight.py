"""Tests for src/tether/eval/preflight.py — Phase 1 eval-as-a-service Day 2."""
from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from tether.eval.preflight import (
    ALL_FAILURE_MODES,
    DEFAULT_PREFLIGHT_TIMEOUT_S,
    PreflightResult,
    PreflightSmokeTest,
)


# ---------------------------------------------------------------------------
# Bounded enum
# ---------------------------------------------------------------------------


def test_all_failure_modes_bounded():
    """Stable across releases — surfaced in CLI + telemetry."""
    assert "ok" in ALL_FAILURE_MODES
    assert "input-hang" in ALL_FAILURE_MODES
    assert "egl-black-frames" in ALL_FAILURE_MODES
    assert "dep-version-conflict" in ALL_FAILURE_MODES
    assert "osmesa-compile-hang" in ALL_FAILURE_MODES
    assert "subprocess-error" in ALL_FAILURE_MODES
    assert "import-error" in ALL_FAILURE_MODES
    assert "unknown" in ALL_FAILURE_MODES


def test_default_preflight_timeout_is_positive():
    assert DEFAULT_PREFLIGHT_TIMEOUT_S > 0


# ---------------------------------------------------------------------------
# PreflightResult dataclass
# ---------------------------------------------------------------------------


def test_preflight_result_is_frozen():
    r = PreflightResult(
        passed=True, failure_mode="ok", elapsed_s=1.0,
        stdout="", stderr="", remediation="",
    )
    with pytest.raises(AttributeError):
        r.passed = False  # type: ignore[misc]


def test_preflight_result_rejects_invalid_failure_mode():
    with pytest.raises(ValueError, match="failure_mode must be one of"):
        PreflightResult(
            passed=False, failure_mode="invented-mode", elapsed_s=0.1,
            stdout="", stderr="", remediation="",
        )


def test_preflight_result_rejects_passed_with_non_ok_mode():
    with pytest.raises(ValueError, match="passed=True but failure_mode"):
        PreflightResult(
            passed=True, failure_mode="import-error", elapsed_s=0.1,
            stdout="", stderr="", remediation="",
        )


def test_preflight_result_rejects_failed_with_ok_mode():
    with pytest.raises(ValueError, match="passed=False but failure_mode='ok'"):
        PreflightResult(
            passed=False, failure_mode="ok", elapsed_s=0.1,
            stdout="", stderr="", remediation="",
        )


def test_preflight_result_passed_with_ok_is_valid():
    r = PreflightResult(
        passed=True, failure_mode="ok", elapsed_s=0.5,
        stdout="PREFLIGHT_OK\n", stderr="", remediation="",
    )
    assert r.passed
    assert r.failure_mode == "ok"


def test_preflight_result_failed_with_failure_mode_is_valid():
    r = PreflightResult(
        passed=False, failure_mode="import-error", elapsed_s=0.5,
        stdout="", stderr="boom", remediation="install libero",
    )
    assert not r.passed
    assert r.failure_mode == "import-error"


# ---------------------------------------------------------------------------
# PreflightSmokeTest.run — input validation
# ---------------------------------------------------------------------------


def test_run_rejects_zero_timeout():
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        PreflightSmokeTest.run(timeout_s=0)


def test_run_rejects_negative_timeout():
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        PreflightSmokeTest.run(timeout_s=-1)


# ---------------------------------------------------------------------------
# PreflightSmokeTest.run — happy path
# ---------------------------------------------------------------------------


def _stub_subprocess_run(monkeypatch, *, returncode: int, stdout: str = "", stderr: str = ""):
    """Replace subprocess.run with a stub that returns a fake CompletedProcess."""
    fake = MagicMock()
    fake.returncode = returncode
    fake.stdout = stdout
    fake.stderr = stderr

    def _run(*args, **kwargs):
        return fake

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _run)


def test_run_returns_passed_when_subprocess_succeeds(monkeypatch):
    _stub_subprocess_run(monkeypatch, returncode=0, stdout="PREFLIGHT_OK\n")
    result = PreflightSmokeTest.run(timeout_s=10)
    assert result.passed
    assert result.failure_mode == "ok"
    assert result.remediation == ""
    assert result.elapsed_s >= 0


def test_run_returns_failed_when_subprocess_zero_but_no_ok_marker(monkeypatch):
    """Defensive: returncode=0 alone is not enough — script must print PREFLIGHT_OK."""
    _stub_subprocess_run(monkeypatch, returncode=0, stdout="something else")
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    # No marker → unknown
    assert result.failure_mode == "unknown"


# ---------------------------------------------------------------------------
# PreflightSmokeTest.run — failure mode parsing
# ---------------------------------------------------------------------------


def test_run_extracts_input_hang_marker(monkeypatch):
    _stub_subprocess_run(
        monkeypatch, returncode=2,
        stderr="PREFLIGHT_FAILURE_MODE=input-hang: stdin not a tty\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "input-hang"
    assert "patch_libero" in result.remediation


def test_run_extracts_egl_black_frames_marker(monkeypatch):
    _stub_subprocess_run(
        monkeypatch, returncode=8,
        stderr="PREFLIGHT_FAILURE_MODE=egl-black-frames: EGL init failed\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "egl-black-frames"
    assert "osmesa" in result.remediation.lower()


def test_run_extracts_dep_version_conflict_marker(monkeypatch):
    _stub_subprocess_run(
        monkeypatch, returncode=3,
        stderr="PREFLIGHT_FAILURE_MODE=dep-version-conflict: ImportError\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "dep-version-conflict"
    assert "robosuite" in result.remediation


def test_run_extracts_import_error_marker(monkeypatch):
    _stub_subprocess_run(
        monkeypatch, returncode=2,
        stderr="PREFLIGHT_FAILURE_MODE=import-error: No module named libero\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "import-error"
    assert "pip install" in result.remediation


def test_run_extracts_osmesa_compile_hang_marker(monkeypatch):
    _stub_subprocess_run(
        monkeypatch, returncode=9,
        stderr="PREFLIGHT_FAILURE_MODE=osmesa-compile-hang: shader crash\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "osmesa-compile-hang"


def test_run_returns_unknown_when_no_marker(monkeypatch):
    _stub_subprocess_run(
        monkeypatch, returncode=99,
        stderr="some random error without marker\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "unknown"
    assert "GitHub issue" in result.remediation


def test_run_returns_unknown_when_marker_value_invalid(monkeypatch):
    """Marker present but value not in bounded enum → unknown."""
    _stub_subprocess_run(
        monkeypatch, returncode=99,
        stderr="PREFLIGHT_FAILURE_MODE=made-up-value: anything\n",
    )
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "unknown"


def test_run_returns_unknown_when_stderr_empty(monkeypatch):
    _stub_subprocess_run(monkeypatch, returncode=99)
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "unknown"


# ---------------------------------------------------------------------------
# PreflightSmokeTest.run — timeout handling
# ---------------------------------------------------------------------------


def test_run_treats_timeout_as_osmesa_compile_hang(monkeypatch):
    """TimeoutExpired → bounded failure_mode='osmesa-compile-hang' (most-likely cause)."""
    def _run_raises(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="python", timeout=10, output="partial", stderr="",
        )

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _run_raises)
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "osmesa-compile-hang"
    assert "preflight-timeout" in result.remediation or "Increase" in result.remediation


def test_run_handles_timeout_with_bytes_stderr(monkeypatch):
    """TimeoutExpired stderr can be bytes — must decode without crashing."""
    def _run_raises(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="python", timeout=10,
            output=b"binary output", stderr=b"binary err",
        )

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _run_raises)
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.stderr == "binary err"
    assert result.stdout == "binary output"


def test_run_handles_timeout_with_none_outputs(monkeypatch):
    """TimeoutExpired output/stderr can be None — handled."""
    def _run_raises(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=10)

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _run_raises)
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "osmesa-compile-hang"
    assert result.stdout == ""
    assert result.stderr == ""


# ---------------------------------------------------------------------------
# PreflightSmokeTest.run — subprocess launch errors
# ---------------------------------------------------------------------------


def test_run_returns_subprocess_error_on_filenotfound(monkeypatch):
    """Bad python_executable → FileNotFoundError → subprocess-error."""
    def _run_raises(*args, **kwargs):
        raise FileNotFoundError("/does/not/exist not found")

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _run_raises)
    result = PreflightSmokeTest.run(timeout_s=10, python_executable="/does/not/exist")
    assert not result.passed
    assert result.failure_mode == "subprocess-error"
    assert "FileNotFoundError" in result.stderr


def test_run_returns_subprocess_error_on_oserror(monkeypatch):
    def _run_raises(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _run_raises)
    result = PreflightSmokeTest.run(timeout_s=10)
    assert not result.passed
    assert result.failure_mode == "subprocess-error"
    assert "OSError" in result.stderr


# ---------------------------------------------------------------------------
# PreflightSmokeTest.run — uses sys.executable by default
# ---------------------------------------------------------------------------


def test_run_uses_sys_executable_when_not_specified(monkeypatch):
    """No python_executable → falls back to sys.executable."""
    captured = {}

    def _capture(*args, **kwargs):
        captured["argv"] = args[0]
        fake = MagicMock(returncode=0, stdout="PREFLIGHT_OK\n", stderr="")
        return fake

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _capture)
    PreflightSmokeTest.run(timeout_s=10)
    assert captured["argv"][0] == sys.executable


def test_run_uses_provided_python_executable(monkeypatch):
    captured = {}

    def _capture(*args, **kwargs):
        captured["argv"] = args[0]
        fake = MagicMock(returncode=0, stdout="PREFLIGHT_OK\n", stderr="")
        return fake

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _capture)
    PreflightSmokeTest.run(timeout_s=10, python_executable="/custom/python3")
    assert captured["argv"][0] == "/custom/python3"


def test_run_passes_smoke_test_script_via_dash_c(monkeypatch):
    """Subprocess invoked as `python -c <script>` so a broken tether install can still probe."""
    captured = {}

    def _capture(*args, **kwargs):
        captured["argv"] = args[0]
        fake = MagicMock(returncode=0, stdout="PREFLIGHT_OK\n", stderr="")
        return fake

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _capture)
    PreflightSmokeTest.run(timeout_s=10)
    assert captured["argv"][1] == "-c"
    # Script should reference LIBERO + the marker convention
    assert "MUJOCO_GL" in captured["argv"][2]
    assert "PREFLIGHT_FAILURE_MODE=" in captured["argv"][2]


def test_run_passes_timeout_to_subprocess(monkeypatch):
    captured = {}

    def _capture(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        fake = MagicMock(returncode=0, stdout="PREFLIGHT_OK\n", stderr="")
        return fake

    monkeypatch.setattr("tether.eval.preflight.subprocess.run", _capture)
    PreflightSmokeTest.run(timeout_s=42.5)
    assert captured["timeout"] == 42.5


# ---------------------------------------------------------------------------
# _extract_failure_mode helper
# ---------------------------------------------------------------------------


def test_extract_returns_unknown_for_empty_string():
    assert PreflightSmokeTest._extract_failure_mode("") == "unknown"


def test_extract_returns_unknown_for_whitespace_only():
    assert PreflightSmokeTest._extract_failure_mode("   \n  ") == "unknown"


def test_extract_finds_marker_in_multiline_stderr():
    stderr = (
        "Some preamble logging\n"
        "PREFLIGHT_FAILURE_MODE=import-error: bad module\n"
        "Trailing trace\n"
    )
    assert PreflightSmokeTest._extract_failure_mode(stderr) == "import-error"


def test_extract_finds_first_valid_marker():
    """Only the first matching marker line is parsed."""
    stderr = (
        "PREFLIGHT_FAILURE_MODE=osmesa-compile-hang: actual cause\n"
        "PREFLIGHT_FAILURE_MODE=import-error: secondary\n"
    )
    assert PreflightSmokeTest._extract_failure_mode(stderr) == "osmesa-compile-hang"


def test_extract_skips_invalid_marker_for_valid_subsequent():
    """Invalid mode in first line is skipped; valid mode in second is returned."""
    stderr = (
        "PREFLIGHT_FAILURE_MODE=fake-mode: noise\n"
        "PREFLIGHT_FAILURE_MODE=import-error: real cause\n"
    )
    assert PreflightSmokeTest._extract_failure_mode(stderr) == "import-error"


def test_extract_strips_colon_separator():
    """Marker format: PREFLIGHT_FAILURE_MODE=<mode>:<details>."""
    assert PreflightSmokeTest._extract_failure_mode(
        "PREFLIGHT_FAILURE_MODE=input-hang"
    ) == "input-hang"


def test_extract_returns_unknown_for_unmapped_marker_value():
    assert PreflightSmokeTest._extract_failure_mode(
        "PREFLIGHT_FAILURE_MODE=fictional-mode: ...\n"
    ) == "unknown"


# ---------------------------------------------------------------------------
# Remediation strings
# ---------------------------------------------------------------------------


def test_remediation_present_for_every_failure_mode():
    """Every non-ok mode in the bounded enum must have non-empty remediation."""
    from tether.eval.preflight import _REMEDIATION_BY_MODE

    for mode in ALL_FAILURE_MODES:
        if mode == "ok":
            assert _REMEDIATION_BY_MODE[mode] == ""
        else:
            assert _REMEDIATION_BY_MODE[mode], f"missing remediation for {mode!r}"
            assert len(_REMEDIATION_BY_MODE[mode]) > 30, (
                f"remediation for {mode!r} too short to be useful"
            )
