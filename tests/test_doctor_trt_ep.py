"""Unit tests for `tether doctor`'s ORT-TRT EP load chain validation.

Per ADR 2026-04-29-ort-trt-ep-first-class-support.md: doctor adds 4 checks:
1. libnvinfer.so.10 loadable
2. libcublas.so.12 loadable
3. libcudnn.so.9 loadable
4. ORT InferenceSession with TRT EP creates + active providers includes it

Tests verify each branch (lib missing / lib loadable / session creates with
TRT EP active / session creates with fallback / session creation throws /
ORT not installed / TRT EP not in available providers list).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tether.cli import _check_trt_ep_load_chain


def _fake_onnx_module():
    """Build a mock `onnx` module with the helpers our function uses.

    Real onnx C-extension re-import in mocked contexts segfaults on macOS,
    so we mock the entire surface our doctor uses (helper.* + TensorProto).
    """
    fake = MagicMock()
    fake.TensorProto.FLOAT = 1
    fake.helper.make_tensor_value_info.return_value = MagicMock()
    fake.helper.make_node.return_value = MagicMock()
    fake.helper.make_graph.return_value = MagicMock()
    fake.helper.make_opsetid.return_value = MagicMock()

    fake_model_proto = MagicMock()
    fake_model_proto.SerializeToString.return_value = b"fake-onnx-bytes"
    fake.helper.make_model.return_value = fake_model_proto

    return fake


@pytest.fixture
def add_capture():
    """Capture all add(name, ok, detail) calls into a dict for inspection."""
    calls = []

    def fake_add(name, ok, detail):
        calls.append({"name": name, "ok": ok, "detail": detail})

    return calls, fake_add


def _result_for(calls, name_substring):
    """Find the dict for a check whose name contains `name_substring`."""
    matches = [c for c in calls if name_substring in c["name"]]
    assert len(matches) == 1, f"Expected one match for {name_substring!r}, got {len(matches)}"
    return matches[0]


# ─── ctypes.CDLL behavior — libs loadable / not ──────────────────────────────


def test_all_libs_loadable_runs_full_chain(add_capture):
    """Happy path: libnvinfer + libcublas + libcudnn all found + load → session check runs."""
    calls, fake_add = add_capture

    fake_cdll = MagicMock(return_value=MagicMock())

    fake_session = MagicMock()
    fake_session.get_providers.return_value = [
        "TensorrtExecutionProvider", "CPUExecutionProvider"
    ]
    fake_ort = MagicMock()
    fake_ort.InferenceSession.return_value = fake_session
    fake_ort.get_available_providers.return_value = [
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
    ]

    with (
        patch("ctypes.CDLL", fake_cdll),
        patch("os.path.exists", return_value=True),  # all libs "found" in candidate dirs
        patch.dict("sys.modules", {"onnxruntime": fake_ort, "onnx": _fake_onnx_module()}),
    ):
        _check_trt_ep_load_chain(fake_add)

    assert _result_for(calls, "libnvinfer.so.10")["ok"] is True
    assert _result_for(calls, "libcublas.so.12")["ok"] is True
    assert _result_for(calls, "libcudnn.so.9")["ok"] is True
    assert _result_for(calls, "ORT-TRT EP active")["ok"] is True


def test_libnvinfer_missing_skips_session_check(add_capture):
    """If libnvinfer doesn't exist in candidate dirs, mark ✗ + skip session check."""
    calls, fake_add = add_capture

    # libnvinfer is NOT found anywhere; libcublas + libcudnn ARE found
    def exists_side_effect(path):
        return "libnvinfer" not in path

    with (
        patch("ctypes.CDLL", return_value=MagicMock()),
        patch("os.path.exists", side_effect=exists_side_effect),
    ):
        _check_trt_ep_load_chain(fake_add)

    nvinfer = _result_for(calls, "libnvinfer.so.10")
    assert nvinfer["ok"] is False
    assert "NOT installed" in nvinfer["detail"]
    assert "pip install" in nvinfer["detail"]  # remediation hint present

    session = _result_for(calls, "ORT-TRT EP active")
    assert session["ok"] is False
    assert "skipped" in session["detail"]


def test_remediation_hints_present_for_each_lib(add_capture):
    """Each missing lib check must include a pip-install hint."""
    calls, fake_add = add_capture

    with (
        patch("ctypes.CDLL", side_effect=OSError("not found")),
        patch("os.path.exists", return_value=False),  # nothing found anywhere
    ):
        _check_trt_ep_load_chain(fake_add)

    for libsub in ["libnvinfer.so.10", "libcublas.so.12", "libcudnn.so.9"]:
        result = _result_for(calls, libsub)
        assert result["ok"] is False, f"{libsub} expected to be ✗"
        assert "pip install" in result["detail"], (
            f"{libsub} missing the pip-install remediation hint"
        )


# ─── ORT-side behavior — session creates / fallback / throws ─────────────────


def test_trt_ep_not_in_available_providers(add_capture):
    """If ORT was built without TRT EP support, surface clearly."""
    calls, fake_add = add_capture

    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider", "CPUExecutionProvider"  # no TRT
    ]

    with (
        patch("ctypes.CDLL", return_value=MagicMock()),
        patch("os.path.exists", return_value=True),
        patch.dict("sys.modules", {"onnxruntime": fake_ort, "onnx": _fake_onnx_module()}),
    ):
        _check_trt_ep_load_chain(fake_add)

    session = _result_for(calls, "ORT-TRT EP active")
    assert session["ok"] is False
    assert "not in onnxruntime's available providers" in session["detail"]


def test_session_creates_but_trt_ep_falls_back(add_capture):
    """ORT silently dropped TRT EP from active providers — surface clearly."""
    calls, fake_add = add_capture

    fake_session = MagicMock()
    # TRT EP requested but ORT fell back to CUDA only
    fake_session.get_providers.return_value = [
        "CUDAExecutionProvider", "CPUExecutionProvider"
    ]
    fake_ort = MagicMock()
    fake_ort.InferenceSession.return_value = fake_session
    fake_ort.get_available_providers.return_value = [
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
    ]

    with (
        patch("ctypes.CDLL", return_value=MagicMock()),
        patch("os.path.exists", return_value=True),
        patch.dict("sys.modules", {"onnxruntime": fake_ort, "onnx": _fake_onnx_module()}),
    ):
        _check_trt_ep_load_chain(fake_add)

    session = _result_for(calls, "ORT-TRT EP active")
    assert session["ok"] is False
    assert "fell back" in session["detail"]
    assert "LD_LIBRARY_PATH" in session["detail"]  # diagnostic hint


def test_session_creation_throws(add_capture):
    """ORT session creation raises — caught + reported with exception type."""
    calls, fake_add = add_capture

    fake_ort = MagicMock()
    fake_ort.InferenceSession.side_effect = RuntimeError("BOOM internal ORT error")
    fake_ort.get_available_providers.return_value = [
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
    ]

    with (
        patch("ctypes.CDLL", return_value=MagicMock()),
        patch("os.path.exists", return_value=True),
        patch.dict("sys.modules", {"onnxruntime": fake_ort, "onnx": _fake_onnx_module()}),
    ):
        _check_trt_ep_load_chain(fake_add)

    session = _result_for(calls, "ORT-TRT EP active")
    assert session["ok"] is False
    assert "RuntimeError" in session["detail"]
    assert "BOOM" in session["detail"]
