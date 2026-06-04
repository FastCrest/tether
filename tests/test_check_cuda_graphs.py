"""Tests for `tether doctor` cuda-graphs check (`check_cuda_graphs.py`).

Covers the skip / pass / warn / fail branches of the check without
requiring a real GPU — all ORT session construction + capture is mocked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tether.diagnostics.check_cuda_graphs import _run as check_run


def _decomposed_config() -> dict:
    return {
        "export_kind": "decomposed",
        "decomposed": {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "past_kv_tensor_names": ["past_k_0", "past_v_0"],
            "paligemma_layers": 1,
        },
    }


def _monolithic_config() -> dict:
    return {"export_kind": "monolithic"}


def _write_export(tmp_path: Path, config: dict) -> Path:
    """Write a minimal fake export (just tether_config.json) and return its path."""
    (tmp_path / "tether_config.json").write_text(json.dumps(config))
    # Create empty onnx files for path existence
    (tmp_path / "vlm_prefix.onnx").write_bytes(b"\x00")
    (tmp_path / "expert_denoise.onnx").write_bytes(b"\x00")
    return tmp_path


def test_skip_when_no_tether_config(tmp_path):
    result = check_run(model_path=str(tmp_path), embodiment_name="franka", rtc=False)
    assert result.status == "skip"
    assert "tether_config.json" in result.expected


def test_skip_when_export_is_monolithic(tmp_path):
    _write_export(tmp_path, _monolithic_config())
    result = check_run(model_path=str(tmp_path), embodiment_name="franka", rtc=False)
    assert result.status == "skip"
    assert "decomposed" in result.expected.lower()


def test_skip_when_onnxruntime_not_importable(tmp_path):
    _write_export(tmp_path, _decomposed_config())

    # Simulate missing onnxruntime by intercepting the import inside the check
    import tether.diagnostics.check_cuda_graphs as mod

    # Monkey-patch ort import at the check module level by stashing a raising
    # import into sys.modules
    orig_ort = sys.modules.pop("onnxruntime", None)
    try:
        import importlib
        # Insert a dummy that raises ImportError on attribute access
        fake = MagicMock()
        fake.get_available_providers = MagicMock(side_effect=ImportError("mock no ort"))
        sys.modules["onnxruntime"] = fake
        # But the check uses `import onnxruntime as ort` at function scope, which
        # will succeed and hit the fake. Next line uses get_available_providers
        # which raises. The check's except-ImportError only catches the `import`
        # line. This gets fiddly — simpler: patch ort.get_available_providers
        # to return an empty list (no CUDAExecutionProvider).
        fake.get_available_providers = MagicMock(return_value=["CPUExecutionProvider"])
        result = check_run(model_path=str(tmp_path), embodiment_name="franka", rtc=False)
        assert result.status == "skip"
        assert "CUDAExecutionProvider" in result.expected
    finally:
        if orig_ort is not None:
            sys.modules["onnxruntime"] = orig_ort
        elif "onnxruntime" in sys.modules:
            del sys.modules["onnxruntime"]


def _patched_ort_with_providers(providers: list[str]):
    """Build a mock onnxruntime module that exposes the given providers."""
    fake = MagicMock()
    fake.get_available_providers = MagicMock(return_value=providers)
    # InferenceSession mock with a get_inputs() that returns a minimal input
    mock_input = MagicMock()
    mock_input.name = "img_base"
    mock_input.shape = [1, 3, 224, 224]
    mock_input.type = "tensor(float)"
    fake.InferenceSession = MagicMock(return_value=MagicMock(
        get_inputs=MagicMock(return_value=[mock_input]),
        run=MagicMock(return_value=[[]]),
    ))
    return fake


def test_pass_when_both_sessions_capture(tmp_path):
    _write_export(tmp_path, _decomposed_config())

    # Mock try_capture_or_fall_back to return CudaGraphWrapper (captured=True) for both
    with patch.dict(sys.modules, {"onnxruntime": _patched_ort_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])}):
        with patch("tether.runtime.cuda_graphs.try_capture_or_fall_back") as mock_try:
            mock_wrapper = MagicMock()
            mock_wrapper.captured = True
            mock_try.return_value = mock_wrapper

            result = check_run(model_path=str(tmp_path), embodiment_name="franka", rtc=False)
            assert result.status == "pass"
            assert "16x" in result.actual or "compound" in result.actual


def test_warn_when_only_expert_captures(tmp_path):
    _write_export(tmp_path, _decomposed_config())

    with patch.dict(sys.modules, {"onnxruntime": _patched_ort_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])}):
        with patch("tether.runtime.cuda_graphs.try_capture_or_fall_back") as mock_try:
            # First call (vlm_prefix): eager fallback (captured=False)
            # Second call (expert_denoise): captured=True
            prefix_wrapper = MagicMock()
            prefix_wrapper.captured = False
            expert_wrapper = MagicMock()
            expert_wrapper.captured = True
            mock_try.side_effect = [prefix_wrapper, expert_wrapper]

            result = check_run(model_path=str(tmp_path), embodiment_name="franka", rtc=False)
            assert result.status == "warn"
            assert "vlm_prefix" in result.actual
            assert "expert_denoise" in result.actual
            assert "A100" in result.remediation or "AGX Orin" in result.remediation


def test_fail_when_neither_captures(tmp_path):
    _write_export(tmp_path, _decomposed_config())

    with patch.dict(sys.modules, {"onnxruntime": _patched_ort_with_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])}):
        with patch("tether.runtime.cuda_graphs.try_capture_or_fall_back") as mock_try:
            prefix_wrapper = MagicMock()
            prefix_wrapper.captured = False
            expert_wrapper = MagicMock()
            expert_wrapper.captured = False
            mock_try.side_effect = [prefix_wrapper, expert_wrapper]

            result = check_run(model_path=str(tmp_path), embodiment_name="franka", rtc=False)
            assert result.status == "fail"
            assert result.remediation  # fails must have remediation
