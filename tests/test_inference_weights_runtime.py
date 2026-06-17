"""Lift #3 Day 3 — InferenceWeightsRuntime tests.

3 cases per the plan spec. Uses a duck-typed ORT-like mock — real ORT
end-to-end parity is Day 4's gate (needs Modal GPU + real ONNX files).
The mock validates the orchestration: name mapping, binding order,
output fanout. The actual ORT IOBinding call is exercised in Day 4.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tether.runtime.inference_weights import (
    InferenceWeightsRuntime,
    WeightBindingError,
    validate_name_mapping,
)


# ─── Test 1: construct + dispatch produces correct output shape ─────


def test_construct_runtime_and_dispatch_returns_correct_output_shape():
    """Build InferenceWeightsRuntime from a flat dict, dispatch a
    synthetic `/act`-shaped input, assert the output dict has the
    correct keys."""
    flat = {
        "vision_backbone.weight": torch.randn(64, 64),
        "llm_backbone.fc.weight": torch.randn(128, 128),
        "vla_head.expert_stack.weight": torch.randn(32, 32),
    }
    expected_names = set(flat.keys())

    # Mock ORT session — exposes get_outputs / get_inputs / io_binding
    # / run_with_iobinding with the surface InferenceWeightsRuntime uses.
    mock_session = MagicMock()
    mock_output_meta = MagicMock()
    mock_output_meta.name = "velocity"
    mock_session.get_outputs.return_value = [mock_output_meta]

    mock_io_binding = MagicMock()
    # Return an OrtValue-like mock for get_outputs after run_with_iobinding.
    mock_ortval_output = MagicMock()
    mock_ortval_output.numpy.return_value = torch.randn(1, 50, 32).numpy()
    mock_io_binding.get_outputs.return_value = [mock_ortval_output]
    mock_session.io_binding.return_value = mock_io_binding

    # Patch the lazy onnxruntime import so we don't hit ImportError on systems
    # without ORT installed. Lift #5 gates on the real lib; Day 3 doesn't.
    import sys as _sys
    import types as _types
    if "onnxruntime" not in _sys.modules:
        _stub_ort = _types.ModuleType("onnxruntime")
        _stub_ort.OrtValue = MagicMock()
        _stub_ort.OrtValue.ortvalue_from_numpy = MagicMock(return_value=MagicMock())
        _sys.modules["onnxruntime"] = _stub_ort

    runtime = InferenceWeightsRuntime(
        flat_weights=flat,
        ort_session=mock_session,
        expected_names=expected_names,
        device="cpu",
        device_id=0,
    )

    assert runtime.num_weight_tensors == 3

    outputs = runtime.predict_action(
        runtime_inputs={
            "noisy_actions": torch.randn(1, 50, 32),
            "timestep": torch.tensor([0.5]),
        },
    )
    assert "velocity" in outputs
    assert outputs["velocity"].shape == (1, 50, 32)


# ─── Test 2: weight-binding drift (missing key) raises ──────────────


def test_missing_key_in_flat_dict_raises_clearly():
    """ORT expects a key that flat_weights doesn't provide. Construction
    raises WeightBindingError listing the missing keys."""
    flat = {
        "vision_backbone.weight": torch.randn(4, 4),
        # llm_backbone.fc.weight is missing
    }
    expected = {"vision_backbone.weight", "llm_backbone.fc.weight"}

    mock_session = MagicMock()

    with pytest.raises(WeightBindingError) as excinfo:
        InferenceWeightsRuntime(
            flat_weights=flat,
            ort_session=mock_session,
            expected_names=expected,
        )
    assert "missing from flat dict" in str(excinfo.value)
    assert "llm_backbone.fc.weight" in str(excinfo.value)


# ─── Test 3: flat dict has extra key raises ─────────────────────────


def test_extra_key_in_flat_dict_raises_clearly():
    """Flat dict has a key that ORT doesn't expect (typo in
    prepare_triton, or weights for a removed component). Construction
    raises WeightBindingError listing the extras."""
    flat = {
        "vision_backbone.weight": torch.randn(4, 4),
        "vla_head.expert_stack.weight": torch.randn(4, 4),
        "typo.weight": torch.randn(4, 4),  # extra
    }
    expected = {"vision_backbone.weight", "vla_head.expert_stack.weight"}

    mock_session = MagicMock()

    with pytest.raises(WeightBindingError) as excinfo:
        InferenceWeightsRuntime(
            flat_weights=flat,
            ort_session=mock_session,
            expected_names=expected,
        )
    assert "extra in flat dict" in str(excinfo.value)
    assert "typo.weight" in str(excinfo.value)


# ─── Bonus: name-mapping helper used directly ───────────────────────


def test_validate_name_mapping_passes_when_exactly_matched():
    """Sanity: the no-drift case doesn't raise."""
    validate_name_mapping(
        flat_keys={"a", "b", "c"},
        expected_names={"a", "b", "c"},
    )


def test_validate_name_mapping_reports_both_missing_and_extra():
    """Both classes of drift surfaced in a single error."""
    with pytest.raises(WeightBindingError) as excinfo:
        validate_name_mapping(
            flat_keys={"a", "b", "extra"},
            expected_names={"a", "c", "d"},
        )
    err = str(excinfo.value)
    assert "missing from flat dict" in err
    assert "extra in flat dict" in err
