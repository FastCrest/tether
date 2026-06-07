"""Tests for validation utilities."""

import numpy as np
import torch
import pytest

from tether.validate import validate_outputs, ValidationResult


class TestValidateOutputs:
    def test_identical_outputs_pass(self):
        ref = np.array([1.0, 2.0, 3.0])
        result = validate_outputs(ref, ref, threshold=0.01)
        assert result.passed
        assert result.max_abs_diff == 0.0
        assert result.cosine_sim == pytest.approx(1.0)

    def test_small_diff_passes(self):
        ref = np.array([1.0, 2.0, 3.0])
        candidate = np.array([1.001, 2.001, 3.001])
        result = validate_outputs(ref, candidate, threshold=0.01)
        assert result.passed

    def test_large_diff_fails(self):
        ref = np.array([1.0, 2.0, 3.0])
        candidate = np.array([1.5, 2.0, 3.0])
        result = validate_outputs(ref, candidate, threshold=0.01)
        assert not result.passed
        assert result.max_abs_diff == pytest.approx(0.5)

    def test_opposite_direction_reports_negative_cosine(self):
        ref = np.array([1.0, 0.0])
        candidate = np.array([-1.0, 0.0])
        result = validate_outputs(ref, candidate, threshold=0.01)
        assert not result.passed
        assert result.cosine_sim == pytest.approx(-1.0)

    def test_orthogonal_vectors_report_zero_cosine(self):
        ref = np.array([1.0, 0.0])
        candidate = np.array([0.0, 1.0])
        result = validate_outputs(ref, candidate, threshold=2.0)
        assert result.cosine_sim == pytest.approx(0.0)

    def test_identical_tiny_outputs_report_unit_cosine(self):
        ref = np.array([1e-6, 0.0])
        result = validate_outputs(ref, ref, threshold=0.01)
        assert result.cosine_sim == pytest.approx(1.0)

    def test_zero_norm_outputs_report_zero_cosine(self):
        ref = np.array([0.0, 0.0])
        result = validate_outputs(ref, ref, threshold=0.01)
        assert result.passed
        assert result.cosine_sim == 0.0

    def test_shape_mismatch_fails(self):
        ref = np.array([1.0, 2.0])
        candidate = np.array([1.0, 2.0, 3.0])
        result = validate_outputs(ref, candidate)
        assert not result.passed
        assert np.isnan(result.cosine_sim)

    def test_torch_tensor_input(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        candidate = torch.tensor([1.001, 2.001, 3.001])
        result = validate_outputs(ref, candidate, threshold=0.01)
        assert result.passed

    def test_to_dict_includes_rounded_cosine(self):
        result = ValidationResult(
            passed=True,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            max_rel_diff=0.0,
            num_elements=3,
            threshold=0.01,
            details="output",
            cosine_sim=0.123456789,
        )
        assert result.to_dict()["cosine_sim"] == 0.123457
