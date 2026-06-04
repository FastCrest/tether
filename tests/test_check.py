"""Tests for training-to-deployment validation checks."""

import torch
import pytest

from tether.validate_training import (
    check_key_structure,
    check_nan_inf,
    CheckResult,
)


class TestKeyStructure:
    def test_valid_smolvla_structure(self):
        sd = {
            "model.vlm_with_expert.vlm.layer.weight": torch.randn(10),
            "model.vlm_with_expert.lm_expert.layer.weight": torch.randn(10),
            "model.action_in_proj.weight": torch.randn(10),
        }
        result = check_key_structure(sd)
        assert result.passed
        assert "VLM" in result.detail

    def test_missing_components(self):
        sd = {"some.random.key": torch.randn(10)}
        result = check_key_structure(sd)
        assert not result.passed


class TestNanInf:
    def test_clean_weights(self):
        sd = {
            "weight_a": torch.randn(10),
            "weight_b": torch.randn(5, 5),
        }
        result = check_nan_inf(sd)
        assert result.passed

    def test_nan_detected(self):
        sd = {
            "clean": torch.randn(10),
            "bad": torch.tensor([1.0, float("nan"), 3.0]),
        }
        result = check_nan_inf(sd)
        assert not result.passed
        assert "bad" in result.detail

    def test_inf_detected(self):
        sd = {"bad": torch.tensor([float("inf")])}
        result = check_nan_inf(sd)
        assert not result.passed
