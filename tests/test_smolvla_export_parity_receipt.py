from __future__ import annotations

import numpy as np
import pytest

from tether.eval.smolvla_export_parity import (
    SmolVLAExportParityError,
    abs_error_distribution,
    build_smolvla_export_parity_receipt,
)

EXPORT_IDENTITY = {
    "schema": 1,
    "kind": "smolvla-monolithic-onnx",
    "source": "smolvla-monolithic-export",
    "revision": "b" * 40,
    "artifact_sha256": "d" * 64,
    "file_count": 1,
    "size_bytes": 4,
    "artifacts": [{"path": "model.onnx", "size_bytes": 4, "sha256": "e" * 64}],
    "hash_contract": "studio-training-artifacts-v1",
    "symlink_policy": "excluded",
}


def _distribution(max_abs=0.002):
    reference = np.zeros((2, 1, 50, 32), np.float32)
    export = reference.copy()
    export.reshape(-1)[:10] = max_abs
    return abs_error_distribution(reference, export)


def _build(**overrides):
    args = {
        "model_source": "lerobot/smolvla_base",
        "model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "tether_commit": "b" * 40,
        "export_identity": EXPORT_IDENTITY,
        "platform_system": "Linux",
        "ort_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "requested_provider": "CUDAExecutionProvider",
        "cpu_fallback_disabled": True,
        "input_seed": 42,
        "noise_seed": 99,
        "num_steps": 10,
        "shared_input_sha256": "c" * 64,
        "reference_shape": [1, 50, 32],
        "export_shape": [1, 50, 32],
        "first_cosine": 0.99999,
        "first_max_abs": 0.001,
        "full_cosine": 0.9999,
        "full_max_abs": 0.002,
        "error_distribution": _distribution(0.002),
        "cases": [{"index": 0}],
    }
    args.update(overrides)
    return build_smolvla_export_parity_receipt(**args)


def test_receipt_matches_the_kind_and_artifact_studio_binds():
    # Studio tether_private/qualification/families.py smolvla Prerequisite.
    receipt = _build()
    assert receipt["kind"] == "smolvla-reference-shared-input-export-parity"
    assert receipt["family"] == "smolvla"
    assert receipt["export_identity"]["kind"] == "smolvla-monolithic-onnx"
    assert receipt["thresholds"] == {"minimum_cosine": 0.999, "maximum_absolute_error_exclusive": 0.1}
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "recorded"
    assert receipt["metrics"]["error_distribution"]["elements"] == 2 * 50 * 32


def test_distribution_reports_percentiles_and_fraction_above_tolerance():
    reference = np.zeros(1000, np.float32)
    export = reference.copy()
    export[-5:] = 0.5  # 0.5% of elements far above the 0.1 bar
    dist = abs_error_distribution(reference, export, tolerance=0.1)
    assert dist["p50"] == 0.0 and dist["p99"] == 0.0
    assert dist["max"] == pytest.approx(0.5)
    assert dist["frac_at_or_above_tolerance"] == pytest.approx(0.005)
    assert dist["mean"] == pytest.approx(0.0025)


def test_worst_case_failure_fails_the_receipt():
    receipt = _build(full_max_abs=0.2, error_distribution=_distribution(0.2))
    assert receipt["verdict"] == "failed"
    assert receipt["external_acceptance"] == "not-run"


def test_distribution_must_agree_with_full_max_abs():
    with pytest.raises(SmolVLAExportParityError, match="error_distribution.max"):
        _build(full_max_abs=0.002, error_distribution=_distribution(0.05))


def test_unordered_distribution_is_rejected():
    dist = _distribution(0.002)
    dist["p95"] = 1.0
    with pytest.raises(SmolVLAExportParityError, match="ordered"):
        _build(error_distribution=dist)


def test_cases_are_required():
    with pytest.raises(SmolVLAExportParityError, match="cases"):
        _build(cases=[])


def test_shape_mismatch_is_refused_by_distribution():
    with pytest.raises(SmolVLAExportParityError, match="shape"):
        abs_error_distribution(np.zeros((1, 50, 32)), np.zeros((1, 49, 32)))
