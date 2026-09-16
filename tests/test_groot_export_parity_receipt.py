from __future__ import annotations

import math

import pytest

from tether.eval.groot_export_parity import (
    GrootExportParityError,
    build_groot_export_parity_receipt,
)


REVISION = "a" * 40
TETHER_COMMIT = "b" * 40
INPUT_SHA = "c" * 64
EXPORT_IDENTITY = {
    "schema": 1,
    "kind": "groot-monolithic-onnx",
    "source": "groot-monolithic-export",
    "revision": TETHER_COMMIT,
    "artifact_sha256": "d" * 64,
    "file_count": 1,
    "size_bytes": 4,
    "artifacts": [{"path": "model.onnx", "size_bytes": 4, "sha256": "e" * 64}],
    "hash_contract": "studio-training-artifacts-v1",
    "symlink_policy": "excluded",
}


def _build(**overrides):
    args = {
        "model_source": "nvidia/GR00T-N1.6-3B",
        "model_revision": REVISION,
        "tether_commit": TETHER_COMMIT,
        "embodiment_id": 3,
        "export_identity": EXPORT_IDENTITY,
        "platform_system": "Darwin",
        "ort_providers": ["CPUExecutionProvider"],
        "requested_provider": "CPUExecutionProvider",
        "cpu_fallback_disabled": False,
        "input_seed": 42,
        "noise_seed": 99,
        "shared_input_sha256": INPUT_SHA,
        "reference_shape": [1, 50, 128],
        "export_shape": [1, 50, 128],
        "first_cosine": 0.999999,
        "first_max_abs": 1e-5,
        "full_cosine": 0.999999,
        "full_max_abs": 2e-5,
    }
    args.update(overrides)
    return build_groot_export_parity_receipt(**args)


def test_receipt_binds_embodiment_and_per_step_semantics():
    receipt = _build()
    assert receipt["family"] == "groot"
    assert receipt["kind"] == "groot-reference-shared-input-export-parity"
    assert receipt["subject_context"] == {
        "embodiment_id": 3,
        "export_semantics": "per-step-velocity",
        "denoise_loop": "external-runtime",
    }
    assert receipt["shared_input"]["num_steps"] == 1
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"


def test_linux_cuda_no_fallback_can_record_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        cpu_fallback_disabled=True,
    )
    assert receipt["external_acceptance"] == "recorded"


def test_stricter_groot_thresholds_are_enforced():
    receipt = _build(full_cosine=0.9995)
    assert receipt["thresholds"] == {
        "minimum_cosine": 0.9999,
        "maximum_absolute_error_exclusive": 1e-3,
    }
    assert receipt["verdict"] == "failed"
    assert receipt["external_acceptance"] == "not-run"


def test_embodiment_must_be_non_negative_integer():
    with pytest.raises(GrootExportParityError, match="embodiment_id"):
        _build(embodiment_id=-1)
    with pytest.raises(GrootExportParityError, match="embodiment_id"):
        _build(embodiment_id=True)


def test_wrong_artifact_kind_is_rejected():
    identity = dict(EXPORT_IDENTITY)
    identity["kind"] = "pi05-monolithic-onnx"
    with pytest.raises(GrootExportParityError, match="groot-monolithic-onnx"):
        _build(export_identity=identity)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_metrics_are_rejected(value):
    with pytest.raises(GrootExportParityError, match="finite number"):
        _build(full_max_abs=value)
