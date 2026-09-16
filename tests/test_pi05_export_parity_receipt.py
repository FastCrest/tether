from __future__ import annotations

import math

import pytest

from tether.eval.pi05_export_parity import (
    Pi05ExportParityError,
    build_pi05_export_parity_receipt,
)


REVISION = "1" * 40
TETHER_COMMIT = "2" * 40
INPUT_SHA = "3" * 64
EXPORT_IDENTITY = {
    "schema": 1,
    "kind": "pi05-monolithic-onnx",
    "source": "pi05-monolithic-export",
    "revision": TETHER_COMMIT,
    "artifact_sha256": "4" * 64,
    "file_count": 1,
    "size_bytes": 4,
    "artifacts": [{"path": "model.onnx", "size_bytes": 4, "sha256": "5" * 64}],
    "hash_contract": "studio-training-artifacts-v1",
    "symlink_policy": "excluded",
}


def _build(**overrides):
    args = {
        "model_source": "lerobot/pi05_base",
        "model_revision": REVISION,
        "tether_commit": TETHER_COMMIT,
        "export_identity": EXPORT_IDENTITY,
        "platform_system": "Darwin",
        "ort_providers": ["CPUExecutionProvider"],
        "requested_provider": "CPUExecutionProvider",
        "cpu_fallback_disabled": False,
        "input_seed": 42,
        "noise_seed": 99,
        "num_steps": 10,
        "shared_input_sha256": INPUT_SHA,
        "reference_shape": [1, 50, 32],
        "export_shape": [1, 50, 32],
        "first_cosine": 0.99999,
        "first_max_abs": 0.001,
        "full_cosine": 0.9999,
        "full_max_abs": 0.002,
    }
    args.update(overrides)
    return build_pi05_export_parity_receipt(**args)


def test_receipt_is_pi05_scoped_and_local_pass_is_not_external():
    receipt = _build()
    assert receipt["family"] == "pi05"
    assert receipt["kind"] == "pi05-reference-shared-input-export-parity"
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"


def test_passing_linux_cuda_no_fallback_records_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        cpu_fallback_disabled=True,
    )
    assert receipt["external_acceptance"] == "recorded"
    assert receipt["execution"]["scope"] == "linux-cuda"


def test_cuda_with_fallback_allowed_does_not_record_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        cpu_fallback_disabled=False,
    )
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"


def test_wrong_artifact_kind_is_rejected():
    identity = dict(EXPORT_IDENTITY)
    identity["kind"] = "pi0-monolithic-onnx"
    with pytest.raises(Pi05ExportParityError, match="pi05-monolithic-onnx"):
        _build(export_identity=identity)


def test_shape_or_metric_failure_cannot_be_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        cpu_fallback_disabled=True,
        export_shape=[1, 49, 32],
    )
    assert receipt["verdict"] == "failed"
    assert receipt["external_acceptance"] == "not-run"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_metrics_are_rejected(value):
    with pytest.raises(Pi05ExportParityError, match="finite number"):
        _build(first_cosine=value)
