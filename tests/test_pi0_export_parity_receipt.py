from __future__ import annotations

import math

import pytest

from tether.eval.pi0_export_parity import (
    Pi0ExportParityError,
    build_pi0_export_parity_receipt,
)


REVISION = "a" * 40
TETHER_COMMIT = "b" * 40
INPUT_SHA = "c" * 64
EXPORT_IDENTITY = {
    "schema": 1,
    "kind": "pi0-monolithic-onnx",
    "source": "pi0-monolithic-export",
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
        "model_source": "lerobot/pi0_base",
        "model_revision": REVISION,
        "tether_commit": TETHER_COMMIT,
        "export_identity": EXPORT_IDENTITY,
        "platform_system": "Darwin",
        "ort_providers": ["CPUExecutionProvider"],
        "requested_provider": "CPUExecutionProvider",
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
    return build_pi0_export_parity_receipt(**args)


def test_passing_local_receipt_does_not_claim_external_acceptance():
    receipt = _build()
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"
    assert receipt["execution"]["scope"] == "local-or-noncuda"


def test_linux_cuda_passing_receipt_records_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
    )
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "recorded"
    assert receipt["execution"]["scope"] == "linux-cuda"


def test_failure_never_records_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        full_cosine=0.98,
    )
    assert receipt["verdict"] == "failed"
    assert receipt["external_acceptance"] == "not-run"


def test_shape_mismatch_fails_even_when_metrics_pass():
    receipt = _build(export_shape=[1, 49, 32])
    assert receipt["verdict"] == "failed"


def test_requested_provider_must_be_active():
    with pytest.raises(Pi0ExportParityError, match="provider was not active"):
        _build(requested_provider="CUDAExecutionProvider")


@pytest.mark.parametrize("value", ["main", "abc", "g" * 40, "a" * 39])
def test_model_revision_must_be_exact_commit(value):
    with pytest.raises(Pi0ExportParityError, match="model_revision"):
        _build(model_revision=value)


def test_export_identity_must_include_model_onnx():
    identity = dict(EXPORT_IDENTITY)
    identity["artifacts"] = [{"path": "weights.data", "size_bytes": 4, "sha256": "e" * 64}]
    with pytest.raises(Pi0ExportParityError, match="model.onnx"):
        _build(export_identity=identity)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_metrics_are_rejected(value):
    with pytest.raises(Pi0ExportParityError, match="finite number"):
        _build(full_max_abs=value)
