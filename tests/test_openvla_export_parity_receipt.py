from __future__ import annotations

import pytest

from tether.eval.openvla_export_parity import (
    MIN_ACTION_TOKEN_AGREEMENT,
    OpenVLAExportParityError,
    build_openvla_export_parity_receipt,
)

REVISION = "a" * 40
TETHER_COMMIT = "b" * 40
INPUT_SHA = "c" * 64
EXPORT_IDENTITY = {
    "schema": 1,
    "kind": "openvla-onnx",
    "source": "openvla-optimum-onnx-export",
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
        "model_source": "openvla/openvla-7b",
        "model_revision": REVISION,
        "tether_commit": TETHER_COMMIT,
        "export_identity": EXPORT_IDENTITY,
        "platform_system": "Linux",
        "ort_providers": ["CUDAExecutionProvider"],
        "requested_provider": "CUDAExecutionProvider",
        "cpu_fallback_disabled": True,
        "input_seed": 42,
        "shared_input_sha256": INPUT_SHA,
        "reference_shape": [1, 7],
        "export_shape": [1, 7],
        "action_dim": 7,
        "vocab_size": 32000,
        "n_action_bins": 256,
        "dataset_name": None,
        "matching_action_tokens": 7,
        "total_action_tokens": 7,
        "first_cosine": 1.0,
        "first_max_abs": 0.0,
        "full_cosine": 1.0,
        "full_max_abs": 0.0,
    }
    args.update(overrides)
    return build_openvla_export_parity_receipt(**args)


def test_linux_cuda_run_without_cpu_fallback_is_recorded():
    receipt = _build()
    assert receipt["family"] == "openvla"
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "recorded"
    assert receipt["shared_input"]["num_steps"] == 1
    assert receipt["subject_context"]["vocab_size"] == 32000
    assert receipt["subject_context"]["action_decode"] == "argmax-bin-centers"
    assert receipt["metrics"]["action_token_agreement"] == 1.0
    assert receipt["thresholds"]["minimum_action_token_agreement"] == MIN_ACTION_TOKEN_AGREEMENT


def test_a_single_disagreeing_action_token_fails_despite_perfect_cosine():
    """The whole point of the extra gate: one wrong argmax is a whole-bin error
    that a seven-dimension cosine can sit above 0.999 through."""
    receipt = _build(matching_action_tokens=6, first_cosine=1.0, full_cosine=1.0)
    assert receipt["metrics"]["action_token_agreement"] == pytest.approx(6 / 7)
    assert receipt["verdict"] == "failed"
    assert receipt["external_acceptance"] == "not-run"


def test_shared_cosine_threshold_is_not_relaxed():
    receipt = _build(full_cosine=0.998)
    assert receipt["verdict"] == "failed"
    assert receipt["thresholds"]["minimum_cosine"] == 0.999


def test_cpu_run_is_never_externally_accepted():
    receipt = _build(
        platform_system="Darwin",
        ort_providers=["CPUExecutionProvider"],
        requested_provider="CPUExecutionProvider",
        cpu_fallback_disabled=False,
    )
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"


def test_decode_parameters_are_bound_into_the_receipt():
    receipt = _build(dataset_name="bridge_orig", vocab_size=32000, n_action_bins=256)
    assert receipt["subject_context"]["dataset_name"] == "bridge_orig"
    assert receipt["subject_context"]["n_action_bins"] == 256


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_dim": 0},
        {"vocab_size": 0},
        {"n_action_bins": 1},
        {"n_action_bins": 40000},
        {"total_action_tokens": 0},
        {"matching_action_tokens": 8},
        {"total_action_tokens": 8},
        {"dataset_name": "  "},
        {"export_identity": {**EXPORT_IDENTITY, "kind": "pi0-monolithic-onnx"}},
        {"full_cosine": float("nan")},
        {"model_revision": "main"},
    ],
)
def test_rejects_unusable_evidence(overrides):
    with pytest.raises(OpenVLAExportParityError):
        _build(**overrides)
