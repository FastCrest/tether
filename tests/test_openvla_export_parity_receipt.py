from __future__ import annotations

import math

import numpy as np
import pytest

from tether.eval.openvla_export_parity import (
    OpenVLAExportParityError,
    build_openvla_export_parity_receipt,
)


REVISION = "a" * 40
TETHER_COMMIT = "b" * 40
NORM_SHA = "c" * 64
INPUT_SHA = "d" * 64
EXPORT_IDENTITY = {
    "schema": 1,
    "kind": "openvla-optimum-onnx",
    "source": "optimum-openvla-export",
    "revision": TETHER_COMMIT,
    "artifact_sha256": "e" * 64,
    "file_count": 2,
    "size_bytes": 8,
    "artifacts": [
        {"path": "model.onnx", "size_bytes": 4, "sha256": "1" * 64},
        {"path": "model.onnx_data", "size_bytes": 4, "sha256": "2" * 64},
    ],
    "hash_contract": "studio-training-artifacts-v1",
    "symlink_policy": "excluded",
}
TOKENS = np.array([31999, 31998, 31997, 31996, 31995, 31994, 31993], dtype=np.int64)
ACTIONS = np.array([-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99], dtype=np.float32)


def _build(**overrides):
    args = {
        "model_source": "openvla/openvla-7b",
        "model_revision": REVISION,
        "tether_commit": TETHER_COMMIT,
        "export_identity": EXPORT_IDENTITY,
        "platform_system": "Darwin",
        "ort_providers": ["CPUExecutionProvider"],
        "requested_provider": "CPUExecutionProvider",
        "cpu_fallback_disabled": False,
        "action_dim": 7,
        "tokenizer_vocab_size": 32000,
        "padded_logit_vocab_size": 32064,
        "n_bin_edges": 256,
        "dataset_name": "bridge_orig",
        "norm_stats_sha256": NORM_SHA,
        "shared_input_sha256": INPUT_SHA,
        "reference_token_ids": TOKENS,
        "export_token_ids": TOKENS.copy(),
        "reference_actions": ACTIONS,
        "export_actions": ACTIONS.copy(),
    }
    args.update(overrides)
    return build_openvla_export_parity_receipt(**args)


def test_exact_tokens_and_actions_pass_but_local_is_not_external():
    receipt = _build()
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"
    assert receipt["metrics"]["exact_token_match"] is True
    assert receipt["metrics"]["action_max_abs"] == pytest.approx(0.0)
    assert receipt["decoder_contract"]["tokenizer_vocab_size"] == 32000
    assert receipt["decoder_contract"]["padded_logit_vocab_size"] == 32064
    assert receipt["decoder_contract"]["n_bin_centers"] == 255


def test_one_token_mismatch_fails_even_when_actions_match():
    changed = TOKENS.copy()
    changed[-1] -= 1
    receipt = _build(export_token_ids=changed)
    assert receipt["metrics"]["exact_token_match"] is False
    assert receipt["verdict"] == "failed"
    assert receipt["external_acceptance"] == "not-run"


def test_action_difference_fails_even_when_tokens_match():
    changed = ACTIONS.copy()
    changed[0] += 1e-3
    receipt = _build(export_actions=changed)
    assert receipt["metrics"]["exact_token_match"] is True
    assert receipt["metrics"]["action_max_abs"] > 1e-6
    assert receipt["verdict"] == "failed"


def test_linux_cuda_no_fallback_can_record_external_acceptance():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        cpu_fallback_disabled=True,
    )
    assert receipt["external_acceptance"] == "recorded"


def test_linux_cuda_with_cpu_fallback_allowed_is_not_external():
    receipt = _build(
        platform_system="Linux",
        ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        requested_provider="CUDAExecutionProvider",
        cpu_fallback_disabled=False,
    )
    assert receipt["verdict"] == "passed"
    assert receipt["external_acceptance"] == "not-run"


def test_artifact_must_contain_onnx_graph():
    identity = dict(EXPORT_IDENTITY)
    identity["artifacts"] = [{"path": "weights.bin", "size_bytes": 4, "sha256": "1" * 64}]
    with pytest.raises(OpenVLAExportParityError, match="ONNX graph"):
        _build(export_identity=identity)


def test_token_and_action_lengths_must_match_action_dim():
    with pytest.raises(OpenVLAExportParityError, match="exactly 7 token"):
        _build(export_token_ids=TOKENS[:-1])
    with pytest.raises(OpenVLAExportParityError, match="exactly 7 action"):
        _build(export_actions=ACTIONS[:-1])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_actions_are_rejected(value):
    changed = ACTIONS.astype(np.float64)
    changed[0] = value
    with pytest.raises(OpenVLAExportParityError, match="finite action"):
        _build(export_actions=changed)


def test_padded_vocab_cannot_be_smaller_than_tokenizer_vocab():
    with pytest.raises(OpenVLAExportParityError, match="cannot be smaller"):
        _build(padded_logit_vocab_size=31999)
