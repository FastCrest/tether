from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

OPENVLA_EXPORT_PARITY_SCHEMA = 1
MARKER = "TETHER_OPENVLA_EXPORT_PARITY_JSON="
MAX_ACTION_ABS_ERROR = 1e-6
SUPPORTED_REFERENCE_RUNTIME = {
    "transformers": "4.40.1",
    "tokenizers": "0.19.1",
    "timm": "0.9.10",
}
SUPPORTED_TORCH_PREFIX = "2.2.0"
_EXACT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OpenVLAExportParityError(ValueError):
    pass


def _exact_revision(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _EXACT_REVISION.fullmatch(normalized):
        raise OpenVLAExportParityError(f"{label} must be an exact 40-hex commit revision.")
    return normalized


def _sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise OpenVLAExportParityError(f"{label} must be a 64-hex SHA-256 digest.")
    return normalized


def _validate_artifact_identity(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise OpenVLAExportParityError("export_identity must use artifact identity schema 1.")
    if value.get("kind") != "openvla-optimum-onnx":
        raise OpenVLAExportParityError("export_identity kind must be openvla-optimum-onnx.")
    _sha256(str(value.get("artifact_sha256") or ""), label="export artifact identity")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise OpenVLAExportParityError("export_identity must retain hashed artifacts.")
    if not any(
        isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and row["path"].lower().endswith(".onnx")
        for row in artifacts
    ):
        raise OpenVLAExportParityError("export_identity must contain at least one ONNX graph.")
    return value


def _tokens(value: object, *, label: str, action_dim: int) -> list[int]:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.shape[0] != action_dim:
        raise OpenVLAExportParityError(f"{label} must contain exactly {action_dim} token IDs.")
    if not np.issubdtype(array.dtype, np.integer):
        raise OpenVLAExportParityError(f"{label} must contain integer token IDs.")
    return [int(item) for item in array.tolist()]


def _actions(value: object, *, label: str, action_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.shape[0] != action_dim:
        raise OpenVLAExportParityError(f"{label} must contain exactly {action_dim} action values.")
    if not np.isfinite(array).all():
        raise OpenVLAExportParityError(f"{label} must contain finite action values.")
    return array


def _runtime_versions(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise OpenVLAExportParityError("reference_runtime must be an object.")
    required = ("torch", "transformers", "tokenizers", "timm")
    result: dict[str, str] = {}
    for key in required:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise OpenVLAExportParityError(f"reference_runtime.{key} is required.")
        result[key] = item.strip()
    return result


def build_openvla_export_parity_receipt(
    *,
    model_source: str,
    model_revision: str,
    tether_commit: str,
    export_identity: dict[str, Any],
    reference_runtime: dict[str, str],
    platform_system: str,
    ort_providers: list[str],
    requested_provider: str,
    cpu_fallback_disabled: bool,
    action_dim: int,
    tokenizer_vocab_size: int,
    padded_logit_vocab_size: int,
    n_bin_edges: int,
    dataset_name: str,
    norm_stats_sha256: str,
    shared_input_sha256: str,
    reference_token_ids: object,
    export_token_ids: object,
    reference_actions: object,
    export_actions: object,
) -> dict[str, Any]:
    """Build the exact tokenized-action/export prerequisite receipt for Studio M4 #54."""

    source = model_source.strip()
    dataset_name = dataset_name.strip()
    if not source or not dataset_name:
        raise OpenVLAExportParityError("model_source and dataset_name are required.")
    model_revision = _exact_revision(model_revision, label="model_revision")
    tether_commit = _exact_revision(tether_commit, label="tether_commit")
    norm_stats_sha256 = _sha256(norm_stats_sha256, label="norm_stats_sha256")
    shared_input_sha256 = _sha256(shared_input_sha256, label="shared_input_sha256")
    export_identity = _validate_artifact_identity(export_identity)
    reference_runtime = _runtime_versions(reference_runtime)

    for label, value in (
        ("action_dim", action_dim),
        ("tokenizer_vocab_size", tokenizer_vocab_size),
        ("padded_logit_vocab_size", padded_logit_vocab_size),
        ("n_bin_edges", n_bin_edges),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OpenVLAExportParityError(f"{label} must be a positive integer.")
    if n_bin_edges < 2:
        raise OpenVLAExportParityError("n_bin_edges must be >= 2.")
    if padded_logit_vocab_size < tokenizer_vocab_size:
        raise OpenVLAExportParityError(
            "padded_logit_vocab_size cannot be smaller than tokenizer_vocab_size."
        )
    if not isinstance(cpu_fallback_disabled, bool):
        raise OpenVLAExportParityError("cpu_fallback_disabled must be boolean.")

    providers = [item for item in ort_providers if isinstance(item, str) and item]
    if not providers:
        raise OpenVLAExportParityError("At least one active ONNX Runtime provider is required.")
    if requested_provider not in providers:
        raise OpenVLAExportParityError("The requested ONNX Runtime provider was not active.")

    reference_tokens = _tokens(
        reference_token_ids,
        label="reference_token_ids",
        action_dim=action_dim,
    )
    export_tokens = _tokens(
        export_token_ids,
        label="export_token_ids",
        action_dim=action_dim,
    )
    for label, tokens in (
        ("reference_token_ids", reference_tokens),
        ("export_token_ids", export_tokens),
    ):
        if any(token < 0 or token >= tokenizer_vocab_size for token in tokens):
            raise OpenVLAExportParityError(
                f"{label} contains an ID outside tokenizer_vocab_size={tokenizer_vocab_size}."
            )

    reference_action_values = _actions(
        reference_actions,
        label="reference_actions",
        action_dim=action_dim,
    )
    export_action_values = _actions(
        export_actions,
        label="export_actions",
        action_dim=action_dim,
    )

    exact_token_match = reference_tokens == export_tokens
    action_abs_error = np.abs(reference_action_values - export_action_values)
    action_max_abs = float(np.max(action_abs_error))
    actions_match = math.isfinite(action_max_abs) and action_max_abs <= MAX_ACTION_ABS_ERROR
    passed = exact_token_match and actions_match
    reference_runtime_supported = all(
        reference_runtime[key] == expected
        for key, expected in SUPPORTED_REFERENCE_RUNTIME.items()
    ) and reference_runtime["torch"].startswith(SUPPORTED_TORCH_PREFIX)
    linux_cuda = (
        platform_system == "Linux"
        and requested_provider == "CUDAExecutionProvider"
        and cpu_fallback_disabled
    )

    return {
        "schema": OPENVLA_EXPORT_PARITY_SCHEMA,
        "kind": "openvla-tokenized-action-export-parity",
        "family": "openvla",
        "model": {"source": source, "revision": model_revision},
        "implementation": {"tether_commit": tether_commit},
        "export_identity": export_identity,
        "reference_runtime": {
            **reference_runtime,
            "supported": reference_runtime_supported,
            "required": {
                **SUPPORTED_REFERENCE_RUNTIME,
                "torch_prefix": SUPPORTED_TORCH_PREFIX,
            },
        },
        "decoder_contract": {
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "padded_logit_vocab_size": padded_logit_vocab_size,
            "n_bin_edges": n_bin_edges,
            "n_bin_centers": n_bin_edges - 1,
            "action_dim": action_dim,
            "dataset_name": dataset_name,
            "norm_stats_sha256": norm_stats_sha256,
            "mapping": "tokenizer-vocab-minus-token-minus-one -> clipped bin center -> q01/q99 unnormalize",
        },
        "execution": {
            "platform_system": platform_system,
            "requested_provider": requested_provider,
            "active_ort_providers": providers,
            "cpu_fallback_disabled": cpu_fallback_disabled,
            "scope": "linux-cuda" if linux_cuda else "local-or-noncuda",
        },
        "shared_input": {"sha256": shared_input_sha256},
        "reference": {
            "token_ids": reference_tokens,
            "actions": reference_action_values.astype(float).tolist(),
        },
        "export": {
            "token_ids": export_tokens,
            "actions": export_action_values.astype(float).tolist(),
        },
        "metrics": {
            "exact_token_match": exact_token_match,
            "action_max_abs": action_max_abs,
        },
        "thresholds": {
            "exact_token_match_required": True,
            "maximum_action_abs_error_inclusive": MAX_ACTION_ABS_ERROR,
        },
        "verdict": "passed" if passed else "failed",
        "external_acceptance": (
            "recorded" if passed and linux_cuda and reference_runtime_supported else "not-run"
        ),
        "limitations": [
            "This receipt proves the tokenized-action export path only for the exact hashed ONNX artifact, model revision, Tether commit, processor input, dataset norm stats and provider recorded here.",
            "OpenVLA reference acceptance is limited to its upstream-supported dependency stack; a numerically passing result on another Transformers stack is not external acceptance.",
            "It does not establish task success, training support, deployment readiness or physical safety.",
        ],
    }


__all__ = [
    "OPENVLA_EXPORT_PARITY_SCHEMA",
    "MARKER",
    "MAX_ACTION_ABS_ERROR",
    "SUPPORTED_REFERENCE_RUNTIME",
    "SUPPORTED_TORCH_PREFIX",
    "OpenVLAExportParityError",
    "build_openvla_export_parity_receipt",
]
