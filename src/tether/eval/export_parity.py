from __future__ import annotations

import json
import math
import re
from typing import Any

REFERENCE_EXPORT_PARITY_SCHEMA = 1
DEFAULT_MIN_COSINE = 0.999
DEFAULT_MAX_ABS_ERROR = 0.1
_EXACT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExportParityError(ValueError):
    pass


def _exact_revision(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _EXACT_REVISION.fullmatch(normalized):
        raise ExportParityError(f"{label} must be an exact 40-hex commit revision.")
    return normalized


def _sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ExportParityError(f"{label} must be a 64-hex SHA-256 digest.")
    return normalized


def _metric(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportParityError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ExportParityError(f"{label} must be a finite number.")
    return result


def _positive_metric(value: object, *, label: str) -> float:
    result = _metric(value, label=label)
    if result <= 0:
        raise ExportParityError(f"{label} must be positive.")
    return result


def _validate_artifact_identity(
    value: dict[str, Any],
    *,
    artifact_kind: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportParityError("export_identity must be an artifact-identity object.")
    if value.get("schema") != 1:
        raise ExportParityError("export_identity must use artifact identity schema 1.")
    if value.get("kind") != artifact_kind:
        raise ExportParityError(f"export_identity kind must be {artifact_kind}.")
    _sha256(str(value.get("artifact_sha256") or ""), label="export artifact identity")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ExportParityError("export_identity must retain at least one hashed artifact.")
    if not any(row.get("path") == "model.onnx" for row in artifacts if isinstance(row, dict)):
        raise ExportParityError("export_identity must include model.onnx.")
    return value


def _normalize_subject_context(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExportParityError("subject_context must be an object when provided.")
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ExportParityError("subject_context must contain finite JSON values.") from exc
    return normalized


def build_reference_export_parity_receipt(
    *,
    family: str,
    receipt_kind: str,
    artifact_kind: str,
    model_source: str,
    model_revision: str,
    tether_commit: str,
    export_identity: dict[str, Any],
    platform_system: str,
    ort_providers: list[str],
    requested_provider: str,
    cpu_fallback_disabled: bool,
    input_seed: int,
    noise_seed: int,
    num_steps: int,
    shared_input_sha256: str,
    reference_shape: list[int],
    export_shape: list[int],
    first_cosine: float,
    first_max_abs: float,
    full_cosine: float,
    full_max_abs: float,
    minimum_cosine: float = DEFAULT_MIN_COSINE,
    maximum_absolute_error: float = DEFAULT_MAX_ABS_ERROR,
    subject_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an exact-reference/shared-input export parity receipt.

    A passing local or CPU run remains useful evidence, but external acceptance
    is recorded only for a Linux CUDA run that explicitly disables CPU EP
    fallback. Callers supply family-specific artifact and receipt kinds while
    sharing one evidence/threshold contract.
    """

    family = family.strip()
    receipt_kind = receipt_kind.strip()
    artifact_kind = artifact_kind.strip()
    source = model_source.strip()
    if not family or not receipt_kind or not artifact_kind or not source:
        raise ExportParityError(
            "family, receipt_kind, artifact_kind and model_source are required."
        )

    model_revision = _exact_revision(model_revision, label="model_revision")
    tether_commit = _exact_revision(tether_commit, label="tether_commit")
    shared_input_sha256 = _sha256(shared_input_sha256, label="shared_input_sha256")
    export_identity = _validate_artifact_identity(export_identity, artifact_kind=artifact_kind)
    minimum_cosine = _positive_metric(minimum_cosine, label="minimum_cosine")
    maximum_absolute_error = _positive_metric(
        maximum_absolute_error,
        label="maximum_absolute_error",
    )
    subject_context = _normalize_subject_context(subject_context)

    if minimum_cosine > 1.0:
        raise ExportParityError("minimum_cosine cannot exceed 1.0.")
    if not isinstance(cpu_fallback_disabled, bool):
        raise ExportParityError("cpu_fallback_disabled must be boolean.")
    if isinstance(input_seed, bool) or not isinstance(input_seed, int):
        raise ExportParityError("input_seed must be an integer.")
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
        raise ExportParityError("noise_seed must be an integer.")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ExportParityError("num_steps must be a positive integer.")
    if (
        not reference_shape
        or not export_shape
        or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in [*reference_shape, *export_shape]
        )
    ):
        raise ExportParityError("reference_shape and export_shape must contain positive integers.")

    providers = [item for item in ort_providers if isinstance(item, str) and item]
    if not providers:
        raise ExportParityError("At least one active ONNX Runtime provider is required.")
    if requested_provider not in providers:
        raise ExportParityError("The requested ONNX Runtime provider was not active.")

    first_cosine = _metric(first_cosine, label="first_cosine")
    first_max_abs = _metric(first_max_abs, label="first_max_abs")
    full_cosine = _metric(full_cosine, label="full_cosine")
    full_max_abs = _metric(full_max_abs, label="full_max_abs")

    shapes_match = reference_shape == export_shape
    passed = (
        shapes_match
        and first_cosine >= minimum_cosine
        and full_cosine >= minimum_cosine
        and first_max_abs < maximum_absolute_error
        and full_max_abs < maximum_absolute_error
    )
    linux_cuda = (
        platform_system == "Linux"
        and requested_provider == "CUDAExecutionProvider"
        and cpu_fallback_disabled
    )

    return {
        "schema": REFERENCE_EXPORT_PARITY_SCHEMA,
        "kind": receipt_kind,
        "family": family,
        "model": {"source": source, "revision": model_revision},
        "implementation": {"tether_commit": tether_commit},
        "subject_context": subject_context,
        "export_identity": export_identity,
        "execution": {
            "platform_system": platform_system,
            "requested_provider": requested_provider,
            "active_ort_providers": providers,
            "cpu_fallback_disabled": cpu_fallback_disabled,
            "scope": "linux-cuda" if linux_cuda else "local-or-noncuda",
        },
        "shared_input": {
            "input_seed": input_seed,
            "noise_seed": noise_seed,
            "num_steps": num_steps,
            "sha256": shared_input_sha256,
            "reference_shape": list(reference_shape),
            "export_shape": list(export_shape),
        },
        "thresholds": {
            "minimum_cosine": minimum_cosine,
            "maximum_absolute_error_exclusive": maximum_absolute_error,
        },
        "metrics": {
            "first_cosine": first_cosine,
            "first_max_abs": first_max_abs,
            "full_cosine": full_cosine,
            "full_max_abs": full_max_abs,
        },
        "verdict": "passed" if passed else "failed",
        "external_acceptance": "recorded" if passed and linux_cuda else "not-run",
        "limitations": [
            "This receipt proves parity only for the exact hashed ONNX artifact, exact model revision, exact Tether commit, provider and shared input recorded here.",
            "It does not establish task success, training support, hardware deployment readiness or physical safety.",
        ],
    }


__all__ = [
    "REFERENCE_EXPORT_PARITY_SCHEMA",
    "DEFAULT_MIN_COSINE",
    "DEFAULT_MAX_ABS_ERROR",
    "ExportParityError",
    "build_reference_export_parity_receipt",
]
