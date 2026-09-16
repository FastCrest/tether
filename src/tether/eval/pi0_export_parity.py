from __future__ import annotations

import math
import re
from typing import Any

PI0_EXPORT_PARITY_SCHEMA = 1
MARKER = "TETHER_PI0_EXPORT_PARITY_JSON="
MIN_COSINE = 0.999
MAX_ABS_ERROR = 0.1
_EXACT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Pi0ExportParityError(ValueError):
    pass


def _exact_revision(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _EXACT_REVISION.fullmatch(normalized):
        raise Pi0ExportParityError(f"{label} must be an exact 40-hex commit revision.")
    return normalized


def _sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise Pi0ExportParityError(f"{label} must be a 64-hex SHA-256 digest.")
    return normalized


def _metric(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Pi0ExportParityError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise Pi0ExportParityError(f"{label} must be a finite number.")
    return result


def _validate_artifact_identity(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Pi0ExportParityError("export_identity must be an artifact-identity object.")
    if value.get("schema") != 1:
        raise Pi0ExportParityError("export_identity must use artifact identity schema 1.")
    if value.get("kind") != "pi0-monolithic-onnx":
        raise Pi0ExportParityError("export_identity kind must be pi0-monolithic-onnx.")
    _sha256(str(value.get("artifact_sha256") or ""), label="export artifact identity")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Pi0ExportParityError("export_identity must retain at least one hashed artifact.")
    if not any(row.get("path") == "model.onnx" for row in artifacts if isinstance(row, dict)):
        raise Pi0ExportParityError("export_identity must include model.onnx.")
    return value


def build_pi0_export_parity_receipt(
    *,
    model_source: str,
    model_revision: str,
    tether_commit: str,
    export_identity: dict[str, Any],
    platform_system: str,
    ort_providers: list[str],
    requested_provider: str,
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
) -> dict[str, Any]:
    """Build the mechanically-bound public prerequisite receipt for Studio M4 #51.

    This contract intentionally separates code/harness availability from external
    Linux-CUDA acceptance. A passing CPU or macOS run remains useful evidence but
    cannot become the external acceptance required by the Studio master spec.
    """

    source = model_source.strip()
    if not source:
        raise Pi0ExportParityError("model_source is required.")
    model_revision = _exact_revision(model_revision, label="model_revision")
    tether_commit = _exact_revision(tether_commit, label="tether_commit")
    shared_input_sha256 = _sha256(shared_input_sha256, label="shared_input_sha256")
    export_identity = _validate_artifact_identity(export_identity)

    if isinstance(input_seed, bool) or not isinstance(input_seed, int):
        raise Pi0ExportParityError("input_seed must be an integer.")
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
        raise Pi0ExportParityError("noise_seed must be an integer.")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise Pi0ExportParityError("num_steps must be a positive integer.")
    if not reference_shape or not export_shape or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
        for dim in [*reference_shape, *export_shape]
    ):
        raise Pi0ExportParityError("reference_shape and export_shape must contain positive integers.")

    providers = [item for item in ort_providers if isinstance(item, str) and item]
    if not providers:
        raise Pi0ExportParityError("At least one active ONNX Runtime provider is required.")
    if requested_provider not in providers:
        raise Pi0ExportParityError("The requested ONNX Runtime provider was not active.")

    first_cosine = _metric(first_cosine, label="first_cosine")
    first_max_abs = _metric(first_max_abs, label="first_max_abs")
    full_cosine = _metric(full_cosine, label="full_cosine")
    full_max_abs = _metric(full_max_abs, label="full_max_abs")

    shapes_match = reference_shape == export_shape
    passed = (
        shapes_match
        and first_cosine >= MIN_COSINE
        and full_cosine >= MIN_COSINE
        and first_max_abs < MAX_ABS_ERROR
        and full_max_abs < MAX_ABS_ERROR
    )
    linux_cuda = platform_system == "Linux" and requested_provider == "CUDAExecutionProvider"
    external_acceptance = "recorded" if passed and linux_cuda else "not-run"

    return {
        "schema": PI0_EXPORT_PARITY_SCHEMA,
        "kind": "pi0-reference-shared-input-export-parity",
        "family": "pi0",
        "model": {"source": source, "revision": model_revision},
        "implementation": {"tether_commit": tether_commit},
        "export_identity": export_identity,
        "execution": {
            "platform_system": platform_system,
            "requested_provider": requested_provider,
            "active_ort_providers": providers,
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
            "minimum_cosine": MIN_COSINE,
            "maximum_absolute_error_exclusive": MAX_ABS_ERROR,
        },
        "metrics": {
            "first_cosine": first_cosine,
            "first_max_abs": first_max_abs,
            "full_cosine": full_cosine,
            "full_max_abs": full_max_abs,
        },
        "verdict": "passed" if passed else "failed",
        "external_acceptance": external_acceptance,
        "limitations": [
            "This receipt proves parity only for the exact hashed ONNX artifact, exact model revision, exact Tether commit, provider and shared input recorded here.",
            "It does not establish task success, training support, hardware deployment readiness or physical safety.",
        ],
    }


__all__ = [
    "PI0_EXPORT_PARITY_SCHEMA",
    "MARKER",
    "MIN_COSINE",
    "MAX_ABS_ERROR",
    "Pi0ExportParityError",
    "build_pi0_export_parity_receipt",
]
