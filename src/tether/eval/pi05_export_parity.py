from __future__ import annotations

from typing import Any

from tether.eval.export_parity import (
    DEFAULT_MAX_ABS_ERROR,
    DEFAULT_MIN_COSINE,
    REFERENCE_EXPORT_PARITY_SCHEMA,
    ExportParityError,
    build_reference_export_parity_receipt,
)

PI05_EXPORT_PARITY_SCHEMA = REFERENCE_EXPORT_PARITY_SCHEMA
MARKER = "TETHER_PI05_EXPORT_PARITY_JSON="
MIN_COSINE = DEFAULT_MIN_COSINE
MAX_ABS_ERROR = DEFAULT_MAX_ABS_ERROR
Pi05ExportParityError = ExportParityError


def build_pi05_export_parity_receipt(
    *,
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
) -> dict[str, Any]:
    """Build the public pi0.5 export-parity prerequisite receipt for Studio M4 #52."""

    return build_reference_export_parity_receipt(
        family="pi05",
        receipt_kind="pi05-reference-shared-input-export-parity",
        artifact_kind="pi05-monolithic-onnx",
        model_source=model_source,
        model_revision=model_revision,
        tether_commit=tether_commit,
        export_identity=export_identity,
        platform_system=platform_system,
        ort_providers=ort_providers,
        requested_provider=requested_provider,
        cpu_fallback_disabled=cpu_fallback_disabled,
        input_seed=input_seed,
        noise_seed=noise_seed,
        num_steps=num_steps,
        shared_input_sha256=shared_input_sha256,
        reference_shape=reference_shape,
        export_shape=export_shape,
        first_cosine=first_cosine,
        first_max_abs=first_max_abs,
        full_cosine=full_cosine,
        full_max_abs=full_max_abs,
        minimum_cosine=MIN_COSINE,
        maximum_absolute_error=MAX_ABS_ERROR,
    )


__all__ = [
    "PI05_EXPORT_PARITY_SCHEMA",
    "MARKER",
    "MIN_COSINE",
    "MAX_ABS_ERROR",
    "Pi05ExportParityError",
    "build_pi05_export_parity_receipt",
]
