from __future__ import annotations

from typing import Any

from tether.eval.export_parity import (
    REFERENCE_EXPORT_PARITY_SCHEMA,
    ExportParityError,
    build_reference_export_parity_receipt,
)

GROOT_EXPORT_PARITY_SCHEMA = REFERENCE_EXPORT_PARITY_SCHEMA
MARKER = "TETHER_GROOT_EXPORT_PARITY_JSON="
MIN_COSINE = 0.9999
MAX_ABS_ERROR = 1e-3
GrootExportParityError = ExportParityError


def build_groot_export_parity_receipt(
    *,
    model_source: str,
    model_revision: str,
    tether_commit: str,
    embodiment_id: int,
    export_identity: dict[str, Any],
    platform_system: str,
    ort_providers: list[str],
    requested_provider: str,
    cpu_fallback_disabled: bool,
    input_seed: int,
    noise_seed: int,
    shared_input_sha256: str,
    reference_shape: list[int],
    export_shape: list[int],
    first_cosine: float,
    first_max_abs: float,
    full_cosine: float,
    full_max_abs: float,
) -> dict[str, Any]:
    """Build the public GR00T per-step export-parity receipt for Studio M4 #53."""

    if isinstance(embodiment_id, bool) or not isinstance(embodiment_id, int):
        raise GrootExportParityError("embodiment_id must be an integer.")
    if embodiment_id < 0:
        raise GrootExportParityError("embodiment_id must be non-negative.")

    return build_reference_export_parity_receipt(
        family="groot",
        receipt_kind="groot-reference-shared-input-export-parity",
        artifact_kind="groot-monolithic-onnx",
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
        num_steps=1,
        shared_input_sha256=shared_input_sha256,
        reference_shape=reference_shape,
        export_shape=export_shape,
        first_cosine=first_cosine,
        first_max_abs=first_max_abs,
        full_cosine=full_cosine,
        full_max_abs=full_max_abs,
        minimum_cosine=MIN_COSINE,
        maximum_absolute_error=MAX_ABS_ERROR,
        subject_context={
            "embodiment_id": embodiment_id,
            "export_semantics": "per-step-velocity",
            "denoise_loop": "external-runtime",
        },
    )


__all__ = [
    "GROOT_EXPORT_PARITY_SCHEMA",
    "MARKER",
    "MIN_COSINE",
    "MAX_ABS_ERROR",
    "GrootExportParityError",
    "build_groot_export_parity_receipt",
]
