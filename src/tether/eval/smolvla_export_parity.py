from __future__ import annotations

import math
from typing import Any

import numpy as np

from tether.eval.export_parity import (
    DEFAULT_MAX_ABS_ERROR,
    DEFAULT_MIN_COSINE,
    REFERENCE_EXPORT_PARITY_SCHEMA,
    ExportParityError,
    build_reference_export_parity_receipt,
)

SMOLVLA_EXPORT_PARITY_SCHEMA = REFERENCE_EXPORT_PARITY_SCHEMA
MARKER = "TETHER_SMOLVLA_EXPORT_PARITY_JSON="
MIN_COSINE = DEFAULT_MIN_COSINE
MAX_ABS_ERROR = DEFAULT_MAX_ABS_ERROR
SmolVLAExportParityError = ExportParityError

_DISTRIBUTION_KEYS = ("mean", "p50", "p95", "p99", "max")


def abs_error_distribution(
    reference: np.ndarray, export: np.ndarray, *, tolerance: float = MAX_ABS_ERROR
) -> dict[str, float | int]:
    """Element-wise |reference - export| summary: mean, p50/p95/p99, max and the
    fraction of elements at or above ``tolerance`` (the receipt's exclusive bar)."""

    if reference.shape != export.shape:
        raise SmolVLAExportParityError(
            f"reference shape {list(reference.shape)} != export shape {list(export.shape)}."
        )
    error = np.abs(reference.astype(np.float64) - export.astype(np.float64)).reshape(-1)
    if error.size == 0 or not np.all(np.isfinite(error)):
        raise SmolVLAExportParityError("absolute errors must be finite and non-empty.")
    p50, p95, p99 = np.percentile(error, [50, 95, 99])
    return {
        "elements": int(error.size),
        "mean": float(error.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(error.max()),
        "tolerance": float(tolerance),
        "frac_at_or_above_tolerance": float(np.mean(error >= tolerance)),
    }


def _validate_distribution(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmolVLAExportParityError(f"{label} must be an object.")
    for key in (*_DISTRIBUTION_KEYS, "tolerance", "frac_at_or_above_tolerance"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise SmolVLAExportParityError(f"{label}.{key} must be a finite number.")
    elements = value.get("elements")
    if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
        raise SmolVLAExportParityError(f"{label}.elements must be a positive integer.")
    ordered = [float(value[key]) for key in ("p50", "p95", "p99", "max")]
    if ordered != sorted(ordered) or float(value["mean"]) > float(value["max"]):
        raise SmolVLAExportParityError(f"{label} percentiles must be ordered up to max.")
    if not 0.0 <= float(value["frac_at_or_above_tolerance"]) <= 1.0:
        raise SmolVLAExportParityError(f"{label}.frac_at_or_above_tolerance must be in [0, 1].")
    return dict(value)


def build_smolvla_export_parity_receipt(
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
    error_distribution: dict[str, Any],
    cases: list[dict[str, Any]],
    subject_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public SmolVLA export-parity receipt Studio's SmolVLA family binds.

    ``first_*``/``full_*`` are the worst values over every case, so the shared
    gate fails if any case misses the bar. ``error_distribution`` is pooled over
    every element of every case's full action chunk and must agree with
    ``full_max_abs``; ``cases`` keeps each case's own metrics.
    """

    distribution = _validate_distribution(error_distribution, label="error_distribution")
    if not math.isclose(distribution["max"], float(full_max_abs), rel_tol=1e-6, abs_tol=1e-9):
        raise SmolVLAExportParityError(
            "error_distribution.max must equal full_max_abs (both are the pooled worst case)."
        )
    if not isinstance(cases, list) or not cases or not all(isinstance(c, dict) for c in cases):
        raise SmolVLAExportParityError("cases must be a non-empty list of objects.")

    receipt = build_reference_export_parity_receipt(
        family="smolvla",
        receipt_kind="smolvla-reference-shared-input-export-parity",
        artifact_kind="smolvla-monolithic-onnx",
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
        subject_context={
            "exporter": "tether-smolvla-monolithic",
            "export_semantics": "sample_actions-euler-baked",
            "metric_aggregation": "worst-case-over-cases",
            **(subject_context or {}),
        },
    )
    receipt["metrics"]["error_distribution"] = distribution
    receipt["metrics"]["cases"] = cases
    receipt["limitations"].append(
        "first_* and full_* are the worst case over every listed case; the error "
        "distribution pools every element of every case's padded 32-dim action chunk."
    )
    return receipt


__all__ = [
    "MARKER",
    "MAX_ABS_ERROR",
    "MIN_COSINE",
    "SMOLVLA_EXPORT_PARITY_SCHEMA",
    "SmolVLAExportParityError",
    "abs_error_distribution",
    "build_smolvla_export_parity_receipt",
]
