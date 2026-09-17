from __future__ import annotations

from typing import Any

from tether.eval.export_parity import (
    DEFAULT_MAX_ABS_ERROR,
    DEFAULT_MIN_COSINE,
    REFERENCE_EXPORT_PARITY_SCHEMA,
    ExportParityError,
    build_reference_export_parity_receipt,
)

OPENVLA_EXPORT_PARITY_SCHEMA = REFERENCE_EXPORT_PARITY_SCHEMA
MARKER = "TETHER_OPENVLA_EXPORT_PARITY_JSON="
MIN_COSINE = DEFAULT_MIN_COSINE
MAX_ABS_ERROR = DEFAULT_MAX_ABS_ERROR
# OpenVLA's action head is `argmax` over the top `n_action_bins` vocab tokens.
# One disagreeing token is a whole-bin jump, so continuous-action cosine is not
# on its own sufficient evidence that the decoder survived export.
MIN_ACTION_TOKEN_AGREEMENT = 1.0
OpenVLAExportParityError = ExportParityError


def _positive_int(value: object, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenVLAExportParityError(f"{label} must be an integer.")
    if value < minimum:
        raise OpenVLAExportParityError(f"{label} must be at least {minimum}.")
    return value


def build_openvla_export_parity_receipt(
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
    shared_input_sha256: str,
    reference_shape: list[int],
    export_shape: list[int],
    action_dim: int,
    vocab_size: int,
    n_action_bins: int,
    dataset_name: str | None,
    matching_action_tokens: int,
    total_action_tokens: int,
    first_cosine: float,
    first_max_abs: float,
    full_cosine: float,
    full_max_abs: float,
) -> dict[str, Any]:
    """Build the public OpenVLA export-parity prerequisite receipt for Studio M4 #54.

    OpenVLA is not on the flow-matching spine: it ships as a shim over
    ``optimum-cli export onnx`` plus :mod:`tether.postprocess.openvla`. The
    subject of this receipt is therefore the pair (exported LM graph, tokenized
    action decoder), and it gates on exact action-token agreement in addition to
    the shared continuous-action cosine and max-absolute-error thresholds.
    """

    action_dim = _positive_int(action_dim, label="action_dim")
    vocab_size = _positive_int(vocab_size, label="vocab_size")
    n_action_bins = _positive_int(n_action_bins, label="n_action_bins", minimum=2)
    total_action_tokens = _positive_int(total_action_tokens, label="total_action_tokens")
    matching_action_tokens = _positive_int(
        matching_action_tokens,
        label="matching_action_tokens",
        minimum=0,
    )
    if matching_action_tokens > total_action_tokens:
        raise OpenVLAExportParityError("matching_action_tokens cannot exceed total_action_tokens.")
    if total_action_tokens % action_dim:
        raise OpenVLAExportParityError(
            "total_action_tokens must be a whole number of action vectors."
        )
    if n_action_bins > vocab_size:
        raise OpenVLAExportParityError("n_action_bins cannot exceed vocab_size.")
    if dataset_name is not None and (not isinstance(dataset_name, str) or not dataset_name.strip()):
        raise OpenVLAExportParityError("dataset_name must be a non-empty string or None.")

    receipt = build_reference_export_parity_receipt(
        family="openvla",
        receipt_kind="openvla-reference-shared-input-export-parity",
        artifact_kind="openvla-onnx",
        model_source=model_source,
        model_revision=model_revision,
        tether_commit=tether_commit,
        export_identity=export_identity,
        platform_system=platform_system,
        ort_providers=ort_providers,
        requested_provider=requested_provider,
        cpu_fallback_disabled=cpu_fallback_disabled,
        input_seed=input_seed,
        # OpenVLA is autoregressive: there is no diffusion noise and no Euler
        # loop, so the noise seed is fixed and num_steps is always 1.
        noise_seed=0,
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
            "action_dim": action_dim,
            "vocab_size": vocab_size,
            "n_action_bins": n_action_bins,
            "dataset_name": dataset_name,
            "action_decode": "argmax-bin-centers",
            "export_semantics": "single-step-autoregressive",
            "exporter": "optimum-cli-onnx",
        },
    )

    agreement = matching_action_tokens / total_action_tokens
    receipt["thresholds"]["minimum_action_token_agreement"] = MIN_ACTION_TOKEN_AGREEMENT
    receipt["metrics"]["action_token_agreement"] = agreement
    receipt["metrics"]["matching_action_tokens"] = matching_action_tokens
    receipt["metrics"]["total_action_tokens"] = total_action_tokens
    if agreement < MIN_ACTION_TOKEN_AGREEMENT:
        receipt["verdict"] = "failed"
        receipt["external_acceptance"] = "not-run"
    receipt["limitations"].append(
        "OpenVLA decodes actions by argmax over vocabulary bins, so a single "
        "disagreeing action token is a whole-bin error. This receipt requires "
        "exact action-token agreement in addition to the continuous-action "
        "thresholds; continuous cosine alone would not establish decoder parity."
    )
    return receipt


__all__ = [
    "OPENVLA_EXPORT_PARITY_SCHEMA",
    "MARKER",
    "MIN_COSINE",
    "MAX_ABS_ERROR",
    "MIN_ACTION_TOKEN_AGREEMENT",
    "OpenVLAExportParityError",
    "build_openvla_export_parity_receipt",
]
