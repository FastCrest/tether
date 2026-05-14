"""Validation harness for embodiment configs.

Two layers:
1. JSON schema validation (jsonschema lib, draft-07) — types, ranges, enums
2. Cross-field validation (Python) — array lengths must match, gripper index
   must be inside action_space, normalization sizes must match action_dim

The cross-field rules can't live in JSON schema cleanly so they're explicit
Python checks with named error slugs (matches the TECHNICAL_PLAN convention).

Usage:
    from reflex.embodiments import EmbodimentConfig
    from reflex.embodiments.validate import validate_embodiment_config

    cfg = EmbodimentConfig.load_preset("franka")
    ok, errors = validate_embodiment_config(cfg)
    if not ok:
        for e in errors:
            print(f"{e['severity']}: {e['slug']}: {e['message']}")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from . import EmbodimentConfig, get_schema_path


class ValidationError(TypedDict):
    """One validation error — keep stable; downstream tools may parse this."""

    slug: str  # short id like "action-dim-mismatch" — for error registry lookup
    severity: str  # "error" | "warn"
    field: str  # dotted path into config, e.g. "normalization.mean_action"
    message: str  # human-readable explanation


def validate_against_schema(cfg_dict: dict[str, Any]) -> list[ValidationError]:
    """Layer 1: validate against the draft-07 JSON schema. Type errors,
    enum violations, missing required fields, range violations.

    Returns a list of ValidationError dicts (empty if cfg passes)."""
    try:
        import jsonschema
    except ImportError:
        return [
            {
                "slug": "jsonschema-not-installed",
                "severity": "error",
                "field": "",
                "message": "jsonschema not installed; pip install jsonschema",
            }
        ]

    with get_schema_path().open() as f:
        schema = json.load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors: list[ValidationError] = []
    for err in validator.iter_errors(cfg_dict):
        errors.append(
            {
                "slug": "schema-violation",
                "severity": "error",
                "field": ".".join(str(p) for p in err.absolute_path),
                "message": err.message,
            }
        )
    return errors


def validate_cross_field(cfg: EmbodimentConfig) -> list[ValidationError]:
    """Layer 2: cross-field rules that JSON schema can't express cleanly."""
    errors: list[ValidationError] = []
    action_dim = cfg.action_dim

    # action_space.ranges length must equal action_space.dim
    ranges = cfg.action_space.get("ranges", [])
    if len(ranges) != action_dim:
        errors.append(
            {
                "slug": "action-ranges-length-mismatch",
                "severity": "error",
                "field": "action_space.ranges",
                "message": (
                    f"action_space.ranges has {len(ranges)} entries but "
                    f"action_space.dim is {action_dim}; lengths must match"
                ),
            }
        )

    # Per-dim range must be (lo, hi) with lo < hi
    for i, r in enumerate(ranges):
        if len(r) == 2 and r[0] >= r[1]:
            errors.append(
                {
                    "slug": "action-range-inverted",
                    "severity": "error",
                    "field": f"action_space.ranges[{i}]",
                    "message": f"range {r} has lo >= hi (must be strictly less)",
                }
            )

    # normalization.mean_action / std_action length must equal action_dim
    if len(cfg.normalization.get("mean_action", [])) != action_dim:
        errors.append(
            {
                "slug": "norm-mean-action-length-mismatch",
                "severity": "error",
                "field": "normalization.mean_action",
                "message": (
                    f"mean_action length {len(cfg.normalization['mean_action'])} "
                    f"!= action_dim {action_dim}"
                ),
            }
        )
    if len(cfg.normalization.get("std_action", [])) != action_dim:
        errors.append(
            {
                "slug": "norm-std-action-length-mismatch",
                "severity": "error",
                "field": "normalization.std_action",
                "message": (
                    f"std_action length {len(cfg.normalization['std_action'])} "
                    f"!= action_dim {action_dim}"
                ),
            }
        )

    # mean_state and std_state lengths must match each other (state_dim
    # is inferred from these — no separate field — but they must agree)
    mean_state_len = len(cfg.normalization.get("mean_state", []))
    std_state_len = len(cfg.normalization.get("std_state", []))
    if mean_state_len != std_state_len:
        errors.append(
            {
                "slug": "norm-state-length-mismatch",
                "severity": "error",
                "field": "normalization.mean_state",
                "message": (
                    f"mean_state length {mean_state_len} != "
                    f"std_state length {std_state_len}"
                ),
            }
        )

    # gripper.component_idx must be inside [0, action_dim) if gripper exists
    if cfg.gripper:
        grip_idx = cfg.gripper.get("component_idx", -1)
        if not 0 <= grip_idx < action_dim:
            errors.append(
                {
                    "slug": "gripper-idx-out-of-range",
                    "severity": "error",
                    "field": "gripper.component_idx",
                    "message": (
                        f"component_idx {grip_idx} is outside action_space "
                        f"[0, {action_dim})"
                    ),
                }
            )

    # control.rtc_execution_horizon is an INTEGER count of actions (per
    # ADR 2026-04-25-auto-calibration-architecture decision #8 — the legacy
    # fractional form is auto-migrated by EmbodimentConfig.from_dict). The
    # horizon must be at least one action (RTC degenerate below that) AND
    # at most chunk_size (can't lock more actions than the chunk holds).
    horizon = cfg.control.get("rtc_execution_horizon", 0)
    chunk_size = cfg.control.get("chunk_size", 0)
    if horizon < 1:
        errors.append(
            {
                "slug": "rtc-horizon-too-short",
                "severity": "warn",
                "field": "control.rtc_execution_horizon",
                "message": (
                    f"rtc_execution_horizon = {horizon} < 1 action; "
                    f"RTC will degenerate. Set to an integer count of "
                    f"actions to lock during replan."
                ),
            }
        )
    elif chunk_size and horizon > chunk_size:
        errors.append(
            {
                "slug": "rtc-horizon-exceeds-chunk",
                "severity": "warn",
                "field": "control.rtc_execution_horizon",
                "message": (
                    f"rtc_execution_horizon = {horizon} exceeds chunk_size "
                    f"= {chunk_size}; horizon caps at chunk_size in practice."
                ),
            }
        )

    # cameras must have unique names
    names = [c.get("name") for c in cfg.cameras]
    if len(names) != len(set(names)):
        errors.append(
            {
                "slug": "duplicate-camera-name",
                "severity": "error",
                "field": "cameras",
                "message": f"camera names must be unique; got {names}",
            }
        )

    return errors


def validate_embodiment_config(cfg: EmbodimentConfig) -> tuple[bool, list[ValidationError]]:
    """Run both layers. Returns (ok, errors). `ok` is True iff there are no
    errors (warnings don't block)."""
    schema_errs = validate_against_schema(cfg.to_dict())
    cross_errs = validate_cross_field(cfg)
    all_errs = schema_errs + cross_errs
    has_blocking = any(e["severity"] == "error" for e in all_errs)
    return (not has_blocking, all_errs)


def format_errors(errors: list[ValidationError]) -> str:
    """Pretty-print a list of validation errors for CLI output."""
    if not errors:
        return "  (no errors)"
    lines = []
    for e in errors:
        marker = "ERROR" if e["severity"] == "error" else "WARN "
        field = e["field"] or "<root>"
        lines.append(f"  {marker} [{e['slug']}] {field}: {e['message']}")
    return "\n".join(lines)


__all__ = [
    "ValidationError",
    "validate_against_schema",
    "validate_cross_field",
    "validate_embodiment_config",
    "format_errors",
]
