"""Tether Check — training-to-deployment validation.

Validates that a freshly trained VLA checkpoint will export cleanly
before shipping to hardware. Catches problems early.

Usage:
    tether check lerobot/smolvla_base --target orin-nano
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a pre-deployment validation check."""

    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error", "warning", "info"


def check_checkpoint_loadable(path_or_id: str) -> CheckResult:
    """Verify checkpoint can be loaded without errors."""
    try:
        from tether.checkpoint import load_checkpoint

        state_dict, config = load_checkpoint(path_or_id)
        total_params = sum(v.numel() for v in state_dict.values())
        return CheckResult(
            name="checkpoint_loadable",
            passed=True,
            detail=f"Loaded {len(state_dict)} tensors, {total_params/1e6:.1f}M params",
        )
    except Exception as e:
        return CheckResult(
            name="checkpoint_loadable",
            passed=False,
            detail=str(e)[:200],
        )


def check_model_size(state_dict: dict[str, torch.Tensor], target: str) -> CheckResult:
    """Check if model fits on target hardware."""
    from tether.config import get_hardware_profile

    hardware = get_hardware_profile(target)
    total_params = sum(v.numel() for v in state_dict.values())
    weight_gb = total_params * 2 / 1e9  # FP16
    runtime_gb = weight_gb * 2.5  # Estimated with activations

    fits = runtime_gb < hardware.memory_gb * 0.8
    return CheckResult(
        name="model_size",
        passed=fits,
        detail=f"Weights: {weight_gb:.2f}GB, estimated runtime: {runtime_gb:.2f}GB, "
               f"target: {hardware.name} ({hardware.memory_gb}GB)",
        severity="error" if not fits else "info",
    )


def check_key_structure(state_dict: dict[str, torch.Tensor]) -> CheckResult:
    """Verify checkpoint has expected VLA structure."""
    has_vlm = any("vlm" in k for k in state_dict.keys())
    has_expert = any("expert" in k or "lm_expert" in k for k in state_dict.keys())
    has_action = any("action" in k for k in state_dict.keys())

    components = []
    if has_vlm:
        components.append("VLM")
    if has_expert:
        components.append("expert")
    if has_action:
        components.append("action_proj")

    passed = len(components) >= 2
    return CheckResult(
        name="key_structure",
        passed=passed,
        detail=f"Found components: {', '.join(components) or 'none'}",
        severity="error" if not passed else "info",
    )


def check_dtype_compatibility(state_dict: dict[str, torch.Tensor], target: str) -> CheckResult:
    """Check if weight dtypes are compatible with target."""
    from tether.config import get_hardware_profile

    hardware = get_hardware_profile(target)
    dtypes = set()
    for v in state_dict.values():
        dtypes.add(str(v.dtype))

    has_fp8 = any("float8" in d for d in dtypes)
    if has_fp8 and not hardware.fp8_support:
        return CheckResult(
            name="dtype_compatibility",
            passed=False,
            detail=f"Checkpoint has FP8 weights but {hardware.name} doesn't support FP8 (SM {hardware.sm_version})",
            severity="error",
        )

    return CheckResult(
        name="dtype_compatibility",
        passed=True,
        detail=f"Weight dtypes: {', '.join(sorted(dtypes))}. Compatible with {hardware.name}.",
    )


def check_nan_inf(state_dict: dict[str, torch.Tensor]) -> CheckResult:
    """Check for NaN or Inf values in weights."""
    bad_keys = []
    for key, tensor in state_dict.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            bad_keys.append(key)

    if bad_keys:
        return CheckResult(
            name="nan_inf",
            passed=False,
            detail=f"Found NaN/Inf in {len(bad_keys)} tensors: {bad_keys[:5]}",
            severity="error",
        )

    return CheckResult(
        name="nan_inf",
        passed=True,
        detail="No NaN or Inf values found",
    )


def run_all_checks(path_or_id: str, target: str = "desktop") -> list[CheckResult]:
    """Run all pre-deployment validation checks."""
    results = []

    # Check 1: Loadable
    load_result = check_checkpoint_loadable(path_or_id)
    results.append(load_result)
    if not load_result.passed:
        return results

    # Load for remaining checks
    from tether.checkpoint import load_checkpoint

    state_dict, config = load_checkpoint(path_or_id)

    # Check 2: Size
    results.append(check_model_size(state_dict, target))

    # Check 3: Structure
    results.append(check_key_structure(state_dict))

    # Check 4: Dtype
    results.append(check_dtype_compatibility(state_dict, target))

    # Check 5: NaN/Inf
    results.append(check_nan_inf(state_dict))

    return results
