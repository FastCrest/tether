"""Validation: compare exported model outputs against PyTorch reference."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    num_elements: int
    threshold: float
    details: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_abs_diff": round(self.max_abs_diff, 6),
            "mean_abs_diff": round(self.mean_abs_diff, 6),
            "max_rel_diff": round(self.max_rel_diff, 6),
            "num_elements": self.num_elements,
            "threshold": self.threshold,
            "details": self.details,
        }


def validate_outputs(
    reference: np.ndarray | torch.Tensor,
    candidate: np.ndarray | torch.Tensor,
    threshold: float = 0.02,
    name: str = "output",
) -> ValidationResult:
    """Compare two outputs and report numerical differences."""
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    if isinstance(candidate, torch.Tensor):
        candidate = candidate.detach().cpu().numpy()

    reference = reference.astype(np.float32)
    candidate = candidate.astype(np.float32)

    if reference.shape != candidate.shape:
        return ValidationResult(
            passed=False,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            num_elements=0,
            threshold=threshold,
            details=f"Shape mismatch: reference {reference.shape} vs candidate {candidate.shape}",
        )

    abs_diff = np.abs(reference - candidate)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())

    denom = np.maximum(np.abs(reference), 1e-8)
    rel_diff = abs_diff / denom
    max_rel = float(rel_diff.max())

    passed = max_abs < threshold
    details = (
        f"{name}: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
        f"max_rel={max_rel:.4f}, threshold={threshold}"
    )
    if passed:
        logger.info("PASS %s", details)
    else:
        logger.warning("FAIL %s", details)

    return ValidationResult(
        passed=passed,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        max_rel_diff=max_rel,
        num_elements=int(reference.size),
        threshold=threshold,
        details=details,
    )


def validate_decomposition(
    original_module: torch.nn.Module,
    decomposed_module: torch.nn.Module,
    dummy_input: torch.Tensor,
    threshold: float = 1e-5,
    name: str = "decomposition",
) -> ValidationResult:
    """Validate that a decomposed module produces the same output."""
    original_module.eval()
    decomposed_module.eval()

    with torch.no_grad():
        ref_output = original_module(dummy_input)
        dec_output = decomposed_module(dummy_input)

    return validate_outputs(ref_output, dec_output, threshold=threshold, name=name)
