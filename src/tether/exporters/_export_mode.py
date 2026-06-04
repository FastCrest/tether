"""Export-mode selection for parallel decomposed export.

Per ADR `01_decisions/2026-04-28-parallel-decomposed-export.md`:
decomposed VLA exports produce two independent ONNX graphs (vlm_prefix +
expert_denoise). They share no state and can run as parallel subprocesses
on hardware that fits both models in VRAM. Roughly halves wall-time on
RTX-class GPUs.

Auto-detection rule:
    parallel iff `2 * estimated_model_vram + buffer < free_vram`

Failure mode discipline (per CLAUDE.md "no silent fallbacks"):
- Auto: probe VRAM, pick mode, log which + why
- Parallel forced: fail loudly with InsufficientVRAMError if doesn't fit
- Sequential forced: always works (the safe baseline)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# 1 GB safety buffer above 2x model size. Empirically chosen — accounts for
# PyTorch + ORT scratch + framework overhead during simultaneous export.
# May need tuning per-model in v0.6.x if real hardware shows wrong selection.
_VRAM_SAFETY_BUFFER_BYTES = 1 * 1024 ** 3

# PyTorch + activations + ORT scratch are typically ~4x the serialized ONNX
# size during export. Conservative estimate to prevent OOM. Tune per ADR's
# 2026-07-28 revisit if production telemetry shows different.
_ONNX_TO_VRAM_MULTIPLIER = 4


class ExportMode(str, Enum):
    AUTO = "auto"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class ExportModeDecision:
    """The outcome of select_mode() — what mode was chosen and why."""

    mode: ExportMode  # Always concrete — never AUTO after select_mode resolves
    reason: str       # Human-readable explanation for logging


class InsufficientVRAMError(RuntimeError):
    """--export-mode parallel was requested but VRAM is insufficient.

    Raised loudly per CLAUDE.md "no silent fallbacks." User chose parallel
    explicitly; if it can't run, they need to know so they can switch to
    auto or sequential — not silently degraded.
    """


def probe_free_vram() -> int | None:
    """Return free GPU VRAM in bytes, or None if no GPU / probe failed.

    None means "we don't know" — caller should default to sequential.
    """
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception as exc:  # noqa: BLE001 — torch.cuda errors are diverse
        logger.warning("VRAM probe failed (%s); defaulting to sequential.", exc)
        return None


def estimate_model_vram_from_onnx(onnx_size_bytes: int) -> int:
    """Estimate the GPU VRAM a single export pass needs, given the ONNX file size.

    Rule of thumb: PyTorch model + activations + ORT scratch = ~4x serialized.
    Conservative — prefers selecting sequential over OOM in parallel.
    """
    return onnx_size_bytes * _ONNX_TO_VRAM_MULTIPLIER


def select_mode(
    requested: ExportMode,
    estimated_per_export_vram: int,
) -> ExportModeDecision:
    """Resolve `requested` to a concrete mode (PARALLEL or SEQUENTIAL).

    Args:
        requested: ExportMode.AUTO / PARALLEL / SEQUENTIAL
        estimated_per_export_vram: estimated VRAM bytes per single export pass
                                    (use estimate_model_vram_from_onnx())

    Returns:
        ExportModeDecision with concrete mode + human-readable reason

    Raises:
        InsufficientVRAMError: if requested == PARALLEL and VRAM doesn't fit
        ValueError: if requested isn't a valid ExportMode

    Per CLAUDE.md "no silent fallbacks": when user explicitly requests
    parallel and we can't run it, we raise — not silently downgrade.
    """
    if requested == ExportMode.SEQUENTIAL:
        return ExportModeDecision(
            mode=ExportMode.SEQUENTIAL,
            reason="explicit --export-mode sequential",
        )

    free_vram = probe_free_vram()
    needed_vram = 2 * estimated_per_export_vram + _VRAM_SAFETY_BUFFER_BYTES

    if requested == ExportMode.PARALLEL:
        if free_vram is None:
            raise InsufficientVRAMError(
                "--export-mode parallel requires a CUDA GPU with VRAM probe. "
                "No GPU detected (or torch.cuda.mem_get_info() failed). "
                "Use --export-mode sequential to force the safe baseline, "
                "or --export-mode auto to let the installer pick."
            )
        if free_vram < needed_vram:
            raise InsufficientVRAMError(
                f"--export-mode parallel needs ~{needed_vram / 1e9:.1f} GB "
                f"free VRAM (2x model size + 1 GB buffer); only "
                f"{free_vram / 1e9:.1f} GB free. Free up GPU memory, use "
                f"a larger GPU, or switch to --export-mode sequential / auto."
            )
        return ExportModeDecision(
            mode=ExportMode.PARALLEL,
            reason=f"explicit --export-mode parallel ({free_vram / 1e9:.1f} GB "
                   f"free, models need ~{needed_vram / 1e9:.1f} GB combined)",
        )

    if requested == ExportMode.AUTO:
        if free_vram is None:
            return ExportModeDecision(
                mode=ExportMode.SEQUENTIAL,
                reason="no GPU detected — sequential is the only option on CPU",
            )
        if free_vram < needed_vram:
            return ExportModeDecision(
                mode=ExportMode.SEQUENTIAL,
                reason=f"only {free_vram / 1e9:.1f} GB VRAM free, parallel "
                       f"would need ~{needed_vram / 1e9:.1f} GB",
            )
        return ExportModeDecision(
            mode=ExportMode.PARALLEL,
            reason=f"{free_vram / 1e9:.1f} GB free, models need "
                   f"~{needed_vram / 1e9:.1f} GB combined",
        )

    raise ValueError(f"Unknown ExportMode: {requested!r}")


def log_decision(decision: ExportModeDecision) -> None:
    """Print a clear, single-line log of the chosen export mode."""
    logger.info("Export mode: %s (%s)", decision.mode.value, decision.reason)
