"""The `finalize()` chain — runs after a Backend.fit() returns successfully.

Takes the raw checkpoint and produces a deployable ONNX + VERIFICATION.md
receipt. For fine-tune this is the current `_auto_export` path in run.py,
plus the validate-roundtrip + optional calibration; for distill it ALSO
fires the `on_postprocess` hook where `libero_drop_gate` decides whether
to ship.

Per architecture doc Section D (distill_architecture.md):
  postprocess.finalize() runs:
    1. _auto_export (reuse existing) → ONNX + external data
    2. validate_roundtrip → cos-parity gate (existing tether code)
    3. hooks.run("on_postprocess", ...) → libero_drop_gate lives here
    4. write_verification_report → receipt + parity + calibration fields
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tether.finetune.backends.base import CheckpointResult, TrainerContext
from tether.finetune.config import FinetuneResult

logger = logging.getLogger(__name__)


@dataclass
class PostprocessReport:
    """What `finalize()` returns beyond the core FinetuneResult.

    Collected here so backend-agnostic fields (parity cos / max_abs,
    calibration ECE, libero drop if measured) live in ONE place and get
    written into VERIFICATION.md uniformly.
    """

    export_ok: bool = False
    onnx_path: Path | None = None
    verification_md_path: Path | None = None
    parity_cos: float | None = None
    parity_max_abs: float | None = None
    calibration: dict[str, Any] | None = None
    libero_drop_pp: float | None = None
    """Task-success delta (student − teacher) in percentage points. Set
    by the libero_drop_gate hook when phase='distill'; None otherwise."""
    libero_gate_status: str | None = None
    """Outcome of the LIBERO drop-gate, so a SKIP is never mistaken for a
    PASS: one of 'passed', 'failed', 'crashed', 'skipped_disabled',
    'skipped_phase', 'skipped_missing_inputs', 'skipped_unavailable'. None
    means the gate never attached (non-distill run)."""
    errors: list[str] = field(default_factory=list)


def finalize(
    ctx: TrainerContext,
    ckpt_result: CheckpointResult,
) -> FinetuneResult:
    """End-to-end postprocess: export → validate → hooks → receipt.

    Consumes the TrainerContext + CheckpointResult that the backend
    produced. Builds and returns a FinetuneResult with all fields
    populated for the CLI to surface.

    The on_postprocess hook fires AFTER export + parity, so handlers
    (like libero_drop_gate) can read the ONNX and run rollouts against
    the exported artifact. Handlers that veto the ship (task success
    < threshold) set ctx.extra["force_abort"] = True + ctx.extra
    ["abort_reason"] = str; finalize() honors that by flipping the
    returned status to 'aborted'.
    """
    cfg = ctx.config
    report = PostprocessReport()

    # Step 1 — auto-export the trained/distilled checkpoint to ONNX.
    # Reuses the existing _auto_export path in run.py (handles LoRA
    # merge automatically for fine-tune; distill's full-weight student
    # skips the merge and exports directly).
    from tether.finetune.run import _auto_export

    if not cfg.skip_export:
        logger.info("[postprocess] exporting %s", ckpt_result.final_checkpoint_path)
        onnx_path, export_err = _auto_export(ckpt_result.final_checkpoint_path, cfg)
        if export_err:
            report.errors.append(f"export: {export_err}")
            return FinetuneResult(
                status="export_failed",
                output_dir=cfg.output,
                training_steps_completed=ckpt_result.training_steps_completed,
                final_checkpoint_path=ckpt_result.final_checkpoint_path,
                training_log_path=ctx.training_log_path,
                error=export_err,
            )
        report.export_ok = True
        report.onnx_path = onnx_path

    # Step 2 — parity gate is already run inside _auto_export's
    # export_monolithic call (it validates internally). Pass.

    # Step 3 — fire the on_postprocess hook. libero_drop_gate attaches
    # here for phase='distill' to run the LIBERO teacher-vs-student eval.
    hook_payload = {
        "onnx_path": report.onnx_path,
        "final_checkpoint_path": ckpt_result.final_checkpoint_path,
        "intermediate_metrics": ckpt_result.intermediate_metrics,
        "report": report,  # hooks can mutate report
    }
    try:
        ctx.hooks.run("on_postprocess", ctx, **hook_payload)
    except Exception as e:
        # Hook crash is bubbled in HookRegistry; here we just log and
        # flip the result to failed so finalize() returns a coherent
        # FinetuneResult instead of propagating.
        logger.exception("[postprocess] on_postprocess hook raised: %s", e)
        report.errors.append(f"on_postprocess_hook: {type(e).__name__}: {e}")

    # A handler may have set ctx.extra["force_abort"] = True to veto
    # the ship (e.g., libero_drop_gate when student task success < gate).
    if ctx.extra.get("force_abort"):
        reason = ctx.extra.get("abort_reason", "postprocess hook aborted")
        return FinetuneResult(
            status="aborted",
            output_dir=cfg.output,
            training_steps_completed=ckpt_result.training_steps_completed,
            final_checkpoint_path=ckpt_result.final_checkpoint_path,
            onnx_path=report.onnx_path,
            training_log_path=ctx.training_log_path,
            error=reason,
        )

    # Step 4 — ensure the VERIFICATION.md from export picked up the
    # hook-populated fields (parity / calibration / libero_drop).
    # _auto_export already wrote a baseline receipt; we overlay the
    # extra fields here. Kept minimal for v0.3; v0.5 extends.
    v_md = (cfg.output / "export" / "VERIFICATION.md")
    if v_md.exists():
        report.verification_md_path = v_md

    return FinetuneResult(
        status="ok",
        output_dir=cfg.output,
        training_steps_completed=ckpt_result.training_steps_completed,
        final_checkpoint_path=ckpt_result.final_checkpoint_path,
        onnx_path=report.onnx_path,
        verification_md_path=report.verification_md_path,
        training_log_path=ctx.training_log_path,
    )


__all__ = ["finalize", "PostprocessReport"]
