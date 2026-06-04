"""Typer command body for `tether distill`.

Replaces the v0.2 DMPO/pi-Flow entrypoint. v0.3 routes to
`run_finetune(cfg_with_phase_distill)` so distillation reuses the
same orchestration path as fine-tune — single source of truth for
preflight, backend dispatch, export, and VERIFICATION.md.

See https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/01_architecture/distill_SYNTHESIS.md for scope.
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def distill_command(
    teacher_export: str = typer.Option(
        ...,
        "--teacher-export",
        help="Path (or HF id) of a tether-exported teacher dir. Must "
             "contain config.json + model.safetensors (the format "
             "`tether export` produces). v0.3 supports pi0 + pi05 only.",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="HF dataset id for distillation. Use the same dataset the "
             "teacher was fine-tuned on (e.g. lerobot/libero). Must be "
             "LeRobotDataset v3 compatible.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        help="Output directory. Contains training/checkpoints/, "
             "training_log.jsonl, export/VERIFICATION.md on success.",
    ),
    num_steps: int = typer.Option(
        10_000,
        "--steps",
        help="Training steps for SnapFlow. 10-20k is typical for pi-family "
             "on LIBERO-scale datasets. Short enough to fit in one "
             "overnight run on an A10.",
    ),
    batch_size: int = typer.Option(8, "--batch-size"),
    learning_rate: float = typer.Option(
        1e-4, "--learning-rate", "--lr",
        help="AdamW learning rate. Paper used 1e-4 on pi0.5; tune down "
             "to 5e-5 if loss diverges in the first 1k steps.",
    ),
    consistency_alpha: float = typer.Option(
        1.0,
        "--consistency-alpha",
        help="Mix weight for the SnapFlow consistency loss term. Paper "
             "uses 1.0; try 0.5 if the student's flow-matching loss "
             "refuses to converge.",
    ),
    precision: str = typer.Option(
        "bf16", "--precision", help="bf16 | fp32 (bf16 halves VRAM use)",
    ),
    target: str = typer.Option(
        "desktop",
        "--target",
        help="Hardware profile for the auto-export step. The distilled "
             "student is exported right after training completes. "
             "Options: desktop | orin | orin-nano | thor.",
    ),
    libero_gate_pp: float = typer.Option(
        5.0,
        "--libero-gate-pp",
        help="Max allowed LIBERO task-success drop (student vs teacher) "
             "in percentage points. If the student's drop exceeds this "
             "gate, the distilled export is NOT shipped (kept on disk "
             "for inspection). Default 5pp is permissive for v0.3 beta.",
    ),
    skip_libero_gate: bool = typer.Option(
        False,
        "--skip-libero-gate",
        help="Disable the LIBERO drop-gate entirely. Useful for quick "
             "smoke tests or when you don't have LIBERO installed. "
             "Distill will ship without task-success verification.",
    ),
    skip_export: bool = typer.Option(
        False,
        "--skip-export",
        help="Train only; skip the auto-export to ONNX. Useful for "
             "training sanity checks.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run preflight validation and exit without training. "
             "Verifies teacher_export + dataset schema.",
    ),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip preflight validation. Escape hatch.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Distill a flow-matching VLA teacher into a 1-step student (SnapFlow).

    v0.3 scope: pi0 + pi0.5 only. SnapFlow produces a student that
    matches the teacher's task success at 5-10× the inference speed,
    per arxiv 2604.05656 (Luan et al. Apr 2026).

    The student is a copy of the teacher with a zero-init target_time
    embedding; training drives the consistency loss so target_time=1
    activates a one-step generation path. The teacher is frozen.

    Example:
        tether distill \\
            --teacher-export ./my_pi0_libero \\
            --dataset lerobot/libero \\
            --output ./distilled_student \\
            --steps 10000
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    from tether.finetune.config import FinetuneConfig
    from tether.finetune.hooks import HookRegistry
    from tether.finetune.hooks.libero_drop_gate import attach_to as attach_libero_gate
    from tether.finetune.run import run_finetune

    extra_args = {
        "consistency_alpha": consistency_alpha,
        "libero_gate_threshold_pp": libero_gate_pp,
    }
    if skip_libero_gate:
        extra_args["libero_gate_skip"] = True

    cfg = FinetuneConfig(
        base="",  # distill mode infers base from teacher_export
        dataset=dataset,
        output=Path(output),
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mode="full",  # SnapFlow trains full weights (not LoRA)
        precision=precision,
        target=target,
        skip_export=skip_export,
        dry_run=dry_run,
        skip_preflight=skip_preflight,
        phase="distill",
        teacher_export=teacher_export,
        distillation_method="snapflow",
        extra_lerobot_args=extra_args,
    )

    # Build hooks registry with libero_drop_gate attached (unless skipped).
    hooks = HookRegistry()
    if not skip_libero_gate:
        attach_libero_gate(hooks)

    console.print(f"[bold]tether distill[/bold] — SnapFlow (v0.3 MVP)")
    console.print(f"  teacher:   {cfg.teacher_export}")
    console.print(f"  dataset:   {cfg.dataset}")
    console.print(f"  output:    {cfg.output}")
    console.print(f"  steps:     {cfg.num_steps}  batch={cfg.batch_size}  "
                  f"lr={cfg.learning_rate}  alpha={consistency_alpha}")
    if skip_libero_gate:
        console.print(f"  [yellow]libero gate: DISABLED[/yellow]")
    else:
        console.print(f"  libero gate: {libero_gate_pp}pp drop threshold")
    console.print()

    result = run_finetune(cfg, hooks=hooks)

    console.print(f"\n[bold]Result[/bold]")
    console.print(f"  status: {result.status}")
    if result.error:
        console.print(f"  [red]error:[/red] {result.error}")
        raise typer.Exit(code=1)
    console.print(f"  checkpoint: {result.final_checkpoint_path}")
    console.print(f"  training_log: {result.training_log_path}")
    if result.onnx_path:
        console.print(f"  [green]onnx:[/green] {result.onnx_path}")
    if result.verification_md_path:
        console.print(f"  verification: {result.verification_md_path}")


__all__ = ["distill_command"]
