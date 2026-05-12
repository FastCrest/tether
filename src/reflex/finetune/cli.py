"""Typer command body for `reflex finetune`.

Registered from src/reflex/cli.py via `from reflex.finetune.cli import
finetune_command; app.command(name="finetune")(finetune_command)`.

Kept in a separate module so the CLI + orchestration can be tested
independently.
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def finetune_command(
    base: str = typer.Option(
        "",
        "--base",
        help="HF model id of the base checkpoint, e.g. lerobot/smolvla_base. "
             "Leave empty for from-scratch training (set --policy + --mode full).",
    ),
    policy: str = typer.Option(
        "auto",
        "--policy",
        help="Policy class. 'auto' (default) infers from --base. Set explicitly "
             "(e.g. 'act') for from-scratch training. Per ADR 2026-05-06.",
    ),
    chunk_size: int = typer.Option(
        50,
        "--chunk-size",
        help="Action chunk size for ACT / diffusion-policy from-scratch "
             "training. Pretrained bases (smolvla / pi0.5) bake this in.",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="HF dataset id to fine-tune on, e.g. lerobot/libero",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        help="Output directory. Will contain the fine-tuned checkpoint + "
             "exported model.onnx + VERIFICATION.md on success.",
    ),
    num_steps: int = typer.Option(
        20000,
        "--steps",
        help="Total training steps. SmolVLA LoRA: 2-20k is typical.",
    ),
    batch_size: int = typer.Option(8, "--batch-size"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "--lr"),
    mode: str = typer.Option(
        "lora",
        "--mode",
        help="Fine-tune mode. v0.3 supports 'lora' only; 'full' and "
             "'lora-cross-embodiment' land in v0.5+.",
    ),
    lora_rank: int = typer.Option(
        32,
        "--lora-rank",
        help="LoRA rank. Default 32; VLAs need higher rank than LLMs "
             "(LoRA-SP arxiv 2603.07404). Bump to 64 for GR00T's DiT.",
    ),
    precision: str = typer.Option(
        "bf16", "--precision", help="bf16 | fp32"
    ),
    seed: int = typer.Option(42, "--seed"),
    target: str = typer.Option(
        "desktop",
        "--target",
        help="Hardware profile for the auto-export step. Options: "
             "desktop | orin | orin-nano | thor",
    ),
    skip_export: bool = typer.Option(
        False,
        "--skip-export",
        help="Train only; don't auto-run reflex export. Useful for "
             "training sanity checks where you don't need ONNX yet.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run preflight validation and exit without training. Fast "
             "way to verify dataset compatibility + dataset-size floors "
             "before committing GPU hours. Writes preflight_report.txt "
             "to the output directory.",
    ),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip preflight validation. Escape hatch for local-dataset "
             "or gated-repo flows where preflight can't resolve schema. "
             "Only set if you know what you're doing.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fine-tune a VLA and auto-export to deployable ONNX.

    Thin orchestrator over lerobot-train. v0.3 supports SmolVLA LoRA
    on any LeRobotDataset v3 dataset. pi0 / pi0.5 / GR00T / openpi-JAX
    backend all land in v0.5+.

    Example:
        reflex finetune \\
            --base lerobot/smolvla_base \\
            --dataset lerobot/libero \\
            --output ./my_smolvla_libero \\
            --steps 10000
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    from reflex.finetune.config import FinetuneConfig
    from reflex.finetune.run import run_finetune

    cfg = FinetuneConfig(
        base=base,
        dataset=dataset,
        output=Path(output),
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mode=mode,
        policy=policy,
        chunk_size=chunk_size,
        lora_rank=lora_rank,
        precision=precision,
        seed=seed,
        target=target,
        skip_export=skip_export,
        dry_run=dry_run,
        skip_preflight=skip_preflight,
    )

    console.print(f"[bold]reflex finetune[/bold] — v0.3 MVP (SmolVLA LoRA)")
    console.print(f"  base:    {cfg.base}")
    console.print(f"  dataset: {cfg.dataset}")
    console.print(f"  output:  {cfg.output}")
    console.print(f"  steps:   {cfg.num_steps}  batch={cfg.batch_size}  "
                  f"lr={cfg.learning_rate}  lora_r={cfg.lora_rank}")
    console.print(f"  backend: lerobot  (openpi-JAX + hf_transformers in v0.5+)")
    console.print()

    result = run_finetune(cfg)

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


__all__ = ["finetune_command"]
