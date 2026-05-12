"""Configs + result shapes for reflex finetune.

Decoupled from run.py so tests + external callers can construct a
FinetuneConfig without importing the full orchestration path (which
would pull lerobot + HF + ORT).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FinetuneConfig:
    """Input to run_finetune(). Flat dataclass; maps ~1:1 to CLI flags.

    v0.3 exposes a narrow slice of what the full architecture doc lists.
    Extra fields are added here as backends / preflight / action-head
    registries land in v0.5+.
    """

    base: str
    """HF model id of the base checkpoint (e.g. lerobot/smolvla_base)."""

    dataset: str
    """HF dataset id to fine-tune on (e.g. lerobot/libero)."""

    output: Path
    """Output directory. Will contain model.onnx + VERIFICATION.md after
    successful run."""

    num_steps: int = 20_000
    """Total training steps. For SmolVLA LoRA, 2-20k is typical depending
    on dataset size."""

    batch_size: int = 8
    learning_rate: float = 1e-4
    mode: str = "lora"
    """One of: lora | lora-cross-embodiment | full. v0.3 supports lora.
    `full` is now permitted for ACT-from-scratch training (policy='act')
    per ADR 2026-05-06-vendor-auto-soarm.md."""

    policy: str = "auto"
    """Policy class to train.
      - 'auto' (default): infer from `base` (e.g. lerobot/smolvla_base → smolvla).
      - 'act' / 'diffusion' / 'pi0' / etc.: explicit lerobot policy.type.
    Used by from-scratch training paths where `base` is empty or a non-pretrained
    sentinel; lerobot-train requires --policy.type either way.

    ACT-from-scratch (vendored from auto_soarm 2026-05-06): set policy='act',
    mode='full', leave base='' (or set to ''). Recipe defaults:
        num_steps=30000, batch_size=8, learning_rate=1e-5, chunk_size=31, seed=1
    """

    chunk_size: int = 50
    """Action chunk size. Used by ACT (chunk_size=31 in auto_soarm's recipe)
    + diffusion-policy. Pretrained policies (smolvla / pi0.5) bake this in;
    we only pass it through when policy != 'auto'."""

    lora_rank: int = 32
    """Default r=32 — VLA intrinsic rank is 4-16x LLM's (LoRA-SP).
    lerobot default is r=16; we override to avoid the norm-stats
    mismatch that drops task success 96%→1% (competitive research #6)."""

    precision: str = "bf16"
    """bf16 by default (SOTA research A4). Options: bf16 | fp32."""

    seed: int = 42
    backend: str = "lerobot"
    """v0.3 supports only lerobot. openpi + hf_transformers in v0.5+."""

    skip_export: bool = False
    """v0.3 auto-exports on success. Set True to skip (e.g. for a quick
    training sanity check that doesn't need ONNX)."""

    target: str = "desktop"
    """Hardware profile for the auto-export step."""

    extra_lerobot_args: dict[str, Any] = field(default_factory=dict)
    """Escape hatch: pass through extra args to lerobot-train. Used for
    features not yet first-class in FinetuneConfig."""

    dry_run: bool = False
    """If True, run preflight checks and exit. No GPU time spent.
    Useful for validating config before committing to a multi-hour run."""

    skip_preflight: bool = False
    """If True, skip preflight validation. Escape hatch for local-dataset
    or gated-repo flows where preflight can't resolve the schema.
    Only set if you know what you're doing."""

    phase: str = "train"
    """One of: 'train' | 'distill'.
    - 'train' (default, v0.3): fine-tune via lerobot-train subprocess
    - 'distill' (v0.3+): SnapFlow distillation. Requires teacher_export.
    Routed to backends via reflex.finetune.backends.resolve_backend()."""

    teacher_export: str | None = None
    """For phase='distill' only: path/HF-id of the teacher's reflex-
    export dir (merged PyTorch checkpoint). None for phase='train'."""

    distillation_method: str = "snapflow"
    """For phase='distill' only: which distillation trainer to use.
    v0.3 supports 'snapflow' only. 'consistency' (DDPM for GR00T)
    lands in v0.5+."""

    variant: str = "default"
    """Architecture variant of the student. v0.5 adds 'state_out' for
    pi0.5 students: strips state from the language prompt and adds an
    explicit state_proj layer that consumes proprio state. Unlocks the
    prefix KV cache in production (lang becomes stable per episode).
    Values:
      - 'default': v0.3.1 SnapFlow student (state-in-lang via the
        default pi0.5 preprocessor + no state_proj layer).
      - 'state_out': pi0.5 only. Applies enable_snapflow_state_out
        to the student + swaps the preprocessor step.
    See reflex_vault 01_architecture/distill_state_out_pi05_design.md."""

    loss_mode: str = "snapflow"
    """Distillation loss formulation. v0.5 retry adds 'teacher_supervised'.
    Values:
      - 'snapflow' (default): self-distillation consistency loss
        (Eq. 11 of arxiv 2604.05656). Teacher is NOT in the loss; the
        student is initialized from teacher weights and the loss enforces
        self-consistency across the velocity field. Works for v0.3.1
        (same architecture, same input modality).
      - 'teacher_supervised': L2(v_student, v_teacher) at random t.
        Teacher actually IN the loss. Required for v0.5 cross-modality
        distillation (state-in-lang teacher → state-out student) where
        SnapFlow's self-consistency converges to the degenerate "ignore
        state" fixed point.
    See reflex_vault 03_experiments/2026-04-22-v0.5-retry-plan-pmcd.md."""

    warm_init_state_proj_from: str = ""
    """Optional HF repo id to warm-init state_proj weights from. Only
    applies when variant='state_out'. Recommended:
    'lerobot/pi0_libero_finetuned_v044' for LIBERO. Empty string =
    small-random init (fallback)."""

    state_sensitivity_alpha: float = 0.0
    """Weight for the state-sensitivity penalty term (only applies
    when loss_mode='teacher_supervised'). 0 disables. Recommended:
    0.1 for early training to push student off the 'ignore state'
    fixed point, then 0 for fine-tuning. Currently constant
    throughout training; curriculum schedule TBD."""

    base_dataset: str | None = None
    """Optional second dataset to MIX with `dataset` during training.
    When set, the dataloader uses ConcatDataset + WeightedRandomSampler
    to draw from both sources per `mix_ratio`. Used by Pro-tier
    self-distilling-serve (per ADR 2026-04-25-self-distilling-serve-
    architecture decision #2): customer's collected data goes in
    `dataset`; the original LeRobot training set goes in `base_dataset`
    so the student doesn't catastrophically forget the base distribution.
    Requires both datasets to share action_dim + chunk_size; preflight
    cross-validates."""

    mix_ratio: float = 0.5
    """When `base_dataset` is set, fraction of each batch drawn from the
    customer `dataset` (vs `base_dataset`). 0.0 = base only (no
    customer adaptation). 1.0 = customer only (max adaptation, max
    catastrophic-forgetting risk). Default 0.5 (50/50) per ADR
    2026-04-25-self-distilling-serve-architecture: balances adaptation
    against base-distribution preservation."""

    def __post_init__(self) -> None:
        self.output = Path(self.output)
        if not (0.0 <= self.mix_ratio <= 1.0):
            raise ValueError(
                f"mix_ratio must be in [0, 1], got {self.mix_ratio}"
            )


@dataclass
class FinetuneResult:
    """Output of run_finetune()."""

    status: str
    """One of: ok | training_failed | export_failed | aborted."""

    output_dir: Path
    training_steps_completed: int = 0
    final_checkpoint_path: Path | None = None

    onnx_path: Path | None = None
    """If skip_export=False and status=="ok", path to the exported ONNX."""

    verification_md_path: Path | None = None
    """Path to VERIFICATION.md written by the auto-export chain."""

    training_log_path: Path | None = None
    """Path to training_log.jsonl (stdout capture of the training run)."""

    error: str | None = None
    """Populated when status != "ok"."""

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        for attr in (
            "final_checkpoint_path",
            "onnx_path",
            "verification_md_path",
            "training_log_path",
        ):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, Path(v))


__all__ = ["FinetuneConfig", "FinetuneResult"]
