"""The run_finetune() orchestration.

v0.3 flow:
  1. Validate config (basic, not full preflight — that's v0.5)
  2. Invoke the chosen backend's fit() — currently lerobot only
  3. Locate the final checkpoint in <output>/checkpoints/
  4. Auto-invoke reflex export on it unless skip_export=True
  5. Return a FinetuneResult with paths + status

v0.5 will add: preflight schema + memory + norm-stats check, parity-gate
at each checkpoint save, calibration-on-holdout, pluggable action-head
registry. See architecture doc Section D for the full target shape.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from reflex.finetune.config import FinetuneConfig, FinetuneResult

logger = logging.getLogger(__name__)


def _validate_config(cfg: FinetuneConfig) -> list[str]:
    """Cheap up-front checks. Returns list of error messages (empty = OK).

    Real preflight (checkpoint loading, dataset schema validation,
    memory budget) is v0.5; this just catches obvious misconfigurations
    before we spin up any training process.
    """
    errs: list[str] = []
    is_distill = getattr(cfg, "phase", "train") == "distill"
    is_from_scratch = (
        getattr(cfg, "policy", "auto") != "auto"
        and getattr(cfg, "mode", "lora") == "full"
    )
    # base is inferred from teacher_export in distill mode, OR not needed
    # when training from-scratch (policy != 'auto' + mode='full').
    if not cfg.base and not is_distill and not is_from_scratch:
        errs.append(
            "base is required (e.g. lerobot/smolvla_base) for fine-tuning. "
            "For ACT-from-scratch training, set --policy act --mode full and "
            "leave --base empty."
        )
    if is_distill and not cfg.teacher_export:
        errs.append("teacher_export is required for phase='distill'")
    if not cfg.dataset:
        errs.append("dataset is required (e.g. lerobot/libero)")
    if cfg.num_steps <= 0:
        errs.append(f"num_steps must be > 0; got {cfg.num_steps}")
    if cfg.batch_size <= 0:
        errs.append(f"batch_size must be > 0; got {cfg.batch_size}")
    if cfg.mode not in ("lora", "lora-cross-embodiment", "full"):
        errs.append(
            f"mode must be one of lora|lora-cross-embodiment|full; got {cfg.mode!r}"
        )
    # v0.3 fine-tune supports LoRA on pretrained bases. `full` is now
    # permitted for from-scratch training (policy != 'auto').
    if not is_distill and cfg.mode != "lora" and not is_from_scratch:
        errs.append(
            f"v0.3 fine-tune supports --mode lora for pretrained bases. "
            f"For from-scratch training, also set --policy <name> (e.g. act). "
            f"Got mode={cfg.mode!r} policy={getattr(cfg, 'policy', 'auto')!r}."
        )
    if is_distill and cfg.mode != "full":
        errs.append(
            f"phase='distill' requires --mode full (SnapFlow trains full weights); got {cfg.mode!r}"
        )
    if cfg.backend != "lerobot":
        errs.append(
            f"v0.3 only supports --backend lerobot; {cfg.backend!r} lands in v0.5+"
        )
    if cfg.precision not in ("bf16", "fp32"):
        errs.append(
            f"precision must be bf16 or fp32; got {cfg.precision!r}"
        )
    return errs


def _infer_policy_type(base: str) -> str:
    """Derive lerobot's policy registry name from the base model id.

    lerobot 0.5.1 registers policies by short name (smolvla, pi0, pi05,
    act, diffusion, vqbet, ...) and requires `--policy.type=<name>` at
    CLI time. The full HF model id (e.g. 'lerobot/smolvla_base') goes
    to `--policy.pretrained_model_path=...` separately.

    Falls back to raising a clear error for unrecognized bases rather
    than guessing. Customers can override via extra_lerobot_args={"policy.type": "..."}.
    """
    base_lower = base.lower()
    if "smolvla" in base_lower:
        return "smolvla"
    if "pi05" in base_lower or "pi0.5" in base_lower or "pi_05" in base_lower:
        return "pi05"
    if "pi0" in base_lower:
        return "pi0"
    if "gr00t" in base_lower or "groot" in base_lower:
        # lerobot 0.5.1 can't load N1.6 per prior Step-3 finding; v0.6 work.
        return "gr00t_n1_5"
    raise ValueError(
        f"Could not infer --policy.type from base={base!r}. "
        f"Supported in v0.3: lerobot/smolvla_base. For other bases, "
        f"pass policy.type explicitly via extra_lerobot_args."
    )


def _build_lerobot_command(cfg: FinetuneConfig) -> list[str]:
    """Construct the lerobot-train invocation.

    lerobot-train accepts draccus-style CLI args (dotted keys). We
    translate our flat FinetuneConfig into the equivalent. Extra knobs
    not yet first-class in FinetuneConfig come through extra_lerobot_args.

    Schema targets lerobot 0.5.1 specifically (the version pinned in the
    Modal image). If lerobot's CLI shifts upstream, update here — the
    generated command is always surfaced in the training log so
    customers can reproduce manually.

    Key arg names per lerobot 0.5.1:
      --policy.pretrained_model_path   (NOT --policy.path)
      --optimizer.lr                   (NOT --policy.optimizer_lr)
      --peft.method_type=lora          (enables PEFT)
      --peft.r                         (LoRA rank)

    cfg.precision is intentionally NOT passed through — lerobot 0.5.1
    doesn't expose a top-level precision flag; it's baked into the
    policy config. v0.5 will add per-policy precision overrides.
    """
    # draccus requires `policy.type` to select which PreTrainedConfig
    # subclass to decode into. Use explicit cfg.policy when set, else
    # infer from the base-model id.
    explicit_policy = getattr(cfg, "policy", "auto")
    if explicit_policy != "auto":
        policy_type = explicit_policy
    else:
        policy_type = _infer_policy_type(cfg.base)
    is_from_scratch = (explicit_policy != "auto" and cfg.mode == "full")

    # lerobot-train wants to OWN its output_dir (errors if pre-existing
    # and resume=False). We keep cfg.output as the reflex orchestration
    # root, and give lerobot a subdirectory it creates fresh on each run.
    lerobot_output = cfg.output / "training"

    # lerobot validates that --policy.repo_id is set (used for Hub
    # uploading). We don't push to Hub, but the validator runs anyway.
    # Pass a placeholder derived from the output dir name.
    # Customers who want to auto-push can override via extra_lerobot_args.
    repo_id = f"local/{cfg.output.name}"

    cmd = [
        "lerobot-train",
        f"--policy.type={policy_type}",
        f"--policy.repo_id={repo_id}",
        f"--policy.push_to_hub=false",
        f"--dataset.repo_id={cfg.dataset}",
        f"--output_dir={lerobot_output}",
        f"--steps={cfg.num_steps}",
        f"--batch_size={cfg.batch_size}",
        f"--optimizer.lr={cfg.learning_rate}",
        f"--seed={cfg.seed}",
    ]
    if not is_from_scratch:
        # Pretrained-base path: pass the HF id / local checkpoint to load weights from.
        cmd.append(f"--policy.pretrained_path={cfg.base}")
    if is_from_scratch and cfg.chunk_size:
        # ACT (and similar chunked policies) need chunk_size; pretrained bases
        # bake this in. auto_soarm convention (per its train.py): set
        # n_action_steps == chunk_size so ACTConfig.__post_init__ doesn't
        # reject the default n_action_steps=100 against a smaller chunk_size.
        cmd.append(f"--policy.chunk_size={cfg.chunk_size}")
        cmd.append(f"--policy.n_action_steps={cfg.chunk_size}")
    if cfg.mode == "lora":
        cmd.extend([
            f"--peft.method_type=lora",
            f"--peft.r={cfg.lora_rank}",
        ])
    for k, v in cfg.extra_lerobot_args.items():
        cmd.append(f"--{k}={v}")
    return cmd


def _run_lerobot_training(
    cfg: FinetuneConfig,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> int:
    """Invoke lerobot-train via subprocess. Streams stdout to log file
    and the root logger. Returns the subprocess exit code.
    """
    cmd = _build_lerobot_command(cfg)
    logger.info("[finetune] exec: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"# reflex finetune — lerobot-train invocation\n")
        log.write(f"# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            # Mirror to our logger at INFO — customers see progress.
            logger.info(line.rstrip())
        proc.wait()
        return proc.returncode


def _locate_checkpoint(output_dir: Path) -> Path | None:
    """Find the final lerobot checkpoint.

    The reflex orchestration dir layout:
        <output>/
            training/               <- lerobot's output_dir
                checkpoints/
                    <step>/pretrained_model/  <- the actual ckpt
            training_log.jsonl
            export/                 <- reflex export output (later)

    We pick the highest-numbered step. Older layouts (where reflex
    itself was the lerobot output_dir) are also tolerated.
    """
    for base in (output_dir / "training" / "checkpoints",
                 output_dir / "checkpoints"):
        if base.exists():
            ckpt_root = base
            break
    else:
        return None
    # lerobot writes step-number dirs like '000200' plus a 'last'
    # pointer (symlink or alias). Prefer numeric dirs — 'last' doesn't
    # resolve reliably on Modal volumes and has no content of its own.
    step_dirs = [
        p for p in ckpt_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    if not step_dirs:
        # Fall back: any dir, including 'last' if that's all we have.
        step_dirs = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not step_dirs:
        return None
    try:
        step_dirs.sort(key=lambda p: int(p.name))
    except ValueError:
        step_dirs.sort(key=lambda p: p.stat().st_mtime)
    final = step_dirs[-1] / "pretrained_model"
    if final.exists():
        return final
    return step_dirs[-1]


def _merge_lora_adapter(checkpoint: Path, base_model_id: str) -> tuple[Path | None, str | None]:
    """Merge a LoRA adapter checkpoint into the base model weights.

    lerobot-train writes LoRA fine-tunes as adapter_model.safetensors +
    adapter_config.json — only the LoRA deltas, no base weights. reflex
    export needs a self-contained checkpoint (with model.safetensors),
    so we load base + apply adapter + merge + save.

    Returns (merged_checkpoint_path, error). On success error is None.
    Writes merged weights to `<checkpoint>/merged/`.
    """
    adapter_cfg = checkpoint / "adapter_config.json"
    adapter_weights = checkpoint / "adapter_model.safetensors"
    if not (adapter_cfg.exists() and adapter_weights.exists()):
        # Not a LoRA checkpoint — caller can use it directly.
        return checkpoint, None

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from peft import PeftModel
    except ImportError as exc:
        return None, f"LoRA merge requires lerobot + peft: {exc}"

    merged_dir = checkpoint / "merged"
    if merged_dir.exists() and any(merged_dir.iterdir()):
        logger.info("[finetune] merged LoRA already exists at %s", merged_dir)
        return merged_dir, None

    try:
        logger.info("[finetune] Merging LoRA adapter into %s base", base_model_id)
        policy = SmolVLAPolicy.from_pretrained(base_model_id)
        # Apply adapter. PeftModel.from_pretrained handles the adapter
        # config + weights.
        merged_policy = PeftModel.from_pretrained(policy, str(checkpoint))
        merged_policy = merged_policy.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_policy.save_pretrained(str(merged_dir))
        # Also copy the pre/post processors + config that lerobot wrote
        # so the merged dir is self-contained for downstream loading.
        import shutil
        for name in (
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "train_config.json",
        ):
            src = checkpoint / name
            if src.exists():
                shutil.copy(src, merged_dir / name)
        # Copy processor safetensors (lerobot suffixes them with step numbers).
        for src in checkpoint.glob("policy_*_processor.safetensors"):
            shutil.copy(src, merged_dir / src.name)
        logger.info("[finetune] LoRA merged to %s", merged_dir)
        return merged_dir, None
    except Exception as exc:
        return None, f"LoRA merge raised: {type(exc).__name__}: {exc}"


def _auto_export(checkpoint: Path, cfg: FinetuneConfig) -> tuple[Path | None, str | None]:
    """Run reflex's existing monolithic export on the fine-tuned checkpoint.

    Returns (onnx_path, error). On success, error is None. If the
    checkpoint is a LoRA adapter (adapter_model.safetensors without
    full base weights), merges it into the base model first.
    """
    # LoRA checkpoints need the base merged in before export can find
    # self-contained weights.
    merged, merge_err = _merge_lora_adapter(checkpoint, cfg.base)
    if merge_err:
        return None, merge_err
    export_src = merged or checkpoint

    try:
        from reflex.exporters.monolithic import export_monolithic
    except ImportError as exc:
        return None, f"reflex.exporters.monolithic import failed: {exc}"

    try:
        result = export_monolithic(
            model_id=str(export_src),
            output_dir=cfg.output / "export",
            target=cfg.target,
        )
        onnx_path = result.get("onnx_path") if isinstance(result, dict) else None
        if onnx_path is None:
            return None, "export_monolithic returned no onnx_path"
        return Path(onnx_path), None
    except Exception as exc:
        return None, f"export_monolithic raised: {type(exc).__name__}: {exc}"


def run_finetune(cfg: FinetuneConfig, *, hooks=None) -> FinetuneResult:
    """Run a fine-tune end-to-end: train → auto-export → receipt.

    v0.3 flow: SmolVLA LoRA via subprocess-lerobot-train, then reflex
    export on the final checkpoint. No parity gate, no calibration, no
    pre-flight — those land in v0.5. A run_finetune() call that reaches
    cfg.num_steps + exports ONNX is a SUCCESS; anything else is an
    error with actionable details in FinetuneResult.error.

    Args:
      cfg: FinetuneConfig.
      hooks: optional HookRegistry. If None, an empty registry is
        created. The distill CLI attaches `libero_drop_gate` here
        before calling run_finetune.
    """
    cfg.output.mkdir(parents=True, exist_ok=True)
    training_log = cfg.output / "training_log.jsonl"

    errs = _validate_config(cfg)
    if errs:
        return FinetuneResult(
            status="aborted",
            output_dir=cfg.output,
            error="config validation failed:\n  " + "\n  ".join(errs),
        )

    # Pre-flight validation (v0.5) — catches top customer pains before
    # any GPU time. Dry-run + skip flags supported.
    if not cfg.skip_preflight:
        from reflex.finetune.preflight import run_preflight
        logger.info("[finetune] running preflight checks...")
        report = run_preflight(cfg)
        logger.info("[finetune] preflight:\n%s", report.render())
        # Persist for customer inspection.
        preflight_path = cfg.output / "preflight_report.txt"
        preflight_path.write_text(report.render() + "\n")
        if report.has_failures:
            return FinetuneResult(
                status="aborted",
                output_dir=cfg.output,
                error=report.error_message(),
            )
        if cfg.dry_run:
            logger.info("[finetune] dry_run=True; skipping training")
            return FinetuneResult(
                status="ok",
                output_dir=cfg.output,
                error=None,
            )

    logger.info("[finetune] start: phase=%s base=%s dataset=%s output=%s steps=%d",
                cfg.phase, cfg.base, cfg.dataset, cfg.output, cfg.num_steps)
    t0 = time.time()

    # Route via resolve_backend() so both phase='train' (fine-tune) and
    # phase='distill' (SnapFlow, Phase B) share one orchestration path.
    # Backend.fit() writes training_log, fires hooks, returns the final
    # checkpoint path; postprocess.finalize() takes over from there.
    from reflex.finetune.backends import TrainerContext, resolve_backend
    from reflex.finetune.hooks import HookRegistry
    from reflex.finetune.postprocess import finalize

    ctx = TrainerContext(
        config=cfg,
        hooks=hooks if hooks is not None else HookRegistry(),
        training_log_path=training_log,
        teacher_path=Path(cfg.teacher_export) if cfg.teacher_export else None,
    )
    backend = resolve_backend(cfg)
    ckpt_result = backend.fit(ctx)
    elapsed = time.time() - t0
    logger.info(
        "[finetune] training status=%s steps=%d elapsed=%.1fs",
        ckpt_result.status, ckpt_result.training_steps_completed, elapsed,
    )

    if ckpt_result.status != "ok":
        return FinetuneResult(
            status=ckpt_result.status,
            output_dir=cfg.output,
            training_steps_completed=ckpt_result.training_steps_completed,
            training_log_path=training_log,
            error=ckpt_result.error,
        )

    if cfg.skip_export:
        logger.info("[finetune] skip_export=True; done (no ONNX)")
        return FinetuneResult(
            status="ok",
            output_dir=cfg.output,
            training_steps_completed=ckpt_result.training_steps_completed,
            final_checkpoint_path=ckpt_result.final_checkpoint_path,
            training_log_path=training_log,
        )

    # Export + on_postprocess hook + VERIFICATION.md all live here.
    return finalize(ctx, ckpt_result)


__all__ = ["run_finetune"]
