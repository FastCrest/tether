"""SnapFlow backend — in-process 1-step self-distillation trainer.

Implements `Backend.fit()` using the math in `tether.distill.snapflow`
and the frozen teacher loader in `tether.distill.teacher_loader`.

## What this backend does

1. Load teacher policy (frozen, eval mode) via teacher_loader.
2. deepcopy() teacher → student. Install the zero-init target_time
   embedding into the student's velocity head (model surgery via
   the policy-specific adapter in `_velocity_adapters`).
3. Build a LeRobotDataset dataloader matching the teacher's expected
   obs_kwargs (image, state, language).
4. Training loop: for each batch, sample t + noise, compute SnapFlow
   loss (flow_matching + consistency), backprop through student only.
5. Save checkpoint every `checkpoint_every` steps via the teacher's
   native `save_pretrained` (same format `_auto_export` consumes).
6. Fire the lifecycle hooks (`on_start`, `on_step`, `on_checkpoint`,
   `on_end`) at the documented points.

## What this backend does NOT do

- No LoRA/adapter logic — SnapFlow trains FULL weights (the student
  is a full-weight copy, no PEFT merge step needed).
- No ONNX export (postprocess.finalize chain owns that).
- No LIBERO eval (libero_drop_gate hook owns that, fired by finalize).
- No VLM prefix caching (current v0.3 recomputes per step for
  simplicity; v0.5 can cache when profiling shows it's the bottleneck).

## The velocity-adapter split

The SnapFlow math is paradigm-agnostic, but actually EXTRACTING the
velocity function from a lerobot PI0Policy / PI05Policy depends on
per-policy internals (expert-stack attribute names, prefix KV cache
construction, etc.). So this module splits:

  - `SnapFlowBackend.fit()` — generic loop (optimizer, dataloader,
    loss step, checkpoint, hooks). Takes velocity_fn callables.
  - `_build_velocity_adapters(teacher, student, policy_type)` —
    policy-specific. For pi0/pi05, wraps the PaliGemma-VLM +
    expert-stack. Lazy-imported per-policy-type so CI can test
    the generic loop without lerobot installed.

## v0.3.1 (target_time properly wired)

pi0/pi05 teacher + student are mutated in place by
`tether.distill.snapflow_pi0_model.enable_snapflow()`. This swaps
`__class__` to SnapFlowPI0Pytorch and attaches a zero-init learnable
`target_time_embed_mlp`. Downstream effect:

  - `student.model.embed_suffix(..., target_time=tensor)` adds the
    learned contribution to the time embedding, so consistency loss
    at target_time=1 is semantically meaningful.
  - `student.model.denoise_step` skips `copy.deepcopy(past_kv)` (safe
    because downstream forward uses `use_cache=False`) and skips the
    hardcoded `fp32` cast (uses `action_out_proj.weight.dtype`).
  - Past KV cache stays graph-attached → VLM receives gradients.

Dataset now uses `delta_timestamps={ACTION: [i/fps for i in range(chunk_size)]}`
so LeRobotDataset yields real action chunks instead of single-frame tiles.

v0.3.1: pi0 + pi05. SmolVLA pending velocity-convergence validation;
GR00T is v0.5+ (different denoising paradigm — DDPM).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from tether.finetune.backends.base import (
    Backend,
    CheckpointResult,
    TrainerContext,
)

logger = logging.getLogger(__name__)


# Default mix coefficient for the SnapFlow consistency term when the
# FinetuneConfig doesn't override via extra_lerobot_args. Paper uses 1.0.
DEFAULT_CONSISTENCY_ALPHA: float = 1.0

# Save a student checkpoint every N steps (same convention as
# lerobot-train). Override via cfg.extra_lerobot_args["checkpoint_every"].
DEFAULT_CHECKPOINT_EVERY: int = 1_000


class SnapFlowBackend:
    """In-process SnapFlow distillation trainer."""

    def fit(self, ctx: TrainerContext) -> CheckpointResult:
        cfg = ctx.config
        if not cfg.teacher_export:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=(
                    "SnapFlowBackend requires cfg.teacher_export to be set. "
                    "Pass --teacher-export <path-or-hf-id> pointing at a "
                    "tether-exported dir (merged PyTorch checkpoint)."
                ),
            )

        # Lazy-import torch + SnapFlow math so a CI run that never touches
        # distill doesn't pay the import cost.
        import torch
        from torch.optim import AdamW

        from tether.distill.snapflow import snapflow_loss_step
        from tether.distill.teacher_loader import load_teacher

        # ---- 1. Load teacher, build student --------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = cfg.precision if cfg.precision in ("bf16", "fp32") else "bf16"
        logger.info("[snapflow] loading teacher from %s on %s", cfg.teacher_export, device)
        loaded = load_teacher(cfg.teacher_export, device=device, dtype=dtype)
        teacher = loaded.policy
        policy_type = loaded.policy_type

        logger.info("[snapflow] building student (fresh load from teacher dir)")
        # deepcopy(teacher) breaks on pi0 because some internal tensors are
        # non-leaf (see pytorch#103001). Load a second PI0Policy from the
        # same checkpoint instead — costs one extra safetensors parse but
        # produces a clean trainable copy.
        from tether.distill.teacher_loader import load_teacher as _load
        student_loaded = _load(cfg.teacher_export, device=device, dtype=dtype)
        student = student_loaded.policy
        for p in student.parameters():
            p.requires_grad = True
        student.train()

        # ---- 2. Enable SnapFlow overrides on both teacher + student --------
        # pi0/pi05: swap __class__ to SnapFlowPI0Pytorch and attach zero-init
        # target_time_embed_mlp. This replaces three v0.3 monkey-patches:
        # deepcopy-free past_kv, no hardcoded fp32 cast, target_time support.
        # Teacher's target_time_embed_mlp stays zero + frozen (we never pass
        # target_time to teacher, so the contribution is 0 either way).
        #
        # Also disable gradient_checkpointing on both. Pi05 enables it by
        # default; with GC on, the gemma stack force-overrides use_cache=False
        # and past_kv comes back EMPTY from the prefix forward, breaking
        # downstream denoise_step (attention mask shape mismatch). We accept
        # the memory cost of disabling GC; A100-80GB has headroom at batch=4.
        if policy_type in ("pi0", "pi05"):
            from tether.distill.snapflow_pi0_model import enable_snapflow
            enable_snapflow(teacher.model)

            variant = getattr(cfg, "variant", "default")
            if variant == "state_out" and policy_type == "pi05":
                from tether.distill.snapflow_pi0_model import enable_snapflow_state_out
                # Student: state-out variant. enable_snapflow_state_out internally
                # calls enable_snapflow first (adds target_time_embed_mlp) then
                # registers state_proj + swaps to SnapFlowPI05StateOutPytorch.
                warm_init_repo = getattr(cfg, "warm_init_state_proj_from", "") or None
                enable_snapflow_state_out(
                    student.model, warm_init_from_pi0=warm_init_repo,
                )
                logger.info(
                    "[snapflow] student uses STATE-OUT variant — will receive "
                    "proprio state via state_proj, not lang tokens. "
                    "Teacher stays on default (state-in-lang) preprocessor path."
                )
                if warm_init_repo:
                    logger.info(
                        "[snapflow] state_proj warm-init from %s", warm_init_repo,
                    )
            elif variant == "state_out":
                return CheckpointResult(
                    final_checkpoint_path=Path(cfg.output),
                    training_steps_completed=0,
                    status="training_failed",
                    error=(
                        f"variant='state_out' is only supported for pi05 "
                        f"(state-in-language is pi0.5 specific); got policy_type={policy_type}"
                    ),
                )
            else:
                enable_snapflow(student.model)

            for p in teacher.model.target_time_embed_mlp.parameters():
                p.requires_grad = False
            for m in (teacher.model, student.model):
                disable_gc = getattr(m, "gradient_checkpointing_disable", None)
                if callable(disable_gc):
                    disable_gc()

        # ---- 3. Install velocity adapters ---------------------------------
        teacher_velocity_fn, student_velocity_fn = _build_velocity_adapters(
            teacher=teacher,
            student=student,
            policy_type=policy_type,
        )

        # ---- 4. Build dataloader + preprocessor ---------------------------
        chunk_size_cfg = getattr(teacher.config, "chunk_size", None)
        try:
            loader = _build_dataloader(
                cfg, policy_type=policy_type, chunk_size=chunk_size_cfg,
            )
        except Exception as e:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=f"dataloader construction failed: {type(e).__name__}: {e}",
            )
        try:
            preprocessor = _build_preprocessor(
                teacher_path=loaded.checkpoint_dir,
                image_key_map=cfg.extra_lerobot_args.get("image_key_map"),
                device=device,
            )
        except Exception as e:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=f"preprocessor construction failed: {type(e).__name__}: {e}",
            )

        # v0.5 state-out: build a second preprocessor for the student
        # with the state-tokenization step swapped for the state-out
        # variant (produces lang_tokens without state appended). Teacher
        # keeps the original preprocessor so its forward stays correct.
        student_preprocessor = preprocessor
        if getattr(cfg, "variant", "default") == "state_out":
            try:
                student_preprocessor = _build_preprocessor(
                    teacher_path=loaded.checkpoint_dir,
                    image_key_map=cfg.extra_lerobot_args.get("image_key_map"),
                    device=device,
                )
                from tether.distill.pi05_state_out_processor import (
                    swap_prepare_step_in_pipeline,
                )
                swap_prepare_step_in_pipeline(
                    student_preprocessor,
                    max_state_dim=getattr(teacher.config, "max_state_dim", 32),
                )
                logger.info(
                    "[snapflow] built separate student preprocessor (state-out)"
                )
            except Exception as e:
                return CheckpointResult(
                    final_checkpoint_path=Path(cfg.output),
                    training_steps_completed=0,
                    status="training_failed",
                    error=f"state-out preprocessor construction failed: {type(e).__name__}: {e}",
                )

        # ---- 5. Optimizer + checkpoint dir --------------------------------
        opt = AdamW(
            (p for p in student.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )
        checkpoint_root = Path(cfg.output) / "training" / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_every = int(
            cfg.extra_lerobot_args.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY)
        )
        consistency_alpha = float(
            cfg.extra_lerobot_args.get("consistency_alpha", DEFAULT_CONSISTENCY_ALPHA)
        )

        # ---- 6. Fire on_start ---------------------------------------------
        ctx.hooks.run("on_start", ctx, config=cfg, policy_type=policy_type)

        # ---- 7. Training loop ---------------------------------------------
        step = 0
        last_ckpt: Path | None = None
        loss_history: list[dict[str, float]] = []
        log_handle = open(ctx.training_log_path, "a", encoding="utf-8")
        try:
            import time as _time
            import torch as _torch
            target_dtype = _torch.bfloat16 if dtype == "bf16" else _torch.float32
            max_action_dim = getattr(teacher.config, "max_action_dim", None)
            chunk_size = getattr(teacher.config, "chunk_size", None)
            variant = getattr(cfg, "variant", "default")
            loss_mode = getattr(cfg, "loss_mode", "snapflow")
            sensitivity_alpha = getattr(cfg, "state_sensitivity_alpha", 0.0)
            heartbeat_every = int(
                cfg.extra_lerobot_args.get("heartbeat_every", 50)
            )
            heartbeat_t0 = _time.time()
            heartbeat_step0 = 0
            for step, batch in enumerate(loader, start=1):
                # Apply lerobot's preprocessor pipeline: rename_map (image keys),
                # tokenizer (task -> language_tokens), device transfer, normalize.
                # v0.5 state-out: preprocess twice — teacher sees state-in-lang,
                # student sees stripped-lang + state as an explicit batch key.
                if variant == "state_out":
                    teacher_batch = preprocessor(batch)
                    student_batch = student_preprocessor(batch)
                    action, noise, t, teacher_obs_kwargs = _prepare_batch(
                        teacher_batch,
                        device=device,
                        compute_dtype=target_dtype,
                        max_action_dim=max_action_dim,
                        chunk_size=chunk_size,
                    )
                    _, _, _, obs_kwargs = _prepare_batch(
                        student_batch,
                        device=device,
                        compute_dtype=target_dtype,
                        max_action_dim=max_action_dim,
                        chunk_size=chunk_size,
                    )
                else:
                    batch = preprocessor(batch)
                    action, noise, t, obs_kwargs = _prepare_batch(
                        batch,
                        device=device,
                        compute_dtype=target_dtype,
                        max_action_dim=max_action_dim,
                        chunk_size=chunk_size,
                    )
                    teacher_obs_kwargs = None

                opt.zero_grad()
                if loss_mode == "teacher_supervised":
                    from tether.distill.snapflow import teacher_supervised_loss_step
                    loss, snap = teacher_supervised_loss_step(
                        student_velocity_fn,
                        teacher_velocity_fn,
                        action=action,
                        noise=noise,
                        t=t,
                        obs_kwargs=obs_kwargs,
                        teacher_obs_kwargs=teacher_obs_kwargs,
                        state_sensitivity_alpha=sensitivity_alpha,
                    )
                else:
                    loss, snap = snapflow_loss_step(
                        student_velocity_fn,
                        teacher_velocity_fn,
                        action=action,
                        noise=noise,
                        t=t,
                        obs_kwargs=obs_kwargs,
                        teacher_obs_kwargs=teacher_obs_kwargs,
                        consistency_alpha=consistency_alpha,
                    )
                loss.backward()
                opt.step()

                loss_history.append(asdict(snap) | {"step": step})
                log_handle.write(
                    json.dumps({"step": step, **asdict(snap), "lr": cfg.learning_rate}) + "\n"
                )
                log_handle.flush()

                ctx.hooks.run(
                    "on_step",
                    ctx,
                    step=step,
                    loss=snap.total,
                    lr=cfg.learning_rate,
                    flow_matching=snap.flow_matching,
                    consistency=snap.consistency,
                )

                if step == 1 or step % heartbeat_every == 0:
                    elapsed = _time.time() - heartbeat_t0
                    steps_in_window = step - heartbeat_step0
                    rate = steps_in_window / max(elapsed, 1e-6)
                    eta_min = (cfg.num_steps - step) / max(rate, 1e-6) / 60.0
                    logger.info(
                        "[snapflow] step %d/%d  total=%.4f  fm=%.4f  cons=%.4f  "
                        "rate=%.2f steps/s  eta=%.1f min",
                        step, cfg.num_steps, snap.total, snap.flow_matching,
                        snap.consistency, rate, eta_min,
                    )
                    heartbeat_t0 = _time.time()
                    heartbeat_step0 = step

                if step % checkpoint_every == 0 or step == cfg.num_steps:
                    last_ckpt = _save_student_checkpoint(
                        student, checkpoint_root, step,
                    )
                    ctx.hooks.run(
                        "on_checkpoint", ctx, step=step, ckpt_path=last_ckpt,
                    )

                if step >= cfg.num_steps:
                    break
        finally:
            log_handle.close()

        # ---- 8. Ensure we have a final checkpoint -------------------------
        if last_ckpt is None:
            last_ckpt = _save_student_checkpoint(
                student, checkpoint_root, step or 0,
            )

        # ---- 9. Provenance stamp ------------------------------------------
        _write_provenance(
            last_ckpt,
            teacher_dir=loaded.checkpoint_dir,
            policy_type=policy_type,
            steps=step,
            consistency_alpha=consistency_alpha,
        )

        # ---- 10. Fire on_end + return -------------------------------------
        ctx.hooks.run(
            "on_end", ctx, status="ok", steps_completed=step,
        )
        return CheckpointResult(
            final_checkpoint_path=last_ckpt,
            training_steps_completed=step,
            status="ok",
            intermediate_metrics={
                "policy_type": policy_type,
                "final_loss": loss_history[-1]["total"] if loss_history else None,
                "final_fm_loss": loss_history[-1]["flow_matching"] if loss_history else None,
                "final_consistency_loss": loss_history[-1]["consistency"] if loss_history else None,
                "consistency_alpha": consistency_alpha,
                "teacher_export": str(loaded.checkpoint_dir),
            },
        )


# ---------------------------------------------------------------------------
# Policy-specific velocity adapters
# ---------------------------------------------------------------------------

def _build_velocity_adapters(
    teacher: Any,
    student: Any,
    policy_type: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return `(teacher_velocity_fn, student_velocity_fn)` for policy_type.

    Each returned callable has signature:
        teacher_velocity_fn(x, t, **obs_kwargs) -> velocity tensor
        student_velocity_fn(x, t, target_time=None, **obs_kwargs) -> velocity tensor

    The student's `target_time` routes through the zero-init
    `target_time_embed_mlp` attached by
    `tether.distill.snapflow_pi0_model.enable_snapflow()`.

    Raises NotImplementedError for policy_types we haven't wired yet
    (SmolVLA v0.3.1; GR00T v0.5+).
    """
    if policy_type == "pi0":
        return _build_pi0_adapters(teacher, student)
    if policy_type == "pi05":
        return _build_pi05_adapters(teacher, student)
    raise NotImplementedError(
        f"No velocity adapter for policy_type={policy_type!r}. "
        f"pi0/pi05 are supported in v0.3; SmolVLA in v0.3.1; "
        f"GR00T in v0.5+."
    )


def _build_pi0_adapters(teacher: Any, student: Any):
    """Delegating alias for the pi0-specific adapter builder."""
    return _build_pi_family_adapters(teacher, student)


def _build_pi05_adapters(teacher: Any, student: Any):
    """Velocity adapters for pi0.5.

    pi0.5 differs from pi0 in three ways that matter here:
      1. ``embed_suffix`` signature is ``(noisy_actions, timestep)`` — no
         state. State isn't projected into the suffix embedding at all
         (pi0.5 relies on language + vision conditioning only).
      2. ``denoise_step`` signature is ``(prefix_pad_masks, past_key_values,
         x_t, timestep)`` — no state. SnapFlowPI05Pytorch override still
         accepts ``state=None`` kwarg for cross-family compatibility but
         it's unused.
      3. PI05Policy has no ``prepare_state`` method; the VLM prefix is
         computed from images + language only.

    The velocity-function interface returned here matches the pi0 path
    so the generic SnapFlow loss loop is paradigm-agnostic.
    """
    import torch
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    def _build_prefix_cache_pi05(policy, obs_kwargs):
        m = policy.model
        images, img_masks = policy._preprocess_images(obs_kwargs)
        lang_tokens = obs_kwargs[OBS_LANGUAGE_TOKENS]
        lang_masks = obs_kwargs[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        _, past_kv = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_kv, prefix_pad_masks

    # v0.5 state-out detection: when the student's class is the
    # state-out variant, its denoise_step requires state=<tensor>.
    def _is_state_out(policy):
        return type(policy.model).__name__ == "SnapFlowPI05StateOutPytorch"

    def _run_denoise_step_pi05(policy, x, t, obs_kwargs, target_time=None):
        action_dtype = policy.model.action_in_proj.weight.dtype
        past_kv, prefix_pad_masks = _build_prefix_cache_pi05(policy, obs_kwargs)
        if x.dtype != action_dtype:
            x = x.to(action_dtype)
        extra = {}
        if _is_state_out(policy):
            # State-out student: denoise_step requires explicit state.
            # The v0.5 preprocessor leaves OBS_STATE in the batch so it
            # reaches obs_kwargs here.
            from lerobot.utils.constants import OBS_STATE
            state_vec = obs_kwargs[OBS_STATE]
            # Pad to max_state_dim — LIBERO state is 8-dim but pi0.5
            # state_proj is sized to max_state_dim=32. Pi0's prepare_state
            # does this padding; pi0.5 didn't need it (state was in lang),
            # so we replicate inline here for the state-out variant.
            max_state_dim = policy.model.state_proj.in_features
            cur = state_vec.shape[-1]
            if cur < max_state_dim:
                import torch.nn.functional as _F
                state_vec = _F.pad(state_vec, (0, max_state_dim - cur))
            extra["state"] = state_vec
        return policy.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_kv,
            x_t=x,
            timestep=t,
            target_time=target_time,
            **extra,
        )

    def teacher_velocity_fn(x, t, **obs_kwargs):
        with torch.no_grad():
            return _run_denoise_step_pi05(teacher, x, t, obs_kwargs, target_time=None)

    def student_velocity_fn(x, t, target_time=None, **obs_kwargs):
        return _run_denoise_step_pi05(student, x, t, obs_kwargs, target_time=target_time)

    return teacher_velocity_fn, student_velocity_fn


def _build_pi_family_adapters(
    teacher: Any,
    student: Any,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Velocity adapters for pi0 / pi05.

    Both pi0 and pi05 wrap a PaliGemma VLM + a Gemma action expert.
    Velocity extraction mirrors lerobot's internal sampling loop
    (modeling_pi0.py:854-886):

      1. `policy._preprocess_images(batch)` → `(images, img_masks)`
      2. `policy.prepare_state(batch)` → state
      3. `policy.model.embed_prefix(...)` → prefix_embs + masks
      4. `policy.model.paligemma_with_expert.forward(use_cache=True)` →
         `past_key_values` (VLM prefix cache)
      5. Per-step: `policy.model.denoise_step(state, prefix_pad_masks,
         past_key_values, x_t, timestep)` → velocity

    We recompute the VLM prefix per velocity call for simplicity. The
    Euler-loop in lerobot caches the prefix once per action query, but
    during training each step uses a fresh batch so there's nothing to
    cache across steps anyway.

    ## target_time wiring (v0.3.1)

    The SnapFlow paper trick — a zero-init `target_time` embedding added
    to the time embedding inside `embed_suffix` — is now live via
    `tether.distill.snapflow_pi0_model.enable_snapflow()`. After that
    call, `student.model.embed_suffix(..., target_time=tensor)` adds
    the learned contribution before the action_time MLP fuses it with
    the action embedding.

    Loss components:
      - Flow-matching: student.denoise_step at random t (target_time=None)
        vs the analytic velocity `u_t = noise - action`.
      - Consistency: student.denoise_step at t with target_time=1 vs the
        teacher's 2-step Euler shortcut from x_t. This trains the student
        to produce the 2-step shortcut velocity in one forward pass at
        target_time=1, which is the SnapFlow 1-NFE inference mode.

    Zero-init on `target_time_embed_mlp.output` means student == teacher
    at init; the consistency loss drives the learned offset.
    """
    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    def _build_prefix_cache(policy, obs_kwargs):
        """Compute (past_kv, prefix_pad_masks, state) from a lerobot batch.

        obs_kwargs is the batch dict minus 'action' — the flattened
        observation/language/state that LeRobotDataset produces AFTER
        lerobot's preprocessor pipeline has run (so language is already
        tokenized into OBS_LANGUAGE_TOKENS + OBS_LANGUAGE_ATTENTION_MASK).
        """
        m = policy.model
        images, img_masks = policy._preprocess_images(obs_kwargs)
        lang_tokens = obs_kwargs[OBS_LANGUAGE_TOKENS]
        lang_masks = obs_kwargs[OBS_LANGUAGE_ATTENTION_MASK]
        state = policy.prepare_state(obs_kwargs)
        # Defensive re-pad: prepare_state SHOULD pad state to max_state_dim,
        # but observed in smoke runs that some preprocessor configs leave
        # the state untouched. Force-pad here so state_proj always sees
        # the expected max_state_dim.
        max_state_dim = getattr(policy.config, "max_state_dim", None)
        if max_state_dim is not None and state.shape[-1] < max_state_dim:
            state = F.pad(state, (0, max_state_dim - state.shape[-1]))
        # lerobot auto-casts state to fp32 if state_proj weights are fp32
        # (modeling_pi0.py:693) but doesn't handle the bf16-weights + fp32-state
        # direction. Cast state to match state_proj weight dtype here.
        state_dtype = m.state_proj.weight.dtype
        if state.dtype != state_dtype:
            state = state.to(state_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        # Mirror lerobot's sample_actions trick: force eager attention
        # before the prefix forward so bf16 models don't hit the SDPA
        # dtype-mismatch-on-bias error (modeling_pi0.py:844).
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        _, past_kv = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        # SnapFlowPI0Pytorch.denoise_step skips the `copy.deepcopy(past_kv)`
        # that the stock lerobot path does (modeling_pi0.py:918), so we no
        # longer need to detach past_kv tensors. Student path now keeps
        # past_kv graph-attached → VLM receives gradients.
        return past_kv, prefix_pad_masks, state

    def _run_denoise_step(policy, x, t, obs_kwargs, target_time=None):
        """Compute velocity via denoise_step.

        Requires ``enable_snapflow()`` to have been called on ``policy.model``
        (SnapFlowBackend does this at load time). That override removes the
        deepcopy + fp32-cast monkey-patches the stock lerobot path needed.
        """
        action_dtype = policy.model.action_in_proj.weight.dtype
        past_kv, prefix_pad_masks, state = _build_prefix_cache(policy, obs_kwargs)
        if x.dtype != action_dtype:
            x = x.to(action_dtype)
        return policy.model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_kv,
            x_t=x,
            timestep=t,
            target_time=target_time,
        )

    def teacher_velocity_fn(x, t, **obs_kwargs):
        with torch.no_grad():
            return _run_denoise_step(teacher, x, t, obs_kwargs, target_time=None)

    def student_velocity_fn(x, t, target_time=None, **obs_kwargs):
        return _run_denoise_step(student, x, t, obs_kwargs, target_time=target_time)

    return teacher_velocity_fn, student_velocity_fn


# ---------------------------------------------------------------------------
# Dataloader + batch prep
# ---------------------------------------------------------------------------

def _build_preprocessor(
    *,
    teacher_path: Path,
    image_key_map: dict | None,
    device: str,
):
    """Load the teacher's preprocessor pipeline with optional rename overrides.

    lerobot stores per-model preprocessor configs at
    `policy_preprocessor.json` alongside the weights. The pipeline does:
      1. rename_observations_processor  — maps dataset image keys to
         what the model expects (empty for `lerobot/pi0_base`; users
         must supply via image_key_map for real datasets)
      2. to_batch_processor             — tensor-dict normalization
      3. pi0_new_line_processor         — appends newline for language
      4. tokenizer_processor            — task string -> language_tokens
      5. device_processor               — moves to cuda
      6. normalizer_processor           — state/action normalization

    We override (1) with the caller's image_key_map and (5) with
    the run's device. Other steps are left as-shipped.
    """
    from lerobot.processor.pipeline import DataProcessorPipeline

    overrides: dict = {}
    if image_key_map:
        overrides["rename_observations_processor"] = {"rename_map": dict(image_key_map)}
    overrides.setdefault("device_processor", {})["device"] = device

    return DataProcessorPipeline.from_pretrained(
        str(teacher_path),
        config_filename="policy_preprocessor.json",
        overrides=overrides,
    )


def _build_dataloader(cfg, *, policy_type: str, chunk_size: int | None = None):
    """Build a LeRobotDataset dataloader for the distillation run.

    When ``chunk_size`` is given, configures ``delta_timestamps`` so the
    dataset yields real ``(B, chunk_size, action_dim)`` action chunks
    (i.e. the 50 future actions pi0 expects to predict in one shot).
    fps is read from LeRobotDatasetMetadata.

    Lazy-imports lerobot.datasets so CI without lerobot installed can
    still test the rest of the loop.
    """
    import torch
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )
    from lerobot.utils.constants import ACTION

    delta_timestamps = None
    if chunk_size is not None and chunk_size > 1:
        meta = LeRobotDatasetMetadata(cfg.dataset)
        fps = meta.fps
        delta_timestamps = {ACTION: [i / fps for i in range(chunk_size)]}
        logger.info(
            "[snapflow] delta_timestamps: ACTION chunk_size=%d @ fps=%d",
            chunk_size, fps,
        )

    dataset = LeRobotDataset(cfg.dataset, delta_timestamps=delta_timestamps)

    # Pro-tier base+customer mix per ADR 2026-04-25-self-distilling-serve-
    # architecture decision #2. When cfg.base_dataset is set, mix the two
    # datasets via ConcatDataset + WeightedRandomSampler so the student
    # adapts to customer data WITHOUT catastrophically forgetting the
    # base distribution. Default mix_ratio=0.5 (50/50). When unset,
    # behaves exactly as before (single-dataset shuffle loader).
    base_dataset_id = getattr(cfg, "base_dataset", None)
    mix_ratio = getattr(cfg, "mix_ratio", 0.5)
    if base_dataset_id:
        base_ds = LeRobotDataset(base_dataset_id, delta_timestamps=delta_timestamps)
        # Cross-validate critical fields so a mismatched base+customer pair
        # fails loud at preflight rather than producing garbage gradients.
        if hasattr(dataset, "features") and hasattr(base_ds, "features"):
            cust_action = dataset.features.get("action") if dataset.features else None
            base_action = base_ds.features.get("action") if base_ds.features else None
            if cust_action is not None and base_action is not None:
                if cust_action.get("shape") != base_action.get("shape"):
                    raise ValueError(
                        f"base_dataset action shape {base_action.get('shape')} "
                        f"does not match dataset action shape {cust_action.get('shape')}"
                    )
        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        combined = ConcatDataset([dataset, base_ds])
        # Weight per-sample so a random draw produces customer with prob
        # `mix_ratio` and base with prob `1 - mix_ratio`.
        n_cust = len(dataset)
        n_base = len(base_ds)
        # Avoid div-by-zero on empty datasets.
        weight_cust = (mix_ratio / max(1, n_cust))
        weight_base = ((1.0 - mix_ratio) / max(1, n_base))
        sample_weights = (
            [weight_cust] * n_cust + [weight_base] * n_base
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_cust + n_base,
            replacement=True,
        )
        loader = torch.utils.data.DataLoader(
            combined,
            batch_size=cfg.batch_size,
            sampler=sampler,  # mutually exclusive with shuffle=True
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        logger.info(
            "[snapflow] base+customer mix: dataset=%s (n=%d) + base_dataset=%s "
            "(n=%d), mix_ratio=%.2f",
            cfg.dataset, n_cust, base_dataset_id, n_base, mix_ratio,
        )
        return loader

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _prepare_batch(
    batch: dict,
    *,
    device: str,
    compute_dtype=None,
    max_action_dim: int | None = None,
    chunk_size: int | None = None,
) -> tuple:
    """Unpack a preprocessed LeRobotDataset batch into (action, noise, t, obs_kwargs).

    The batch has already been through the lerobot preprocessor pipeline,
    so tensors are on-device and language is tokenized. We split the
    action ground-truth out and leave everything else in obs_kwargs for
    the velocity fn.

    When `compute_dtype` is set (e.g. torch.bfloat16), action + noise
    are cast so the flow-matching chain runs in the model's dtype and
    avoids Linear-layer dtype mismatches downstream.

    When `max_action_dim` is set, the action is right-padded with zeros
    to that width so the flow-matching chain stays at the model's
    expected action dim (pi0/pi05 expect 32 even if the dataset is 7-dim).
    """
    import torch
    import torch.nn.functional as F

    action = batch["action"]
    if action.device.type != device.split(":")[0]:
        action = action.to(device)
    if compute_dtype is not None and action.dtype != compute_dtype:
        action = action.to(compute_dtype)
    # pi0/pi05 training expects (B, chunk_size, action_dim). The dataloader
    # configures delta_timestamps so LeRobotDataset yields a real chunk,
    # but older smoke paths / non-delta_timestamps datasets may still
    # deliver a single frame — fall back to tiling in that case.
    if chunk_size is not None and action.ndim == 2:
        logger.warning(
            "[snapflow] dataset delivered single-step action (ndim=2); "
            "tiling across chunk_size=%d — configure delta_timestamps "
            "on the dataset for a real action chunk.",
            chunk_size,
        )
        action = action.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()
    if max_action_dim is not None and action.shape[-1] < max_action_dim:
        action = F.pad(action, (0, max_action_dim - action.shape[-1]))
    batch_size = action.shape[0]
    noise = torch.randn_like(action)
    t = torch.rand(batch_size, device=device)
    if compute_dtype is not None:
        t = t.to(compute_dtype)

    obs_kwargs = {k: v for k, v in batch.items() if k != "action"}
    return action, noise, t, obs_kwargs


# ---------------------------------------------------------------------------
# Checkpoint save + provenance
# ---------------------------------------------------------------------------

def _save_student_checkpoint(
    student: Any,
    checkpoint_root: Path,
    step: int,
) -> Path:
    """Save the student under the standard lerobot pretrained_model/
    layout so `_auto_export` can consume it unchanged.

    Directory: <root>/<step:08d>/pretrained_model/
      ├── config.json           (canonical; written by save_pretrained)
      ├── model.safetensors     (student's full weights)
      └── distill_provenance.json  (added by _write_provenance)

    config.json carries the canonical lerobot `type` field; tether
    distill metadata lives in distill_provenance.json (NOT polluting
    config.json -- breaks PI05Config strict-decode at export time).
    """
    ckpt_dir = checkpoint_root / f"{step:08d}" / "pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained writes config.json + model.safetensors atomically.
    save_fn = getattr(student, "save_pretrained", None)
    if save_fn is None:
        raise AttributeError(
            f"student policy has no save_pretrained — SnapFlow expects a "
            f"HuggingFace-style model. Got {type(student).__name__}."
        )
    save_fn(str(ckpt_dir))

    # Tether distill metadata is written to distill_provenance.json
    # (see _write_provenance below), NOT to config.json. Polluting the
    # canonical lerobot/HF config.json with `_reflex_*` fields breaks
    # PI05Config / PI0Config strict-decode at load time -- caught by
    # 2026-04-25 distill smoke v3 + v5 (reflex_context experiments).
    # snapflow_pi0_model.py:_pick_policy_class still falls back to
    # `_tether_distill_teacher_type` for back-compat with old
    # checkpoints, but new SnapFlow checkpoints use the canonical
    # `type` field already written by save_pretrained.

    logger.info("[snapflow] checkpoint saved: %s", ckpt_dir)
    return ckpt_dir


def _write_provenance(
    ckpt_dir: Path,
    *,
    teacher_dir: Path,
    policy_type: str,
    steps: int,
    consistency_alpha: float,
) -> None:
    """Write a distill_provenance.json next to the checkpoint so
    VERIFICATION.md can reference where the teacher came from.

    Fields:
      - teacher_export: absolute path
      - policy_type: pi0 | pi05 | ...
      - steps: training steps completed
      - consistency_alpha: mix coef used
      - method: "snapflow"
    """
    prov = {
        "method": "snapflow",
        "policy_type": policy_type,
        "teacher_export": str(teacher_dir),
        "steps": steps,
        "consistency_alpha": consistency_alpha,
        "paper": "arxiv.org/abs/2604.05656",
    }
    (ckpt_dir / "distill_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )


__all__ = ["SnapFlowBackend"]
