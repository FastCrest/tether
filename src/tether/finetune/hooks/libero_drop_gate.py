"""LIBERO drop-gate hook — task-success kill-gate for distillation runs.

Attaches to the `on_postprocess` lifecycle hook. After the student has
been trained + exported + parity-validated, this hook:

  1. Runs the TEACHER on N LIBERO tasks (baseline task-success rate).
  2. Runs the STUDENT (1-NFE path via target_time=1) on the same tasks.
  3. Computes `drop_pp = teacher_success - student_success` (percent pts).
  4. If `drop_pp > gate_threshold`, sets `ctx.extra["force_abort"]`
     and `ctx.extra["abort_reason"]` so finalize() flips the result
     to 'aborted'.

## Why this hook exists

SnapFlow's value proposition is "1-step inference WITHOUT losing task
success." If the student's task-success drops too far below the
teacher, we've traded latency for capability — the student is strictly
worse. The drop-gate makes that trade-off explicit: ship only if the
student is within `gate_threshold` of the teacher.

Paper baseline: pi0.5 student at 98.75% vs teacher at 97.75% on LIBERO
(student was BETTER). v0.3 gate threshold: 5 pp — permissive for
early runs, tightens later.

## Scope (v0.3)

- Small N (8-16 tasks) so the gate is cheap enough to run on every
  distill output. Full LIBERO suite (40+ tasks) is a separate
  benchmark command.
- Runs LOCALLY with whatever hardware the orchestrator has. No Modal
  here — if Modal is needed, wire it into the CLI not the hook.
- If LIBERO infra is unavailable (no env, no GPU), logs a warning and
  skips the gate (doesn't abort). This is intentional: we don't want
  a missing optional dep to kill an otherwise-successful distill run.

## Configuration via ctx.config.extra_lerobot_args

- `libero_gate_threshold_pp: float = 5.0` — max permissible drop
- `libero_gate_tasks: int = 8` — how many tasks to eval
- `libero_gate_rollouts_per_task: int = 3` — episodes per task
- `libero_gate_skip: bool = False` — disable the gate entirely
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Defaults tuned for cheap-enough-to-always-run (<10 min on an A10).
# See architecture doc Section E for the rationale on these numbers.
DEFAULT_GATE_THRESHOLD_PP: float = 5.0
DEFAULT_NUM_TASKS: int = 8
DEFAULT_ROLLOUTS_PER_TASK: int = 3


def libero_drop_gate(ctx, **payload) -> None:
    """on_postprocess handler. Signature matches HookRegistry contract:
    `(ctx: TrainerContext, **payload) -> None`.

    Payload keys (provided by postprocess.finalize):
      - onnx_path: Path | None
      - final_checkpoint_path: Path
      - intermediate_metrics: dict (from CheckpointResult)
      - report: PostprocessReport (mutable — set libero_drop_pp)

    Side effect: may set ctx.extra["force_abort"] = True + reason.
    """
    cfg = ctx.config
    extra = cfg.extra_lerobot_args or {}

    if extra.get("libero_gate_skip"):
        logger.info("[libero_gate] skipped via extra_lerobot_args.libero_gate_skip")
        return

    # Only meaningful for distill runs — guard against accidental attach
    # to a fine-tune run where there's no teacher.
    if getattr(cfg, "phase", "train") != "distill":
        logger.debug("[libero_gate] phase != 'distill'; skipping")
        return

    threshold = float(extra.get("libero_gate_threshold_pp", DEFAULT_GATE_THRESHOLD_PP))
    num_tasks = int(extra.get("libero_gate_tasks", DEFAULT_NUM_TASKS))
    rollouts = int(extra.get("libero_gate_rollouts_per_task", DEFAULT_ROLLOUTS_PER_TASK))

    teacher_export = cfg.teacher_export
    student_ckpt = payload.get("final_checkpoint_path")
    if teacher_export is None or student_ckpt is None:
        logger.warning(
            "[libero_gate] missing teacher_export or student ckpt in payload; "
            "skipping (teacher=%r, student=%r)",
            teacher_export, student_ckpt,
        )
        return

    try:
        teacher_success, student_success = _run_teacher_student_rollouts(
            teacher_export=teacher_export,
            student_checkpoint=student_ckpt,
            num_tasks=num_tasks,
            rollouts_per_task=rollouts,
        )
    except _LiberoUnavailable as e:
        logger.warning(
            "[libero_gate] LIBERO infra unavailable (%s); skipping gate. "
            "Distill will ship without task-success verification.", e,
        )
        return
    except Exception as e:
        logger.exception("[libero_gate] rollouts crashed: %s", e)
        ctx.extra["force_abort"] = True
        ctx.extra["abort_reason"] = f"libero_gate crashed: {type(e).__name__}: {e}"
        return

    drop_pp = (teacher_success - student_success) * 100.0
    logger.info(
        "[libero_gate] teacher=%.2f%% student=%.2f%% drop=%.2f pp (threshold=%.2f pp)",
        teacher_success * 100, student_success * 100, drop_pp, threshold,
    )

    # Surface the number into the PostprocessReport regardless of pass/fail.
    report = payload.get("report")
    if report is not None:
        report.libero_drop_pp = drop_pp

    if drop_pp > threshold:
        ctx.extra["force_abort"] = True
        ctx.extra["abort_reason"] = (
            f"LIBERO drop {drop_pp:.2f}pp exceeds gate threshold {threshold:.2f}pp. "
            f"Teacher={teacher_success * 100:.2f}%, "
            f"student={student_success * 100:.2f}% on {num_tasks} tasks × "
            f"{rollouts} rollouts. Distill checkpoint kept on disk; "
            f"export artifact is NOT shipped."
        )
    else:
        logger.info(
            "[libero_gate] PASS: drop %.2fpp <= threshold %.2fpp",
            drop_pp, threshold,
        )


# ---------------------------------------------------------------------------
# Rollout execution
# ---------------------------------------------------------------------------

class _LiberoUnavailable(RuntimeError):
    """Raised when the LIBERO harness can't run (missing sim, no GPU, etc.).
    The gate silently skips on this rather than aborting."""


def _run_teacher_student_rollouts(
    *,
    teacher_export: str,
    student_checkpoint: Any,
    num_tasks: int,
    rollouts_per_task: int,
) -> tuple[float, float]:
    """Run rollouts for teacher + student on LIBERO tasks.

    Returns (teacher_success_rate, student_success_rate) as floats in [0, 1].

    Lazy-imports the LIBERO harness so a CI run that never touches the
    gate doesn't fail on missing libero installs. Raises _LiberoUnavailable
    if the harness isn't installed.
    """
    try:
        # tether.safety.libero or a libero helper module — lazy imported.
        from tether.libero_harness import run_task_suite  # type: ignore[import]
    except ImportError as e:
        raise _LiberoUnavailable(f"libero_harness not importable: {e}")

    task_ids = list(range(num_tasks))
    teacher_rate = run_task_suite(
        policy_ref=teacher_export,
        tasks=task_ids,
        rollouts=rollouts_per_task,
        use_one_step=False,   # teacher uses full Euler loop
    )
    student_rate = run_task_suite(
        policy_ref=str(student_checkpoint),
        tasks=task_ids,
        rollouts=rollouts_per_task,
        use_one_step=True,    # student uses 1-NFE path (target_time=1)
    )
    return teacher_rate, student_rate


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def attach_to(hooks, *, threshold_pp: float | None = None) -> None:
    """Convenience: register `libero_drop_gate` on a HookRegistry.

    Called by the CLI when wiring up a distill run. A threshold override
    here takes precedence over cfg.extra_lerobot_args.libero_gate_threshold_pp
    ONLY if set via this helper (keeps CLI flag > config).
    """
    if threshold_pp is not None:
        def handler(ctx, **payload):
            cfg_extra = (ctx.config.extra_lerobot_args or {})
            cfg_extra["libero_gate_threshold_pp"] = threshold_pp
            ctx.config.extra_lerobot_args = cfg_extra
            return libero_drop_gate(ctx, **payload)
        hooks.register("on_postprocess", handler)
    else:
        hooks.register("on_postprocess", libero_drop_gate)


__all__ = ["libero_drop_gate", "attach_to"]
