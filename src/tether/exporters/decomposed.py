"""Decomposed pi0.5 ONNX export — vlm_prefix.onnx + expert_denoise.onnx.

Design doc: `reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md`.

## What this module does

Exports pi0.5 (optionally a SnapFlow-distilled student) as TWO ONNX
graphs instead of one:

1. ``vlm_prefix.onnx`` — takes the vision + language inputs, returns a
   flat tuple of 36 ``past_k_i`` / ``past_v_i`` tensors (18 paligemma
   layers × 2) plus ``prefix_pad_masks``. Runs the full VLM forward
   pass once per observation.

2. ``expert_denoise.onnx`` — takes the 36 past_kv tensors +
   ``prefix_pad_masks`` + ``state`` (pi05 has no state input actually —
   see note below) + ``noise``, runs the action-expert denoising Euler
   loop (1 step for distilled students with ``target_time=1``, 10 steps
   for the teacher), returns the action chunk.

Serve layer (see ``tether.runtime.pi05_decomposed_server``) hashes the
VLM inputs per-call and reuses the last ``past_kv`` output when the
observation hasn't meaningfully changed — the 3–4× deployment speedup
described in the design doc.

## Structural notes

- pi0.5 has no ``state`` input at the Policy level; state is tokenized
  into the language prompt upstream. The expert-denoise wrapper here
  still has no ``state`` input.
- paligemma layer count: 18. kv_heads=1, head_dim=256. Each K or V
  tensor is ``(B, 1, seq_len, 256)``.
- seq_len is dynamic per observation (tokenized-language length +
  vision patches) but bounded by the preprocessor's config.
- The expert stack's attention reads the paligemma past_kv as
  "past tokens" — this is the cross-attention pattern that makes
  caching work.

## Dependencies

Same as ``monolithic`` extra: transformers==5.3.0, lerobot==0.5.1,
onnx-diagnostic>=0.9. ``apply_export_patches`` from ``monolithic.py``
is invoked here to get the shared denoise-step + cache-freeze patches.

## What this module does NOT do

- Does NOT implement the serve-layer cache (that's
  ``tether.runtime.pi05_decomposed_server``).
- Does NOT implement the perceptual-hash obs-matcher (same server).
- Does NOT handle SmolVLA — that already has a decomposed path in
  ``tether.runtime.vlm_orchestrator``.
- Does NOT handle pi0 (the non-.5 variant). pi0 decomposed export is
  a separate future goal (pi0 has state_proj, different suffix shape).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from tether.exporters._export_mode import (
    ExportMode,
    estimate_model_vram_from_onnx,
    log_decision,
    select_mode,
)

logger = logging.getLogger(__name__)

# pi0.5 constants — pulled from get_gemma_config("gemma_2b") in
# lerobot/policies/pi05/modeling_pi05.py line 322-330.
PI05_PALIGEMMA_LAYERS: int = 18  # pi05 paligemma has 18 transformer layers
PI05_KV_HEADS: int = 1
PI05_HEAD_DIM: int = 256
_PI05_BATCH_SIZE: int = 1
_PI05_IMAGE_SIZE: int = 224
_PI05_LANG_TOKENS: int = 200
_PI05_VISION_PATCHES_PER_VIEW: int = 256

# Conservative pre-export estimate used by --export-mode auto. The decomposed
# pi0.5 path has historically produced roughly 13 GiB of ONNX/external data.
# Using that estimate keeps auto on the sequential baseline for most hardware,
# while still allowing explicit or truly high-VRAM parallel runs.
_PI05_ESTIMATED_ONNX_BYTES: int = int(13.0 * 1024 ** 3)


def export_pi05_decomposed(
    model_id: str,
    output_dir: str | Path,
    *,
    num_steps: int = 1,
    target: str = "desktop",
    student_checkpoint: str | Path | None = None,
    variant: str = "default",
    export_mode: ExportMode | str = ExportMode.AUTO,
    per_step_expert: bool = False,
) -> dict[str, Any]:
    """Export pi0.5 as ``vlm_prefix.onnx`` + ``expert_denoise.onnx``.

    Args:
        model_id: HF repo id for the pi0.5 base/variant (e.g.
            ``"lerobot/pi05_libero_finetuned_v044"``). Ignored if
            ``student_checkpoint`` is provided.
        output_dir: where to write the two ONNX files + tether_config.json.
        num_steps: denoising steps baked into expert_denoise.onnx.
            1 for distilled students (target_time=1 path); 10 for the
            canonical teacher.
            When ``per_step_expert=True`` this only configures runtime
            metadata; the export itself is single-step regardless.
        target: target hardware profile; passed through to tether_config.
        student_checkpoint: optional path to a SnapFlow-distilled
            checkpoint dir. When set, loads via ``load_snapflow_student``
            and enables the ``target_time_embed_mlp`` path. Must use
            ``num_steps=1`` in this mode.
        export_mode: ``auto`` selects parallel only when the VRAM probe says
            two independent policy loads fit. ``parallel`` fails loudly if
            the probe says it will not fit. ``sequential`` preserves the
            historical single-process export path.
        per_step_expert: when True, export the expert as a single-Euler-step
            graph ``(x_t, t, past_kv) → v_t`` instead of unrolling the loop
            into the ONNX. The runtime drives the Euler loop in Python,
            unlocking RTC's per-step guidance hook. Default False preserves
            the baked-loop shape that's been the customer-facing default
            since the decomposed export shipped. Spec:
            ``reflex_context/features/03_export/per-step-expert-export.md``.

    Returns dict with paths + byte sizes + sanity metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_mode = ExportMode(export_mode)
    estimated_vram = estimate_model_vram_from_onnx(_PI05_ESTIMATED_ONNX_BYTES)
    decision = select_mode(requested_mode, estimated_vram)
    log_decision(decision)

    if decision.mode == ExportMode.PARALLEL:
        return _export_pi05_decomposed_parallel(
            model_id=model_id,
            output_dir=output_dir,
            num_steps=num_steps,
            target=target,
            student_checkpoint=student_checkpoint,
            variant=variant,
            export_mode_reason=decision.reason,
            per_step_expert=per_step_expert,
        )

    return _export_pi05_decomposed_sequential(
        model_id=model_id,
        output_dir=output_dir,
        num_steps=num_steps,
        target=target,
        student_checkpoint=student_checkpoint,
        variant=variant,
        export_mode_reason=decision.reason,
        per_step_expert=per_step_expert,
    )


def _export_pi05_decomposed_sequential(
    *,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    target: str,
    student_checkpoint: str | Path | None,
    variant: str,
    export_mode_reason: str,
    per_step_expert: bool = False,
) -> dict[str, Any]:
    """Historical pi0.5 decomposed export path: one policy load, two passes."""
    policy = _load_pi05_policy(model_id, num_steps, student_checkpoint, variant)
    past_kv_names = _past_kv_names()
    prefix_seq_len = _prefix_seq_len()

    prefix_meta = _export_pi05_prefix_pass(policy, output_dir, past_kv_names)

    # Free the prefix wrapper before building the expert — on A100-80GB we
    # OOM'd with both loaded + a second prefix forward for dummy inputs.
    import gc
    gc.collect()

    expert_meta = _export_pi05_expert_pass(
        policy=policy,
        output_dir=output_dir,
        num_steps=num_steps,
        variant=variant,
        per_step_expert=per_step_expert,
        past_kv_names=past_kv_names,
        prefix_seq_len=prefix_seq_len,
    )
    _assert_matching_export_metadata(prefix_meta, expert_meta)

    return _write_decomposed_export_result(
        model_id=model_id,
        output_dir=output_dir,
        num_steps=num_steps,
        target=target,
        student_checkpoint=student_checkpoint,
        variant=variant,
        past_kv_names=past_kv_names,
        chunk_size=int(prefix_meta["chunk_size"]),
        action_dim=int(prefix_meta["action_dim"]),
        export_mode=ExportMode.SEQUENTIAL,
        export_mode_reason=export_mode_reason,
        per_step_expert=per_step_expert,
    )


def _export_pi05_decomposed_parallel(
    *,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    target: str,
    student_checkpoint: str | Path | None,
    variant: str,
    export_mode_reason: str,
    per_step_expert: bool = False,
) -> dict[str, Any]:
    """Run prefix and expert export in separate spawned processes."""
    past_kv_names = _past_kv_names()
    prefix_seq_len = _prefix_seq_len()
    prefix_meta, expert_meta = _run_parallel_pi05_exports(
        model_id=model_id,
        output_dir=output_dir,
        num_steps=num_steps,
        student_checkpoint=student_checkpoint,
        variant=variant,
        past_kv_names=past_kv_names,
        prefix_seq_len=prefix_seq_len,
        per_step_expert=per_step_expert,
    )
    _assert_matching_export_metadata(prefix_meta, expert_meta)

    return _write_decomposed_export_result(
        model_id=model_id,
        output_dir=output_dir,
        num_steps=num_steps,
        target=target,
        student_checkpoint=student_checkpoint,
        variant=variant,
        past_kv_names=past_kv_names,
        chunk_size=int(prefix_meta["chunk_size"]),
        action_dim=int(prefix_meta["action_dim"]),
        export_mode=ExportMode.PARALLEL,
        export_mode_reason=export_mode_reason,
        per_step_expert=per_step_expert,
    )


def _run_parallel_pi05_exports(
    *,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    student_checkpoint: str | Path | None,
    variant: str,
    past_kv_names: list[str],
    prefix_seq_len: int,
    per_step_expert: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Submit the two independent export passes to spawned worker processes."""
    ctx = mp.get_context("spawn")
    student = str(student_checkpoint) if student_checkpoint is not None else None
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        prefix_future = pool.submit(
            _export_pi05_prefix_worker,
            model_id,
            str(output_dir),
            num_steps,
            student,
            variant,
            past_kv_names,
        )
        expert_future = pool.submit(
            _export_pi05_expert_worker,
            model_id,
            str(output_dir),
            num_steps,
            student,
            variant,
            past_kv_names,
            prefix_seq_len,
            per_step_expert,
        )
        return prefix_future.result(), expert_future.result()


def _export_pi05_prefix_worker(
    model_id: str,
    output_dir: str,
    num_steps: int,
    student_checkpoint: str | None,
    variant: str,
    past_kv_names: list[str],
) -> dict[str, Any]:
    policy = _load_pi05_policy(model_id, num_steps, student_checkpoint, variant)
    return _export_pi05_prefix_pass(policy, Path(output_dir), past_kv_names)


def _export_pi05_expert_worker(
    model_id: str,
    output_dir: str,
    num_steps: int,
    student_checkpoint: str | None,
    variant: str,
    past_kv_names: list[str],
    prefix_seq_len: int,
    per_step_expert: bool = False,
) -> dict[str, Any]:
    policy = _load_pi05_policy(model_id, num_steps, student_checkpoint, variant)
    return _export_pi05_expert_pass(
        policy=policy,
        output_dir=Path(output_dir),
        num_steps=num_steps,
        variant=variant,
        per_step_expert=per_step_expert,
        past_kv_names=past_kv_names,
        prefix_seq_len=prefix_seq_len,
    )


def _load_pi05_policy(
    model_id: str,
    num_steps: int,
    student_checkpoint: str | Path | None,
    variant: str,
):
    """Load and patch the pi0.5 policy exactly once for one export process."""
    _require_decomposed_deps()

    import torch

    from tether.exporters.monolithic import (
        apply_export_patches,
        _force_eager_attn,
        _apply_pi05_denoise_step_patch,
    )

    apply_export_patches()

    # Student path handles target_time_embed_mlp weights; base path is a plain
    # PI05Policy. This function is process-local by design for parallel mode.
    t0 = time.time()
    if student_checkpoint is not None:
        if num_steps != 1:
            raise ValueError(
                f"student_checkpoint requires num_steps=1 (SnapFlow "
                f"distilled students use a single 1-NFE denoise call); "
                f"got num_steps={num_steps}"
            )
        from tether.distill.snapflow_pi0_model import load_snapflow_student
        logger.info("[decomposed] Loading SnapFlow student from %s", student_checkpoint)
        policy = load_snapflow_student(student_checkpoint)
        if variant == "state_out":
            # The v0.5 student needs the state-out class swap + state_proj
            # registered. load_snapflow_student installed default
            # SnapFlowPI05Pytorch; reset class then upgrade.
            from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
            from tether.distill.snapflow_pi0_model import enable_snapflow_state_out
            policy.model.__class__ = PI05Pytorch
            enable_snapflow_state_out(policy.model)
            logger.info("[decomposed] enabled state-out variant on student")
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        logger.info("[decomposed] Loading %s", model_id)
        # Apply the same torch.compile suppression as monolithic export
        # — LIBERO-finetuned pi0.5 configs set compile_model=True.
        _orig_compile = torch.compile
        torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))
        try:
            policy = PI05Policy.from_pretrained(model_id)
        finally:
            torch.compile = _orig_compile

    policy.eval().to("cpu").to(torch.float32)
    _gc_disable = getattr(policy.model, "gradient_checkpointing_disable", None)
    if callable(_gc_disable):
        _gc_disable()
    _force_eager_attn(policy.model)
    _apply_pi05_denoise_step_patch()  # reuses monolithic's F.pad mask fix
    logger.info("[decomposed] Loaded in %.1fs", time.time() - t0)
    return policy


def _export_pi05_prefix_pass(
    policy,
    output_dir: Path,
    past_kv_names: list[str],
) -> dict[str, Any]:
    import torch
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    from tether.exporters.monolithic import _fix_onnx_where_dtype_mismatches

    cfg = policy.config
    B = _PI05_BATCH_SIZE
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    prefix_wrapper = Pi05PrefixWrapper(policy.model).eval()

    prefix_dummy = dict(
        img_base=torch.randn(B, 3, _PI05_IMAGE_SIZE, _PI05_IMAGE_SIZE, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, _PI05_IMAGE_SIZE, _PI05_IMAGE_SIZE, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, _PI05_IMAGE_SIZE, _PI05_IMAGE_SIZE, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        # Preprocessor pads pi0.5 lang prompts to 200 tokens at runtime;
        # match so the exported ONNX doesn't ARG-fail when LIBERO feeds
        # 200-token prompts. 3*256 vision + 200 lang = 968 prefix_seq_len.
        lang_tokens=torch.randint(0, 257152, (B, _PI05_LANG_TOKENS), dtype=torch.long),
        lang_masks=torch.ones(B, _PI05_LANG_TOKENS, dtype=torch.bool),
    )

    prefix_output_names = past_kv_names + ["prefix_pad_masks"]

    prefix_path = output_dir / "vlm_prefix.onnx"
    logger.info("[decomposed] Exporting prefix → %s", prefix_path)
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep_prefix = torch.export.export(
            prefix_wrapper, tuple(prefix_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[decomposed] prefix torch.export: %.1fs", time.time() - t0)

    t0 = time.time()
    torch.onnx.export(
        ep_prefix, tuple(prefix_dummy.values()), str(prefix_path),
        input_names=list(prefix_dummy.keys()),
        output_names=prefix_output_names,
        opset_version=19,
    )
    logger.info("[decomposed] prefix ONNX conversion: %.1fs", time.time() - t0)

    prefix_fixes = _fix_onnx_where_dtype_mismatches(prefix_path)
    logger.info("[decomposed] prefix Cast fixes: %d", prefix_fixes)

    del prefix_wrapper, ep_prefix
    return {
        "chunk_size": int(chunk),
        "action_dim": int(action_dim),
        "prefix_seq_len": _prefix_seq_len(),
        "prefix_cast_fixes": int(prefix_fixes),
    }


def _export_pi05_expert_pass(
    *,
    policy,
    output_dir: Path,
    num_steps: int,
    variant: str,
    past_kv_names: list[str],
    prefix_seq_len: int,
    per_step_expert: bool = False,
) -> dict[str, Any]:
    """Export the expert ONNX graph.

    Two shapes:
    - **baked-loop** (``per_step_expert=False``, default): single ONNX call
      runs ``num_steps`` Euler iterations internally, returning fully-denoised
      actions. Input ``noise``, output ``actions``.
    - **per-step** (``per_step_expert=True``): single ONNX call runs ONE Euler
      step, returning velocity ``v_t``. Caller drives the Euler loop in Python.
      Input ``x_t`` + ``t``, output ``v_t``. Spec:
      ``reflex_context/features/03_export/per-step-expert-export.md``.
    """
    import torch
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    from tether.exporters.monolithic import _fix_onnx_where_dtype_mismatches

    cfg = policy.config
    B = _PI05_BATCH_SIZE
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    # pi05 prefix_seq_len must match exactly what vlm_prefix.onnx will
    # emit at runtime. With lang_tokens=(B,200) dummies above and 3 vision
    # views × 256 patches each, the natural prefix seq_len is
    # 3×256 + 200 = 968. Both ONNX graphs are shape-specialized to their
    # export-time seq_len so they MUST match or attention will compute
    # on zero-padded tail positions. (For production with longer lang
    # prompts, both graphs would need dynamic seq_len via
    # torch.export Dim — deferred until parity is green.)
    past_kv_shape = (B, PI05_KV_HEADS, prefix_seq_len, PI05_HEAD_DIM)
    past_kv_dummies = [
        torch.randn(past_kv_shape, dtype=torch.float32)
        for _ in range(PI05_PALIGEMMA_LAYERS * 2)
    ]
    prefix_pad_masks_dummy = torch.ones(B, prefix_seq_len, dtype=torch.bool)

    if per_step_expert:
        expert_wrapper = Pi05ExpertPerStepWrapper(policy.model).eval()
    else:
        expert_wrapper = Pi05ExpertWrapper(policy.model, num_steps).eval()

    expert_dummy = {}
    for idx, t in enumerate(past_kv_dummies):
        expert_dummy[past_kv_names[idx]] = t
    expert_dummy["prefix_pad_masks"] = prefix_pad_masks_dummy

    if per_step_expert:
        # Per-step shape: input is (x_t, t) instead of noise; output is v_t.
        expert_dummy["x_t"] = torch.randn(B, chunk, action_dim, dtype=torch.float32)
        # Scalar timestep, broadcast to (B,). Use t=1.0 (start of Euler trajectory).
        expert_dummy["t"] = torch.full((B,), 1.0, dtype=torch.float32)
        output_names = ["v_t"]
    else:
        expert_dummy["noise"] = torch.randn(B, chunk, action_dim, dtype=torch.float32)
        output_names = ["actions"]

    if variant == "state_out":
        # Add state input to the expert ONNX graph (matches the runtime
        # signature of SnapFlowPI05StateOutPytorch.denoise_step).
        state_dim = policy.model.state_proj.in_features
        expert_dummy["state"] = torch.randn(B, state_dim, dtype=torch.float32)

    expert_path = output_dir / "expert_denoise.onnx"
    shape_label = "per-step" if per_step_expert else f"baked num_steps={num_steps}"
    logger.info(
        "[decomposed] Exporting expert (%s) → %s", shape_label, expert_path,
    )
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep_expert = torch.export.export(
            expert_wrapper, tuple(expert_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[decomposed] expert torch.export: %.1fs", time.time() - t0)

    t0 = time.time()
    # optimize=False disables torch.onnx.export's constant-folding pass for
    # the expert graph. The default optimize=True folds the float64 sin/cos
    # of `create_sinusoidal_pos_embedding(timestep, dim, ...)` in
    # FP32-precision arithmetic, producing a `_to_copy` constant that
    # differs from a true float64 compute by ~3e-5 max_abs. Without this
    # fix, baked num_steps=10 carries that ~3e-5 time-emb drift through the
    # entire expert stack and the per-step ONNX (which keeps Sin/Cos
    # dynamic) cannot match it (cell 1 measured cos=0.998 / max_abs=0.24
    # vs gate's cos≥0.99999 / max_abs≤1e-5). With optimize=False both
    # baked and per-step compute time embedding via runtime Sin/Cos in
    # float64, so the precision pathway is symmetric. Marginal node-count
    # increase, immaterial runtime cost (sin/cos << gemm).
    # Reproduced + isolated 2026-04-30; see
    # 03_experiments/2026-04-30-per-step-parity-modal-a100.md.
    torch.onnx.export(
        ep_expert, tuple(expert_dummy.values()), str(expert_path),
        input_names=list(expert_dummy.keys()),
        output_names=output_names,
        opset_version=19,
        optimize=False,
    )
    logger.info("[decomposed] expert ONNX conversion: %.1fs", time.time() - t0)

    expert_fixes = _fix_onnx_where_dtype_mismatches(expert_path)
    logger.info("[decomposed] expert Cast fixes: %d", expert_fixes)

    del expert_wrapper, ep_expert
    return {
        "chunk_size": int(chunk),
        "action_dim": int(action_dim),
        "prefix_seq_len": int(prefix_seq_len),
        "expert_cast_fixes": int(expert_fixes),
        "per_step_expert": bool(per_step_expert),
    }


def _write_decomposed_export_result(
    *,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    target: str,
    student_checkpoint: str | Path | None,
    variant: str,
    past_kv_names: list[str],
    chunk_size: int,
    action_dim: int,
    export_mode: ExportMode,
    export_mode_reason: str,
    per_step_expert: bool = False,
) -> dict[str, Any]:
    prefix_path = output_dir / "vlm_prefix.onnx"
    expert_path = output_dir / "expert_denoise.onnx"

    tether_cfg = {
        "model_id": model_id if student_checkpoint is None else str(student_checkpoint),
        "model_type": "pi05_decomposed_student" if student_checkpoint else "pi05_decomposed",
        "target": target,
        "num_denoising_steps": num_steps,
        "chunk_size": chunk_size,
        "action_chunk_size": chunk_size,
        "action_dim": action_dim,
        "opset": 19,
        "export_kind": "decomposed",
        "export_mode": export_mode.value,
        "export_mode_reason": export_mode_reason,
        "decomposed": {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "paligemma_layers": PI05_PALIGEMMA_LAYERS,
            "kv_heads": PI05_KV_HEADS,
            "head_dim": PI05_HEAD_DIM,
            "past_kv_tensor_names": past_kv_names,
            "variant": variant,
            "expert_takes_state": variant == "state_out",
            # When True, expert_denoise.onnx takes (x_t, t, past_kv) → v_t
            # and the runtime drives the Euler loop in Python. Default False
            # preserves the baked-loop shape (noise → actions, single call).
            # Spec: features/03_export/per-step-expert-export.md.
            "per_step_expert": bool(per_step_expert),
        },
    }
    (output_dir / "tether_config.json").write_text(json.dumps(tether_cfg, indent=2))
    try:
        from tether.verification_report import write_verification_report
        write_verification_report(output_dir, parity=None)
    except Exception:
        pass

    size_prefix = prefix_path.stat().st_size / 1e6
    size_expert = expert_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data"))
    external_mb = sum(f.stat().st_size for f in data_files) / 1e6

    return {
        "status": "ok",
        "vlm_prefix_onnx": str(prefix_path),
        "expert_denoise_onnx": str(expert_path),
        "vlm_prefix_mb": size_prefix,
        "expert_denoise_mb": size_expert,
        "external_data_mb": external_mb,
        "total_mb": size_prefix + size_expert + external_mb,
        "num_steps": num_steps,
        "paligemma_layers": PI05_PALIGEMMA_LAYERS,
        "export_mode": export_mode.value,
    }


def _past_kv_names() -> list[str]:
    names = []
    for layer_idx in range(PI05_PALIGEMMA_LAYERS):
        names.append(f"past_k_{layer_idx}")
        names.append(f"past_v_{layer_idx}")
    return names


def _prefix_seq_len() -> int:
    return 3 * _PI05_VISION_PATCHES_PER_VIEW + _PI05_LANG_TOKENS


def _assert_matching_export_metadata(
    prefix_meta: dict[str, Any],
    expert_meta: dict[str, Any],
) -> None:
    for key in ("chunk_size", "action_dim", "prefix_seq_len"):
        if int(prefix_meta[key]) != int(expert_meta[key]):
            raise RuntimeError(
                f"decomposed export metadata mismatch for {key}: "
                f"prefix={prefix_meta[key]!r}, expert={expert_meta[key]!r}"
            )


class Pi05PrefixWrapper:
    """VLM prefix wrapper for pi0.5. Runs the paligemma forward pass
    and returns a flat tuple of (past_k_0, past_v_0, ..., past_k_17,
    past_v_17, prefix_pad_masks).

    Defined as a lazy class built on first instantiation so we don't
    force lerobot import at module load. See ``_build_prefix_class``.
    """
    def __new__(cls, pi05_model):
        impl = _build_prefix_class()
        return impl(pi05_model)


class Pi05ExpertWrapper:
    """Expert denoising wrapper for pi0.5. Takes 36 flat past_kv tensors
    + prefix_pad_masks + noise, runs the Euler loop (num_steps iterations;
    1 for SnapFlow students with target_time=1), returns actions.
    """
    def __new__(cls, pi05_model, num_steps):
        impl = _build_expert_class()
        return impl(pi05_model, num_steps)


class Pi05ExpertPerStepWrapper:
    """Per-step expert wrapper for pi0.5. Takes 36 flat past_kv tensors +
    prefix_pad_masks + x_t + t (and optional state). Returns v_t — the
    velocity from ONE denoise step, not the fully-denoised actions.

    The runtime drives the Euler loop in Python:
        for i in range(num_steps):
            t = 1.0 + i × dt
            v_t = expert_per_step(x_t, t, past_kv, ...)
            x_t = x_t + v_t × dt

    This unlocks RTC's per-step guidance hook (lerobot RTCProcessor.denoise_step)
    + future per-step caching patterns (Dexmal D.8, sub-frame interpolation C.5).

    See features/03_export/per-step-expert-export.md for the spec.
    Used when ``export_pi05_decomposed(..., per_step_expert=True)``.
    """
    def __new__(cls, pi05_model):
        impl = _build_expert_per_step_class()
        return impl(pi05_model)


_PREFIX_CLASS: Any = None
_EXPERT_CLASS: Any = None
_EXPERT_PER_STEP_CLASS: Any = None


def _build_prefix_class():
    global _PREFIX_CLASS
    if _PREFIX_CLASS is not None:
        return _PREFIX_CLASS

    import torch
    import torch.nn as nn
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    class _Pi05PrefixWrapper(nn.Module):
        def __init__(self, pi05_model):
            super().__init__()
            self.model = pi05_model

        def forward(
            self,
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
        ):
            images = [img_base, img_wrist_l, img_wrist_r]
            img_masks = [mask_base, mask_wrist_l, mask_wrist_r]

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d)
            self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            # Flatten DynamicCache → tuple of tensors. transformers 5.3
            # DynamicCache has a `.layers[i]` list of `DynamicLayer`
            # objects, each exposing `.keys` and `.values` tensors. The
            # onnx-diagnostic torch_export_patches context may strip
            # to_legacy_cache off the class, so we pull the tensors
            # directly from the per-layer objects.
            flat: list = []
            for layer_idx in range(PI05_PALIGEMMA_LAYERS):
                layer = past_key_values.layers[layer_idx]
                flat.append(layer.keys)
                flat.append(layer.values)
            flat.append(prefix_pad_masks)
            return tuple(flat)

    _PREFIX_CLASS = _Pi05PrefixWrapper
    return _PREFIX_CLASS


def _build_expert_class():
    global _EXPERT_CLASS
    if _EXPERT_CLASS is not None:
        return _EXPERT_CLASS

    import torch
    import torch.nn as nn

    # Build the list of past_k_i / past_v_i kwarg names that the
    # wrapper's forward will accept. Needed so torch.export can trace
    # the signature.
    kv_param_names = []
    for layer_idx in range(PI05_PALIGEMMA_LAYERS):
        kv_param_names.append(f"past_k_{layer_idx}")
        kv_param_names.append(f"past_v_{layer_idx}")

    class _Pi05ExpertWrapper(nn.Module):
        """Runs the Euler denoise loop using a past_kv reconstructed
        from flat tensor inputs. ``num_steps`` is baked in at init.

        v0.5 state-out: when ``pi05_model`` has a ``state_proj`` module
        (i.e. it's a SnapFlowPI05StateOutPytorch), the wrapper expects
        an additional ``state`` tensor at the END of args, and threads it
        through denoise_step. Default variant keeps the original signature.
        """

        def __init__(self, pi05_model, num_steps):
            super().__init__()
            self.model = pi05_model
            self.n_steps = num_steps
            # SnapFlow student path: if the model has target_time_embed_mlp,
            # we pass target_time=1 to denoise_step. Otherwise plain teacher.
            self._is_snapflow = hasattr(pi05_model, "target_time_embed_mlp")
            # v0.5 state-out path: model has explicit state_proj layer.
            self._is_state_out = hasattr(pi05_model, "state_proj")

        def forward(self, *args):
            # args layout (default): 36 past_kv tensors + prefix_pad_masks + noise.
            # args layout (state_out): same + state tensor (last position).
            past_flat = args[:PI05_PALIGEMMA_LAYERS * 2]
            prefix_pad_masks = args[PI05_PALIGEMMA_LAYERS * 2]
            noise = args[PI05_PALIGEMMA_LAYERS * 2 + 1]
            state = args[PI05_PALIGEMMA_LAYERS * 2 + 2] if self._is_state_out else None

            # Reconstruct a proper DynamicCache by populating per-layer
            # via .update() — transformers 5.3 removed from_legacy_cache
            # as a classmethod on DynamicCache. update() appends to an
            # empty cache layer so the first call with a given layer_idx
            # initializes that layer's K/V. pi_gemma forward needs a
            # real DynamicCache (isinstance check) so we can't pass a
            # shim.
            from transformers.cache_utils import DynamicCache
            past_kv = DynamicCache()
            for i in range(PI05_PALIGEMMA_LAYERS):
                past_kv.update(
                    key_states=past_flat[2 * i],
                    value_states=past_flat[2 * i + 1],
                    layer_idx=i,
                    cache_kwargs=None,
                )

            action_dtype = self.model.action_in_proj.weight.dtype
            if noise.dtype != action_dtype:
                noise = noise.to(action_dtype)

            dt = -1.0 / self.n_steps
            x_t = noise
            for step in range(self.n_steps):
                time_val = 1.0 + step * dt
                time_tensor = torch.full(
                    (x_t.shape[0],), time_val,
                    dtype=torch.float32, device=x_t.device,
                )
                # State-out variant: pass state= to denoise_step. Default
                # path leaves it unset.
                state_kw = {"state": state} if self._is_state_out else {}
                if self._is_snapflow:
                    target_time_tensor = torch.ones_like(time_tensor)
                    v_t = self.model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_kv,
                        x_t=x_t,
                        timestep=time_tensor,
                        target_time=target_time_tensor,
                        **state_kw,
                    )
                else:
                    v_t = self.model.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_kv,
                        x_t=x_t,
                        timestep=time_tensor,
                        **state_kw,
                    )
                x_t = x_t + dt * v_t

            return x_t.to(noise.dtype)

    _EXPERT_CLASS = _Pi05ExpertWrapper
    return _EXPERT_CLASS


def _build_expert_per_step_class():
    """Per-step wrapper factory — single Euler step, no internal loop.

    Mirrors ``_build_expert_class()`` but the wrapper does ONE denoise step
    per forward call instead of unrolling N steps into the ONNX graph. The
    runtime drives the Euler loop in Python, calling this wrapper N times.

    Spec: ``features/03_export/per-step-expert-export.md``
    Research: ``features/03_export/per-step-expert-export_research.md``
    """
    global _EXPERT_PER_STEP_CLASS
    if _EXPERT_PER_STEP_CLASS is not None:
        return _EXPERT_PER_STEP_CLASS

    import torch
    import torch.nn as nn

    class _Pi05ExpertPerStepWrapper(nn.Module):
        """Single Euler step. Inputs: 36 past_kv tensors + prefix_pad_masks +
        x_t (current latent shape (B, T, A)) + t (scalar timestep tensor (B,))
        + optional state. Output: v_t (velocity at this step, shape (B, T, A)).

        Per-step contract matches OpenPI Pi0
        (``openpi/src/openpi/models_pytorch/pi0_pytorch.py:298-313``) — the
        closest precedent. See research sidecar Lens 1 for the survey.
        """

        def __init__(self, pi05_model):
            super().__init__()
            self.model = pi05_model
            # SnapFlow student path — pass target_time=1 to denoise_step.
            self._is_snapflow = hasattr(pi05_model, "target_time_embed_mlp")
            # v0.5 state-out path — model has explicit state_proj layer.
            self._is_state_out = hasattr(pi05_model, "state_proj")

        def forward(self, *args):
            # args layout (default):    36 past_kv + prefix_pad_masks + x_t + t.
            # args layout (state_out):  same + state tensor (last position).
            past_flat = args[:PI05_PALIGEMMA_LAYERS * 2]
            prefix_pad_masks = args[PI05_PALIGEMMA_LAYERS * 2]
            x_t = args[PI05_PALIGEMMA_LAYERS * 2 + 1]
            t = args[PI05_PALIGEMMA_LAYERS * 2 + 2]
            state = args[PI05_PALIGEMMA_LAYERS * 2 + 3] if self._is_state_out else None

            # Reconstruct DynamicCache identically to the baked-loop wrapper
            # (see _Pi05ExpertWrapper above) — same source of past_kv, same
            # layer-by-layer .update() pattern. Bit-for-bit equivalent.
            from transformers.cache_utils import DynamicCache
            past_kv = DynamicCache()
            for i in range(PI05_PALIGEMMA_LAYERS):
                past_kv.update(
                    key_states=past_flat[2 * i],
                    value_states=past_flat[2 * i + 1],
                    layer_idx=i,
                    cache_kwargs=None,
                )

            action_dtype = self.model.action_in_proj.weight.dtype
            if x_t.dtype != action_dtype:
                x_t = x_t.to(action_dtype)

            # Single denoise step. No Euler accumulation — that's the caller's
            # responsibility. Returning v_t (velocity), not actions.
            state_kw = {"state": state} if self._is_state_out else {}
            if self._is_snapflow:
                target_time_tensor = torch.ones_like(t)
                v_t = self.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_kv,
                    x_t=x_t,
                    timestep=t,
                    target_time=target_time_tensor,
                    **state_kw,
                )
            else:
                v_t = self.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_kv,
                    x_t=x_t,
                    timestep=t,
                    **state_kw,
                )

            return v_t

    _EXPERT_PER_STEP_CLASS = _Pi05ExpertPerStepWrapper
    return _EXPERT_PER_STEP_CLASS


class _FlatCache:
    """Minimal DynamicCache-shaped shim that wraps a flat tuple of
    (K_0, V_0, K_1, V_1, ..., K_{N-1}, V_{N-1}) tensors. Exposes
    ``.key_cache[i]``, ``.value_cache[i]``, and ``get_seq_length()``
    — the only attributes pi_gemma's forward path reads from past_kv
    under our denoise_step patch.
    """
    def __init__(self, flat: tuple, num_layers: int):
        self.key_cache = [flat[2 * i] for i in range(num_layers)]
        self.value_cache = [flat[2 * i + 1] for i in range(num_layers)]
        self.is_initialized = True

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.key_cache:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])

    def __len__(self) -> int:
        return len(self.key_cache)


def _require_decomposed_deps() -> None:
    """Decomposed export shares the monolithic ``[monolithic]`` extra."""
    from tether.exporters.monolithic import _require_monolithic_deps
    _require_monolithic_deps()


__all__ = [
    "PI05_PALIGEMMA_LAYERS",
    "PI05_KV_HEADS",
    "PI05_HEAD_DIM",
    "Pi05ExpertWrapper",
    "Pi05ExpertPerStepWrapper",
    "export_pi05_decomposed",
]
