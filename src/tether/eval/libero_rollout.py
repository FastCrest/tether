"""LIBERO rollout primitive — extracted from scripts/modal_libero_pi05_decomposed.py
so multiple Modal scripts can share the proven loop.

Lifted verbatim (modulo signature) on 2026-05-20 as part of fluxvla-lift-program
lift #4 prerequisite per `01_decisions/2026-05-19-fluxvla-lift-program.md`.
Behavior must remain bit-identical to the original — the existing Modal scripts
import from here as their only change.

The rollout primitive is:

- Pure(ish): no Modal-specific decorations, no volume.commit() calls. Caller
  handles Modal/volume orchestration.
- Lazy LIBERO import: `libero` package + `mujoco` are imported inside the
  function, not at module load. The `tether` package itself does NOT depend on
  LIBERO; only callers that actually run a rollout pay the dep cost.
- Inference-object-agnostic: takes a `Pi05DecomposedInference` (or duck-typed
  equivalent — see `InferenceProtocol` below) so future exporters (DreamZero,
  fast-kernels Pi0.5, etc.) can swap in without rewriting the loop.
- Per-episode error isolation: an error inside one episode adds a row to
  `results["errors"]` but the loop continues to the next episode/task.

What lives HERE (the primitive):
- LIBERO env construction + reset/step lifecycle
- Per-step preprocessor → inference → postprocessor pipeline
- Action chunk plan dispatch + replan-on-empty
- Per-episode video frame capture (optional)
- Aggregate results dict

What lives in the CALLER (Modal scripts):
- Modal image + GPU + volume choice
- HF checkpoint download
- ONNX export (if needed before rollout)
- Final results persistence + announce

Cross-references:
- Original location: `scripts/modal_libero_pi05_decomposed.py:178-573`
- Caller: `scripts/modal_libero_pi05_decomposed.py` (refactored to thin wrapper)
- Caller: `scripts/modal_fluxvla_checkpoint_eval.py` (new for lift #4)
- ADR: `01_decisions/2026-05-19-fluxvla-lift-program.md`
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# Per FluxVLA's libero_eval_runner.py:267-276 and the original
# modal_libero_pi05_decomposed.py — identical constants.
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Standard LIBERO startup-stabilization action — drops objects to the table
# before the policy starts acting. Matches FluxVLA's num_steps_wait=10 default.
LIBERO_DUMMY_ACTION: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


class InferenceProtocol(Protocol):
    """Minimum interface a rollout-time inference object must satisfy.

    Pi05DecomposedInference satisfies this. Future exporters (DreamZero,
    fast-kernels Pi0.5, GR00T DiT) implement these three methods to plug in.
    """

    def reset_cache(self) -> None: ...

    def predict_action_chunk(
        self,
        *,
        img_base: Any,
        img_wrist_l: Any,
        img_wrist_r: Any,
        mask_base: Any,
        mask_wrist_l: Any,
        mask_wrist_r: Any,
        lang_tokens: Any,
        lang_masks: Any,
        noise: Any,
        state: Any,
        episode_id: str,
    ) -> Any: ...

    def get_stats(self) -> dict[str, Any]: ...


def run_libero_rollout(
    *,
    inference: InferenceProtocol | None = None,
    policy: Any,  # PI05Policy or load_snapflow_student output — must expose .config + ._preprocess_images
    preprocessor: Any,  # PolicyProcessorPipeline
    postprocessor: Any,  # PolicyProcessorPipeline
    task_suite_name: str = "libero_10",
    num_episodes: int = 1,
    task_indices: list[int] | None = None,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
    save_video_dir: str = "",
    label: str = "rollout",
    use_native: bool = False,
    capture_trajectories: bool = False,
) -> dict[str, Any]:
    """Run LIBERO rollouts through the given inference + processor pipeline.

    Behaviorally identical to the original modal_libero_pi05_decomposed.run_decomposed_libero
    rollout body. Returns the same shape of results dict.

    Args match the original Modal function 1:1, plus `label` for log clarity.

    Returns:
        results dict with shape:
        {
            "model": str,                  # `label` arg, used as model id in logs
            "suite": str,                  # task_suite_name
            "num_episodes_per_task": int,
            "max_steps": int,
            "resize_size": int,
            "replan_steps": int,
            "num_steps_wait": int,
            "per_task": [{
                "task_idx": int,
                "task_description": str,
                "episodes": [{"ep": int, "success": bool, "steps": int}],
                "success": int,            # successes within task
                "total": int,              # episodes within task
            }],
            "total_success": int,
            "total_eps": int,
            "success_rate_pct": float,
            "cache_stats": dict,           # from inference.get_stats()
            "errors": [...],
        }
    """
    # Lazy imports — LIBERO + mujoco only needed at rollout time, not at module load.
    import collections
    import math
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import torch

    # The Pi05DecomposedInference module uses logger.info(...) for provider
    # diagnostics; default root handler is WARN which swallows those.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # PyTorch 2.6 default-weights_only-True refuses LIBERO init-state pickles.
    _orig_torch_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_load

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE, ACTION,
    )

    cfg = policy.config
    chunk_size = cfg.chunk_size
    action_dim_pad = cfg.max_action_dim
    real_action_dim = cfg.output_features[ACTION].shape[0]

    # ─── LIBERO setup ────────────────────────────────────────────────
    np.random.seed(seed)
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    if task_suite_name not in TASK_SUITE_MAX_STEPS:
        raise KeyError(
            f"task_suite_name={task_suite_name!r} not in TASK_SUITE_MAX_STEPS. "
            f"Known: {sorted(TASK_SUITE_MAX_STEPS)}"
        )
    max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
    print(f"[{label}] suite={task_suite_name}, num_tasks={num_tasks}, max_steps={max_steps}")

    def _quat2axisangle(quat):
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def _resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
        import cv2
        h, w = img.shape[:2]
        if h != w:
            side = max(h, w)
            pad_top = (side - h) // 2
            pad_bot = side - h - pad_top
            pad_left = (side - w) // 2
            pad_right = side - w - pad_left
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bot, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0],
            )
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    def _to_tensor(img_np_hwc: np.ndarray):
        # HWC uint8 → NCHW float32 [0,1] (standard lerobot format)
        t = torch.from_numpy(img_np_hwc).float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).to("cuda")

    def _build_env(task):
        task_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": str(task_bddl),
            "camera_heights": 256,
            "camera_widths": 256,
        }
        return OffScreenRenderEnv(**env_args)

    def _build_batch(obs, task_description):
        # 180° flip on both cameras matches lerobot's LIBERO preprocessing convention
        # (and FluxVLA's eval_utils.py:98-99). Critical — getting this wrong silently
        # drops success rate by ~30%.
        img = _resize_with_pad(obs["agentview_image"][::-1, ::-1], resize_size)
        wrist_img = _resize_with_pad(
            obs["robot0_eye_in_hand_image"][::-1, ::-1], resize_size,
        )
        state = np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]).astype(np.float32)
        return {
            "observation.images.image": _to_tensor(img),
            "observation.images.image2": _to_tensor(wrist_img),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to("cuda"),
            "task": [task_description],
        }

    # ─── Results ─────────────────────────────────────────────────────
    results = {
        "model": label,
        "suite": task_suite_name,
        "num_episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "resize_size": resize_size,
        "replan_steps": replan_steps,
        "num_steps_wait": num_steps_wait,
        "seed": seed,
        "per_task": [],
        "total_success": 0,
        "total_eps": 0,
        "cache_stats": None,  # filled at end
        "errors": [],
    }
    tasks_to_run = task_indices if task_indices is not None else list(range(num_tasks))
    print(f"[{label}] Running tasks: {tasks_to_run}")

    for task_idx in tasks_to_run:
        task = task_suite.get_task(task_idx)
        task_description = task.language
        print(f"\n[{label}] TASK {task_idx}: {task_description!r}")
        initial_states = task_suite.get_task_init_states(task_idx)
        env = _build_env(task)
        task_result = {
            "task_idx": task_idx,
            "task_description": task_description,
            "episodes": [],
            "success": 0,
            "total": 0,
        }

        for ep in range(num_episodes):
            try:
                env.reset()
                init_idx = ep % len(initial_states)
                obs = env.set_init_state(initial_states[init_idx])
                policy.reset()
                if inference is not None:
                    inference.reset_cache()
                action_plan = collections.deque()
                t = 0
                done = False
                video_frames = [] if save_video_dir else None
                ep_applied_actions: list = []  # per-step executed action (capture_trajectories)
                ep_eef_positions: list = []
                if video_frames is not None:
                    video_frames.append(
                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    )

                while t < max_steps + num_steps_wait:
                    try:
                        if t < num_steps_wait:
                            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                            if video_frames is not None:
                                video_frames.append(
                                    np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                )
                            t += 1
                            continue

                        if not action_plan:
                            batch = _build_batch(obs, task_description)
                            batch_pp = preprocessor(batch)
                            batch_pp = {
                                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                                for k, v in batch_pp.items()
                            }

                            if use_native:
                                with torch.no_grad():
                                    action = policy.select_action(batch_pp)
                                post = postprocessor(action.detach().cpu())
                                chunk_np_post = (
                                    post.detach().cpu().numpy()
                                    if hasattr(post, "detach")
                                    else np.asarray(post)
                                )
                                if chunk_np_post.ndim == 3:
                                    chunk_np_post = chunk_np_post[0]
                                if chunk_np_post.ndim == 1:
                                    chunk_np_post = chunk_np_post[np.newaxis, :]
                                chunk_np_post = chunk_np_post[:, :7]
                                action_plan.extend(chunk_np_post[:replan_steps])
                            else:
                                with torch.no_grad():
                                    images, img_masks = policy._preprocess_images(batch_pp)
                                    lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
                                    lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
                                    bsize = images[0].shape[0]
                                    noise = torch.randn(
                                        bsize, chunk_size, action_dim_pad,
                                        device=images[0].device, dtype=torch.float32,
                                    )
                                    state_np = (
                                        batch_pp[OBS_STATE].cpu().numpy()
                                        if OBS_STATE in batch_pp else None
                                    )
                                    _episode_id = f"t{task_idx}_ep{ep}"
                                    chunk_np = inference.predict_action_chunk(
                                        img_base=images[0].cpu().numpy(),
                                        img_wrist_l=images[1].cpu().numpy(),
                                        img_wrist_r=images[2].cpu().numpy(),
                                        mask_base=img_masks[0].cpu().numpy(),
                                        mask_wrist_l=img_masks[1].cpu().numpy(),
                                        mask_wrist_r=img_masks[2].cpu().numpy(),
                                        lang_tokens=lang_tokens.cpu().numpy(),
                                        lang_masks=lang_masks.cpu().numpy(),
                                        noise=noise.cpu().numpy(),
                                        state=state_np,
                                        episode_id=_episode_id,
                                    )
                                    chunk = torch.from_numpy(chunk_np).to(images[0].device)
                                    chunk = chunk[:, :, :real_action_dim]

                                post = postprocessor(chunk.detach().cpu())
                                chunk_np_post = (
                                    post.detach().cpu().numpy()
                                    if hasattr(post, "detach")
                                    else np.asarray(post)
                                )
                                if chunk_np_post.ndim == 3:
                                    chunk_np_post = chunk_np_post[0]
                                chunk_np_post = chunk_np_post[:, :7]
                                action_plan.extend(chunk_np_post[:replan_steps])

                        action = action_plan.popleft()
                        if capture_trajectories:
                            # The executed 7-dim action — identical layout for the
                            # native and the optimized arm, so the two-sample test
                            # compares like with like (model-internal chunk shapes
                            # differ between arms and are NOT comparable).
                            ep_applied_actions.append(
                                np.asarray(action, dtype=np.float32).reshape(-1)[:7]
                            )
                        obs, _, done, info = env.step(np.asarray(action).tolist())
                        if capture_trajectories and "robot0_eef_pos" in obs:
                            ep_eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                        if video_frames is not None:
                            video_frames.append(
                                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            )
                        t += 1
                        if done:
                            break

                    except Exception as step_exc:
                        tb = traceback.format_exc()
                        results["errors"].append({
                            "task_idx": task_idx, "ep": ep, "t": t,
                            "error": f"{step_exc}",
                            "traceback": tb.splitlines()[-5:],
                        })
                        raise

                success = bool(done)
                episode_rec = {"ep": ep, "success": success, "steps": t}
                if capture_trajectories:
                    episode_rec["actions"] = [a.tolist() for a in ep_applied_actions]
                    episode_rec["eef_positions"] = [p.tolist() for p in ep_eef_positions]
                task_result["episodes"].append(episode_rec)
                task_result["total"] += 1
                if success:
                    task_result["success"] += 1
                print(f"  ep {ep}: {'✓' if success else '✗'} (steps={t})")

                if video_frames is not None and len(video_frames) > 0:
                    Path(save_video_dir).mkdir(parents=True, exist_ok=True)
                    tag = "S" if success else "F"
                    out = Path(save_video_dir) / (
                        f"{label}_t{task_idx}_ep{ep}_seed{seed}_steps{t}_{tag}.npz"
                    )
                    np.savez_compressed(str(out), frames=np.array(video_frames, dtype=np.uint8))
                    print(f"    frames → {out} ({len(video_frames)} frames)")

            except Exception as ep_exc:
                print(f"  ep {ep}: ERROR {ep_exc}")
                task_result["episodes"].append({
                    "ep": ep, "success": False, "error": str(ep_exc),
                })
                task_result["total"] += 1

        env.close()
        results["per_task"].append(task_result)
        results["total_success"] += task_result["success"]
        results["total_eps"] += task_result["total"]
        print(f"  TASK {task_idx}: {task_result['success']}/{task_result['total']}")

    results["cache_stats"] = inference.get_stats() if inference is not None else {}
    if results["total_eps"]:
        results["success_rate_pct"] = 100.0 * results["total_success"] / results["total_eps"]
    print(
        f"\n[{label}] TOTAL: {results['total_success']}/{results['total_eps']} "
        f"= {results.get('success_rate_pct', 0):.1f}%"
    )
    print(f"[{label}] CACHE STATS: {results['cache_stats']}")
    return results


def load_pi05_policy_and_processors(
    *,
    student_checkpoint: str,
    decomposed_dir: str,
    preprocessor_ref: str | None = None,
    force_teacher: bool = False,
) -> tuple[Any, Any, Any]:
    """Load PyTorch policy (for config + _preprocess_images) + processor pipelines.

    Extracted from the same Modal script for reuse. Returns (policy, preprocessor,
    postprocessor). Handles the SnapFlow-student vs fallback-HF dispatch.

    Args:
        force_teacher: when True, load via PI05Policy.from_pretrained even if
            model.safetensors exists (for non-SnapFlow fine-tunes like FluxVLA).

    Caller is expected to already have torch + lerobot importable.
    """
    import json as _json
    from pathlib import Path

    import torch

    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    student_ckpt_path = Path(student_checkpoint)
    if not force_teacher and (student_ckpt_path / "model.safetensors").exists():
        print(f"[load] Loading SnapFlow student from {student_checkpoint}")
        from tether.distill.snapflow_pi0_model import load_snapflow_student
        policy = load_snapflow_student(student_checkpoint)
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        if force_teacher and (student_ckpt_path / "model.safetensors").exists():
            print(f"[load] Loading teacher fine-tune from {student_checkpoint}")
            policy = PI05Policy.from_pretrained(student_checkpoint)
        else:
            fallback = preprocessor_ref or "lerobot/pi05_libero_finetuned_v044"
            print(
                f"[load] No model.safetensors at {student_checkpoint}; "
                f"loading PI05Policy from {fallback} (config + preprocessing only — "
                f"inference still runs through decomposed ONNX)"
            )
            policy = PI05Policy.from_pretrained(fallback)
    policy.eval().to("cuda").to(torch.float32)

    # Student-distillation checkpoints don't always ship the processor JSONs —
    # fall back to the teacher HF repo for baseline preprocessor + normalizer.
    from huggingface_hub import snapshot_download
    proc_ref = preprocessor_ref or student_checkpoint
    if proc_ref and not Path(proc_ref).exists():
        proc_ref = snapshot_download(proc_ref)
    print(f"[load] Using processor configs from: {proc_ref}")
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=proc_ref,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=proc_ref,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # v0.5 state-out detection: if decomposed export was built with
    # expert_takes_state=True, swap the prepare step to the state-out version.
    decomposed_cfg_path = Path(decomposed_dir) / "tether_config.json"
    is_state_out_export = False
    if decomposed_cfg_path.exists():
        with decomposed_cfg_path.open() as _f:
            _dcfg = _json.load(_f)
        is_state_out_export = (
            _dcfg.get("decomposed", {}).get("expert_takes_state", False)
        )
    if is_state_out_export:
        from tether.distill.pi05_state_out_processor import swap_prepare_step_in_pipeline
        from lerobot.utils.constants import ACTION
        max_state_dim = policy.config.max_action_dim  # pi0.5: 32
        swap_prepare_step_in_pipeline(preprocessor, max_state_dim=max_state_dim)
        print(
            f"[load] Detected state-out export — swapped preprocessor to "
            f"Pi05PrepareTokenizerStateOutStep (max_state_dim={max_state_dim})"
        )

    return policy, preprocessor, postprocessor
