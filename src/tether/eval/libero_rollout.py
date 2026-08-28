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
import hashlib
from typing import Any, Protocol

from tether.verification_evidence import normalize_verification_device

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
PAIRED_SEED_PROTOCOL = "sha256-v1"


def paired_policy_call_seed(base_seed: int, task_idx: int, ep: int, call: int) -> int:
    """Stable seed shared by native diffusion and exported protocol noise."""

    payload = f"{base_seed}:{task_idx}:{ep}:{call}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def seed_verification_random_sources(
    seed: int, *, torch_module: Any, device: str
) -> None:
    """Seed Python, NumPy, torch CPU, and the selected CUDA RNG identically."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch_module.manual_seed(seed)
    if device == "cuda":
        torch_module.cuda.manual_seed_all(seed)


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


class VerificationPolicyContext:
    """Weight-free policy surface used by the isolated optimized arm."""

    def __init__(self, *, model_type: str, config: Any, device: str) -> None:
        self.model_type = model_type
        self.config = config
        self.device = device

    def reset(self) -> None:
        return None

    def _preprocess_images(self, batch: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        import torch

        present = [key for key in self.config.image_features if key in batch]
        missing = [key for key in self.config.image_features if key not in batch]
        if not present:
            raise ValueError("all configured image features are missing")

        images: list[Any] = []
        masks: list[Any] = []
        if self.model_type == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

            for key in present:
                image = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
                image = image.to(self.device, dtype=torch.float32)
                if self.config.resize_imgs_with_padding is not None:
                    image = resize_with_pad(
                        image,
                        *self.config.resize_imgs_with_padding,
                        pad_value=0,
                    )
                image = image * 2.0 - 1.0
                mask_key = f"{key}_padding_mask"
                mask = (
                    batch[mask_key].bool().to(self.device)
                    if mask_key in batch
                    else torch.ones(
                        image.shape[0], dtype=torch.bool, device=self.device
                    )
                )
                images.append(image)
                masks.append(mask)
            for index in range(len(missing)):
                if index >= self.config.empty_cameras:
                    break
                images.append(torch.ones_like(images[-1]) * -1)
                masks.append(torch.zeros_like(masks[-1]))
            return images, masks

        if self.model_type in {
            "pi0",
            "pi05",
            "pi05_decomposed",
            "pi05_decomposed_student",
        }:
            if self.model_type == "pi0":
                from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch
            else:
                from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

            for key in present:
                image = batch[key].to(self.device, dtype=torch.float32)
                channels_first = image.shape[1] == 3
                if channels_first:
                    image = image.permute(0, 2, 3, 1)
                if image.shape[1:3] != self.config.image_resolution:
                    image = resize_with_pad_torch(
                        image, *self.config.image_resolution
                    )
                image = image * 2.0 - 1.0
                if channels_first:
                    image = image.permute(0, 3, 1, 2)
                images.append(image)
                masks.append(
                    torch.ones(
                        image.shape[0], dtype=torch.bool, device=self.device
                    )
                )
            for _ in missing:
                images.append(torch.ones_like(images[-1]) * -1)
                masks.append(torch.zeros_like(masks[-1]))
            return images, masks

        raise ValueError(
            f"weight-free verification preprocessing does not support {self.model_type!r}"
        )


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
    safety_config: str | None = None,
    safety_limits: Any | None = None,
    safety_config_sha256: str | None = None,
    verification_device: str = "cpu",
    memory_sampler: Any | None = None,
    action_guard: Any | None = None,
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

    from tether.eval.evidence_capture import (
        CanonicalSafetyLimits,
        apply_action_guard,
        canonicalize_safety_limits,
        validate_safety_limits,
    )
    from tether.verification_evidence import EvidenceValue, UnavailableReason

    verification_device = normalize_verification_device(verification_device)
    seed_verification_random_sources(
        seed, torch_module=torch, device=verification_device
    )

    if safety_config and safety_limits is not None:
        raise ValueError("pass safety_config or canonical safety_limits, not both")
    canonical_safety: CanonicalSafetyLimits | None = None
    if action_guard is None and safety_limits is not None:
        if not isinstance(safety_limits, CanonicalSafetyLimits):
            raise TypeError("safety_limits must be CanonicalSafetyLimits")
        canonical_safety = safety_limits
        if safety_config_sha256 != canonical_safety.sha256:
            raise ValueError("canonical safety-config SHA-256 mismatch")
        from tether.safety import ActionGuard

        action_guard = ActionGuard(
            limits=canonical_safety.to_safety_limits(),
            mode="clamp",
            max_consecutive_clamps=0,
        )
    elif action_guard is None and safety_config:
        from tether.safety import ActionGuard, SafetyLimits

        action_guard = ActionGuard(
            limits=SafetyLimits.from_json(safety_config),
            mode="clamp",
            max_consecutive_clamps=0,
        )
    if action_guard is not None:
        validate_safety_limits(action_guard.limits, action_dim=7)
        runtime_safety = canonicalize_safety_limits(action_guard.limits)
        if (
            safety_config_sha256 is not None
            and runtime_safety.sha256 != safety_config_sha256
        ):
            raise ValueError("runtime safety limits differ from expected SHA-256")
        canonical_safety = canonical_safety or runtime_safety
        if runtime_safety.sha256 != canonical_safety.sha256:
            raise ValueError("runtime safety limits differ from canonical value")
        safety_config_sha256 = canonical_safety.sha256

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
        return t.permute(2, 0, 1).unsqueeze(0).to(verification_device)

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
            "observation.state": torch.from_numpy(state)
            .unsqueeze(0)
            .to(verification_device),
            "task": [task_description],
        }

    # ─── Results ─────────────────────────────────────────────────────
    results: dict[str, Any] = {
        "model": label,
        "suite": task_suite_name,
        "num_episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "resize_size": resize_size,
        "replan_steps": replan_steps,
        "num_steps_wait": num_steps_wait,
        "seed": seed,
        "seed_protocol": PAIRED_SEED_PROTOCOL,
        "verification_device": verification_device,
        "per_task": [],
        "total_success": 0,
        "total_eps": 0,
        "cache_stats": None,  # filled at end
        "errors": [],
        "safety_evidence": (
            EvidenceValue.available(
                {
                    "sha256": safety_config_sha256,
                    "limits": canonical_safety.to_dict(),
                }
            ).to_dict()
            if action_guard is not None and canonical_safety is not None
            else EvidenceValue.unavailable(UnavailableReason.CAPTURE_DISABLED).to_dict()
        ),
        "memory_evidence": EvidenceValue.unavailable(
            UnavailableReason.CAPTURE_DISABLED
        ).to_dict(),
    }
    tasks_to_run = task_indices if task_indices is not None else list(range(num_tasks))
    print(f"[{label}] Running tasks: {tasks_to_run}")

    for task_idx in tasks_to_run:
        task = task_suite.get_task(task_idx)
        task_description = task.language
        print(f"\n[{label}] TASK {task_idx}: {task_description!r}")
        initial_states = task_suite.get_task_init_states(task_idx)
        env = _build_env(task)
        task_result: dict[str, Any] = {
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
                action_plan: collections.deque[Any] = collections.deque()
                t = 0
                done = False
                video_frames: list[Any] | None = [] if save_video_dir else None
                ep_applied_actions: list = []  # per-step executed action (capture_trajectories)
                ep_action_timestamps: list[float] = []
                ep_eef_positions: list = []
                ep_joint_velocities: list = []
                ep_inference_latency_ms: list[float] = []
                ep_inference_call_seeds: list[int] = []
                ep_inference_call_steps: list[int] = []
                ep_safety_clamp_count = 0
                ep_previous_guarded_action = None
                inference_call = 0
                seed_verification_random_sources(
                    paired_policy_call_seed(seed, task_idx, ep, -1),
                    torch_module=torch,
                    device=verification_device,
                )
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
                            if memory_sampler is not None:
                                memory_sampler.start()
                            batch = _build_batch(obs, task_description)
                            batch_pp = preprocessor(batch)
                            batch_pp = {
                                k: (
                                    v.to(verification_device)
                                    if isinstance(v, torch.Tensor)
                                    else v
                                )
                                for k, v in batch_pp.items()
                            }
                            call_seed = paired_policy_call_seed(
                                seed, task_idx, ep, inference_call
                            )
                            seed_verification_random_sources(
                                call_seed,
                                torch_module=torch,
                                device=verification_device,
                            )
                            ep_inference_call_seeds.append(call_seed)
                            ep_inference_call_steps.append(t)
                            inference_call += 1
                            state_tensor = batch_pp.get(OBS_STATE)
                            if not isinstance(state_tensor, torch.Tensor):
                                raise ValueError(
                                    "processed batch is missing tensor observation state"
                                )
                            shared_noise = torch.randn(
                                state_tensor.shape[0],
                                chunk_size,
                                action_dim_pad,
                                device=verification_device,
                                dtype=torch.float32,
                            )

                            if use_native:
                                if not hasattr(policy, "predict_action_chunk"):
                                    raise TypeError(
                                        "native verification policy must expose "
                                        "predict_action_chunk(batch, noise=...)"
                                    )
                                inference_started = time.perf_counter()
                                with torch.no_grad():
                                    chunk = policy.predict_action_chunk(
                                        batch_pp, noise=shared_noise
                                    )
                                inference_latency_ms = (
                                    time.perf_counter() - inference_started
                                ) * 1000.0
                                post = postprocessor(chunk.detach().cpu())
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
                                assert inference is not None
                                with torch.no_grad():
                                    images, img_masks = policy._preprocess_images(batch_pp)
                                    lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
                                    lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
                                    bsize = images[0].shape[0]
                                    if bsize != shared_noise.shape[0]:
                                        raise ValueError(
                                            "preprocessed image batch size differs from state"
                                        )
                                    state_np = (
                                        batch_pp[OBS_STATE].cpu().numpy()
                                        if OBS_STATE in batch_pp else None
                                    )
                                    _episode_id = f"t{task_idx}_ep{ep}"
                                    inference_started = time.perf_counter()
                                    chunk_np = inference.predict_action_chunk(
                                        img_base=images[0].cpu().numpy(),
                                        img_wrist_l=images[1].cpu().numpy(),
                                        img_wrist_r=images[2].cpu().numpy(),
                                        mask_base=img_masks[0].cpu().numpy(),
                                        mask_wrist_l=img_masks[1].cpu().numpy(),
                                        mask_wrist_r=img_masks[2].cpu().numpy(),
                                        lang_tokens=lang_tokens.cpu().numpy(),
                                        lang_masks=lang_masks.cpu().numpy(),
                                        noise=shared_noise.cpu().numpy(),
                                        state=state_np,
                                        episode_id=_episode_id,
                                    )
                                    inference_latency_ms = (
                                        time.perf_counter() - inference_started
                                    ) * 1000.0
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

                            if capture_trajectories:
                                ep_inference_latency_ms.append(inference_latency_ms)

                        action = action_plan.popleft()
                        action_array, clamp_count = apply_action_guard(
                            np.asarray(action, dtype=np.float32).reshape(-1)[:7],
                            action_guard,
                            previous_action=ep_previous_guarded_action,
                        )
                        ep_previous_guarded_action = action_array
                        ep_safety_clamp_count += clamp_count
                        if capture_trajectories:
                            # The executed 7-dim action — identical layout for the
                            # native and the optimized arm, so the two-sample test
                            # compares like with like (model-internal chunk shapes
                            # differ between arms and are NOT comparable).
                            ep_applied_actions.append(
                                action_array
                            )
                            ep_action_timestamps.append(time.perf_counter())
                        obs, _, done, info = env.step(action_array.tolist())
                        if capture_trajectories and "robot0_eef_pos" in obs:
                            ep_eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                        if capture_trajectories:
                            for velocity_key in (
                                "robot0_joint_vel",
                                "robot0_joint_velocities",
                                "robot0_joint_qvel",
                            ):
                                if velocity_key in obs:
                                    ep_joint_velocities.append(
                                        np.asarray(obs[velocity_key], dtype=np.float32)
                                        .reshape(-1)
                                        .tolist()
                                    )
                                    break
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
                episode_rec: dict[str, Any] = {
                    "ep": ep,
                    "success": success,
                    "steps": t,
                }
                if capture_trajectories:
                    episode_rec["actions"] = [a.tolist() for a in ep_applied_actions]
                    episode_rec["action_timestamps_s"] = ep_action_timestamps
                    episode_rec["eef_positions"] = [p.tolist() for p in ep_eef_positions]
                    episode_rec["inference_latency_ms"] = ep_inference_latency_ms
                    episode_rec["inference_call_seeds"] = ep_inference_call_seeds
                    episode_rec["inference_call_steps"] = ep_inference_call_steps
                    if ep_joint_velocities:
                        episode_rec["joint_velocities"] = ep_joint_velocities
                if action_guard is not None:
                    episode_rec["safety_clamp_count"] = ep_safety_clamp_count
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
    if action_guard is not None and canonical_safety is not None:
        final_safety = canonicalize_safety_limits(action_guard.limits)
        if final_safety.sha256 != canonical_safety.sha256:
            raise RuntimeError("safety limits mutated during verification rollout")
    if memory_sampler is not None:
        results["memory_evidence"] = memory_sampler.stop().to_dict()
    if results["total_eps"]:
        results["success_rate_pct"] = 100.0 * results["total_success"] / results["total_eps"]
    print(
        f"\n[{label}] TOTAL: {results['total_success']}/{results['total_eps']} "
        f"= {results.get('success_rate_pct', 0):.1f}%"
    )
    print(f"[{label}] CACHE STATS: {results['cache_stats']}")
    return results


def _load_exact_reference_policy(model_type: str, checkpoint: str) -> Any:
    """Load the declared native family from the exact export provenance ref."""
    import importlib

    targets = {
        "pi05": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        "pi05_decomposed": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        "pi0": ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
        "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    }
    if model_type == "pi05_decomposed_student":
        module = importlib.import_module("tether.distill.snapflow_pi0_model")
        return module.load_snapflow_student(checkpoint)
    try:
        module_name, class_name = targets[model_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported native verification model_type={model_type!r}"
        ) from exc
    policy_class = getattr(importlib.import_module(module_name), class_name)
    return policy_class.from_pretrained(checkpoint)


def _load_exact_reference_config(model_type: str, checkpoint: str) -> Any:
    import importlib

    targets = {
        "pi05": ("lerobot.policies.pi05.configuration_pi05", "PI05Config"),
        "pi05_decomposed": (
            "lerobot.policies.pi05.configuration_pi05",
            "PI05Config",
        ),
        "pi05_decomposed_student": (
            "lerobot.policies.pi05.configuration_pi05",
            "PI05Config",
        ),
        "pi0": ("lerobot.policies.pi0.configuration_pi0", "PI0Config"),
        "smolvla": (
            "lerobot.policies.smolvla.configuration_smolvla",
            "SmolVLAConfig",
        ),
    }
    try:
        module_name, class_name = targets[model_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported weight-free verification model_type={model_type!r}"
        ) from exc
    config_class = getattr(importlib.import_module(module_name), class_name)
    return config_class.from_pretrained(checkpoint)


def _load_processor_pipelines(
    *,
    processor_ref: str,
    device: str,
) -> tuple[Any, Any]:
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    resolved_ref = processor_ref
    if not Path(resolved_ref).exists():
        resolved_ref = snapshot_download(resolved_ref)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=resolved_ref,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=resolved_ref,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_verification_policy_context_and_processors(
    *,
    checkpoint: str,
    model_type: str,
    preprocessor_ref: str | None,
    decomposed_dir: str,
    device: str,
) -> tuple[VerificationPolicyContext, Any, Any]:
    """Load exact config/processors without allocating native model weights."""

    import json
    from pathlib import Path

    device = normalize_verification_device(device)
    config = _load_exact_reference_config(model_type, checkpoint)
    context = VerificationPolicyContext(
        model_type=model_type,
        config=config,
        device=device,
    )
    preprocessor, postprocessor = _load_processor_pipelines(
        processor_ref=preprocessor_ref or checkpoint,
        device=device,
    )
    config_path = Path(decomposed_dir) / "tether_config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text())
        if payload.get("decomposed", {}).get("expert_takes_state", False):
            from tether.distill.pi05_state_out_processor import (
                swap_prepare_step_in_pipeline,
            )

            swap_prepare_step_in_pipeline(
                preprocessor, max_state_dim=config.max_action_dim
            )
    return context, preprocessor, postprocessor


def load_pi05_policy_and_processors(
    *,
    student_checkpoint: str,
    decomposed_dir: str,
    preprocessor_ref: str | None = None,
    force_teacher: bool = False,
    model_type: str = "pi05",
    require_exact_checkpoint: bool = False,
    device: str = "cuda",
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

    student_ckpt_path = Path(student_checkpoint)
    if require_exact_checkpoint:
        print(f"[load] Loading exact {model_type} reference from {student_checkpoint}")
        policy = _load_exact_reference_policy(model_type, student_checkpoint)
    elif not force_teacher and (student_ckpt_path / "model.safetensors").exists():
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
    policy.eval().to(device).to(torch.float32)

    # Student-distillation checkpoints don't always ship processor JSONs.
    proc_ref = preprocessor_ref or student_checkpoint
    print(f"[load] Using processor configs from: {proc_ref}")
    preprocessor, postprocessor = _load_processor_pipelines(
        processor_ref=proc_ref,
        device=device,
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
        max_state_dim = policy.config.max_action_dim  # pi0.5: 32
        swap_prepare_step_in_pipeline(preprocessor, max_state_dim=max_state_dim)
        print(
            f"[load] Detected state-out export — swapped preprocessor to "
            f"Pi05PrepareTokenizerStateOutStep (max_state_dim={max_state_dim})"
        )

    return policy, preprocessor, postprocessor
