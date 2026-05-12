"""Modal: LIBERO-10 task-success through the decomposed pi0.5 ONNX chain
(``vlm_prefix.onnx`` → ``expert_denoise.onnx``), with optional
perceptual-hash prefix cache.

Closes Verification Plan steps 2+3 of
``reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md``:

- ``--cache none``: decomposed chain with no cache. Should match the
  monolithic student's 29/30 exactly (cos=1.0 parity already verified).
- ``--cache phash``: VLM is skipped whenever the perceptual hash of the
  3 camera images AND the exact hash of the language tokens matches the
  prior frame inside the TTL window. If task-success holds at 29/30, the
  cache is safe. If it drops, hamming threshold is too loose.

Reuses the LIBERO rollout loop + preprocessing from
``modal_libero_lerobot_native.py``, but swaps the policy forward with
``Pi05DecomposedInference``.

Usage:
    # Cache off — parity run:
    modal run scripts/modal_libero_pi05_decomposed.py \\
      --student-checkpoint /onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model \\
      --decomposed-dir /onnx_out/distill_v031_pi05_libero_r4/decomposed_v2 \\
      --cache none --tasks all --num-episodes 3

    # Cache on — measures hit-rate × retention:
    modal run scripts/modal_libero_pi05_decomposed.py \\
      --student-checkpoint /onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model \\
      --decomposed-dir /onnx_out/distill_v031_pi05_libero_r4/decomposed_v2 \\
      --cache phash --tasks all --num-episodes 3
"""
import os
import subprocess
import modal

app = modal.App("reflex-libero-pi05-decomposed")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

TASK_SUITE_MAX_STEPS = {
    "libero_10": 520,
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_90": 400,
}
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

# Image recipe duplicated from modal_libero_lerobot_native.py — that's
# the proven one for pi0.5 LIBERO rollouts. osmesa render + pinned mujoco
# + PYTHONPATH /opt/LIBERO all matter.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "build-essential",
        "libosmesa6", "libosmesa6-dev",
        "clang",
    )
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "nvidia-curand-cu12>=10.0,<12.0",
        "nvidia-cufft-cu12>=11.0,<13.0",
        "nvidia-cusparse-cu12>=12.0,<13.0",
        "nvidia-cusolver-cu12>=11.0,<13.0",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cuda-nvrtc-cu12>=12.0,<13.0",
        "onnxscript>=0.1",
        "mujoco==3.3.2",
        "robosuite==1.4.1",
        "h5py",
        "bddl==1.0.1",
        "future",
        "robomimic",
        "hydra-core>=1.1",
        "easydict",
        "einops",
        "opencv-python-headless",
        "gym",
        "gymnasium",
        "lerobot==0.5.1",
        "num2words",
        "imageio",
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "reflex-vla[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
        # Point onnxruntime-gpu at the CUDA libs bundled as pip packages
        # so the CUDAExecutionProvider actually loads on A100. Previous
        # attempts missed libcudart (cuda-runtime) + libcudnn path —
        # onnxruntime silently fell back to CPU each time.
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
    .run_commands("mkdir -p /tmp/libero_data")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def run_decomposed_libero(
    student_checkpoint: str,
    decomposed_dir: str,
    cache_mode: str = "none",
    num_episodes: int = 1,
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    cache_ttl_sec: float = 0.2,
    cache_max_age_steps: int = 0,
    action_cache_max_age_steps: int = 2,
    phash_hamming: int = 6,
    preprocessor_ref: str = "lerobot/pi05_libero_finetuned_v044",
    seed: int = 7,
    save_video_dir: str = "",
):
    """LIBERO rollout through the decomposed ONNX chain. Mirrors
    ``modal_libero_lerobot_native.run_ported_libero`` but swaps the
    forward path for ``Pi05DecomposedInference``.

    save_video_dir: if non-empty, write per-episode MP4 of agentview camera
    to that path inside the container (typically a /onnx_out subpath so it
    persists on the volume). Filename encodes task/ep/seed/steps/success.
    """
    import collections
    import logging
    import math
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import torch

    # The Pi05DecomposedInference module uses logger.info(...) for provider
    # diagnostics; default root handler is WARN which swallows those.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # PyTorch 2.6 default-weights_only-True refuses LIBERO init-state pickles.
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load

    # ─── Load PyTorch policy (preprocessing + config only) ──────────
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, ACTION,
    )

    # Two ways to source the policy (config + _preprocess_images helper):
    # 1) SnapFlow student dir on the volume (has model.safetensors) — preferred
    #    when the student PyTorch checkpoint lives alongside the decomposed
    #    ONNX export.
    # 2) Fallback: load PI05Policy from preprocessor_ref (HF id). Inference
    #    actually runs through ONNX so the policy weights here are unused —
    #    only config + _preprocess_images matter.
    student_ckpt_path = Path(student_checkpoint)
    if (student_ckpt_path / "model.safetensors").exists():
        print(f"[decomposed] Loading SnapFlow student from {student_checkpoint}")
        from reflex.distill.snapflow_pi0_model import load_snapflow_student
        policy = load_snapflow_student(student_checkpoint)
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        fallback = preprocessor_ref or "lerobot/pi05_libero_finetuned_v044"
        print(f"[decomposed] No model.safetensors at {student_checkpoint}; "
              f"loading PI05Policy from {fallback} (config + preprocessing only — "
              f"inference still runs through decomposed ONNX)")
        policy = PI05Policy.from_pretrained(fallback)
    policy.eval().to("cuda").to(torch.float32)
    cfg = policy.config
    chunk_size = cfg.chunk_size
    action_dim_pad = cfg.max_action_dim
    real_action_dim = cfg.output_features[ACTION].shape[0]

    # Student-distillation checkpoints don't always ship the processor
    # JSONs — fall back to the teacher HF repo (pi05_libero_finetuned_v044
    # by default) which has the baseline preprocessor + normalizer stats.
    from huggingface_hub import snapshot_download
    proc_ref = preprocessor_ref or student_checkpoint
    if proc_ref and not Path(proc_ref).exists():
        proc_ref = snapshot_download(proc_ref)
    print(f"[decomposed] Using processor configs from: {proc_ref}")
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

    # v0.5 state-out: if the decomposed export was built with
    # expert_takes_state=True, the student expects state via the
    # state_proj input AND a stripped lang prompt. Default preprocessor
    # bakes state into lang ("Task: ..., State: ...") which (a) double-
    # feeds state to the model, (b) makes lang_tokens drift per frame
    # → kills the prefix cache hit rate. Swap to the state-out step.
    import json as _json
    decomposed_cfg_path = Path(decomposed_dir) / "reflex_config.json"
    is_state_out_export = False
    if decomposed_cfg_path.exists():
        with decomposed_cfg_path.open() as _f:
            _dcfg = _json.load(_f)
        is_state_out_export = (
            _dcfg.get("decomposed", {}).get("expert_takes_state", False)
        )
    if is_state_out_export:
        from reflex.distill.pi05_state_out_processor import swap_prepare_step_in_pipeline
        max_state_dim = action_dim_pad  # pi0.5 max_state_dim == max_action_dim == 32
        swap_prepare_step_in_pipeline(preprocessor, max_state_dim=max_state_dim)
        print(
            f"[decomposed] Detected state-out export — swapped preprocessor "
            f"to Pi05PrepareTokenizerStateOutStep (max_state_dim={max_state_dim})"
        )

    print(f"[decomposed] Policy + processors ready. chunk_size={chunk_size}, "
          f"max_action_dim={action_dim_pad}, real_action_dim={real_action_dim}")

    # ─── Load decomposed ONNX inference ──────────────────────────────
    from reflex.runtime.pi05_decomposed_server import Pi05DecomposedInference
    # Map CLI cache_mode to class params:
    #   none    → enable_cache=False
    #   phash   → enable_cache=True, cache_level='prefix' (VLM-skip on hit)
    #   action  → enable_cache=True, cache_level='action' (full-forward skip)
    #             + cache_ignore_lang=True (pi0.5 state-in-lang needs this)
    #   episode → enable_cache=True, cache_level='episode' (lang-only cache,
    #             image ignored) — THE MOAT for v0.5 state-out pi0.5
    _enable_cache = cache_mode in ("phash", "action", "episode")
    if cache_mode == "action":
        _cache_level = "action"
    elif cache_mode == "episode":
        _cache_level = "episode"
    elif cache_mode == "phash":
        _cache_level = "prefix"
    else:
        _cache_level = "prefix"  # dummy; enable_cache=False
    _ignore_lang = cache_mode == "action"
    inference = Pi05DecomposedInference(
        export_dir=decomposed_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        enable_cache=_enable_cache,
        cache_ttl_sec=cache_ttl_sec,
        cache_max_age_steps=cache_max_age_steps,
        phash_hamming_threshold=phash_hamming,
        cache_level=_cache_level,
        action_cache_max_age_steps=action_cache_max_age_steps,
        cache_ignore_lang=_ignore_lang,
    )
    print(f"[decomposed] Pi05DecomposedInference ready. cache_mode={cache_mode}, "
          f"cache_level={_cache_level}, ignore_lang={_ignore_lang}, "
          f"action_max_age_steps={action_cache_max_age_steps}, "
          f"phash_threshold={phash_hamming}")

    # ─── LIBERO setup ────────────────────────────────────────────────
    np.random.seed(seed)
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
    print(f"[decomposed] suite={task_suite_name}, num_tasks={num_tasks}, max_steps={max_steps}")

    def _quat2axisangle(quat):
        if quat[3] > 1.0: quat[3] = 1.0
        elif quat[3] < -1.0: quat[3] = -1.0
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
            img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
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
        img = _resize_with_pad(obs["agentview_image"][::-1, ::-1], resize_size)
        wrist_img = _resize_with_pad(obs["robot0_eye_in_hand_image"][::-1, ::-1], resize_size)
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
        "model": f"decomposed:{decomposed_dir}",
        "cache_mode": cache_mode,
        "suite": task_suite_name,
        "num_episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "resize_size": resize_size,
        "replan_steps": replan_steps,
        "num_steps_wait": num_steps_wait,
        "cache_ttl_sec": cache_ttl_sec,
        "phash_hamming": phash_hamming,
        "per_task": [],
        "total_success": 0,
        "total_eps": 0,
        "cache_stats": None,  # filled at end
        "errors": [],
    }
    tasks_to_run = task_indices if task_indices is not None else list(range(num_tasks))
    print(f"[decomposed] Running tasks: {tasks_to_run}")

    for task_idx in tasks_to_run:
        task = task_suite.get_task(task_idx)
        task_description = task.language
        print(f"\n[decomposed] TASK {task_idx}: {task_description!r}")
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
                inference.reset_cache()  # fresh cache per episode
                action_plan = collections.deque()
                t = 0
                done = False
                video_frames = [] if save_video_dir else None
                if video_frames is not None:
                    video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

                while t < max_steps + num_steps_wait:
                    try:
                        if t < num_steps_wait:
                            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                            if video_frames is not None:
                                video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            t += 1
                            continue

                        if not action_plan:
                            batch = _build_batch(obs, task_description)
                            batch_pp = preprocessor(batch)
                            batch_pp = {
                                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                                for k, v in batch_pp.items()
                            }
                            with torch.no_grad():
                                from lerobot.utils.constants import OBS_STATE
                                images, img_masks = policy._preprocess_images(batch_pp)
                                lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
                                lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
                                bsize = images[0].shape[0]
                                noise = torch.randn(
                                    bsize, chunk_size, action_dim_pad,
                                    device=images[0].device, dtype=torch.float32,
                                )
                                # v0.5: pass state explicitly. predict_action_chunk
                                # accepts state=None for default exports and uses
                                # it for state-out exports (auto-detects from
                                # reflex_config.json's expert_takes_state flag).
                                state_np = batch_pp[OBS_STATE].cpu().numpy() if OBS_STATE in batch_pp else None
                                # Episode id stable within one LIBERO episode
                                # → lang-only cache hits after the first frame.
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
                                # Trim padded max_action_dim → real env action dim
                                chunk = chunk[:, :, :real_action_dim]

                            # Postprocessor is a PolicyProcessorPipeline whose
                            # to_transition/to_output converters accept a raw
                            # tensor and return a tensor — pass the chunk
                            # directly (same path as modal_libero_lerobot_native).
                            post = postprocessor(chunk.detach().cpu())
                            chunk_np_post = (
                                post.detach().cpu().numpy()
                                if hasattr(post, "detach")
                                else np.asarray(post)
                            )
                            if chunk_np_post.ndim == 3:
                                chunk_np_post = chunk_np_post[0]  # (chunk, N)
                            chunk_np_post = chunk_np_post[:, :7]  # LIBERO 7D
                            action_plan.extend(chunk_np_post[:replan_steps])

                        action = action_plan.popleft()
                        obs, _, done, info = env.step(np.asarray(action).tolist())
                        if video_frames is not None:
                            video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
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
                task_result["episodes"].append({
                    "ep": ep, "success": success, "steps": t,
                })
                task_result["total"] += 1
                if success:
                    task_result["success"] += 1
                print(f"  ep {ep}: {'✓' if success else '✗'} (steps={t})")
                if video_frames is not None and len(video_frames) > 0:
                    Path(save_video_dir).mkdir(parents=True, exist_ok=True)
                    tag = "S" if success else "F"
                    out = Path(save_video_dir) / (
                        f"student_t{task_idx}_ep{ep}_seed{seed}_steps{t}_{tag}.npz"
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

    results["cache_stats"] = inference.get_stats()
    if results["total_eps"]:
        results["success_rate_pct"] = 100.0 * results["total_success"] / results["total_eps"]
    print(f"\n[decomposed] TOTAL: {results['total_success']}/{results['total_eps']} "
          f"= {results.get('success_rate_pct', 0):.1f}%")
    print(f"[decomposed] CACHE STATS: {results['cache_stats']}")
    return results


@app.local_entrypoint()
def main(
    student_checkpoint: str = "/onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model",
    decomposed_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/decomposed_v2",
    cache: str = "none",
    num_episodes: int = 1,
    tasks: str = "0",
    suite: str = "libero_10",
    cache_ttl_sec: float = 0.2,
    cache_max_age_steps: int = 0,
    action_cache_max_age_steps: int = 2,
    phash_hamming: int = 6,
    preprocessor_ref: str = "lerobot/pi05_libero_finetuned_v044",
    seed: int = 7,
    save_video_dir: str = "",
):
    """
    --student-checkpoint   Path to SnapFlow student dir on volume (for
                           policy.config only — inference actually runs
                           through the decomposed ONNX).
    --decomposed-dir       Dir with vlm_prefix.onnx + expert_denoise.onnx
                           + reflex_config.json.
    --cache                'none' | 'phash' (VLM-skip cache) | 'action'
                           (full-forward skip + cache_ignore_lang for pi0.5)
    --num-episodes         Episodes per task.
    --tasks "0" | "0,1" | "all"
    --suite                libero_10 (default) | others.
    --cache-ttl-sec        TTL after which cache entry is stale (default 0.2s).
    --phash-hamming        Per-image hamming distance threshold (default 6).
    --preprocessor-ref     HF repo id OR local path for the preprocessor +
                           postprocessor JSONs. Defaults to the teacher
                           (pi05_libero_finetuned_v044) because student
                           checkpoints don't ship processor configs.
    """
    if tasks == "all":
        task_list = None
    else:
        task_list = [int(t) for t in tasks.split(",")]
    if cache not in {"none", "phash", "action", "episode"}:
        raise ValueError(f"--cache must be 'none'|'phash'|'action'|'episode', got {cache!r}")

    print(f"Running decomposed LIBERO {suite}: cache={cache}, "
          f"tasks={task_list or 'all'}, {num_episodes} eps each")
    r = run_decomposed_libero.remote(
        student_checkpoint=student_checkpoint,
        decomposed_dir=decomposed_dir,
        cache_mode=cache,
        num_episodes=num_episodes,
        task_suite_name=suite,
        task_indices=task_list,
        cache_ttl_sec=cache_ttl_sec,
        cache_max_age_steps=cache_max_age_steps,
        action_cache_max_age_steps=action_cache_max_age_steps,
        phash_hamming=phash_hamming,
        preprocessor_ref=preprocessor_ref,
        seed=seed,
        save_video_dir=save_video_dir,
    )
    print("\n=== RESULT ===")
    print(f"  model: {r.get('model')}")
    print(f"  cache: {r.get('cache_mode')}")
    print(f"  success_rate: {r.get('success_rate_pct', 0):.1f}%")
    print(f"  total: {r['total_success']}/{r['total_eps']}")
    print(f"  cache_stats: {r.get('cache_stats')}")
    print(f"  errors: {len(r.get('errors', []))}")
    for task in r.get("per_task", []):
        print(f"  task {task['task_idx']}: "
              f"{task['success']}/{task['total']} — "
              f"{task['task_description'][:60]}")
