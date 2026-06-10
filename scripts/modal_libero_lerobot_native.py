"""LIBERO-10 eval — port of OpenPI's battle-tested rollout.

This is a rewrite (2026-04-19). Prior iterations (9 runs) got 0% because our
script missed critical pieces from the reference implementation. This version
is a line-by-line port of:

    openpi/examples/libero/main.py (battle-tested OpenPI LIBERO rollout)

cloned at `/Users/romirjain/Desktop/building projects/openpi/`, adapted for
in-process SmolVLAPolicy (OpenPI uses a WebsocketClient to a separate model
server; we call policy.select_action directly).

10 fixes applied vs prior version:
  1. max_steps=520 for libero_10 (was 300 — truncated episodes)
  2. Use OffScreenRenderEnv directly (was lerobot's gym-wrapped LiberoEnv —
     obs was nested under 'pixels' with missing state)
  3. Init state rotation: env.set_init_state(initial_states[episode_idx])
     per episode (fixes lerobot#2375 trap where every episode hits init_0)
  4. num_steps_wait=10 zero-action settling at episode start (objects are
     still falling)
  5. Correct _quat2axisangle (magnitude-preserving, copied from robosuite)
  6. Resize images to 224×224 with pad before policy inference
  7. replan_steps=5 action-plan deque (was: select_action every env step)
  8. Env resolution 256×256 (was: lerobot's 360×360 default)
  9. env.seed(7) for reproducibility
 10. 180° H+W image flip via numpy slicing (matches OpenPI exactly)

Usage:
    modal run scripts/modal_libero_lerobot_native.py --tasks 0 --num-episodes 1
    modal run scripts/modal_libero_lerobot_native.py --tasks all --num-episodes 5
"""
import os
import subprocess
import modal

app = modal.App("tether-libero-lerobot-native")


def _hf_secret():
    """HF token secret. Needed for gated upstream dependencies (e.g.
    pi0.5 pulls google/paligemma-3b-pt-224 at tokenizer init).

    Dispatch (same pattern as modal_tether_distill.py):
      1. Local HF_TOKEN env var → wrap in ephemeral Secret.
      2. Pre-registered Modal secret 'huggingface' → use as-is.
      3. Empty dict fallback → works for fully-public models only.
    """
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


# Image: Python 3.12 + lerobot 0.5.1 + LIBERO + MuJoCo + robosuite 1.4.1.
# mujoco==3.3.2 per lerobot#2258 (older versions render different colors).
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
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "mujoco==3.3.2",  # pinned per lerobot#2258
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
        "imageio",  # replay video save (optional)
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    # Pull tether so `tether.distill.snapflow_pi0_model` is importable
    # when --snapflow-student is used. Uses the github-token secret to
    # clone the private repo. Cheap — this only pulls the Python package.
    .run_commands(
        f'pip install "fastcrest-tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
    })
    .run_commands("mkdir -p /tmp/libero_data")
)


# ─── Constants matching OpenPI reference ─────────────────────────────
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


_onnx_output_volume = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    secrets=[_hf_secret()],
    volumes={
        "/onnx_out": _onnx_output_volume,
        "/root/.cache/huggingface": _hf_cache_volume,
    },
)
def run_ported_libero(
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    num_episodes: int = 1,
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
    snapflow_student: str = "",
    snapflow_onnx: str = "",
    preprocessor_ref: str = "",
    save_video_dir: str = "",
):
    """Port of openpi/examples/libero/main.py rolled out end-to-end.

    Returns per-task success summary. Uses OffScreenRenderEnv directly
    (NOT lerobot's gym-wrapped LiberoEnv) for OpenPI-style obs access.
    """
    import collections
    import math
    import time
    import traceback
    import numpy as np
    import torch

    # PyTorch 2.6+ changed torch.load default to weights_only=True, which
    # refuses to unpickle LIBERO's init_state files (they embed numpy globals).
    # LIBERO is a trusted local source; force weights_only=False globally.
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load

    # ─── Load policy (in-process, not via websocket) ─────────────────
    print(f"[ported] Loading {model_id}...")
    t0 = time.time()
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download

    # Dispatch:
    # - snapflow_onnx set → load ONNX session via onnxruntime (no PyTorch
    #   inference at all, ORT does it). Still load load_snapflow_student
    #   for the policy.config + _preprocess_images helpers.
    # - snapflow_student set → load PyTorch student, use sample_actions_1step.
    # - else: dispatch normal lerobot policy class by model_id prefix.
    use_1nfe = bool(snapflow_student)
    use_onnx = bool(snapflow_onnx)
    ort_session = None

    if use_onnx and not snapflow_student:
        raise ValueError(
            "--snapflow-onnx requires --snapflow-student too — the harness "
            "still needs the PyTorch policy for preprocessing helpers "
            "(_preprocess_images / config)."
        )

    if use_1nfe:
        from tether.distill.snapflow_pi0_model import load_snapflow_student
        print(f"[ported] Loading SnapFlow student from {snapflow_student} (1-NFE inference)")
        policy = load_snapflow_student(snapflow_student)
        detected_type = "snapflow_student_onnx" if use_onnx else "snapflow_student"
        repo_dir = preprocessor_ref or snapflow_student
        if use_onnx:
            import onnxruntime as ort
            print(f"[ported] Loading ONNX session from {snapflow_onnx}")
            ort_session = ort.InferenceSession(snapflow_onnx, providers=["CPUExecutionProvider"])
            ort_input_dtypes = {
                i.name: i.type for i in ort_session.get_inputs()
            }
            print(f"[ported] ORT inputs: {[(n, t) for n, t in ort_input_dtypes.items()]}")
    else:
        model_key = model_id.split("/")[-1].lower()
        if model_key.startswith("pi05") or model_key.startswith("pi0.5") or "pi05" in model_key:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            policy_cls = PI05Policy
            detected_type = "pi05"
        elif model_key.startswith("pi0") or "pi0" in model_key:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            policy_cls = PI0Policy
            detected_type = "pi0"
        else:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            policy_cls = SmolVLAPolicy
            detected_type = "smolvla"
        print(f"[ported] Detected policy type: {detected_type} ({policy_cls.__name__})")
        policy = policy_cls.from_pretrained(model_id)
        repo_dir = snapshot_download(model_id)

    policy.eval().to("cuda").to(torch.float32)

    # v0.5 state-out detection: if the loaded student is the state-out
    # variant, the preprocessor must be patched to strip state from the
    # language prompt and emit OBS_STATE as a separate batch key.
    is_state_out = (
        snapflow_student
        and type(policy.model).__name__ == "SnapFlowPI05StateOutPytorch"
    )
    if is_state_out:
        print(f"[ported] Detected v0.5 state-out student — will swap preprocessor")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    if is_state_out:
        from tether.distill.pi05_state_out_processor import swap_prepare_step_in_pipeline
        max_state_dim = getattr(policy.config, "max_state_dim", 32)
        swap_prepare_step_in_pipeline(preprocessor, max_state_dim=max_state_dim)
        print(f"[ported] Swapped preprocessor for state-out (max_state_dim={max_state_dim})")
    # Postprocessor unnormalizes policy output back to env action space.
    # Without this, predict_action_chunk returns normalized (zero-mean,
    # unit-variance) values which the env interprets as wildly wrong
    # deltas. This is what was missing in the 0/3 run on romirj profile.
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print(f"[ported] Policy + pre+post processors loaded in {time.time()-t0:.1f}s")

    # ─── Set up LIBERO suite ─────────────────────────────────────────
    np.random.seed(seed)
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
    print(f"[ported] suite={task_suite_name}, num_tasks={num_tasks_in_suite}, "
          f"max_steps={max_steps}")

    # Helper — _quat2axisangle, verbatim from OpenPI (robosuite formula).
    def _quat2axisangle(quat):
        if quat[3] > 1.0: quat[3] = 1.0
        elif quat[3] < -1.0: quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    # Helper — resize with pad (matches openpi_client.image_tools.resize_with_pad).
    # Pads to square first, then resizes. Preserves aspect ratio.
    def _resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image
        h, w = img.shape[:2]
        # Pad to square with zeros
        if h > w:
            pad = (h - w) // 2
            img = np.pad(img, [(0, 0), (pad, h - w - pad), (0, 0)], mode="constant")
        elif w > h:
            pad = (w - h) // 2
            img = np.pad(img, [(pad, w - h - pad), (0, 0), (0, 0)], mode="constant")
        # Resize
        pil = Image.fromarray(img)
        pil = pil.resize((size, size), Image.BILINEAR)
        return np.asarray(pil)

    def _build_env(task):
        task_bddl_file = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)
        return env

    def _build_batch(obs, task_description):
        """Build batch in SmolVLA's expected format, mirroring OpenPI obs."""
        # Images — 180° flip per LIBERO convention, resize to 224 with pad
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = _resize_with_pad(img, resize_size)
        wrist_img = _resize_with_pad(wrist_img, resize_size)

        def _to_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            return t.permute(2, 0, 1).unsqueeze(0).to("cuda")

        # State: eef_pos(3) + axis-angle(3) + gripper_qpos(2) = 8D
        state = np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]).astype(np.float32)

        batch = {
            "observation.images.image": _to_tensor(img),
            "observation.images.image2": _to_tensor(wrist_img),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to("cuda"),
            "task": [task_description],
        }
        return batch

    # ─── Results struct ──────────────────────────────────────────────
    results = {
        "model": model_id,
        "harness": "openpi-port-lerobot-native",
        "suite": task_suite_name,
        "num_episodes_per_task": num_episodes,
        "max_steps": max_steps,
        "resize_size": resize_size,
        "replan_steps": replan_steps,
        "num_steps_wait": num_steps_wait,
        "per_task": [],
        "total_success": 0,
        "total_eps": 0,
        "errors": [],
    }

    tasks_to_run = task_indices if task_indices is not None else list(range(num_tasks_in_suite))
    print(f"[ported] Running tasks: {tasks_to_run}")

    # ─── Main loop: tasks × episodes ─────────────────────────────────
    for task_idx in tasks_to_run:
        task = task_suite.get_task(task_idx)
        task_description = task.language
        print(f"\n[ported] TASK {task_idx}: {task_description!r}")
        initial_states = task_suite.get_task_init_states(task_idx)
        print(f"[ported] {len(initial_states)} init states available")

        env = _build_env(task)
        task_start = time.time()
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
                # CRITICAL: rotate init state per episode (fixes #2375)
                init_idx = ep % len(initial_states)
                obs = env.set_init_state(initial_states[init_idx])
                policy.reset()
                action_plan = collections.deque()
                t = 0
                done = False
                video_frames = [] if save_video_dir else None
                if video_frames is not None:
                    video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

                while t < max_steps + num_steps_wait:
                    try:
                        # num_steps_wait: let objects settle
                        if t < num_steps_wait:
                            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                            if video_frames is not None:
                                video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            t += 1
                            continue

                        # Dump obs schema on first real step
                        if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                            obs_info = {
                                k: (obs[k].shape if hasattr(obs[k], "shape") else type(obs[k]).__name__)
                                for k in sorted(obs.keys())
                                if any(x in k.lower() for x in ["image", "eef", "gripper", "joint"])
                            }
                            print(f"[debug] obs keys: {obs_info}")

                        if not action_plan:
                            batch = _build_batch(obs, task_description)
                            batch_pp = preprocessor(batch)
                            batch_pp = {
                                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                                for k, v in batch_pp.items()
                            }
                            if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                                print(f"[debug] batch_pp keys: {sorted(batch_pp.keys())}")
                            with torch.no_grad():
                                if use_onnx:
                                    # SnapFlow 1-NFE via ONNX (FP32 or FP16).
                                    # Same preprocessing path as PyTorch
                                    # 1-NFE; only difference is ORT executes
                                    # the model. We sample noise here so the
                                    # ORT input is deterministic per call.
                                    from lerobot.utils.constants import (
                                        OBS_LANGUAGE_ATTENTION_MASK,
                                        OBS_LANGUAGE_TOKENS, ACTION,
                                    )
                                    images, img_masks = policy._preprocess_images(batch_pp)
                                    lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
                                    lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
                                    cfg = policy.config
                                    chunk_size = cfg.chunk_size
                                    action_dim_pad = cfg.max_action_dim
                                    bsize = images[0].shape[0]
                                    noise = torch.randn(
                                        bsize, chunk_size, action_dim_pad,
                                        device=images[0].device, dtype=torch.float32,
                                    )
                                    np_inputs = {
                                        "img_base": images[0].cpu().numpy(),
                                        "img_wrist_l": images[1].cpu().numpy(),
                                        "img_wrist_r": images[2].cpu().numpy(),
                                        "mask_base": img_masks[0].cpu().numpy(),
                                        "mask_wrist_l": img_masks[1].cpu().numpy(),
                                        "mask_wrist_r": img_masks[2].cpu().numpy(),
                                        "lang_tokens": lang_tokens.cpu().numpy(),
                                        "lang_masks": lang_masks.cpu().numpy(),
                                        "noise": noise.cpu().numpy(),
                                    }
                                    # FP16 export was built with keep_io_types=False;
                                    # cast float inputs to fp16 to match the session.
                                    import numpy as _np
                                    for k, v in list(np_inputs.items()):
                                        meta_type = ort_input_dtypes.get(k, "")
                                        if "float16" in meta_type and v.dtype == _np.float32:
                                            np_inputs[k] = v.astype(_np.float16)
                                    chunk_np = ort_session.run(["actions"], np_inputs)[0]
                                    if chunk_np.dtype != _np.float32:
                                        chunk_np = chunk_np.astype(_np.float32)
                                    chunk = torch.from_numpy(chunk_np).to(images[0].device)
                                    orig_dim = cfg.output_features[ACTION].shape[0]
                                    chunk = chunk[:, :, :orig_dim]
                                elif use_1nfe:
                                    # SnapFlow 1-NFE: single denoise_step
                                    # at target_time=1 via the subclass override.
                                    # Three signatures depending on policy:
                                    #   pi0:                state required
                                    #   pi0.5 (default):    no state
                                    #   pi0.5 (state-out, v0.5): state required
                                    from lerobot.utils.constants import (
                                        OBS_LANGUAGE_ATTENTION_MASK,
                                        OBS_LANGUAGE_TOKENS,
                                        OBS_STATE,
                                    )
                                    images, img_masks = policy._preprocess_images(batch_pp)
                                    lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
                                    lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
                                    if hasattr(policy, "prepare_state"):
                                        # pi0 path: needs state
                                        state_t = policy.prepare_state(batch_pp)
                                        chunk = policy.model.sample_actions_1step(
                                            images, img_masks, lang_tokens, lang_masks, state_t,
                                        )
                                    elif is_state_out:
                                        # pi0.5 state-out variant (v0.5): state from preprocessor
                                        state_t = batch_pp[OBS_STATE]
                                        # Pad state to max_state_dim if needed
                                        max_sd = policy.model.state_proj.in_features
                                        if state_t.shape[-1] < max_sd:
                                            import torch.nn.functional as F
                                            state_t = F.pad(state_t, (0, max_sd - state_t.shape[-1]))
                                        chunk = policy.model.sample_actions_1step(
                                            images, img_masks, lang_tokens, lang_masks, state_t,
                                        )
                                    else:
                                        # pi0.5 default path: no state
                                        chunk = policy.model.sample_actions_1step(
                                            images, img_masks, lang_tokens, lang_masks,
                                        )
                                    # predict_action_chunk trims max_action_dim
                                    # (32) down to the real env action dim (7
                                    # for LIBERO) before returning. Our 1-NFE
                                    # path bypasses that wrapper, so replicate
                                    # the trim here.
                                    from lerobot.utils.constants import ACTION
                                    orig_dim = policy.config.output_features[ACTION].shape[0]
                                    chunk = chunk[:, :, :orig_dim]
                                else:
                                    chunk = policy.predict_action_chunk(batch_pp)
                            # Apply postprocessor (unnormalizes back to env
                            # action space). chunk shape (1, chunk_size, N);
                            # postprocessor operates on CPU tensors.
                            post = postprocessor(chunk.detach().cpu())
                            chunk_np = (
                                post.detach().cpu().numpy()
                                if hasattr(post, "detach")
                                else np.asarray(post)
                            )
                            if chunk_np.ndim == 3:
                                chunk_np = chunk_np[0]  # → (chunk_size, N)
                            # Trim to 7-dim LIBERO action
                            chunk_np = chunk_np[:, :7]
                            action_plan.extend(chunk_np[:replan_steps])
                            if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                                print(f"[debug] first action: {chunk_np[0]}")

                        action = action_plan.popleft()
                        obs, _, done, info = env.step(action.tolist())
                        if video_frames is not None:
                            video_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        if done:
                            task_result["success"] += 1
                            results["total_success"] += 1
                            break
                        t += 1
                    except Exception as e:
                        err_tb = traceback.format_exc()
                        print(f"  step error: {e}")
                        print(err_tb[-800:])
                        results["errors"].append({
                            "task": task_idx, "ep": ep,
                            "error": str(e), "tb": err_tb[-400:],
                        })
                        break

                # Cast to Python primitives so the result dict deserializes
                # cleanly on any local Python (numpy.bool_ pickles with numpy
                # globals → Modal client needs numpy to unpack, avoidable).
                task_result["episodes"].append({
                    "ep": int(ep),
                    "init_idx": int(init_idx),
                    "steps": int(t),
                    "success": bool(done),
                })
                task_result["total"] += 1
                results["total_eps"] += 1
                print(f"  ep {ep} (init_idx={init_idx}): "
                      f"{'SUCCESS' if done else 'fail'} at {t} steps "
                      f"({time.time()-task_start:.1f}s total)")
                if video_frames is not None and len(video_frames) > 0:
                    from pathlib import Path as _P
                    _P(save_video_dir).mkdir(parents=True, exist_ok=True)
                    tag = "S" if done else "F"
                    out = _P(save_video_dir) / (
                        f"teacher_t{task_idx}_ep{ep}_seed{seed}_steps{t}_{tag}.npz"
                    )
                    np.savez_compressed(str(out), frames=np.array(video_frames, dtype=np.uint8))
                    print(f"    frames → {out} ({len(video_frames)} frames)")
            except Exception as e:
                err_tb = traceback.format_exc()
                print(f"  episode error: {e}")
                print(err_tb[-1000:])
                results["errors"].append({
                    "task": task_idx, "ep": ep,
                    "error": str(e), "tb": err_tb[-400:],
                })
                task_result["total"] += 1
                results["total_eps"] += 1

        results["per_task"].append(task_result)
        print(f"[ported] task {task_idx} done: "
              f"{task_result['success']}/{task_result['total']}")
        try:
            env.close()
        except Exception:
            pass

    success_rate = (
        100.0 * results["total_success"] / results["total_eps"]
        if results["total_eps"] else 0.0
    )
    results["success_rate_pct"] = round(success_rate, 1)
    print(f"\n====== {task_suite_name} (OpenPI-ported) ======")
    print(f"  Model: {model_id}")
    print(f"  Success: {results['total_success']}/{results['total_eps']} "
          f"= {success_rate:.1f}%")
    return results


@app.local_entrypoint()
def main(
    num_episodes: int = 1,
    tasks: str = "0",
    suite: str = "libero_10",
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    snapflow_student: str = "",
    snapflow_onnx: str = "",
    preprocessor_ref: str = "",
    save_video_dir: str = "",
):
    """
    --num-episodes N          episodes per task (OpenPI default: 50)
    --tasks "0"               single task
    --tasks "0,1,2"           multiple
    --tasks "all"             all tasks in suite
    --suite                   libero_10|libero_spatial|libero_object|libero_goal|libero_90
    --model-id                HF repo id (when not using snapflow student).
    --snapflow-student PATH   Path to a SnapFlow distilled student checkpoint
                              dir on the Modal volume (e.g.
                              /onnx_out/distill_v031_pi05_libero/training/
                              checkpoints/00010000/pretrained_model). When
                              set, loads via load_snapflow_student and uses
                              1-NFE inference (sample_actions_1step).
    --preprocessor-ref HFID   Fallback HF id for preprocessor pipeline when
                              the student checkpoint dir is missing the
                              policy_preprocessor.json (defaults to the
                              student dir itself — tether saves processors
                              alongside safetensors).
    """
    if tasks == "all":
        task_list = None  # run all
    else:
        task_list = [int(t) for t in tasks.split(",")]
    which = f"snapflow-student={snapflow_student}" if snapflow_student else f"model={model_id}"
    print(f"Running OpenPI-port LIBERO {suite}: {which} tasks={task_list or 'all'}, "
          f"{num_episodes} eps each")
    r = run_ported_libero.remote(
        model_id=model_id,
        num_episodes=num_episodes,
        task_suite_name=suite,
        task_indices=task_list,
        snapflow_student=snapflow_student,
        snapflow_onnx=snapflow_onnx,
        preprocessor_ref=preprocessor_ref,
        save_video_dir=save_video_dir,
    )
    print("\n=== RESULT ===")
    print(f"  model: {r.get('model')}")
    print(f"  success_rate: {r.get('success_rate_pct', '?')}%")
    print(f"  total: {r['total_success']}/{r['total_eps']}")
    print(f"  errors: {len(r.get('errors', []))}")
    for task in r.get("per_task", []):
        print(f"  task {task['task_idx']}: "
              f"{task['success']}/{task['total']} — "
              f"{task['task_description'][:60]}")
