"""LIBERO-10 eval via the **monolithic ONNX** path (customer-facing).

Counterpart to `modal_libero_lerobot_native.py` which runs the native
PyTorch path and landed 10/25 = 40% success at N=25. This script runs
the SAME 25-episode harness but swaps the core forward for the
exported `smolvla_libero_monolithic/model.onnx` — the artifact shipped
to customers as the ONNX engine source.

Goal: confirm that cos=+1.000000 parity translates to matching
task success. If native=40% and ONNX≈40%, the "cos=1.0 implies
deployable" claim is load-bearing. If ONNX<30%, something in the
export is dropping task-relevant information.

Model path: `/onnx_out/smolvla_libero_monolithic/model.onnx` on the
`pi0-onnx-outputs` Modal volume (exported via
`modal run scripts/modal_smolvla_monolithic_export.py
 --model-id HuggingFaceVLA/smolvla_libero
 --out-subdir smolvla_libero_monolithic`).

The 10 ONNX inputs mirror `SmolVLAMonolithicWrapper.forward`:
  img_cam1..3, mask_cam1..3, lang_tokens, lang_masks, state, noise

We use the same SmolVLAPolicy instance to build these — specifically
`prepare_images` / `prepare_state` to get pre-SigLIP images and padded
state, then pull `observation.language.tokens` / `attention_mask`
from the preprocessor output. The ONNX does the rest (VLM + denoise).

Usage:
    modal run scripts/modal_libero_monolithic_onnx.py
    modal run scripts/modal_libero_monolithic_onnx.py --tasks 0,1,2,3,4 --num-episodes 5
"""
import os
import subprocess
import modal

app = modal.App("tether-libero-monolithic-onnx")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
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

# Reuse the smolvla export volume (pi0-onnx-outputs) so we don't need
# to rebuild or re-export.
hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


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
        "onnxruntime-gpu>=1.20",
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


# ─── Constants (match native harness) ─────────────────────────────────
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def run_libero_onnx(
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    onnx_subdir: str = "smolvla_libero_monolithic",
    num_episodes: int = 1,
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
):
    """Port of native LIBERO harness — ONNX variant."""
    import collections
    import math
    import time
    import traceback
    from pathlib import Path
    import numpy as np
    import onnxruntime as ort
    import torch

    # PyTorch 2.6+ weights_only=True refuses LIBERO init states.
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ─── Load SmolVLAPolicy (for preprocessor/postprocessor + prepare_* helpers) ──
    # We use the policy to build ONNX inputs but bypass its forward. This is
    # the minimal-diff way to run the ONNX vs native comparison — same
    # pre/post-processing, only the core forward differs.
    print(f"[onnx] Loading {model_id} for preprocessor...")
    t0 = time.time()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download

    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval().to("cuda").to(torch.float32)
    repo_dir = snapshot_download(model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print(f"[onnx] Policy + pre/post in {time.time()-t0:.1f}s")

    # Config dims (for noise shape + unpad)
    original_action_dim = policy.config.action_feature.shape[0]
    max_action_dim = policy.config.max_action_dim
    chunk_size = policy.config.chunk_size
    print(f"[onnx] action_dim orig={original_action_dim} max={max_action_dim} "
          f"chunk={chunk_size}")

    # ─── Load ONNX ───────────────────────────────────────────────────
    onnx_path = Path(ONNX_OUTPUT_PATH) / onnx_subdir / "model.onnx"
    if not onnx_path.exists():
        return {"status": "fail", "reason": f"{onnx_path} not found"}

    size_gb = onnx_path.stat().st_size / 1e9
    data_files = list(onnx_path.parent.glob("*.data")) + list(onnx_path.parent.glob("*.bin"))
    data_gb = sum(f.stat().st_size for f in data_files) / 1e9
    print(f"[onnx] Loading {onnx_path} ({size_gb:.1f}GB + {data_gb:.1f}GB external)...")

    t0 = time.time()
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    # Probe input names so we know which camera-naming convention this export uses.
    # SmolVLA exports use img_cam1/cam2/cam3; pi05 exports use img_base/wrist_l/wrist_r.
    # Both have lang_tokens/lang_masks/state/noise but the camera names differ.
    # Without this probe, the wrong feed dict gets a "Required inputs missing" error
    # at first sess.run (caught 2026-04-26 firing teacher pi05 eval through this script).
    _input_names = {inp.name for inp in sess.get_inputs()}
    _input_shapes = {inp.name: inp.shape for inp in sess.get_inputs()}
    if "img_cam1" in _input_names:
        _cam_keys = ("img_cam1", "img_cam2", "img_cam3", "mask_cam1", "mask_cam2", "mask_cam3")
        print("[onnx] cam naming: SmolVLA-style (cam1/cam2/cam3)")
    elif "img_base" in _input_names:
        _cam_keys = ("img_base", "img_wrist_l", "img_wrist_r", "mask_base", "mask_wrist_l", "mask_wrist_r")
        print("[onnx] cam naming: pi05-style (base/wrist_l/wrist_r)")
    else:
        raise RuntimeError(
            f"Unknown camera-naming convention in ONNX inputs: {sorted(_input_names)}. "
            f"Expected either img_cam1 (SmolVLA) or img_base (pi05) as first image input."
        )
    # Probe shape of cam3 / wrist_r so the empty-camera padding matches the
    # ONNX's expected resolution. Pi05's empty_camera_0 is 224x224 while
    # cam1/cam2 are 256x256; SmolVLA uses 256x256 for all 3. Without this
    # probe, the cam3 zero-tensor pad has the wrong HxW and ORT throws
    # "INVALID_ARGUMENT: Got invalid dimensions for input: img_wrist_r".
    _cam3_shape = _input_shapes[_cam_keys[2]]  # e.g. ['batch', 3, 224, 224]
    print(f"[onnx] cam3 expected shape: {_cam3_shape}")
    used_providers = sess.get_providers()
    print(f"[onnx]   loaded in {time.time()-t0:.1f}s — providers={used_providers}")
    input_names = [i.name for i in sess.get_inputs()]
    print(f"[onnx]   inputs: {input_names}")

    # Detect the ONNX's expected lang_seq. Earlier exports hardcoded 16;
    # if a future export uses dynamic or seq=48, honor it. Falls back to
    # 16 if the shape is symbolic or missing (safer for task 0-3, will
    # silently truncate task 4's 21-token prompt — see
    # reflex_context/06_experiments/task34_gap_audit.md).
    expected_lang_seq = 16
    for i in sess.get_inputs():
        if i.name == "lang_tokens":
            shape = i.shape
            # Dim 1 is the seq length; pick static int, else stay 16.
            if len(shape) >= 2 and isinstance(shape[1], int):
                expected_lang_seq = int(shape[1])
            break
    print(f"[onnx]   lang_seq (detected): {expected_lang_seq}")

    # ─── LIBERO setup ────────────────────────────────────────────────
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
    print(f"[onnx] suite={task_suite_name}, num_tasks={num_tasks_in_suite}, "
          f"max_steps={max_steps}")

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
        from PIL import Image
        h, w = img.shape[:2]
        if h > w:
            pad = (h - w) // 2
            img = np.pad(img, [(0, 0), (pad, h - w - pad), (0, 0)], mode="constant")
        elif w > h:
            pad = (w - h) // 2
            img = np.pad(img, [(pad, w - h - pad), (0, 0), (0, 0)], mode="constant")
        pil = Image.fromarray(img)
        pil = pil.resize((size, size), Image.BILINEAR)
        return np.asarray(pil)

    def _build_env(task):
        task_bddl_file = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(seed)
        return env

    def _build_batch(obs, task_description):
        """Identical to native harness."""
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = _resize_with_pad(img, resize_size)
        wrist_img = _resize_with_pad(wrist_img, resize_size)

        def _to_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            return t.permute(2, 0, 1).unsqueeze(0).to("cuda")

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

    def _onnx_predict_chunk(batch_pp) -> torch.Tensor:
        """Run the monolithic ONNX. Returns raw chunk [B, chunk, max_action_dim].

        We reuse the policy's prepare_images/prepare_state for parity with
        the native path — these are the exact same transforms the ONNX
        export traced. The preprocessor already added
        observation.language.tokens / attention_mask.
        """
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        images, img_masks = policy.prepare_images(batch_pp)
        # The ONNX wrapper expects exactly 3 cameras. LIBERO only has 2
        # (agentview + wrist); pad the 3rd slot with a zero image + empty
        # mask. Use the ONNX's expected cam3 shape (probed at session-load
        # time) -- pi05's empty_camera_0 is 224x224 while cam1/cam2 are
        # 256x256 (SmolVLA uses 256x256 for all 3). Caught 2026-04-26.
        while len(images) < 3:
            if len(images) == 2 and isinstance(_cam3_shape, list) and len(_cam3_shape) == 4:
                # _cam3_shape: ['batch'|N, C, H, W]
                _cam3_h = _cam3_shape[2] if isinstance(_cam3_shape[2], int) else images[0].shape[-2]
                _cam3_w = _cam3_shape[3] if isinstance(_cam3_shape[3], int) else images[0].shape[-1]
                pad_img = torch.full(
                    (images[0].shape[0], images[0].shape[1], _cam3_h, _cam3_w),
                    -1.0, dtype=images[0].dtype, device=images[0].device,
                )
            else:
                pad_img = torch.ones_like(images[0]) * -1.0
            images.append(pad_img)
            img_masks.append(torch.zeros_like(img_masks[0]))
        state = policy.prepare_state(batch_pp)
        lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
        lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]
        # Pad/trim to whatever the ONNX expects (detected at load time).
        # Historically seq=16; re-exports with dynamic seq or seq=48 will
        # work through the same path.
        target_seq = expected_lang_seq
        cur_seq = lang_tokens.shape[1]
        if cur_seq < target_seq:
            pad_len = target_seq - cur_seq
            B = lang_tokens.shape[0]
            tok_pad = torch.zeros(B, pad_len, dtype=lang_tokens.dtype,
                                  device=lang_tokens.device)
            mask_pad = torch.zeros(B, pad_len, dtype=lang_masks.dtype,
                                   device=lang_masks.device)
            lang_tokens = torch.cat([lang_tokens, tok_pad], dim=1)
            lang_masks = torch.cat([lang_masks, mask_pad], dim=1)
        elif cur_seq > target_seq:
            lang_tokens = lang_tokens[:, :target_seq]
            lang_masks = lang_masks[:, :target_seq]

        # Generate noise the same way SmolVLA.sample_actions does when None.
        B = state.shape[0]
        noise = torch.randn(
            B, chunk_size, max_action_dim,
            device=state.device, dtype=state.dtype,
        )

        # Assemble 10-input feed
        def _np(t, dtype=None):
            a = t.detach().cpu().numpy()
            return a.astype(dtype) if dtype is not None else a

        # Resize each camera tensor to the ONNX's expected shape if it differs.
        # SmolVLA: all 3 cams 256x256. Pi05: image/image2 256x256, empty_camera 224x224
        # per preprocessor JSON, but the actual ONNX trace shape is what matters --
        # probed at session-load time. Use F.interpolate to handle any HxW mismatch.
        # Caught 2026-04-26: cam2 (img_wrist_l) feed at 256x256 but pi05 ONNX
        # expects 224x224.
        def _resize_to_expected(tensor: torch.Tensor, cam_key: str) -> torch.Tensor:
            shape = _input_shapes.get(cam_key)
            if not (isinstance(shape, list) and len(shape) == 4):
                return tensor
            exp_h = shape[2] if isinstance(shape[2], int) else None
            exp_w = shape[3] if isinstance(shape[3], int) else None
            if exp_h is None or exp_w is None:
                return tensor
            if tensor.shape[-2] == exp_h and tensor.shape[-1] == exp_w:
                return tensor
            import torch.nn.functional as F
            return F.interpolate(
                tensor, size=(exp_h, exp_w), mode="bilinear", align_corners=False
            )

        feed = {
            _cam_keys[0]: _np(_resize_to_expected(images[0], _cam_keys[0]), np.float32),
            _cam_keys[1]: _np(_resize_to_expected(images[1], _cam_keys[1]), np.float32),
            _cam_keys[2]: _np(_resize_to_expected(images[2], _cam_keys[2]), np.float32),
            _cam_keys[3]: _np(img_masks[0], bool),
            _cam_keys[4]: _np(img_masks[1], bool),
            _cam_keys[5]: _np(img_masks[2], bool),
            "lang_tokens": _np(lang_tokens, np.int64),
            "lang_masks": _np(lang_masks, bool),
            "state": _np(state, np.float32),
            "noise": _np(noise, np.float32),
        }
        # pi05 export inputs DON'T include 'state' (state is fed via lang prompt
        # for state-in exports, or via state_proj for state-out — but the
        # monolithic export doesn't expose state_proj as a separate input).
        # SmolVLA includes 'state'. Drop the key if not in inputs.
        if "state" not in _input_names:
            feed.pop("state", None)

        out = sess.run(["actions"], feed)[0]  # [B, chunk, max_action_dim]
        return torch.from_numpy(out)

    # ─── Results struct ──────────────────────────────────────────────
    results = {
        "model": model_id,
        "harness": "onnx-monolithic",
        "suite": task_suite_name,
        "onnx_path": str(onnx_path),
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
    print(f"[onnx] Running tasks: {tasks_to_run}")

    for task_idx in tasks_to_run:
        task = task_suite.get_task(task_idx)
        task_description = task.language
        print(f"\n[onnx] TASK {task_idx}: {task_description!r}")
        initial_states = task_suite.get_task_init_states(task_idx)

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
                init_idx = ep % len(initial_states)
                obs = env.set_init_state(initial_states[init_idx])
                policy.reset()
                action_plan = collections.deque()
                t = 0
                done = False

                while t < max_steps + num_steps_wait:
                    try:
                        if t < num_steps_wait:
                            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

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

                            # THE SWAP: ONNX replaces policy.predict_action_chunk
                            raw_chunk = _onnx_predict_chunk(batch_pp)  # CPU tensor

                            # Unpad to original_action_dim (native does this too)
                            chunk = raw_chunk[:, :, :original_action_dim]

                            # Postprocess (unnormalize)
                            post = postprocessor(chunk)
                            chunk_np = (
                                post.detach().cpu().numpy()
                                if hasattr(post, "detach")
                                else np.asarray(post)
                            )
                            if chunk_np.ndim == 3:
                                chunk_np = chunk_np[0]
                            chunk_np = chunk_np[:, :7]
                            action_plan.extend(chunk_np[:replan_steps])
                            if t == num_steps_wait and ep == 0 and task_idx == tasks_to_run[0]:
                                print(f"[debug] first action: {chunk_np[0]}")

                        action = action_plan.popleft()
                        obs, _, done, info = env.step(action.tolist())
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
        print(f"[onnx] task {task_idx} done: "
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
    print(f"\n====== {task_suite_name} (ONNX monolithic) ======")
    print(f"  Model: {model_id}")
    print(f"  Success: {results['total_success']}/{results['total_eps']} "
          f"= {success_rate:.1f}%")
    return results


@app.local_entrypoint()
def main(
    num_episodes: int = 1,
    tasks: str = "0",
    suite: str = "libero_10",
    onnx_subdir: str = "smolvla_libero_monolithic",
    seed: int = 7,
):
    """
    --num-episodes N: episodes per task (native used 5)
    --tasks "0"       single task
    --tasks "0,1,2,3,4"   N=25 matching native run
    --tasks "all"     all 10 tasks
    --onnx-subdir     subfolder under /onnx_out/ (default smolvla_libero_monolithic)
    --seed            RNG seed for LIBERO envs, NumPy, and Torch noise
    """
    if tasks == "all":
        task_list = None
    else:
        task_list = [int(t) for t in tasks.split(",")]
    print(f"Running ONNX-monolithic LIBERO {suite}: tasks={task_list or 'all'}, "
          f"{num_episodes} eps each")
    r = run_libero_onnx.remote(
        num_episodes=num_episodes,
        task_suite_name=suite,
        task_indices=task_list,
        onnx_subdir=onnx_subdir,
        seed=seed,
    )
    print("\n=== RESULT ===")
    # Early-return failure path (e.g., ONNX missing on volume) — surface
    # the status + reason so operators don't see opaque '?' counts.
    # Caught by 2026-04-25 eval-as-a-service Modal smoke validation.
    if r.get("status") == "fail":
        print("  status: FAIL")
        print(f"  reason: {r.get('reason', '(no reason)')}")
        return
    print(f"  success_rate: {r.get('success_rate_pct', '?')}%")
    print(f"  total: {r.get('total_success', '?')}/{r.get('total_eps', '?')}")
    print(f"  errors: {len(r.get('errors', []))}")
    for task in r.get("per_task", []):
        print(f"  task {task['task_idx']}: "
              f"{task['success']}/{task['total']} — "
              f"{task['task_description'][:60]}")
