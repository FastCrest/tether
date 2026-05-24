"""Lift #5 L3 side-by-side — native lerobot vs Triton on same LIBERO tasks.

Root-cause diagnostic for the 0/2 L3 diagnostic result. Runs BOTH arms:
  ARM A: native lerobot (PI05Policy.predict_action_chunk, fp32)
  ARM B: Triton (Pi05FastKernelsInference via TritonLIBEROAdapter, bf16)

Same tasks × same episodes × same init states. Compares success rates +
prints first-action values to diagnose systematic drift.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_l3_side_by_side.py
"""
import os
import subprocess

import modal

app = modal.App("reflex-fast-kernels-l3-side-by-side")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "lift/5-day1-2-vendor-triton-kernels"


_HEAD = _repo_head_sha()


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "ninja-build", "clang", "build-essential",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "libosmesa6", "libosmesa6-dev",
        "gnupg", "wget",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb"
        " && apt-get update"
        " && apt-get install -y cuda-toolkit-12-4 --no-install-recommends"
        " && rm cuda-keyring_1.1-1_all.deb",
    )
    .pip_install(
        "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "triton>=3.1", "ninja",
        "mujoco==3.3.2", "robosuite==1.4.1",
        "h5py", "bddl==1.0.1", "future", "robomimic",
        "hydra-core>=1.1", "easydict", "einops",
        "opencv-python-headless", "gym", "gymnasium",
        "lerobot==0.5.1", "num2words", "imageio",
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
    })
    .run_commands("mkdir -p /tmp/libero_data")
)


@app.function(
    image=image, gpu="A100-40GB", timeout=7200,
    secrets=[_hf_secret()],
)
def run_side_by_side(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    task_indices: list[int] | None = None,
    num_episodes: int = 3,
) -> dict:
    """Side-by-side: native lerobot vs Triton on same LIBERO tasks."""
    import collections
    import math
    import time

    import numpy as np
    import torch

    # torch.load patch for LIBERO init states
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load

    if task_indices is None:
        task_indices = [3, 4, 6]  # single-object tasks with higher baseline success

    # Disable PyTorch's Triton autotuner — it blocks the GPU for 30+ seconds
    # per matmul shape on first call, causing Modal to kill the task for
    # "failed to respond to cancellation." Set before any torch operations.
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[sbs] Side-by-side — model={model_id}, tasks={task_indices}, N={num_episodes}", flush=True)
    print(f"[sbs] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    t_total = time.time()

    # ── Load policy (CPU — shared for native ARM + preprocessing) ─────
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).cpu()
    policy.eval()
    print(f"[sbs] [{time.time()-t0:.1f}s] PI05Policy loaded (CPU)", flush=True)

    # ── Build Triton adapter ─────────────────────────────────────────
    t0 = time.time()
    from reflex.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter
    adapter = TritonLIBEROAdapter.from_policy(policy, capture=True)
    print(f"[sbs] [{time.time()-t0:.1f}s] Triton adapter ready", flush=True)

    # ── Load preprocessor / postprocessor ─────────────────────────────
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(repo_id=model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print(f"[sbs] Pre/post processors loaded", flush=True)

    # ── LIBERO setup ──────────────────────────────────────────────────
    np.random.seed(42)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path

    max_steps = 520
    resize_size = 224
    replan_steps = 5
    num_steps_wait = 10
    LIBERO_DUMMY_ACTION = np.zeros(7)

    def _quat2axisangle(quat):
        if quat[3] > 1.0: quat[3] = 1.0
        elif quat[3] < -1.0: quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def _resize_with_pad(img, size):
        from PIL import Image as PILImage
        h, w = img.shape[:2]
        if h > w:
            pad = (h - w) // 2
            img = np.pad(img, [(0, 0), (pad, h - w - pad), (0, 0)], mode="constant")
        elif w > h:
            pad = (w - h) // 2
            img = np.pad(img, [(pad, w - h - pad), (0, 0), (0, 0)], mode="constant")
        pil = PILImage.fromarray(img)
        pil = pil.resize((size, size), PILImage.BILINEAR)
        return np.asarray(pil)

    def _build_env(task):
        task_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl),
            camera_heights=256, camera_widths=256,
        )
        env.seed(42)
        return env

    def _build_batch(obs, task_description):
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

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()

    # ── Run both ARMs ─────────────────────────────────────────────────
    def _run_arm(arm_name, get_action_chunk):
        """Generic LIBERO eval loop. get_action_chunk(batch_pp) -> np.ndarray."""
        arm_results = {"tasks": [], "total_success": 0, "total_eps": 0}
        for task_idx in task_indices:
            task = task_suite.get_task(task_idx)
            task_desc = task.language
            initial_states = task_suite.get_task_init_states(task_idx)
            env = _build_env(task)
            task_success = 0
            print(f"\n[sbs] [{arm_name}] TASK {task_idx}: {task_desc!r}", flush=True)

            for ep in range(num_episodes):
                try:
                    env.reset()
                    obs = env.set_init_state(initial_states[ep % len(initial_states)])
                    policy.reset()
                    action_plan = collections.deque()
                    t = 0
                    first_action_logged = False

                    while t < max_steps + num_steps_wait:
                        if t < num_steps_wait:
                            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        if not action_plan:
                            batch = _build_batch(obs, task_desc)
                            batch_pp = preprocessor(batch)
                            batch_pp = {
                                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                                for k, v in batch_pp.items()
                            }
                            chunk_np = get_action_chunk(batch_pp)
                            action_plan.extend(chunk_np[:replan_steps])

                            if not first_action_logged:
                                print(f"[sbs] [{arm_name}] first action: {chunk_np[0][:4]}...", flush=True)
                                first_action_logged = True

                        action = action_plan.popleft()
                        obs, _, done, info = env.step(action)
                        t += 1

                        # Progress logging every 100 steps
                        if t % 100 == 0:
                            elapsed = time.time() - t_total
                            print(f"[sbs] [{arm_name}] task {task_idx} ep {ep} step {t}/{max_steps+num_steps_wait} ({elapsed:.0f}s elapsed)", flush=True)

                        if done:
                            break

                    success = bool(env.is_success()["task"]) if hasattr(env, "is_success") else False
                    task_success += int(success)
                    print(f"[sbs] [{arm_name}] ep {ep}: {'SUCCESS' if success else 'FAIL'} (t={t})", flush=True)
                except Exception as e:
                    import traceback
                    print(f"[sbs] [{arm_name}] ep {ep} ERROR: {e}", flush=True)
                    traceback.print_exc()

            env.close()
            rate = task_success / max(num_episodes, 1)
            arm_results["tasks"].append({
                "task_idx": task_idx, "task_desc": task_desc,
                "success": task_success, "total": num_episodes, "rate": rate,
            })
            arm_results["total_success"] += task_success
            arm_results["total_eps"] += num_episodes
            print(f"[sbs] [{arm_name}] TASK {task_idx}: {task_success}/{num_episodes} ({rate:.0%})", flush=True)
        return arm_results

    # ── ARM A: native lerobot ─────────────────────────────────────────
    print(f"\n[sbs] {'='*60}", flush=True)
    print(f"[sbs] ARM A: native lerobot (PI05Policy.predict_action_chunk, fp32)", flush=True)
    print(f"[sbs] {'='*60}", flush=True)

    # Move policy to CUDA for native ARM
    policy.to("cuda")

    def _native_action_chunk(batch_pp):
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch_pp)
        post = postprocessor(chunk.detach().cpu())
        chunk_np = post.detach().cpu().numpy() if hasattr(post, "detach") else np.asarray(post)
        if chunk_np.ndim == 3:
            chunk_np = chunk_np[0]
        return chunk_np[:, :7]

    native_results = _run_arm("NATIVE", _native_action_chunk)

    # Move policy back to CPU to free VRAM for Triton ARM
    policy.to("cpu")
    torch.cuda.empty_cache()

    # ── ARM B: Triton ─────────────────────────────────────────────────
    print(f"\n[sbs] {'='*60}", flush=True)
    print(f"[sbs] ARM B: Triton (Pi05FastKernelsInference, bf16)", flush=True)
    print(f"[sbs] {'='*60}", flush=True)

    def _triton_action_chunk(batch_pp):
        chunk = adapter.predict_chunk(batch_pp)
        post = postprocessor(chunk.detach().cpu())
        chunk_np = post.detach().cpu().numpy() if hasattr(post, "detach") else np.asarray(post)
        if chunk_np.ndim == 3:
            chunk_np = chunk_np[0]
        return chunk_np[:, :7]

    triton_results = _run_arm("TRITON", _triton_action_chunk)

    # ── Compare ───────────────────────────────────────────────────────
    native_rate = native_results["total_success"] / max(native_results["total_eps"], 1)
    triton_rate = triton_results["total_success"] / max(triton_results["total_eps"], 1)
    delta = triton_rate - native_rate

    print(f"\n[sbs] {'='*60}", flush=True)
    print(f"[sbs] NATIVE:  {native_results['total_success']}/{native_results['total_eps']} ({native_rate:.0%})", flush=True)
    print(f"[sbs] TRITON:  {triton_results['total_success']}/{triton_results['total_eps']} ({triton_rate:.0%})", flush=True)
    print(f"[sbs] Delta:   {delta:+.0%}", flush=True)
    print(f"[sbs] Total time: {time.time()-t_total:.1f}s", flush=True)
    print(f"[sbs] {'='*60}", flush=True)

    return {
        "native": native_results,
        "triton": triton_results,
        "native_rate": native_rate,
        "triton_rate": triton_rate,
        "delta": delta,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Lift #5 L3 side-by-side: native lerobot vs Triton")
    print("=" * 70)
    result = run_side_by_side.remote()
    print("\n" + "=" * 70)
    print(f"NATIVE: {result['native']['total_success']}/{result['native']['total_eps']} ({result['native_rate']:.0%})")
    print(f"TRITON: {result['triton']['total_success']}/{result['triton']['total_eps']} ({result['triton_rate']:.0%})")
    print(f"DELTA:  {result['delta']:+.0%}")
    print("=" * 70)
