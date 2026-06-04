"""Lift #5 L3 diagnostic spike — verify Triton ARM produces non-zero LIBERO success.

Fires N=1 episode on 2 LIBERO tasks with the Triton path. Per
feedback_validate_baseline_and_check_modal_midflight: verify baseline > 0
before the full N=100 gate (~$60).

If this spike gets 0/2, there's a fundamental bug in the Triton → LIBERO
pipeline. If it gets ≥1/2, the pipeline works and we fire the full gate.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_l3_diagnostic.py
"""
import os
import subprocess

import modal

app = modal.App("tether-fast-kernels-l3-diagnostic")


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
    # debian_slim base (same as the working modal_libero_lerobot_native.py).
    # The nvidia/cuda-devel base timed out on Modal image build (~5 GB pull).
    # Instead: install CUDA dev headers via the NVIDIA apt repo so nvcc is
    # available for JIT C++ extension compile, without the full devel image.
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "ninja-build", "clang", "build-essential",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "libosmesa6", "libosmesa6-dev",
        "gnupg", "wget",
    )
    .run_commands(
        # Install CUDA toolkit 12.4 headers + nvcc via NVIDIA apt repo.
        # This gives us nvcc for JIT without the full 5GB cuda-devel image.
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb"
        " && apt-get update"
        " && apt-get install -y cuda-toolkit-12-4 --no-install-recommends"
        " && rm cuda-keyring_1.1-1_all.deb",
    )
    .pip_install(
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "triton>=3.1", "ninja",
        # LIBERO deps (matching the working modal_libero_lerobot_native.py image)
        "mujoco==3.3.2",
        "robosuite==1.4.1",
        "h5py", "bddl==1.0.1", "future", "robomimic",
        "hydra-core>=1.1", "easydict", "einops",
        "opencv-python-headless", "gym", "gymnasium",
        "lerobot==0.5.1",
        "num2words", "imageio",
    )
    # Clone + install LIBERO the same way the working eval script does
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    # Patch LIBERO (fixes pickle issues with numpy globals)
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
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
def run_l3_diagnostic(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    task_indices: list[int] | None = None,
    num_episodes: int = 1,
) -> dict:
    """L3 diagnostic: N=1 per task × 2 tasks with Triton ARM.

    Returns per-task success + aggregate.
    """
    import collections
    import math
    import time

    import numpy as np
    import torch

    if task_indices is None:
        task_indices = [0, 1]

    print(f"[l3-diag] L3 diagnostic — model={model_id}, tasks={task_indices}, N={num_episodes}", flush=True)
    print(f"[l3-diag] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    t_total = time.time()

    # ── Load policy + build Triton adapter ────────────────────────────
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    # Keep policy on CPU — only needed for _preprocess_images + preprocessor.
    # The Triton adapter handles CUDA via its own VLA. Loading both to CUDA
    # OOMs on A100-40GB (~26 GB each = 52 GB > 40 GB).
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).cpu()
    policy.eval()
    print(f"[l3-diag] [{time.time()-t0:.1f}s] PI05Policy loaded", flush=True)

    t0 = time.time()
    from tether.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter
    # Build from a CPU copy to avoid weight corruption
    policy_cpu = PI05Policy.from_pretrained(model_id)
    policy_cpu = policy_cpu.to(dtype=torch.float32).to("cpu")
    adapter = TritonLIBEROAdapter.from_policy(policy_cpu, capture=True)
    del policy_cpu
    print(f"[l3-diag] [{time.time()-t0:.1f}s] Triton adapter ready", flush=True)

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
    print(f"[l3-diag] Pre/post processors loaded", flush=True)

    # ── LIBERO setup ──────────────────────────────────────────────────
    # PyTorch 2.6+ changed torch.load default to weights_only=True, which
    # refuses to unpickle LIBERO's init_state files (they embed numpy globals).
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load

    np.random.seed(42)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path

    task_suite_name = "libero_10"
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    max_steps = 520
    resize_size = 224
    replan_steps = 5
    num_steps_wait = 10
    LIBERO_DUMMY_ACTION = np.zeros(7)
    LIBERO_ENV_RESOLUTION = 256

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
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
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

    # ── Eval loop ─────────────────────────────────────────────────────
    results = {"tasks": [], "total_success": 0, "total_eps": 0}

    for task_idx in task_indices:
        task = task_suite.get_task(task_idx)
        task_desc = task.language
        initial_states = task_suite.get_task_init_states(task_idx)
        env = _build_env(task)
        task_success = 0

        print(f"\n[l3-diag] TASK {task_idx}: {task_desc!r}", flush=True)

        for ep in range(num_episodes):
            try:
                env.reset()
                obs = env.set_init_state(initial_states[ep % len(initial_states)])
                adapter.reset()
                action_plan = collections.deque()
                t = 0
                done = False

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

                        # Triton ARM
                        chunk = adapter.predict_chunk(batch_pp)

                        # Denormalize via postprocessor (same pattern as native ARM
                        # in modal_libero_lerobot_native.py lines 548-558).
                        # Postprocessor takes a raw tensor, NOT a dict.
                        post = postprocessor(chunk.detach().cpu())
                        chunk_np = (
                            post.detach().cpu().numpy()
                            if hasattr(post, "detach")
                            else np.asarray(post)
                        )
                        if chunk_np.ndim == 3:
                            chunk_np = chunk_np[0]  # (1, chunk_size, N) → (chunk_size, N)
                        chunk_np = chunk_np[:, :7]  # trim to LIBERO 7-dim action
                        action_plan.extend(chunk_np[:replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, info = env.step(action)
                    t += 1

                    if done:
                        break

                success = bool(env.is_success()["task"]) if hasattr(env, "is_success") else False
                task_success += int(success)
                print(f"[l3-diag]   ep {ep}: {'SUCCESS' if success else 'FAIL'} (t={t})", flush=True)

            except Exception as e:
                import traceback
                print(f"[l3-diag]   ep {ep} ERROR: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

        env.close()
        rate = task_success / max(num_episodes, 1)
        results["tasks"].append({
            "task_idx": task_idx,
            "task_desc": task_desc,
            "success": task_success,
            "total": num_episodes,
            "rate": rate,
        })
        results["total_success"] += task_success
        results["total_eps"] += num_episodes
        print(f"[l3-diag] TASK {task_idx}: {task_success}/{num_episodes} ({rate:.0%})", flush=True)

    agg_rate = results["total_success"] / max(results["total_eps"], 1)
    verdict = "PASS" if results["total_success"] > 0 else "FAIL"
    print(f"\n[l3-diag] {'='*60}", flush=True)
    print(f"[l3-diag] Aggregate: {results['total_success']}/{results['total_eps']} ({agg_rate:.0%})", flush=True)
    print(f"[l3-diag] VERDICT: {verdict} (gate: > 0 success for diagnostic)", flush=True)
    print(f"[l3-diag] Total time: {time.time()-t_total:.1f}s", flush=True)
    print(f"[l3-diag] {'='*60}", flush=True)

    results["verdict"] = verdict
    results["agg_rate"] = agg_rate
    return results


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Lift #5 L3 diagnostic spike — Triton ARM LIBERO eval")
    print("=" * 70)
    result = run_l3_diagnostic.remote()
    print("\n" + "=" * 70)
    for k, v in result.items():
        if k == "tasks":
            for t in v:
                print(f"  task {t['task_idx']}: {t['success']}/{t['total']} ({t['rate']:.0%}) — {t['task_desc']}")
        else:
            print(f"  {k}={v}")
    print("=" * 70)
