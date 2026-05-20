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

    Rollout body extracted on 2026-05-20 to ``src/reflex/eval/libero_rollout.py``
    so multiple Modal scripts can share the proven loop (lift #4 of
    fluxvla-lift-program). Behavior is bit-identical to the pre-refactor
    inline version.

    save_video_dir: if non-empty, write per-episode MP4 of agentview camera
    to that path inside the container (typically a /onnx_out subpath so it
    persists on the volume). Filename encodes task/ep/seed/steps/success.
    """
    import logging

    # The Pi05DecomposedInference module uses logger.info(...) for provider
    # diagnostics; default root handler is WARN which swallows those.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load policy + processors via the shared helper (extracted 2026-05-20).
    from reflex.eval.libero_rollout import load_pi05_policy_and_processors
    policy, preprocessor, postprocessor = load_pi05_policy_and_processors(
        student_checkpoint=student_checkpoint,
        decomposed_dir=decomposed_dir,
        preprocessor_ref=preprocessor_ref,
    )
    cfg = policy.config
    print(
        f"[decomposed] Policy + processors ready. chunk_size={cfg.chunk_size}, "
        f"max_action_dim={cfg.max_action_dim}"
    )

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

    # ─── Run rollout via shared helper (extracted 2026-05-20) ────────
    from reflex.eval.libero_rollout import run_libero_rollout
    results = run_libero_rollout(
        inference=inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        resize_size=resize_size,
        replan_steps=replan_steps,
        num_steps_wait=num_steps_wait,
        seed=seed,
        save_video_dir=save_video_dir,
        label=f"decomposed:{Path(decomposed_dir).name}",
    )
    # Add caller-specific metadata that the shared helper doesn't know about.
    results["cache_mode"] = cache_mode
    results["cache_ttl_sec"] = cache_ttl_sec
    results["phash_hamming"] = phash_hamming
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
