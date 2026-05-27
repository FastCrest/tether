"""Modal: Run FluxVLA's own eval harness on their pi0.5 LIBERO-10 checkpoint.

Instead of converting to lerobot format and running through reflex's export
pipeline (as modal_fluxvla_checkpoint_eval.py does), this script runs FluxVLA's
native code directly:

1. Build PI05FlowMatching model from FluxVLA's mmengine config.
2. Load the published checkpoint weights via safetensors.
3. Build the eval dataset transforms and denormalize_action from their config.
4. Run the LIBERO rollout loop (single-GPU, no torch.distributed) replicating
   FluxVLA's libero_eval_runner.py lines 229-338.

This validates FluxVLA's published 97.85% LIBERO-10 average using THEIR code
and THEIR checkpoint, as a ground-truth baseline for comparison with reflex's
pipeline.

Usage:
    modal run scripts/modal_fluxvla_native_eval.py --smoke
    modal run scripts/modal_fluxvla_native_eval.py --num-episodes 50
    modal run scripts/modal_fluxvla_native_eval.py --suites libero_object

Source attribution (Apache 2.0):
- Checkpoint: huggingface.co/limxdynamics/FluxVLAEngine
- FluxVLA codebase: reference/FluxVLA/ (local mount)
"""
from __future__ import annotations

import os
import modal

app = modal.App("reflex-fluxvla-native-eval")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pinned FluxVLA HF reference.
FLUXVLA_HF_REPO = "limxdynamics/FluxVLAEngine"
FLUXVLA_SUBDIR = "pi05_paligemma_libero_10_full_finetune_bs64"
FLUXVLA_CHECKPOINT_FILE = "checkpoints/step-038064-epoch-24-loss=0.0170.safetensors"

# FluxVLA's published numbers for verification.
FLUXVLA_PUBLISHED = {
    "libero_spatial": 98.6,
    "libero_object": 99.0,
    "libero_goal": 97.8,
    "libero_10": 96.0,
    "average": 97.85,
}

# LIBERO suite max steps — match FluxVLA's libero_eval_runner.py:267-276.
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

FLUXVLA_SRC = "/opt/FluxVLA"


# ---------------------------------------------------------------------------
# Image: proven LIBERO recipe from modal_fluxvla_checkpoint_eval.py
# + FluxVLA source mounted and pip-installed (no CUDA ext build)
# + mmengine + tensorflow (FluxVLA eval_utils uses tf for image resize)
# ---------------------------------------------------------------------------
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
        "torch==2.6.0",
        "torchvision==0.21.0",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers>=4.48,<4.54",
        "numpy<2",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
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
        "num2words",
        "imageio",
        "imageio-ffmpeg",
        # FluxVLA-specific deps
        "mmengine==0.10.7",
        "accelerate==0.33.0",
        "sentencepiece",
        "tensorflow>=2.15",
        "draccus",
        "rich",
        "tqdm",
        "datasets>=4.0",
        "jsonlines",
    )
    # Clone + patch LIBERO (same as modal_fluxvla_checkpoint_eval.py)
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    # Mount FluxVLA source and install WITHOUT building CUDA extensions.
    # We only need PI05FlowMatching (pure PyTorch), not PI05FlowMatchingInference
    # (which requires custom Triton/CUDA kernels).
    .add_local_dir(
        os.path.join(REPO_ROOT, "reference", "FluxVLA"),
        remote_path=FLUXVLA_SRC,
        copy=True,
    )
    # No pip install needed — PYTHONPATH includes FLUXVLA_SRC (set in .env below).
    # FluxVLA's setup.py imports torch at top level which isn't available during
    # image build, and editable installs fail on Modal's pip version. PYTHONPATH
    # is sufficient since we only need the Python modules, not the CUDA extensions.
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "TORCHINDUCTOR_DISABLE": "1",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": f"/opt/LIBERO:{FLUXVLA_SRC}",
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


def _download_fluxvla_checkpoint(cache_dir: str) -> str:
    """Download FluxVLA's pi05 LIBERO-10 finetune from HF.

    Returns (ckpt_path, ckpt_parent_dir) where ckpt_parent_dir contains
    dataset_statistics.json (FluxVLA convention: sibling of checkpoints/).
    """
    import logging
    from pathlib import Path
    from huggingface_hub import snapshot_download

    cache = Path(cache_dir)
    subdir_path = cache / FLUXVLA_SUBDIR
    ckpt_path = subdir_path / FLUXVLA_CHECKPOINT_FILE

    if ckpt_path.exists():
        logging.info("FluxVLA checkpoint already cached at %s", ckpt_path)
        return str(ckpt_path)

    logging.info("Downloading FluxVLA checkpoint from HF (~13 GB)...")
    snapshot_download(
        repo_id=FLUXVLA_HF_REPO,
        allow_patterns=[f"{FLUXVLA_SUBDIR}/*"],
        local_dir=str(cache),
    )
    if not ckpt_path.exists():
        raise RuntimeError(
            f"Expected checkpoint at {ckpt_path} after download. "
            f"Check HF auth + {FLUXVLA_HF_REPO} repo layout."
        )
    return str(ckpt_path)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,  # 3 hours
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUT: onnx_output},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_fluxvla_native_eval(
    num_episodes: int = 50,
    smoke: bool = False,
    suites: list[str] | None = None,
    seed: int = 7,
):
    """Run FluxVLA's native eval harness on their LIBERO-10 checkpoint."""
    import gc
    import json
    import logging
    import time
    from pathlib import Path

    import torch
    import tqdm

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("fluxvla_native_eval")

    if smoke:
        num_episodes = 1
        suites = ["libero_object"]

    if suites is None:
        suites = list(TASK_SUITE_MAX_STEPS.keys())

    log.info("=== FluxVLA native pi0.5 LIBERO-10 eval ===")
    log.info("Suites: %s", suites)
    log.info("Episodes per task: %d", num_episodes)
    log.info("Seed: %d", seed)
    log.info("Target (FluxVLA published): %s", FLUXVLA_PUBLISHED)

    start = time.time()

    # -----------------------------------------------------------------------
    # Stage 1: Download checkpoint
    # -----------------------------------------------------------------------
    log.info("[Stage 1/3] Download FluxVLA checkpoint...")
    ckpt_path = _download_fluxvla_checkpoint(
        f"{HF_CACHE_PATH}/fluxvla_pi05_libero10"
    )
    log.info("Checkpoint: %s", ckpt_path)
    hf_cache.commit()

    # dataset_statistics.json lives at the checkpoint's grandparent
    # (FluxVLA convention: <run_dir>/dataset_statistics.json, checkpoints/ is a child)
    ckpt_parent = str(Path(ckpt_path).resolve().parent.parent)
    data_stat_path = os.path.join(ckpt_parent, "dataset_statistics.json")
    assert os.path.exists(data_stat_path), (
        f"dataset_statistics.json not found at {data_stat_path}"
    )
    log.info("Dataset stats: %s", data_stat_path)

    # -----------------------------------------------------------------------
    # Stage 2: Build model + load weights using FluxVLA's own machinery
    # -----------------------------------------------------------------------
    log.info("[Stage 2/3] Build model + load weights...")

    # Import FluxVLA's builders (triggers mmengine registry population)
    from mmengine.config import Config as MmConfig
    from fluxvla.engines import build_vla_from_cfg, build_dataset_from_cfg, build_transform_from_cfg
    from fluxvla.engines.utils.torch_utils import set_seed_everywhere
    from fluxvla.engines.utils.eval_utils import get_libero_env, get_libero_dummy_action
    from safetensors.torch import load_file

    # Load the inference config
    config_path = os.path.join(
        FLUXVLA_SRC, "configs", "pi05",
        "pi05_paligemma_libero_10_full_inference.py",
    )
    cfg = MmConfig.fromfile(config_path)

    # Build the NON-inference model (PI05FlowMatching, pure PyTorch SDPA).
    # The inference_model variant (PI05FlowMatchingInference) requires custom
    # Triton/CUDA kernels we don't want to build. PI05FlowMatching.predict_action
    # uses standard PyTorch and produces identical outputs.
    log.info("Building PI05FlowMatching model (pure PyTorch)...")
    model_cfg = dict(cfg.model)
    # Override pretrained_name_or_path — the config points to a local training
    # path that doesn't exist here.  We load the fine-tuned checkpoint directly
    # via load_state_dict below.
    model_cfg["pretrained_name_or_path"] = None
    vla = build_vla_from_cfg(model_cfg).eval()

    # Load checkpoint weights
    log.info("Loading checkpoint weights from %s", ckpt_path)
    state_dict = load_file(ckpt_path, device="cpu")
    vla.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()

    # Load norm stats into model
    with open(data_stat_path, "r") as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats

    # Move to GPU in bf16
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    vla.to(device=device, dtype=torch.bfloat16)
    log.info("Model loaded on %s in bf16", device)

    # -----------------------------------------------------------------------
    # Stage 3: Run LIBERO eval per suite
    # -----------------------------------------------------------------------
    log.info("[Stage 3/3] Run LIBERO eval...")

    # Build eval dataset transform and denormalize_action from config
    eval_cfg = cfg.eval

    # The eval config's dataset and denormalize_action need norm_stats injected
    eval_dataset_cfg = dict(eval_cfg.dataset)
    eval_dataset_cfg["norm_stats"] = data_stat_path

    eval_denorm_cfg = dict(eval_cfg.denormalize_action)
    eval_denorm_cfg["norm_stats"] = data_stat_path

    eval_chunk_size = eval_cfg.get("eval_chunk_size", 10)
    num_steps_wait = eval_cfg.get("num_steps_wait", 10)
    mixed_precision_dtype = torch.bfloat16

    results = {"suites": {}, "fluxvla_published": FLUXVLA_PUBLISHED}

    for suite in suites:
        log.info("--- LIBERO suite: %s (N=%d) ---", suite, num_episodes)
        suite_start = time.time()

        set_seed_everywhere(seed)

        # Build dataset transform with suite-specific settings
        suite_dataset_cfg = dict(eval_dataset_cfg)
        suite_dataset_cfg["task_suite_name"] = suite
        suite_dataset_cfg["norm_stats_key"] = f"{suite}_no_noops"
        dataset_transform = build_dataset_from_cfg(suite_dataset_cfg)

        # Build denormalize_action
        denorm_action = build_transform_from_cfg(eval_denorm_cfg)

        # Get LIBERO benchmark
        from libero.libero import benchmark as libero_benchmark
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        num_tasks = task_suite.n_tasks

        unnorm_key = suite
        # FluxVLA uses the _no_noops suffix for norm_stats key
        candidate_unnorm_key = f"{unnorm_key}_no_noops"
        if (unnorm_key not in vla.norm_stats
                and candidate_unnorm_key in vla.norm_stats):
            unnorm_key = candidate_unnorm_key
        norm_stats_key = f"{suite}_no_noops"

        total_episodes = 0
        total_successes = 0

        max_steps = TASK_SUITE_MAX_STEPS[suite]

        pbar = tqdm.tqdm(
            total=num_tasks * num_episodes,
            desc=f"Eval {suite}",
            dynamic_ncols=True,
        )

        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            task_successes = 0

            for trial_id in range(num_episodes):
                log.info(
                    "Task %d/%d, Trial %d/%d",
                    task_id + 1, num_tasks, trial_id + 1, num_episodes,
                )

                # Initialize environment
                env, task_description = get_libero_env(task, resolution=256)
                env.reset()
                obs = env.set_init_state(initial_states[trial_id])
                is_new_episode = True

                t = 0
                done = False

                while t < max_steps + num_steps_wait:
                    # Wait for objects to settle
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action()
                        )
                        t += 1
                        continue

                    # Prepare observation for the dataset transform
                    obs["task_description"] = task_description
                    obs["is_new_episode"] = is_new_episode
                    batch, replay_img = dataset_transform(obs)
                    is_new_episode = False
                    batch["unnorm_key"] = unnorm_key

                    # Run model inference
                    with torch.autocast(
                        "cuda",
                        dtype=mixed_precision_dtype,
                        enabled=True,
                    ):
                        with torch.no_grad():
                            actions = vla.predict_action(**batch)

                    # Handle action shape: [batch, chunk, action_dim] or [batch, action_dim]
                    if len(actions.shape) == 3:
                        actions = actions[
                            0, :eval_chunk_size, :
                        ].float().cpu().numpy()
                    else:
                        assert len(actions.shape) == 2, (
                            f"Unexpected action shape: {actions.shape}"
                        )
                        actions = actions[0, None, :].float().cpu().numpy()

                    # Execute action chunk
                    for action in actions:
                        inputs = dict(
                            action=action,
                            task_suite_name=suite,
                            norm_stats_key=norm_stats_key,
                        )
                        action_denormed = denorm_action(inputs)
                        obs, reward, done, info = env.step(
                            action_denormed.tolist()
                        )
                        if done:
                            total_successes += 1
                            task_successes += 1
                            break
                        t += 1

                    if done:
                        break

                total_episodes += 1
                pbar.update(1)
                env.close()

            log.info(
                "Task %d: %d/%d successes (%.1f%%)",
                task_id,
                task_successes,
                num_episodes,
                100 * task_successes / max(num_episodes, 1),
            )

        pbar.close()

        suite_pct = 100 * total_successes / max(total_episodes, 1)
        target = FLUXVLA_PUBLISHED.get(suite, 0)
        suite_result = {
            "successes": total_successes,
            "total": total_episodes,
            "pct": suite_pct,
            "target_pct": target,
            "delta_pp": suite_pct - target,
            "elapsed_sec": time.time() - suite_start,
        }
        results["suites"][suite] = suite_result
        log.info(
            "Suite %s: %d/%d (%.1f%%, target %.1f%%, delta %+.1fpp) in %.0fs",
            suite,
            total_successes,
            total_episodes,
            suite_pct,
            target,
            suite_pct - target,
            time.time() - suite_start,
        )

    # Aggregate
    total_s = sum(r["successes"] for r in results["suites"].values())
    total_e = sum(r["total"] for r in results["suites"].values())
    avg_pct = 100 * total_s / max(total_e, 1)
    results["aggregate"] = {
        "total_successes": total_s,
        "total_episodes": total_e,
        "average_pct": avg_pct,
        "fluxvla_target_pct": FLUXVLA_PUBLISHED["average"],
        "delta_pct": avg_pct - FLUXVLA_PUBLISHED["average"],
    }
    results["elapsed_sec"] = time.time() - start

    log.info("=== Results ===")
    for suite, r in results["suites"].items():
        log.info(
            "  %s: %.1f%% (target %.1f%%, delta %+.1fpp)",
            suite, r["pct"], r["target_pct"], r["delta_pp"],
        )
    log.info(
        "  AVERAGE: %.2f%% (target %.2f%%, delta %+.2fpp)",
        avg_pct,
        FLUXVLA_PUBLISHED["average"],
        avg_pct - FLUXVLA_PUBLISHED["average"],
    )

    # Persist artifact
    artifact_dir = Path(ONNX_OUT) / "fluxvla_native_eval_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"native_eval_seed{seed}_n{num_episodes}.json"
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)
    onnx_output.commit()
    log.info("Artifact: %s", artifact_path)

    return results


@app.local_entrypoint()
def main(
    num_episodes: int = 50,
    smoke: bool = False,
    suites: str = "",
    seed: int = 7,
):
    """Local entrypoint.

    Example:
        modal run scripts/modal_fluxvla_native_eval.py --smoke
        modal run scripts/modal_fluxvla_native_eval.py --num-episodes 50
        modal run scripts/modal_fluxvla_native_eval.py --suites libero_object,libero_spatial
    """
    parsed_suites: list[str] | None = None
    if suites:
        parsed_suites = [s.strip() for s in suites.split(",") if s.strip()]

    results = run_fluxvla_native_eval.remote(
        num_episodes=num_episodes,
        smoke=smoke,
        suites=parsed_suites,
        seed=seed,
    )

    print("=" * 70)
    print("FluxVLA native pi0.5 LIBERO-10 eval -- results")
    print("=" * 70)
    import json
    print(json.dumps(results, indent=2))
