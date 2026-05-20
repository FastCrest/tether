"""Modal: LIBERO-10 eval against FluxVLA's published pi0.5 fine-tuned checkpoint.

Validates the LIBERO 97.85%-average claim from FluxVLA's README against our
reflex export + serve pipeline. Closes the customer-visible LIBERO benchmark
gap (we currently report 64% on `lerobot/pi05_libero_finetuned_v044`; FluxVLA
publishes 97.85% on their finetune).

Lift #4 of the fluxvla-lift-program — `01_decisions/2026-05-19-fluxvla-lift-program.md`.

Pipeline:

1. Download `limxdynamics/FluxVLAEngine/pi05_paligemma_libero_10_full_finetune_bs64`
   from HF (cached on Modal volume after first run).
2. Convert FluxVLA's raw training safetensors → lerobot-format HF layout
   (one-off shim until lift #1 BaseVLA spine + name_mapping land).
3. Run reflex's standard pi0.5 export pipeline against the converted checkpoint
   (decomposed VLM-prefix + per-step expert ONNX). Includes parity verification
   (cos = +1.0 hard gate vs PyTorch reference).
4. Run LIBERO eval against the export at N=50 trials/task across all 4 LIBERO-10
   subsuites (Spatial, Object, Goal, Long). Uses the proven rollout loop from
   `modal_libero_pi05_decomposed.py`.
5. Write per-suite + aggregate numbers to a JSON artifact on the volume. Compare
   against FluxVLA's published 97.85% average.

Methodology gates (per `02_research/competitors/fluxvla.md`):

- 180° image rotation matching their `eval_utils.py:98-99` — confirmed in
  reflex's LIBERO wrapper, mirrored here.
- `num_steps_wait=10` dummy-action grace at episode start — already standard
  in our LIBERO loop.
- Per-suite `max_steps` (Spatial 220, Object 280, Goal 300, Long 520) match
  FluxVLA's table.
- `eval_chunk_size=10` matches their inference config.

Cost estimate: ~$15-20 (4 LIBERO subsuites × 50 episodes × ~30s per episode
on A100 = ~100 min wall clock × A100-80GB hourly rate).

Usage:
    modal run scripts/modal_fluxvla_checkpoint_eval.py
    # Quick smoke (1 task, 1 episode):
    modal run scripts/modal_fluxvla_checkpoint_eval.py --smoke
    # Full eval (default 50/task across all 4 suites):
    modal run scripts/modal_fluxvla_checkpoint_eval.py --num-episodes 50

Source attribution (Apache 2.0):
- Checkpoint: huggingface.co/limxdynamics/FluxVLAEngine (subdir
  pi05_paligemma_libero_10_full_finetune_bs64)
- Source paper / methodology: see fluxvla.limxdynamics.com + their README
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("reflex-fluxvla-checkpoint-eval")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUILD_BUST = _build_bust()

# Pinned FluxVLA HF reference. If they update upstream, we re-pin deliberately.
FLUXVLA_HF_REPO = "limxdynamics/FluxVLAEngine"
FLUXVLA_SUBDIR = "pi05_paligemma_libero_10_full_finetune_bs64"
# Specific checkpoint file inside the subdir, per FluxVLA's README documentation:
# https://github.com/FluxVLA/FluxVLA/blob/main/README.md (search for "step-028548")
FLUXVLA_CHECKPOINT_FILE = "step-028548-epoch-18-loss=0.0111.safetensors"

# FluxVLA's published numbers for verification (their README table, 2026-04-08+):
FLUXVLA_PUBLISHED = {
    "libero_spatial": 98.6,
    "libero_object": 99.0,
    "libero_goal": 97.8,
    "libero_10": 96.0,  # Long, ± 1.0
    "average": 97.85,
}

# LIBERO suite constants — match FluxVLA's libero_eval_runner.py:267-276
# and the existing modal_libero_pi05_decomposed.py.
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,  # Long
}
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

# Where the converted (FluxVLA → lerobot format) checkpoint lands on the volume.
# Persistent across runs so the conversion only happens once.
CONVERTED_CHECKPOINT_DIR = f"{ONNX_OUT}/fluxvla_pi05_libero10_converted"
# Where the exported decomposed ONNX lands.
EXPORTED_ONNX_DIR = f"{ONNX_OUT}/fluxvla_pi05_libero10_export"

# Same image recipe as modal_libero_pi05_decomposed.py (the proven LIBERO+reflex
# image). osmesa + pinned mujoco + PYTHONPATH /opt/LIBERO all matter.
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
        # Use ORT 1.25.1+ for Blackwell support per v0.9.2 ADR
        "onnxruntime-gpu>=1.25.1",
        "nvidia-cudnn-cu12>=9.5",
        "nvidia-cublas-cu12>=12.6",
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
        f'echo "build_bust={_BUILD_BUST}"',
        f'pip install "reflex-vla[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
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


def _download_fluxvla_checkpoint(target_dir: str) -> str:
    """Pull FluxVLA's pi05 LIBERO-10 finetune from HF.

    Returns the local path to the safetensors file. Caches on Modal volume.
    """
    import logging
    from pathlib import Path
    from huggingface_hub import snapshot_download

    target = Path(target_dir)
    if (target / FLUXVLA_CHECKPOINT_FILE).exists():
        logging.info("FluxVLA checkpoint already cached at %s", target)
        return str(target / FLUXVLA_CHECKPOINT_FILE)

    logging.info("Downloading FluxVLA checkpoint from HF (~13 GB)...")
    snapshot_download(
        repo_id=FLUXVLA_HF_REPO,
        allow_patterns=[f"{FLUXVLA_SUBDIR}/*"],
        local_dir=str(target.parent),
    )
    ckpt_path = target.parent / FLUXVLA_SUBDIR / FLUXVLA_CHECKPOINT_FILE
    if not ckpt_path.exists():
        raise RuntimeError(
            f"Expected checkpoint at {ckpt_path} after HF download. "
            f"Either the FluxVLA upstream layout changed or the snapshot "
            f"download failed silently. Check HF auth + the {FLUXVLA_HF_REPO} repo."
        )
    return str(ckpt_path)


def _convert_fluxvla_to_lerobot(
    fluxvla_safetensors_path: str,
    output_dir: str,
) -> str:
    """Convert FluxVLA's raw training safetensors → lerobot-format HF layout.

    One-off shim. The general-purpose version of this lives in lift #1
    (basevla-spine) once that lands as a per-VLA name_mapping pattern.

    Produces:
        output_dir/
        ├── model.safetensors           ← weights, key-renamed
        ├── config.json                  ← pi0.5 config
        ├── preprocessor_config.json     ← LIBERO dataset stats
        ├── policy_preprocessor_*.safetensors
        └── policy_postprocessor_*.safetensors

    Returns the output_dir path.

    NOTE: this function is a STUB until we inspect FluxVLA's actual state_dict
    layout. The conversion will likely need:
        - Strip FluxVLA-specific module prefixes (e.g., `model.` → ``)
        - Rename `paligemma_with_expert.paligemma.model.language_model.*` →
          lerobot's expected names
        - Generate preprocessor configs from FluxVLA's dataset_statistics.json
          (which lives next to the checkpoint per their convention)
    """
    import json
    import logging
    from pathlib import Path
    from safetensors.torch import load_file, save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Detect already-converted (cached)
    if (output / "model.safetensors").exists() and (output / "config.json").exists():
        logging.info("Converted checkpoint already at %s", output)
        return str(output)

    src = Path(fluxvla_safetensors_path)
    src_dir = src.parent

    logging.info("Loading FluxVLA state_dict from %s", src)
    state_dict = load_file(str(src))
    logging.info("FluxVLA state_dict has %d entries", len(state_dict))

    # Print a sample of keys for debugging (first 10).
    keys_sample = sorted(state_dict.keys())[:10]
    for k in keys_sample:
        logging.info("  key: %s  shape: %s", k, tuple(state_dict[k].shape))

    # FluxVLA name mapping — converts their training-time module prefixes to
    # lerobot pi0.5 expected naming. Built up empirically from the sample above
    # when first fired. THIS BLOCK NEEDS REAL VALUES once we see actual keys.
    name_mapping = {
        # Example placeholder (will be tuned on first fire):
        # "module.paligemma_with_expert.paligemma.model.language_model": "model.language_model",
        # "module.paligemma_with_expert.gemma_expert": "model.gemma_expert",
        # "module.action_in_proj": "model.action_in_proj",
        # ...
    }

    def rename(k: str) -> str:
        for src_prefix, dst_prefix in name_mapping.items():
            if k.startswith(src_prefix):
                return dst_prefix + k[len(src_prefix):]
        return k

    renamed = {rename(k): v for k, v in state_dict.items()}
    logging.info("After rename: %d entries", len(renamed))
    save_file(renamed, str(output / "model.safetensors"))

    # Copy/generate config.json from FluxVLA's checkpoint directory or
    # synthesize from their training config.
    fluxvla_config_candidates = [
        src_dir / "config.json",
        src_dir / "pi05_config.json",
    ]
    config_src = None
    for candidate in fluxvla_config_candidates:
        if candidate.exists():
            config_src = candidate
            break

    if config_src is None:
        # Fallback: pull base pi0.5 config from lerobot
        logging.warning(
            "No config.json found at %s; falling back to lerobot/pi05_base config",
            src_dir,
        )
        from huggingface_hub import hf_hub_download
        config_src = Path(hf_hub_download(repo_id="lerobot/pi05_base", filename="config.json"))

    with open(config_src) as f:
        config = json.load(f)
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Preprocessor configs: FluxVLA may ship dataset_statistics.json adjacent
    # to the checkpoint; lerobot expects policy_preprocessor_*.safetensors.
    # If we can't generate these on first fire, the eval will surface the
    # gap loudly (the existing reflex export pipeline checks for them).
    stats_src = src_dir / "dataset_statistics.json"
    if stats_src.exists():
        # Convert FluxVLA's dataset_statistics → lerobot's preprocessor format.
        # Stub — will be filled in once first fire surfaces the exact shape needed.
        import shutil
        shutil.copy(stats_src, output / "dataset_statistics.json")
        logging.info("Copied dataset_statistics.json (lerobot-format conversion pending first-fire validation)")
    else:
        logging.warning(
            "No dataset_statistics.json next to checkpoint. "
            "reflex serve will fall back to HF teacher preprocessor (lerobot/pi05_libero_finetuned_v044). "
            "Confirm this is the right fallback for FluxVLA's training data."
        )

    return str(output)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,  # 3 hours — covers full N=50 × 4 subsuites
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret(), modal.Secret.from_name("github-token")],
)
def run_fluxvla_libero_eval(
    num_episodes: int = 50,
    smoke: bool = False,
    suites: list[str] | None = None,
    seed: int = 7,  # FluxVLA's published seed
    save_video_dir: str = "",
):
    """Pull FluxVLA's checkpoint → convert → export → eval on LIBERO-10."""
    import json
    import logging
    import time
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if smoke:
        num_episodes = 1
        suites = ["libero_object"]  # smallest, fastest

    if suites is None:
        suites = list(TASK_SUITE_MAX_STEPS.keys())

    logging.info("=== FluxVLA pi0.5 LIBERO-10 eval ===")
    logging.info("Suites: %s", suites)
    logging.info("Episodes per task: %d", num_episodes)
    logging.info("Seed: %d", seed)
    logging.info("Target (FluxVLA published): %s", FLUXVLA_PUBLISHED)

    start = time.time()

    # Stage 1: Download FluxVLA checkpoint
    logging.info("[Stage 1/4] Download FluxVLA checkpoint...")
    fluxvla_ckpt = _download_fluxvla_checkpoint(
        f"{HF_CACHE_PATH}/fluxvla_pi05_libero10",
    )
    logging.info("Downloaded: %s", fluxvla_ckpt)

    # Stage 2: Convert to lerobot format
    logging.info("[Stage 2/4] Convert to lerobot format...")
    converted_dir = _convert_fluxvla_to_lerobot(
        fluxvla_ckpt,
        CONVERTED_CHECKPOINT_DIR,
    )
    logging.info("Converted: %s", converted_dir)

    # Stage 3: Export via reflex
    logging.info("[Stage 3/4] Run reflex export pipeline...")
    if not Path(EXPORTED_ONNX_DIR + "/vlm_prefix.onnx").exists():
        from reflex.exporters.pi05_exporter import export_pi05_decomposed
        export_pi05_decomposed(
            checkpoint_path=converted_dir,
            output_dir=EXPORTED_ONNX_DIR,
            target="desktop",  # A100-equivalent
            verify_parity=True,  # cos=+1.0 hard gate
        )
        onnx_output.commit()
        logging.info("Exported to %s", EXPORTED_ONNX_DIR)
    else:
        logging.info("Export already cached at %s", EXPORTED_ONNX_DIR)

    # Stage 4: Run LIBERO eval per suite
    logging.info("[Stage 4/4] Run LIBERO eval...")
    results = {"suites": {}, "fluxvla_published": FLUXVLA_PUBLISHED}

    for suite in suites:
        logging.info("--- LIBERO suite: %s (N=%d) ---", suite, num_episodes)
        suite_start = time.time()
        # Delegate to the proven rollout loop in modal_libero_pi05_decomposed.
        # This function will be importable in-container because reflex-vla is
        # installed from the same SHA.
        from scripts.modal_libero_pi05_decomposed import run_decomposed_libero
        # Note: run_decomposed_libero is a @app.function decorated remotely —
        # we call its underlying function. For Phase 1, we inline the loop
        # here in a TODO callout (next fire iteration).
        suite_result = _run_libero_suite(
            export_dir=EXPORTED_ONNX_DIR,
            suite=suite,
            num_episodes=num_episodes,
            seed=seed,
            save_video_dir=save_video_dir,
        )
        results["suites"][suite] = suite_result
        logging.info(
            "Suite %s: %d/%d (%.1f%%) in %.0fs",
            suite,
            suite_result["successes"],
            suite_result["total"],
            100 * suite_result["successes"] / max(suite_result["total"], 1),
            time.time() - suite_start,
        )

    # Aggregate
    total_successes = sum(r["successes"] for r in results["suites"].values())
    total_episodes = sum(r["total"] for r in results["suites"].values())
    avg_pct = 100 * total_successes / max(total_episodes, 1)
    results["aggregate"] = {
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "average_pct": avg_pct,
        "fluxvla_target_pct": FLUXVLA_PUBLISHED["average"],
        "delta_pct": avg_pct - FLUXVLA_PUBLISHED["average"],
    }
    results["elapsed_sec"] = time.time() - start

    logging.info("=== Results ===")
    for suite, r in results["suites"].items():
        target = FLUXVLA_PUBLISHED.get(suite, "?")
        logging.info(
            "  %s: %.1f%% (target %s%%, delta %+.1fpp)",
            suite,
            100 * r["successes"] / max(r["total"], 1),
            target,
            (100 * r["successes"] / max(r["total"], 1)) - (target if isinstance(target, (int, float)) else 0),
        )
    logging.info(
        "  AVERAGE: %.2f%% (target %.2f%%, delta %+.2fpp)",
        avg_pct,
        FLUXVLA_PUBLISHED["average"],
        avg_pct - FLUXVLA_PUBLISHED["average"],
    )

    # Persist artifact
    artifact_dir = Path(ONNX_OUT) / "fluxvla_eval_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"eval_seed{seed}_n{num_episodes}.json"
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)
    onnx_output.commit()
    logging.info("Artifact: %s", artifact_path)

    return results


def _run_libero_suite(
    export_dir: str,
    suite: str,
    num_episodes: int,
    seed: int,
    save_video_dir: str,
) -> dict:
    """Run LIBERO rollouts for a single suite at N=num_episodes/task.

    Wires the shared rollout helper extracted to src/reflex/eval/libero_rollout.py
    on 2026-05-20. Returns the rollout dict shape; caller aggregates per-suite.
    """
    from reflex.eval.libero_rollout import (
        load_pi05_policy_and_processors,
        run_libero_rollout,
    )
    from reflex.runtime.pi05_decomposed_server import Pi05DecomposedInference

    # Load policy + processors from the converted (lerobot-format) checkpoint.
    # The student_checkpoint arg is the converted dir — it has model.safetensors,
    # so the SnapFlow-student branch fires (which then dispatches to the
    # FluxVLA-derived weights via load_snapflow_student or the fallback PI05Policy
    # path depending on what's present after _convert_fluxvla_to_lerobot).
    policy, preprocessor, postprocessor = load_pi05_policy_and_processors(
        student_checkpoint=CONVERTED_CHECKPOINT_DIR,
        decomposed_dir=export_dir,
        preprocessor_ref="lerobot/pi05_libero_finetuned_v044",  # baseline preprocessor stats
    )

    inference = Pi05DecomposedInference(
        export_dir=export_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        enable_cache=False,  # cache modes are an orthogonal lift; this eval is bare-baseline
    )

    rollout_results = run_libero_rollout(
        inference=inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=suite,
        num_episodes=num_episodes,
        seed=seed,
        save_video_dir=save_video_dir,
        label=f"fluxvla:{suite}",
    )
    # Caller wants {"successes": int, "total": int, "per_task": [...]}; map.
    return {
        "successes": rollout_results["total_success"],
        "total": rollout_results["total_eps"],
        "per_task": rollout_results["per_task"],
        "errors": rollout_results.get("errors", []),
        "cache_stats": rollout_results.get("cache_stats"),
    }


@app.local_entrypoint()
def main(
    num_episodes: int = 50,
    smoke: bool = False,
    suites: str = "",  # comma-separated, e.g. "libero_object,libero_spatial"
    seed: int = 7,
    save_video_dir: str = "",
):
    """Local entrypoint — fires the Modal eval.

    Example:
        modal run scripts/modal_fluxvla_checkpoint_eval.py
        modal run scripts/modal_fluxvla_checkpoint_eval.py --smoke
        modal run scripts/modal_fluxvla_checkpoint_eval.py \\
            --suites libero_object,libero_spatial --num-episodes 10
    """
    parsed_suites: list[str] | None = None
    if suites:
        parsed_suites = [s.strip() for s in suites.split(",") if s.strip()]

    results = run_fluxvla_libero_eval.remote(
        num_episodes=num_episodes,
        smoke=smoke,
        suites=parsed_suites,
        seed=seed,
        save_video_dir=save_video_dir,
    )

    print("=" * 70)
    print("FluxVLA pi0.5 LIBERO-10 eval — results")
    print("=" * 70)
    import json
    print(json.dumps(results, indent=2))
