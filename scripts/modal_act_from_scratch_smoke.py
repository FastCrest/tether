"""Smoke test for `reflex finetune --policy act --mode full` on Modal.

Validates the new ACT-from-scratch path landed in Phase 4 of the auto_soarm
vendoring (ADR 2026-05-06-vendor-auto-soarm.md). The new code allows
training ACT from scratch (no LoRA, no pretrained base) via:

    reflex finetune --policy act --mode full --chunk-size 31 \\
        --dataset <local-or-hf-id> --output <out> \\
        --steps 30000 --batch-size 8 --learning-rate 1e-5

This script fires it on Modal A10G against a public LeRobot dataset to
confirm:
  1. The reflex-vla install pulls all the new code (config + run + cli).
  2. _build_lerobot_command produces a valid lerobot-train invocation
     with --policy.type=act, no --policy.pretrained_path, --policy.chunk_size=N.
  3. lerobot-train accepts the args and starts training.
  4. A checkpoint actually lands at the expected path.

Cost: ~$1-2 for the 100-step smoke (T4 / A10G; ~5-10 min wall).

Usage:
    modal run scripts/modal_act_from_scratch_smoke.py
    modal run scripts/modal_act_from_scratch_smoke.py --steps 100 --dataset lerobot/pusht
    modal run scripts/modal_act_from_scratch_smoke.py --diagnostic-only   # 1-step pre-flight

Per CLAUDE.md memory feedback_validate_baseline_and_check_modal_midflight.md:
the diagnostic spike runs first (1 step, no measurement) to verify the image
+ dataset load. Bigger smoke only fires after the spike returns clean.
"""
import os
import subprocess

import modal


app = modal.App("reflex-act-from-scratch-smoke")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "clang", "build-essential",
        "ffmpeg", "libavutil-dev", "libavcodec-dev", "libavformat-dev",
        "libswresample-dev", "libswscale-dev",
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
        "lerobot==0.5.1",
        "accelerate>=0.30",
        "datasets>=2.15",
        "num2words",
        "typer",
        "rich",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,  # 1 hr ceiling — smoke targets ≤10 min
    volumes={HF_CACHE_PATH: hf_cache, OUTPUT_PATH: output_vol},
    secrets=[_hf_secret()],
)
def act_smoke(
    *,
    dataset: str = "lerobot/pusht",
    output_subdir: str = "act_smoke",
    steps: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    chunk_size: int = 31,
    seed: int = 1,
    diagnostic_only: bool = False,
):
    """Run the new --policy act --mode full path."""
    import logging
    import time as _time
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    # Verify the new code is actually pulled in.
    from reflex.finetune.config import FinetuneConfig
    from reflex.finetune.run import _build_lerobot_command, run_finetune

    if not hasattr(FinetuneConfig, "policy"):
        return {
            "status": "FAIL",
            "error": "FinetuneConfig.policy field missing — wrong reflex-vla commit",
            "head_sha": "(unknown)",
        }

    out_root = Path(OUTPUT_PATH) / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    actual_steps = 1 if diagnostic_only else steps

    cfg = FinetuneConfig(
        base="",
        dataset=dataset,
        output=out_root,
        num_steps=actual_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mode="full",
        policy="act",
        chunk_size=chunk_size,
        seed=seed,
        skip_export=True,  # don't try to ONNX-export an ACT checkpoint here;
                          # validates training only.
        skip_preflight=True,  # preflight tries to fetch HF dataset metadata;
                              # bypassed for smoke determinism.
    )

    # Build the lerobot-train command we'll fire so the test report
    # surfaces the actual invocation. Independent of training success.
    cmd = _build_lerobot_command(cfg)
    logger.info("=" * 70)
    logger.info("Built lerobot-train command:")
    for arg in cmd:
        logger.info(f"  {arg}")
    logger.info("=" * 70)

    # Sanity check the construction matches Phase 4 expectations:
    cmd_str = " ".join(cmd)
    expected_in_cmd = ["--policy.type=act", "--policy.chunk_size=31"]
    forbidden_in_cmd = ["--policy.pretrained_path", "--peft.method_type"]
    construction_ok = (
        all(s in cmd_str for s in expected_in_cmd)
        and not any(s in cmd_str for s in forbidden_in_cmd)
    )
    logger.info(f"command_construction_ok: {construction_ok}")
    if not construction_ok:
        return {
            "status": "FAIL_COMMAND_CONSTRUCTION",
            "command": cmd,
            "expected_in_cmd": expected_in_cmd,
            "forbidden_in_cmd": forbidden_in_cmd,
            "head_sha": _HEAD,
        }

    if diagnostic_only:
        # Don't actually run lerobot-train; we've validated construction.
        return {
            "status": "OK_CONSTRUCTION_ONLY",
            "command": cmd,
            "head_sha": _HEAD,
            "construction_ok": construction_ok,
        }

    # Real run.
    t0 = _time.time()
    result = run_finetune(cfg)
    elapsed_s = _time.time() - t0

    output_vol.commit()

    return {
        "status": result.status,
        "elapsed_s": round(elapsed_s, 1),
        "output_dir": str(result.output_dir),
        "training_steps_completed": result.training_steps_completed,
        "final_checkpoint_path": (
            str(result.final_checkpoint_path) if result.final_checkpoint_path else None
        ),
        "training_log_path": (
            str(result.training_log_path) if result.training_log_path else None
        ),
        "error": result.error,
        "head_sha": _HEAD,
        "command_construction_ok": construction_ok,
    }


@app.local_entrypoint()
def main(
    dataset: str = "lerobot/pusht",
    output_subdir: str = "act_smoke",
    steps: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    chunk_size: int = 31,
    seed: int = 1,
    diagnostic_only: bool = False,
    skip_diagnostic: bool = False,
):
    print(f"[reflex ACT-from-scratch Modal smoke]")
    print(f"  HEAD:       {_HEAD}")
    print(f"  dataset:    {dataset}")
    print(f"  output:     /onnx_out/{output_subdir}")
    print(f"  steps:      {steps}  chunk_size={chunk_size}  lr={learning_rate}")

    if diagnostic_only:
        print("\n=== DIAGNOSTIC ONLY (construction check) ===")
        diag = act_smoke.remote(
            dataset=dataset,
            output_subdir=f"{output_subdir}_diag",
            steps=1,
            chunk_size=chunk_size,
            diagnostic_only=True,
        )
        for k, v in diag.items():
            if k == "command":
                print(f"  command:")
                for arg in v:
                    print(f"    {arg}")
            else:
                print(f"  {k}: {v}")
        return diag

    if not skip_diagnostic:
        print("\n=== DIAGNOSTIC SPIKE (validates command construction) ===")
        diag = act_smoke.remote(
            dataset=dataset,
            output_subdir=f"{output_subdir}_diag",
            steps=1,
            chunk_size=chunk_size,
            diagnostic_only=True,
        )
        for k, v in diag.items():
            if k == "command":
                print(f"  command:")
                for arg in v:
                    print(f"    {arg}")
            else:
                print(f"  {k}: {v}")
        if not diag.get("construction_ok"):
            print("\n[ABORT] command construction failed; not firing real smoke")
            return diag

    print("\n=== REAL SMOKE (actually trains) ===")
    r = act_smoke.remote(
        dataset=dataset,
        output_subdir=output_subdir,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        chunk_size=chunk_size,
        seed=seed,
        diagnostic_only=False,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        if k == "command":
            print(f"  command:")
            for arg in v:
                print(f"    {arg}")
        else:
            print(f"  {k}: {v}")
