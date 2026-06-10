"""Run `tether distill` on Modal — GPU-accelerated, no local GPU needed.

SnapFlow 1-step self-distillation for flow-matching VLAs. v0.3 supports
pi0 + pi0.5 teachers. Student is a copy of the teacher with a zero-init
target_time embedding; the consistency loss trains it to produce a
1-NFE velocity when target_time=1.

Teacher must be a tether-exported dir on the shared volume (same path
as `tether export` produces). Either run `modal_tether_finetune.py`
first to create the teacher, or upload your own checkpoint.

Usage:
    modal run scripts/modal_tether_distill.py \\
        --teacher-export /onnx_out/my_pi0_libero_teacher \\
        --dataset lerobot/libero \\
        --output-subdir pi0_snapflow_student \\
        --steps 10000

    # Bigger consistency-alpha if the student underfits:
    modal run scripts/modal_tether_distill.py \\
        --teacher-export /onnx_out/my_pi05_teacher \\
        --dataset my-org/my-demos \\
        --output-subdir pi05_snapflow \\
        --steps 20000 \\
        --consistency-alpha 1.5

Cost notes: pi0-scale SnapFlow at 10k steps on A100-80GB is ~1-2 hr
($3-6). pi0.5 is ~30% slower. The LIBERO gate adds ~5-10 min at end.
"""
import os
import subprocess
import modal

app = modal.App("tether-distill")


def _hf_secret():
    """Prefer a local HF_TOKEN env var; fall back to the persistent
    Modal secret named 'huggingface' (same pattern as
    modal_customer_dogfood.py / modal_smolvla_libero_parity.py)."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


def _repo_head_sha() -> str:
    """Return the git HEAD SHA. When running via modal, falls back to 'main'
    on the server side (the Modal build container has no .git)."""
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
        return sha
    except Exception:
        return "main"


def _build_bust_marker() -> str:
    """Unique string used in the image's run_commands to bust Modal's
    layer cache on every local `modal run`. Without this, when the
    HEAD-SHA fetch falls back to the literal string 'main', Modal reuses
    a stale pip-install layer even when the repo has new commits."""
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUILD_BUST = _build_bust_marker()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"

# Distill needs the same image as finetune + the SnapFlow deps land
# automatically via the `tether[monolithic]` wheel install.
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
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20",
        "onnxscript>=0.1",
        "lerobot==0.5.1",
        "peft>=0.10",
        "accelerate>=0.30",
        "datasets>=2.15",
        "num2words",
        "typer",
        "rich",
    )
    .run_commands(
        # Local timestamp in an echo cache-busts Modal's layer cache even
        # when _HEAD falls back to 'main' on the build server.
        f'echo "build_bust={_BUILD_BUST}"',
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A100-80GB",  # pi0/pi0.5 need the VRAM; teacher + student held in memory
    timeout=28800,     # 8 hr — distill + LIBERO gate. 10k steps fits in 2hr.
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def distill_modal(
    teacher_export: str,
    dataset: str = "lerobot/libero",
    output_subdir: str = "distill_snapflow",
    num_steps: int = 10_000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    consistency_alpha: float = 1.0,
    precision: str = "bf16",
    target: str = "desktop",
    libero_gate_pp: float = 5.0,
    skip_libero_gate: bool = False,
    skip_export: bool = False,
    image_key_map_json: str = "",
    variant: str = "default",
    loss_mode: str = "snapflow",
    warm_init_state_proj_from: str = "",
    state_sensitivity_alpha: float = 0.0,
    checkpoint_every: int = 1000,
    heartbeat_every: int = 50,
):
    """Run tether.finetune.run_finetune(phase='distill') on Modal.

    Set ``variant='state_out'`` for the v0.5 pi0.5 state-out student
    (strips proprio state from lang, adds explicit state_proj).
    Unlocks the prefix KV cache in production deployment. Requires
    teacher_export to be a pi0.5 model.
    """
    import logging
    from pathlib import Path

    from tether.finetune.config import FinetuneConfig
    from tether.finetune.hooks import HookRegistry
    from tether.finetune.hooks.libero_drop_gate import attach_to as attach_libero_gate
    from tether.finetune.run import run_finetune

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    output = Path(ONNX_OUTPUT_PATH) / output_subdir
    output.mkdir(parents=True, exist_ok=True)

    extra_args = {
        "consistency_alpha": consistency_alpha,
        "libero_gate_threshold_pp": libero_gate_pp,
        "checkpoint_every": checkpoint_every,
        "heartbeat_every": heartbeat_every,
    }
    if skip_libero_gate:
        extra_args["libero_gate_skip"] = True
    if image_key_map_json:
        import json as _json
        extra_args["image_key_map"] = _json.loads(image_key_map_json)

    cfg = FinetuneConfig(
        base="",
        dataset=dataset,
        output=output,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mode="full",
        precision=precision,
        target=target,
        skip_export=skip_export,
        phase="distill",
        teacher_export=teacher_export,
        distillation_method="snapflow",
        variant=variant,
        loss_mode=loss_mode,
        warm_init_state_proj_from=warm_init_state_proj_from,
        state_sensitivity_alpha=state_sensitivity_alpha,
        extra_lerobot_args=extra_args,
    )

    hooks = HookRegistry()
    if not skip_libero_gate:
        attach_libero_gate(hooks)

    result = run_finetune(cfg, hooks=hooks)

    # Make outputs visible to downstream Modal functions.
    onnx_output.commit()

    return {
        "status": result.status,
        "output_dir": str(result.output_dir),
        "training_steps_completed": result.training_steps_completed,
        "final_checkpoint_path": (
            str(result.final_checkpoint_path)
            if result.final_checkpoint_path else None
        ),
        "onnx_path": str(result.onnx_path) if result.onnx_path else None,
        "verification_md_path": (
            str(result.verification_md_path)
            if result.verification_md_path else None
        ),
        "training_log_path": (
            str(result.training_log_path) if result.training_log_path else None
        ),
        "error": result.error,
    }


@app.local_entrypoint()
def main(
    teacher_export: str,
    dataset: str = "lerobot/libero",
    output_subdir: str = "distill_snapflow",
    steps: int = 10_000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    consistency_alpha: float = 1.0,
    precision: str = "bf16",
    target: str = "desktop",
    libero_gate_pp: float = 5.0,
    skip_libero_gate: bool = False,
    skip_export: bool = False,
    image_key_map_json: str = "",
    variant: str = "default",
    loss_mode: str = "snapflow",
    warm_init_state_proj_from: str = "",
    state_sensitivity_alpha: float = 0.0,
    checkpoint_every: int = 1000,
    heartbeat_every: int = 50,
):
    print(f"[tether distill on Modal — SnapFlow]")
    print(f"  teacher: {teacher_export}")
    print(f"  dataset: {dataset}")
    print(f"  output:  /onnx_out/{output_subdir}")
    print(f"  steps:   {steps}  batch={batch_size}  lr={learning_rate}  "
          f"alpha={consistency_alpha}")
    print(f"  checkpoint_every={checkpoint_every}  heartbeat_every={heartbeat_every}")
    if variant != "default":
        print(f"  variant: {variant}")
    if loss_mode != "snapflow":
        print(f"  loss_mode: {loss_mode}")
    if warm_init_state_proj_from:
        print(f"  warm_init_state_proj_from: {warm_init_state_proj_from}")
    if state_sensitivity_alpha > 0:
        print(f"  state_sensitivity_alpha: {state_sensitivity_alpha}")
    if skip_libero_gate:
        print(f"  libero gate: DISABLED")
    else:
        print(f"  libero gate: {libero_gate_pp}pp drop threshold")
    r = distill_modal.remote(
        teacher_export=teacher_export,
        dataset=dataset,
        output_subdir=output_subdir,
        num_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        consistency_alpha=consistency_alpha,
        precision=precision,
        target=target,
        libero_gate_pp=libero_gate_pp,
        skip_libero_gate=skip_libero_gate,
        skip_export=skip_export,
        image_key_map_json=image_key_map_json,
        variant=variant,
        loss_mode=loss_mode,
        warm_init_state_proj_from=warm_init_state_proj_from,
        state_sensitivity_alpha=state_sensitivity_alpha,
        checkpoint_every=checkpoint_every,
        heartbeat_every=heartbeat_every,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
