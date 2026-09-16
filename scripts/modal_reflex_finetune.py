"""Run `tether finetune` on Modal — GPU-accelerated, no local GPU needed.

This is the v0.3 MVP scaffold. Wraps `tether.finetune.run_finetune` on
an A10G (default) or A100 (for larger base models). Writes output to
the shared gr00t-onnx-outputs volume so subsequent `tether serve` or
parity runs can pick up the exported ONNX without re-copying the
6-12GB file across the network.

Usage:
    modal run scripts/modal_tether_finetune.py \\
        --base lerobot/smolvla_base \\
        --dataset lerobot/libero \\
        --output-subdir my_smolvla_libero \\
        --steps 5000

    # With a larger GPU for pi0-scale models (v0.5+):
    modal run scripts/modal_tether_finetune.py \\
        --base lerobot/pi0_base \\
        --dataset my-org/my-demos \\
        --output-subdir my_pi0_run \\
        --steps 20000 \\
        --gpu A100-80GB

Cost notes: SmolVLA LoRA on a 500-sample dataset for 2000 steps runs
~10-15 min on A10G ($0.30-0.50). pi0 LoRA is 3-5x slower due to the
bigger backbone — budget accordingly.
"""
import os
import json
import modal

app = modal.App("tether-finetune")


def _hf_secret():
    """Prefer a local HF_TOKEN env var; fall back to the persistent
    Modal secret named 'huggingface'."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "clang", "build-essential",
        # ffmpeg + codecs — torchcodec (lerobot's dataset video decoder)
        # dlopens libavutil/libavcodec/libavformat at import time. Without
        # these, dataset loading fails with OSError before training starts.
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
        # SmolVLM's processor requires num2words at import time; lerobot
        # doesn't pull it transitively. Without this the policy load
        # fails with "Package num2words is required to run SmolVLM
        # processor." (v6 smoke test failure mode).
        "num2words",
        "typer",
        "rich",
    )
    .add_local_dir("src", "/opt/tether/src", copy=True)
    .add_local_file("pyproject.toml", "/opt/tether/pyproject.toml", copy=True)
    .add_local_file("README.md", "/opt/tether/README.md", copy=True)
    .run_commands('pip install "/opt/tether[monolithic]"')
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    retries=0,
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def finetune_modal(
    base: str = "lerobot/smolvla_base",
    base_revision: str = "",
    dataset: str = "lerobot/libero",
    dataset_revision: str = "",
    output_subdir: str = "finetune_smolvla_libero",
    num_steps: int = 5000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    precision: str = "bf16",
    seed: int = 42,
    target: str = "desktop",
    skip_export: bool = False,
):
    """Run tether.finetune.run_finetune on Modal."""
    import logging
    from pathlib import Path

    from tether.finetune import FinetuneConfig, run_finetune

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    output = Path(ONNX_OUTPUT_PATH) / output_subdir
    output.mkdir(parents=True, exist_ok=True)

    cfg = FinetuneConfig(
        base=base,
        base_revision=base_revision or None,
        dataset=dataset,
        dataset_revision=dataset_revision or None,
        output=output,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        precision=precision,
        seed=seed,
        target=target,
        skip_export=skip_export,
    )
    result = run_finetune(cfg)

    # Make outputs visible to downstream Modal functions via volume.
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
    base: str = "lerobot/smolvla_base",
    base_revision: str = "",
    dataset: str = "lerobot/libero",
    dataset_revision: str = "",
    output_subdir: str = "finetune_smolvla_libero",
    steps: int = 5000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    precision: str = "bf16",
    seed: int = 42,
    target: str = "desktop",
    skip_export: bool = False,
):
    print(f"[tether finetune on Modal]")
    print(f"  base:    {base}")
    print(f"  dataset: {dataset}")
    print(f"  output:  /onnx_out/{output_subdir}")
    print(f"  steps:   {steps}  batch={batch_size}  lr={learning_rate}  "
          f"lora_r={lora_rank}")
    r = finetune_modal.remote(
        base=base,
        base_revision=base_revision,
        dataset=dataset,
        dataset_revision=dataset_revision,
        output_subdir=output_subdir,
        num_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        precision=precision,
        seed=seed,
        target=target,
        skip_export=skip_export,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
    envelope = {
        "schema_version": 1,
        "base": base,
        "base_revision": base_revision or None,
        "dataset": dataset,
        "dataset_revision": dataset_revision or None,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "result": r,
    }
    print("TETHER_MODAL_TRAIN_RESULT_JSON=" + json.dumps(envelope, sort_keys=True, separators=(",", ":")))
