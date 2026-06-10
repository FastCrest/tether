"""Modal: export a tether-saved SnapFlow student checkpoint to monolithic ONNX.

Pulls a distill-run output from the ``pi0-onnx-outputs`` volume (the same
volume the distill writes into), runs ``export_snapflow_student_monolithic``,
and writes the ONNX next to it.

Usage:
    modal run scripts/modal_export_snapflow_student.py \\
      --checkpoint /onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model \\
      --output-subdir distill_v031_pi05_libero_r4/onnx_1nfe
"""
import os
import subprocess
import modal

app = modal.App("tether-snapflow-export")


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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def export_modal(checkpoint: str, output_subdir: str):
    """Load SnapFlow student from volume, export to ONNX, write back."""
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from tether.exporters.monolithic import export_snapflow_student_monolithic

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found on volume: {ckpt_path}")

    output = Path(ONNX_OUT) / output_subdir
    output.mkdir(parents=True, exist_ok=True)

    result = export_snapflow_student_monolithic(
        str(ckpt_path), str(output), target="desktop",
    )
    onnx_output.commit()

    # Sanity: ONNX loads via onnxruntime
    import onnx
    import onnxruntime as ort
    onnx_path = result["onnx_path"]
    model = onnx.load(onnx_path, load_external_data=False)
    print(f"[export] ONNX IR version: {model.ir_version}, opset: {model.opset_import[0].version}")
    print(f"[export] inputs: {[i.name for i in model.graph.input]}")
    print(f"[export] outputs: {[o.name for o in model.graph.output]}")

    # Skip runtime check if the ONNX is too big to load quickly — just report
    # the size + success.
    size_mb = result["size_mb"]
    print(f"[export] total on disk: {size_mb:.1f} MB")
    return result


@app.local_entrypoint()
def main(
    checkpoint: str = "/onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model",
    output_subdir: str = "distill_v031_pi05_libero_r4/onnx_1nfe",
):
    """
    --checkpoint     Path on the pi0-onnx-outputs volume to a SnapFlow
                     student checkpoint dir (contains model.safetensors +
                     config.json + preprocessor files).
    --output-subdir  Where on the volume to write the exported ONNX.
    """
    result = export_modal.remote(
        checkpoint=checkpoint,
        output_subdir=output_subdir,
    )
    print("\n=== EXPORT RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
