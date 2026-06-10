"""Modal: export pi05_libero_finetuned_v044 to monolithic ONNX at num_steps=10.

The existing modal_pi05_monolithic_export.py hardcodes output_dir to
`/onnx_out/monolithic/` which would overwrite. This script writes to
`/onnx_out/pi05_libero_finetuned_v044/onnx_num_steps_10/` and uses the
library function `tether.exporters.monolithic.export_pi05_monolithic`.

Usage:
    modal run scripts/modal_export_pi05_libero_teacher.py
"""
import os
import subprocess
import modal

app = modal.App("tether-export-pi05-libero-teacher")


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
        f'pip install "fastcrest-tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
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
    timeout=7200,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def export_modal(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    num_steps: int = 10,
    output_subdir: str = "pi05_libero_finetuned_v044/onnx_num_steps_10",
):
    """Export pi0.5 LIBERO teacher monolithic ONNX at num_steps=10."""
    import logging
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from tether.exporters.monolithic import export_pi05_monolithic

    out = Path(ONNX_OUT) / output_subdir
    out.mkdir(parents=True, exist_ok=True)

    result = export_pi05_monolithic(
        model_id=model_id,
        output_dir=str(out),
        num_steps=num_steps,
        target="desktop",
    )
    onnx_output.commit()

    # Sanity: ONNX loads
    import onnx
    onnx_path = result["onnx_path"]
    model = onnx.load(onnx_path, load_external_data=False)
    print(f"[export] IR: {model.ir_version}, opset: {model.opset_import[0].version}")
    print(f"[export] inputs: {[i.name for i in model.graph.input]}")
    print(f"[export] outputs: {[o.name for o in model.graph.output]}")
    return result


@app.local_entrypoint()
def main(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    num_steps: int = 10,
    output_subdir: str = "pi05_libero_finetuned_v044/onnx_num_steps_10",
):
    r = export_modal.remote(
        model_id=model_id, num_steps=num_steps, output_subdir=output_subdir,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
