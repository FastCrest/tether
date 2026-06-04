"""Convert an FP32 monolithic ONNX to FP16 on Modal, targeting Orin Nano 8GB fit.

The heavy pi0/pi0.5 monolithic ONNX exports (12.5GB / 13GB) don't load on
Orin Nano 8GB at FP32. FP16 conversion halves on-disk size (pi0 → ~6.3GB
estimated) and brings them into reach. SmolVLA (1.6GB) and GR00T (4.4GB)
already fit.

This script:
1. Loads a FP32 monolithic ONNX from the `pi0-onnx-outputs` Modal volume.
2. Runs `tether.exporters.fp16_convert.convert_fp32_to_fp16` with the
   LayerNorm-adjacent op blocklist.
3. Writes the FP16 ONNX + `.bin` external data alongside the original.
4. Reports size reduction.

Parity validation is a separate script — this one just does the conversion.

Usage:
    # Convert pi0 FP32 → FP16
    modal run scripts/modal_fp16_convert.py \\
        --src-subdir pi0_monolithic --dst-subdir pi0_monolithic_fp16

    # Convert pi0.5 FP32 → FP16
    modal run scripts/modal_fp16_convert.py \\
        --src-subdir pi05_monolithic --dst-subdir pi05_monolithic_fp16
"""
import os
import subprocess
import modal

app = modal.App("tether-fp16-convert")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()

onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
ONNX_OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxconverter-common>=1.14",  # FP16 converter
        "numpy",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
            secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    # FP16 conversion is CPU-bound (pure numpy reinterpretation); no GPU needed.
    # But we want enough RAM to load a 12.5GB onnx + convert → use a big CPU box.
    cpu=4,
    memory=32768,  # 32 GB — enough headroom for pi0 (12.5GB on disk + working copies)
    timeout=1800,
    volumes={ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def convert_modal(
    src_subdir: str = "pi0_monolithic",
    dst_subdir: str = "pi0_monolithic_fp16",
    model_filename: str = "model.onnx",
):
    """Run FP16 conversion on a monolithic ONNX sitting on the volume."""
    import time
    from pathlib import Path

    from tether.exporters.fp16_convert import convert_fp32_to_fp16

    src_path = Path(ONNX_OUTPUT_PATH) / src_subdir / model_filename
    dst_path = Path(ONNX_OUTPUT_PATH) / dst_subdir / model_filename

    if not src_path.exists():
        return {
            "status": "fail",
            "reason": f"{src_path} not found on volume",
        }

    print(f"[fp16] {src_path} → {dst_path}")
    t0 = time.time()
    result = convert_fp32_to_fp16(src_path, dst_path)
    elapsed = time.time() - t0

    # Commit the volume write so subsequent parity runs see the new file.
    onnx_output.commit()

    print(f"[fp16] DONE in {elapsed:.1f}s")
    print(f"[fp16]   {result['src_bytes'] / 1e9:.2f} GB → "
          f"{result['dst_bytes'] / 1e9:.2f} GB "
          f"(-{result['reduction_ratio'] * 100:.1f}%)")

    return {
        "status": "ok",
        "elapsed_s": elapsed,
        **result,
    }


@app.local_entrypoint()
def main(
    src_subdir: str = "pi0_monolithic",
    dst_subdir: str = "pi0_monolithic_fp16",
):
    """
    --src-subdir: the FP32 ONNX's subdir under /onnx_out/ on the volume.
    --dst-subdir: where the FP16 output goes.
    """
    r = convert_modal.remote(src_subdir=src_subdir, dst_subdir=dst_subdir)
    print("\n=== RESULT ===")
    for k, v in r.items():
        if isinstance(v, (int, float, str, bool, list)):
            print(f"  {k}: {v}")
