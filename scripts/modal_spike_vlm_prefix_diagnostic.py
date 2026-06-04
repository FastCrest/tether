"""Modal: vlm_prefix Where+Cast diagnostic spike (validates root-cause hypothesis
for the 2026-04-24 A100 spike's capture_vs_eager_diverged failure).

Runs scripts/spike_vlm_prefix_diagnostic.py on Modal A100-40GB against the
pi0-onnx-outputs volume. ~30 min, ~$1-2 Modal.

Usage:
    modal run scripts/modal_spike_vlm_prefix_diagnostic.py
    modal run scripts/modal_spike_vlm_prefix_diagnostic.py --output-subdir distill_v050r2_decomposed
"""
from __future__ import annotations

import modal

app = modal.App("tether-vlm-prefix-diagnostic")

onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=False)
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.from_registry("nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0",
        "onnx>=1.16",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
    })
    .add_local_file("scripts/spike_vlm_prefix_diagnostic.py", "/root/diagnose.py", copy=True)
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={ONNX_OUT: onnx_output},
    timeout=900,
)
def diagnose(output_subdir: str):
    import subprocess
    import sys
    from pathlib import Path

    export_dir = Path(ONNX_OUT) / output_subdir
    if not export_dir.exists():
        print(f"FAIL: {export_dir} does not exist")
        return {"passed": False, "reason": "missing_export_dir"}

    rc = subprocess.run(
        [sys.executable, "/root/diagnose.py", "--export-dir", str(export_dir)],
        capture_output=False,
    ).returncode
    return {"passed": rc == 0, "exit_code": rc}


@app.local_entrypoint()
def main(output_subdir: str = "distill_v050r2_decomposed"):
    result = diagnose.remote(output_subdir=output_subdir)
    print()
    print("=" * 60)
    print("DIAGNOSTIC RESULT")
    print("=" * 60)
    print(result)
