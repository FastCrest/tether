"""Modal probe: print nvidia-smi + torch/ORT CUDA versions on A10G.

Diagnostic for the 2026-04-24 vlm_prefix OOM on A10G hypothesis: Modal's
host driver may be < 12.4, triggering the vLLM #5517 cuda-graph memory
overhead that inflates capture footprint ~3x.

Per Modal docs, container CUDA libs are what the app sees but the HOST
driver determines cuda graph capture memory pattern.

Usage:
    modal run scripts/modal_probe_a10g_driver.py
"""
from __future__ import annotations

import modal

app = modal.App("tether-probe-a10g-driver")

image = (
    modal.Image.from_registry("nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch==2.5.1",
        "onnxruntime-gpu==1.20.1",
    )
)


@app.function(image=image, gpu="A10G", timeout=120)
def probe_a10g():
    import subprocess
    import torch
    import onnxruntime as ort

    print("=" * 60)
    print("A10G driver + CUDA runtime probe")
    print("=" * 60)

    print("\n[nvidia-smi]")
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(out.stdout)

    print("\n[nvidia-smi --query-gpu=driver_version,compute_cap]")
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,compute_cap,memory.total,memory.free",
         "--format=csv"],
        capture_output=True, text=True,
    )
    print(out.stdout)

    print("\n[torch]")
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
    print(f"  torch.cuda.get_device_name = {torch.cuda.get_device_name(0)}")

    print("\n[ort]")
    print(f"  ort.__version__ = {ort.__version__}")
    print(f"  ort.get_available_providers = {ort.get_available_providers()}")

    return {"probe_complete": True}


@app.function(image=image, gpu="A100-40GB", timeout=120)
def probe_a100():
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,compute_cap,memory.total,memory.free",
         "--format=csv"],
        capture_output=True, text=True,
    )
    print(f"[A100-40GB]")
    print(out.stdout)
    return {"probe_complete": True}


@app.local_entrypoint()
def main():
    print("Probing A10G...")
    probe_a10g.remote()
    print()
    print("Probing A100 for comparison...")
    probe_a100.remote()
