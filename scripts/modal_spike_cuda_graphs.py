"""Modal: Day-0 ORT CUDA graphs verification spike on real pi0.5 decomposed export.

Mounts the existing pi0-onnx-outputs Modal volume (where modal_export_pi05_decomposed.py
writes its artifacts) and runs scripts/spike_cuda_graphs_ort.py against the
specified subdir. Verifies whether ORT's enable_cuda_graph flag captures cleanly
on our actual exported ONNX files (post-export Where Cast workaround,
baked Euler loop, RoPE reshape).

Usage:
    modal run scripts/modal_spike_cuda_graphs.py
    modal run scripts/modal_spike_cuda_graphs.py --output-subdir pi05_decomposed_smoke

Reference: features/01_serve/subfeatures/_perf_compound/cuda-graphs/cuda-graphs_plan.md
ADR:       01_decisions/2026-04-24-cuda-graphs-architecture.md

Cost: ~$0.50 on A10G for ~5-10 min wall-clock.
"""
from __future__ import annotations

import modal

app = modal.App("tether-cuda-graphs-spike")

onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=False)
ONNX_OUT = "/onnx_out"

image = (
    # Use nvidia/cuda:12.5 base for cuBLAS / cuDNN availability that matches
    # onnxruntime-gpu 1.20.1's expected runtime. A10G host driver version
    # (separate from container CUDA libs) determines CUDA graph memory
    # overhead per vLLM #5517 — probe modal_probe_a10g_driver.py if spike
    # vlm_prefix OOMs to check driver version.
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
    .add_local_file("scripts/spike_cuda_graphs_ort.py", "/root/spike.py", copy=True)
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={ONNX_OUT: onnx_output},
    timeout=600,
)
def spike(output_subdir: str, atol: float):
    import subprocess
    import sys
    from pathlib import Path

    import onnxruntime as ort

    print("=" * 60)
    print(f"Modal spike: ORT={ort.__version__}, providers={ort.get_available_providers()}")
    print("=" * 60)

    export_dir = Path(ONNX_OUT) / output_subdir
    if not export_dir.exists():
        print(f"FAIL: export_dir {export_dir} does not exist on the pi0-onnx-outputs volume")
        print("Did you run modal_export_pi05_decomposed.py first?")
        print(f"Available subdirs in {ONNX_OUT}:")
        try:
            for p in Path(ONNX_OUT).iterdir():
                if p.is_dir():
                    onnx_files = sorted(f.name for f in p.glob("*.onnx"))
                    print(f"  {p.name}/  {onnx_files}")
        except Exception as e:
            print(f"  (could not list: {e})")
        return {"passed": False, "reason": "export_dir_missing"}

    print(f"Export dir: {export_dir}")
    print(f"Files in export_dir: {sorted(p.name for p in export_dir.iterdir())}")

    rc = subprocess.run(
        [sys.executable, "/root/spike.py", "--export-dir", str(export_dir), "--atol", str(atol)],
        capture_output=False,
    ).returncode

    return {"passed": rc == 0, "exit_code": rc}


@app.local_entrypoint()
def main(
    output_subdir: str = "pi05_decomposed_smoke",
    atol: float = 1e-6,
):
    result = spike.remote(output_subdir=output_subdir, atol=atol)
    print()
    print("=" * 60)
    print("SPIKE RESULT")
    print("=" * 60)
    print(result)
    if result.get("passed"):
        print("PASS — proceed with cuda-graphs Day 1+ implementation")
    else:
        print(f"FAIL — see contingency triage in cuda-graphs_research.md Lens 7")
