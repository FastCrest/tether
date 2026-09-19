"""Modal A100 diagnostic (M4 #53): is the CUDA-only 1e-3 gap TF32?

Same artifact + inputs throughout; only the compute backend varies:

  A. torch-CPU reference (the harness reference, fp32-strict)
  B. torch-CUDA reference, TF32 allowed (PyTorch default on Ampere+)
  C. torch-CUDA reference, TF32 disabled (allow_tf32 False, strict fp32)
  D. ORT-CUDA session (the receipt subject)

Predictions under the TF32 theory (cublas default math in ORT-CUDA):
  B vs D ~= 1e-6  (both TF32 -> agree; explains April's number if April
                   ran its reference on CUDA)
  C vs D ~= 1e-3  (strict vs TF32 -> the receipt gap)
  C vs A ~= 1e-7  (sanity: strict torch-CUDA == torch-CPU)

Read-only against the retained export. No receipt, no gate touched.

Usage:
    modal run scripts/modal_groot_tf32_test.py
"""

import modal

app = modal.App("tether-groot-tf32-test")
_hf_cache_volume = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out"

TETHER_REVISION = "4cf9854a72ca9c97ec6a7a398f4246650b001d2a"
TETHER_REPO = "https://github.com/FastCrest/tether.git"
GROOT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers==5.3.0",
        "onnx>=1.16",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "lerobot==0.5.1",
        "num2words",
        "accelerate",
        "draccus",
        "numpy",
        "Pillow",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
    )
    .pip_install(
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cufft-cu12>=11.0,<12.0",
        "nvidia-curand-cu12>=10.0,<11.0",
        extra_options="--no-deps",
    )
    .pip_install("onnxruntime-gpu>=1.20,<1.24")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
)
def run_tf32_test() -> dict:
    import subprocess
    import sys
    from pathlib import Path

    import numpy as _np
    import torch as _torch

    print(
        f"[tf32] torch defaults: matmul.allow_tf32="
        f"{_torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={_torch.backends.cudnn.allow_tf32}",
        flush=True,
    )

    pin_dir = Path("/root/tether-pin")
    if pin_dir.exists():
        subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(["git", "clone", "--quiet", TETHER_REPO, str(pin_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(pin_dir), "checkout", "--quiet", TETHER_REVISION],
        check=True,
    )
    sys.path.insert(0, str(pin_dir / "src"))

    from huggingface_hub import snapshot_download

    from tether.checkpoint import load_checkpoint
    from tether.exporters.gr00t import build_gr00t_full_stack

    snap = snapshot_download("nvidia/GR00T-N1.6-3B", revision=GROOT_REVISION)
    state_dict, _ = load_checkpoint(snap)
    full, metadata = build_gr00t_full_stack(state_dict, embodiment_id=0)
    full.eval()
    raw_action_dim = int(metadata["raw_action_dim"])

    input_rng = _np.random.RandomState(42)
    noise_rng = _np.random.RandomState(99)
    shared = {
        "noisy_actions": noise_rng.randn(1, 50, raw_action_dim).astype(_np.float32),
        "timestep": _np.array([input_rng.uniform(0.05, 0.95)], dtype=_np.float32),
        "position_ids": _np.arange(50, dtype=_np.int64)[None, :],
    }

    def run_ref(model, device):
        na = _torch.from_numpy(shared["noisy_actions"]).to(device)
        ts = _torch.from_numpy(shared["timestep"]).to(device)
        pi = _torch.from_numpy(shared["position_ids"]).to(device)
        with _torch.no_grad():
            return model(na, ts, pi).cpu().numpy().astype(_np.float32)

    # A. harness reference: CPU, fp32-strict.
    ref_cpu = run_ref(full, "cpu")
    print("[tf32] ref_cpu done", flush=True)

    # B/C. CUDA references on a copy (keep the CPU module untouched).
    import copy as _copy

    full_cuda = _copy.deepcopy(full).to("cuda")
    assert _torch.backends.cuda.matmul.allow_tf32 is True
    ref_cuda_tf32 = run_ref(full_cuda, "cuda")
    print("[tf32] ref_cuda_tf32 done", flush=True)

    _torch.backends.cuda.matmul.allow_tf32 = False
    _torch.backends.cudnn.allow_tf32 = False
    ref_cuda_strict = run_ref(full_cuda, "cuda")
    print("[tf32] ref_cuda_strict done", flush=True)
    del full_cuda
    _torch.cuda.empty_cache()

    import onnxruntime as _ort

    export_path = Path(PARITY_OUT_PATH) / "groot" / "export" / "model.onnx"
    so = _ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    cuda_sess = _ort.InferenceSession(
        str(export_path), sess_options=so, providers=["CUDAExecutionProvider"]
    )
    out_cuda = _np.asarray(cuda_sess.run(None, shared)[0], dtype=_np.float32)
    print("[tf32] ort_cuda done", flush=True)

    def cmp(name, a, b):
        e = float(_np.abs(a - b).max())
        print(f"[tf32] {name}: max_abs={e:.4e}", flush=True)
        return e

    return {
        "B_tf32_vs_D_ort_cuda": cmp("B_ref_cuda_tf32_vs_D_ort_cuda", ref_cuda_tf32, out_cuda),
        "C_strict_vs_D_ort_cuda": cmp("C_ref_cuda_strict_vs_D_ort_cuda", ref_cuda_strict, out_cuda),
        "C_strict_vs_A_cpu": cmp("C_ref_cuda_strict_vs_A_ref_cpu", ref_cuda_strict, ref_cpu),
        "B_tf32_vs_A_cpu": cmp("B_ref_cuda_tf32_vs_A_ref_cpu", ref_cuda_tf32, ref_cpu),
        "A_cpu_vs_D_ort_cuda": cmp("A_ref_cpu_vs_D_ort_cuda", ref_cpu, out_cuda),
    }


@app.local_entrypoint()
def main() -> None:
    result = run_tf32_test.remote()
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
