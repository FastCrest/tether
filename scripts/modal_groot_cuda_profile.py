"""Modal A100 diagnostic (M4 #53): ORT-CPU vs ORT-CUDA vs torch profiles.

The rebuilt export passes placement and cosine but fails full_max_abs on
CUDA (1.26e-03) while ORT-CPU matches torch to 1.37e-06. Same artifact,
same inputs — only the EP differs. This job runs all three on one A100
container and prints per-position max_abs profiles for:

  torch-CPU (reference) vs ORT-CPU   — export fidelity on CPU,
  torch-CPU (reference) vs ORT-CUDA  — the receipt comparison,
  ORT-CPU vs ORT-CUDA                — pure backend-kernel delta.

Read-only against the retained export. No receipt, no gate touched.

Usage:
    modal run scripts/modal_groot_cuda_profile.py
"""

import modal

app = modal.App("tether-groot-cuda-profile")
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
def run_profile() -> dict:
    import subprocess
    import sys
    from pathlib import Path

    import numpy as _np
    import torch as _torch

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
    with _torch.no_grad():
        ref = full(
            _torch.from_numpy(shared["noisy_actions"]),
            _torch.from_numpy(shared["timestep"]),
            _torch.from_numpy(shared["position_ids"]),
        ).numpy().astype(_np.float32)

    import onnxruntime as _ort

    print(f"[profile] ort {_ort.__version__}", flush=True)
    export_path = Path(PARITY_OUT_PATH) / "groot" / "export" / "model.onnx"

    cpu_sess = _ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    out_cpu = _np.asarray(cpu_sess.run(None, shared)[0], dtype=_np.float32)

    so = _ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    cuda_sess = _ort.InferenceSession(
        str(export_path), sess_options=so, providers=["CUDAExecutionProvider"]
    )
    print(f"[profile] cuda providers: {cuda_sess.get_providers()}", flush=True)
    out_cuda = _np.asarray(cuda_sess.run(None, shared)[0], dtype=_np.float32)

    def prof(name, a, b):
        e = _np.abs(a - b)
        p = e[0].max(axis=1)
        print(f"[profile] {name}: max={float(e.max()):.4e} " + " ".join(
            f"{i}:{v:.1e}" for i, v in enumerate(p)
        ), flush=True)
        return float(e.max())

    r = {
        "torch_vs_ort_cpu": prof("torch_vs_ort_cpu", ref, out_cpu),
        "torch_vs_ort_cuda": prof("torch_vs_ort_cuda", ref, out_cuda),
        "ort_cpu_vs_ort_cuda": prof("ort_cpu_vs_ort_cuda", out_cpu, out_cuda),
    }
    flat = int(_np.argmax(_np.abs(ref - out_cuda)))
    pos, dim = divmod(flat, raw_action_dim)
    print(
        f"[profile] cuda argmax pos={pos} dim={dim} ref={float(ref[0,pos,dim]):.6f} "
        f"cuda={float(out_cuda[0,pos,dim]):.6f} cpu-ort={float(out_cpu[0,pos,dim]):.6f}",
        flush=True,
    )
    r["cuda_argmax"] = {"pos": pos, "dim": dim}
    return r


@app.local_entrypoint()
def main() -> None:
    result = run_profile.remote()
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
