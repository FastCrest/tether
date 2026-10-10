"""Modal: pi0 export-parity qualification run (Tether Studio M4 #51).

Runs the receipt-grade harness `scripts/local_pi0_monolithic_parity.py` on
Modal Linux + CUDA, following the proven pi0.5 refix pattern (M4 #52 refix,
receipt `parity-gpu-receipts:/pi05/pi05-export-parity-receipt.json`):

- `lerobot/pi0_base` pinned to PI0_BASE_REVISION (last revision before the
  relative-action processor steps broke lerobot 0.5.1 loads).
- Tether implementation pinned to TETHER_REVISION (exact 40-hex commit
  carrying the Pad-free bool mask build in `apply_export_patches`).
- A100-40GB: the fp32 reference + ~13 GB session cannot co-reside on A10G
  (pi0.5 refix OOM'd allocating a 128 MB arena there); this is a tier
  change only, gates unmodified.
- In-function self-kill watchdog bounds GPU burn.

Usage:
    modal run scripts/modal_pi0_export_parity_pinned.py
"""

import os

import modal


# Last `lerobot/pi0_base` revision before the relative-action processor
# steps landed (same hole as pi0.5's `a538eb27` pin).
PI0_BASE_REVISION = "26b99b9439acb1e352439e34ee9c67af0d76efa3"

# Exact Tether implementation commit under test: Pad-free bool mask build
# in the pi0 `apply_export_patches` path (Concat helper, no ONNX Pad).
TETHER_REVISION = "0e08a30d8f458aefef443908265142f3390fb21a"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-pi0-export-parity-pinned")
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out"


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # Install lerobot first — it pins torch>=2.7 and hub>=1.0
        "lerobot==0.5.1",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",  # onnx-diagnostic soft-dep
        "scipy",  # onnx-diagnostic patches_transformers_qwen2_5 transitively
        "numpy",
        "accelerate",
        "draccus",
    )
    # transformers==5.3.0 EXACTLY: the monolithic exporter refuses anything
    # else (5.4+ has a q_length regression in masking_utils.sdpa_mask).
    .pip_install("transformers==5.3.0")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)

# GPU-aware image for the CUDAExecutionProvider parity session. Base image
# has only `onnxruntime` (CPU). cudnn + cublas alone is NOT enough: ORT
# fails to create the CUDA EP without the CUDA runtime, silently placing
# every node on CPU and then failing the no-fallback gate. Verified
# 2026-09-18 (round-1 vehicle defects).
gpu_image = (
    image.pip_install(
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
    image=gpu_image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_parity(
    model_revision: str = PI0_BASE_REVISION,
    tether_revision: str = TETHER_REVISION,
    num_steps: int = 10,
    reuse_export: bool = False,
    watchdog_seconds: float = 3300,
) -> dict:
    import subprocess
    import sys
    import time
    from pathlib import Path

    import os as _os
    import signal as _signal
    import threading as _threading

    _WATCHDOG_SECONDS = float(
        _os.environ.get("PARITY_WATCHDOG_SECONDS", str(watchdog_seconds))
    )

    def _self_kill() -> None:
        print(
            f"[watchdog] deadline of {_WATCHDOG_SECONDS:.0f}s reached; killing this "
            "container rather than burning GPU time on a hung step.",
            flush=True,
        )
        _os.kill(_os.getpid(), _signal.SIGKILL)

    _watchdog = _threading.Timer(_WATCHDOG_SECONDS, _self_kill)
    _watchdog.daemon = True
    _watchdog.start()

    _started_at = time.monotonic()

    def _progress(stage: str) -> None:
        print(f"[progress] {time.monotonic() - _started_at:7.1f}s  {stage}", flush=True)

    _progress("container up, watchdog armed")
    summary: dict = {
        "model_source": "lerobot/pi0_base",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "provider": "CUDAExecutionProvider",
        "num_steps": num_steps,
    }

    # Fail fast if the CUDA EP cannot be created (a broken CUDA/cuDNN
    # image otherwise surfaces much later as a no-fallback gate failure
    # after the reference has already run).
    import numpy as _np

    import onnx as _onnx
    import onnxruntime as _ort

    _node = _onnx.helper.make_node("Add", ["x", "y"], ["z"])
    _graph = _onnx.helper.make_graph(
        [_node],
        "smoke",
        [
            _onnx.helper.make_tensor_value_info("x", _onnx.TensorProto.FLOAT, [1]),
            _onnx.helper.make_tensor_value_info("y", _onnx.TensorProto.FLOAT, [1]),
        ],
        [_onnx.helper.make_tensor_value_info("z", _onnx.TensorProto.FLOAT, [1])],
    )
    _smoke = _onnx.helper.make_model(_graph, opset_imports=[_onnx.helper.make_opsetid("", 19)])
    _smoke.ir_version = 10
    _so = _ort.SessionOptions()
    _so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    _smoke_sess = _ort.InferenceSession(
        _smoke.SerializeToString(), sess_options=_so, providers=["CUDAExecutionProvider"]
    )
    _smoke_providers = _smoke_sess.get_providers()
    # Membership (not equality): ORT always lists CPUExecutionProvider, but
    # with cpu-ep fallback disabled it is inert -- same semantics as the
    # receipt gate (`requested_provider in active providers`).
    assert _smoke_providers[0] == "CUDAExecutionProvider", (
        f"CUDA EP smoke providers: {_smoke_providers}"
    )
    _z = _smoke_sess.run(
        None,
        {"x": _np.ones((1,), dtype=_np.float32), "y": _np.ones((1,), dtype=_np.float32)},
    )[0]
    assert float(_z[0]) == 2.0
    summary["cuda_ep_smoke"] = f"ok (ort {_ort.__version__})"
    _progress(f"CUDA EP smoke ok (ort {_ort.__version__})")

    # 1. Pin the implementation: clone + checkout the exact Tether commit.
    pin_dir = Path("/root/tether-pin")
    if pin_dir.exists():
        subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(
        ["git", "clone", "--quiet", TETHER_REPO, str(pin_dir)], check=True
    )
    subprocess.run(
        ["git", "-C", str(pin_dir), "checkout", "--quiet", tether_revision],
        check=True,
    )
    got = subprocess.check_output(
        ["git", "-C", str(pin_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    assert got == tether_revision, f"tether pin mismatch: {got} != {tether_revision}"
    summary["tether_commit"] = got
    _progress(f"tether pinned at {got[:12]}")
    sys.path.insert(0, str(pin_dir / "src"))

    # 2. Pin the model: snapshot the exact revision, export from the local
    # snapshot dir so artifact and reference agree by construction.
    from huggingface_hub import snapshot_download

    _progress("snapshot_download lerobot/pi0_base@" + model_revision[:12])
    snap = snapshot_download("lerobot/pi0_base", revision=model_revision)
    summary["snapshot_dir"] = snap

    from tether.exporters.monolithic import export_pi0_monolithic

    export_dir = Path(PARITY_OUT_PATH) / "pi0" / "export"
    existing = export_dir / "model.onnx"
    # model.onnx may be graph-only with weights in model.onnx.data: count
    # the whole artifact dir, which is also what the receipt hashes.
    existing_bytes = (
        sum(f.stat().st_size for f in export_dir.glob("*") if f.is_file())
        if existing.is_file()
        else 0
    )
    if reuse_export and existing_bytes > 10**9:
        export_result = {
            "status": "reused",
            "onnx_path": str(existing),
            "size_mb": existing_bytes / 1e6,
            "num_steps": num_steps,
        }
        _progress(f"reusing retained export ({export_result['size_mb']:.1f}MB)")
    else:
        _progress("export_pi0_monolithic from pinned snapshot")
        export_result = export_pi0_monolithic(snap, export_dir, num_steps=num_steps)
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

    # Release the export phase's cached GPU blocks before the parity
    # subprocess: the harness places the reference on CUDA, and the
    # parent's caching-allocator reservation would otherwise OOM it
    # (observed on pi0.5: parent held 22.03/22.06 GiB, child got 16 MiB).
    import gc as _gc

    _gc.collect()
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 -- diagnostic only
        print(f"[warn] cuda cache release failed: {exc!r}", flush=True)
    _progress("parent GPU cache released")

    # 3. Receipt-grade parity with CUDA, CPU fallback disabled.
    receipt_path = Path(PARITY_OUT_PATH) / "pi0" / "pi0-export-parity-receipt.json"
    cmd = [
        sys.executable,
        str(pin_dir / "scripts" / "local_pi0_monolithic_parity.py"),
        "--model-source", "lerobot/pi0_base",
        "--model-revision", model_revision,
        "--onnx-dir", str(export_dir),
        "--provider", "CUDAExecutionProvider",
        "--num-steps", str(num_steps),
        "--receipt", str(receipt_path),
    ]
    _progress("running receipt-grade parity (CUDA, no CPU fallback)")
    harness_env = dict(_os.environ)
    harness_env["PYTHONPATH"] = str(pin_dir / "src") + _os.pathsep + harness_env.get(
        "PYTHONPATH", ""
    )
    proc = subprocess.run(
        cmd, cwd=str(pin_dir), capture_output=True, text=True, env=harness_env
    )
    print(proc.stdout[-6000:], flush=True)
    print(proc.stderr[-3000:], flush=True)
    summary["harness_returncode"] = proc.returncode

    import json as _json

    if receipt_path.is_file():
        receipt = _json.loads(receipt_path.read_text())
        summary["verdict"] = receipt.get("verdict")
        summary["external_acceptance"] = receipt.get("external_acceptance")
        summary["first_cosine"] = receipt.get("first_cosine")
        summary["first_max_abs"] = receipt.get("first_max_abs")
        summary["full_cosine"] = receipt.get("full_cosine")
        summary["full_max_abs"] = receipt.get("full_max_abs")
        summary["ort_providers"] = receipt.get("ort_providers")
        summary["platform"] = receipt.get("platform_system")
        print("[receipt] " + _json.dumps(receipt, sort_keys=True), flush=True)
    else:
        summary["verdict"] = "no-receipt"

    _parity_out_volume.commit()
    _watchdog.cancel()
    _progress("done")
    return summary


@app.local_entrypoint()
def main(
    tether_revision: str = TETHER_REVISION,
    reuse_export: bool = False,
    num_steps: int = 10,
) -> None:
    result = run_export_parity.remote(
        tether_revision=tether_revision,
        reuse_export=reuse_export,
        num_steps=num_steps,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
