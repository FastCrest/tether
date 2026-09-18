"""Modal: GR00T export-parity qualification run (Tether Studio M4 #53).

Runs the receipt-grade harness `scripts/local_groot_monolithic_parity.py` on
Modal Linux + CUDA, following the proven pi0 pattern (exact 40-hex model
commit pin, no floating HEAD, plus an in-function self-kill watchdog):

- `nvidia/GR00T-N1.6-3B` is pinned to GROOT_REVISION (the repo's single
  squashed main commit; floating HEAD is refused regardless).
- The export is built from a local snapshot of that exact revision, so the
  hashed artifact and the reference revision in the receipt agree by
  construction (the snapshot dir is passed as `model_id`, no repo edits).
- The Tether implementation under test is pinned to TETHER_REVISION (an
  exact 40-hex commit): the container clones the repo and checks out that
  commit, and the harness records `git rev-parse HEAD` from that checkout.
- A self-kill wall-clock watchdog bounds GPU burn: `timeout=` on the
  decorator is the platform's promise; a hung download or stuck CUDA call
  can sit in the container without tripping it.

GR00T's gate (docs/groot-export-parity.md) is cosine >= 0.9999 and max
absolute error < 1e-3 on the per-step velocity graph; the harness enforces
it, this script only reports it.

Usage:
    modal run scripts/modal_groot_export_parity_pinned.py
"""

import os

import modal


# `nvidia/GR00T-N1.6-3B` main is a single squashed commit (2025-12-15).
# Pinned exactly; floating HEAD is refused.
GROOT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"

# Exact Tether implementation commit under test. This vehicle adds only new
# files; src/ and the harness are byte-identical to this commit.
TETHER_REVISION = "8f464ebdff9cda09d9f30fca0af5b409cfd794b6"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-groot-export-parity-pinned")
_hf_cache_volume = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out"


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers==5.3.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "numpy",
        "Pillow",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
    )
    # onnxruntime-gpu for the CUDAExecutionProvider parity session (base
    # image carries CPU ORT only); same stanza as the pi0.5 export script.
    .pip_install(
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
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
            "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_parity(
    model_revision: str = GROOT_REVISION,
    tether_revision: str = TETHER_REVISION,
    embodiment_id: int = 0,
    reuse_export: bool = True,
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
        "model_source": "nvidia/GR00T-N1.6-3B",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "provider": "CUDAExecutionProvider",
        "embodiment_id": embodiment_id,
    }

    # 1. Pin the implementation: clone + checkout the exact Tether commit.
    pin_dir = Path("/root/tether-pin")
    if pin_dir.exists():
        subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(["git", "clone", "--quiet", TETHER_REPO, str(pin_dir)], check=True)
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

    _progress("snapshot_download nvidia/GR00T-N1.6-3B@" + model_revision[:12])
    snap = snapshot_download("nvidia/GR00T-N1.6-3B", revision=model_revision)
    summary["snapshot_dir"] = snap

    from tether.exporters.monolithic import export_gr00t_monolithic

    export_dir = Path(PARITY_OUT_PATH) / "groot" / "export"
    existing = export_dir / "model.onnx"
    if reuse_export and existing.is_file() and existing.stat().st_size > 10**6:
        export_result = {
            "status": "reused",
            "onnx_path": str(existing),
            "size_mb": existing.stat().st_size / 1e6,
            "embodiment_id": embodiment_id,
        }
        _progress(f"reusing retained export ({export_result['size_mb']:.1f}MB)")
    else:
        _progress("export_gr00t_monolithic from pinned snapshot")
        export_result = export_gr00t_monolithic(
            snap, export_dir, embodiment_id=embodiment_id
        )
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

    # 3. Receipt-grade parity with CUDA, CPU fallback disabled.
    receipt_path = Path(PARITY_OUT_PATH) / "groot" / "groot-export-parity-receipt.json"
    cmd = [
        sys.executable,
        str(pin_dir / "scripts" / "local_groot_monolithic_parity.py"),
        "--model-source", "nvidia/GR00T-N1.6-3B",
        "--model-revision", model_revision,
        "--embodiment-id", str(embodiment_id),
        "--onnx-dir", str(export_dir),
        "--provider", "CUDAExecutionProvider",
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
def main() -> None:
    result = run_export_parity.remote()
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
