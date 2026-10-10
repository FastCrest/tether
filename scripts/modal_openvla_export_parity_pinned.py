"""Modal: OpenVLA export-parity qualification run (Tether Studio M4 #54).

Runs the receipt-grade harness `scripts/local_openvla_monolithic_parity.py`
on Modal Linux + CUDA (A100-40GB), following the proven pi0.5 refix pattern:

- `openvla/openvla-7b` pinned to OPENVLA_REVISION (exact 40-hex).
- Tether implementation pinned to TETHER_REVISION (exact 40-hex commit
  carrying `export_openvla_monolithic` + the fused-vision fixture fix).
- The export is REUSED from the shared volume (`openvla/export/`): it was
  built on cheap CPU by `modal_openvla_export_cpu_pinned.py` from the same
  pinned source and passed the CPU numerics pre-check (verdict passed,
  7/7 tokens, cos 1.0). This job only adds the CUDA placement gate +
  the receipt. Set reuse_export=False to rebuild (CPU-traced; slow).
- modeling_prismatic.py pins its own deps (timm 0.9.x, transformers
  4.40.1); the image below honors them. The reference runs on CPU
  (fp32); only the ORT session uses CUDA.
- In-function self-kill watchdog bounds GPU burn.

Usage:
    modal run scripts/modal_openvla_export_parity_pinned.py
"""

import os

import modal


# `openvla/openvla-7b` HEAD (2026-02-17 README update; weights/code stable
# since 2024-09-16). Pinned exactly; floating HEAD is refused.
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# Exact Tether implementation commit under test. Override at the CLI; the
# final report pins the exact hash per job.
TETHER_REVISION = "d414707232e641e7c3cf7be3adb60395786b7e33"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-openvla-export-parity-pinned")
_hf_cache_volume = modal.Volume.from_name(
    "openvla-hf-cache", create_if_missing=True
)
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


# Reference-load stack first (contemporary pins the snapshot demands),
# then the GPU ORT + CUDA libs for the parity session.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "timm==0.9.16",
        "numpy==1.26.4",
        "Pillow",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23",
        "accelerate",
        "scipy",
        "onnx>=1.16",
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
    timeout=3600,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_parity(
    model_revision: str = OPENVLA_REVISION,
    tether_revision: str = TETHER_REVISION,
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
        "model_source": "openvla/openvla-7b",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "provider": "CUDAExecutionProvider",
    }

    # Fail fast if the CUDA EP cannot be created.
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
    _smoke = _onnx.helper.make_model(_graph, opset_imports=[_onnx.helper.make_opsetid("", 18)])
    _smoke.ir_version = 9
    _so = _ort.SessionOptions()
    _so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    _smoke_sess = _ort.InferenceSession(
        _smoke.SerializeToString(), sess_options=_so, providers=["CUDAExecutionProvider"]
    )
    _smoke_providers = _smoke_sess.get_providers()
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

    # 2. Reuse the CPU-built export (same pinned source, CPU-verified).
    # Rebuild only on explicit request (CPU-traced; slow but GPU-free).
    export_dir = Path(PARITY_OUT_PATH) / "openvla" / "export"
    existing = export_dir / "model.onnx"
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
        }
        _progress(f"reusing retained export ({export_result['size_mb']:.1f}MB)")
    else:
        from huggingface_hub import snapshot_download

        _progress("snapshot_download openvla/openvla-7b@" + model_revision[:12])
        snap = snapshot_download("openvla/openvla-7b", revision=model_revision)
        from tether.exporters.openvla import export_openvla_monolithic

        _progress("export_openvla_monolithic from pinned snapshot")
        export_result = export_openvla_monolithic(snap, export_dir)
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

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
    receipt_path = Path(PARITY_OUT_PATH) / "openvla" / "openvla-export-parity-receipt.json"
    cmd = [
        sys.executable,
        str(pin_dir / "scripts" / "local_openvla_monolithic_parity.py"),
        "--model-source", "openvla/openvla-7b",
        "--model-revision", model_revision,
        "--onnx-dir", str(export_dir),
        "--provider", "CUDAExecutionProvider",
        "--action-dim", "7",
        "--trust-remote-code",
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
        summary["metrics"] = receipt.get("metrics")
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
    reuse_export: bool = True,
) -> None:
    result = run_export_parity.remote(
        tether_revision=tether_revision,
        reuse_export=reuse_export,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
