"""Modal diagnostic: which nodes of the retained pi0.5 monolithic export fall
outside CUDAExecutionProvider, and do the parity numerics pass when CPU
fallback is (non-qualifyingly) allowed.

Read-only wrt the export: opens /parity_out/pi05/export/model.onnx from the
shared receipts volume, prints per-node EP placement from ORT verbose logs
plus a node op-type census, then runs one parity comparison with fallback
allowed to learn whether the numerics (not the gate) pass.

Usage:
    modal run scripts/modal_pi05_ep_placement_diag.py
"""

import os

import modal

app = modal.App("tether-pi05-ep-placement-diag")
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
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "onnx>=1.16",
        "numpy",
    )
    .pip_install(
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
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
            "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_diag(watchdog_seconds: float = 900) -> dict:
    import collections
    import io
    import logging
    import sys
    import time
    from contextlib import redirect_stderr
    from pathlib import Path

    import os as _os
    import signal as _signal
    import threading as _threading

    _watchdog = _threading.Timer(
        float(_os.environ.get("PARITY_WATCHDOG_SECONDS", str(watchdog_seconds))),
        lambda: (_os.kill(_os.getpid(), _signal.SIGKILL)),
    )
    _watchdog.daemon = True
    _watchdog.start()
    out: dict = {}
    onnx_path = Path(PARITY_OUT_PATH) / "pi05" / "export" / "model.onnx"

    import onnx

    model = onnx.load(str(onnx_path))
    census: dict = collections.Counter(
        f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in model.graph.node
    )
    out["node_count"] = len(model.graph.node)
    out["op_census"] = dict(sorted(census.items(), key=lambda kv: -kv[1]))
    out["ir_version"] = model.ir_version
    out["opset"] = [(o.domain, o.version) for o in model.opset_import]
    out["graph_inputs"] = [i.name for i in model.graph.input]
    out["graph_outputs"] = [o.name for o in model.graph.output]
    print(f"[diag] nodes={out['node_count']} opset={out['opset']}", flush=True)
    for op, count in out["op_census"].items():
        print(f"[diag]   {count:6d}  {op}", flush=True)

    import onnxruntime as ort

    print(f"[diag] ort version={ort.__version__}", flush=True)
    print(f"[diag] available providers={ort.get_available_providers()}", flush=True)

    # Placement probe: verbose session build, fallback ALLOWED, capture logs.
    opts = ort.SessionOptions()
    opts.log_severity_level = 0
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logging.getLogger().addHandler(handler)
    try:
        with redirect_stderr(buf):
            sess = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        out["session_built_fallback_allowed"] = True
        out["active_providers"] = list(sess.get_providers())
        print(f"[diag] active providers={out['active_providers']}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- diagnostic record
        out["session_built_fallback_allowed"] = False
        out["session_error"] = repr(exc)[:1000]
        print(f"[diag] session build failed: {out['session_error']}", flush=True)
    finally:
        logging.getLogger().removeHandler(handler)
    logs = buf.getvalue()
    out["verbose_log_bytes"] = len(logs)
    # Keep only placement-relevant lines.
    keep = [
        line
        for line in logs.splitlines()
        if any(
            key in line
            for key in (
                "assigned to",
                "Memcpy",
                "fallback",
                "not supported",
                "unsupported",
                "CPUExecutionProvider",
            )
        )
    ]
    out["placement_lines"] = keep[:120]
    for line in out["placement_lines"]:
        print(f"[ortlog] {line[:400]}", flush=True)

    _watchdog.cancel()
    print("[diag] done", flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    result = run_diag.remote()
    print("\n=== RESULT ===")
    for key, value in result.items():
        if key == "placement_lines":
            print(f"  {key}: ({len(value)} lines, see log above)")
        else:
            print(f"  {key}: {value}")
