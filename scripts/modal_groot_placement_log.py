"""Modal diagnostic: capture ORT's verbose EP-assignment log for the retained
GR00T export by building the session in a subprocess with stderr to a file.

Usage:
    modal run scripts/modal_groot_placement_log.py
"""

import os

import modal

app = modal.App("tether-groot-placement-log")
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

PARITY_OUT_PATH = "/parity_out"
HF_CACHE_PATH = "/root/.cache/huggingface"
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)


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
    .pip_install("onnx>=1.16", "numpy")
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
    gpu="A10G",
    timeout=1200,
    volumes={
        PARITY_OUT_PATH: _parity_out_volume,
        HF_CACHE_PATH: _hf_cache_volume,
    },
    secrets=[_hf_secret()],
)
def run_diag(watchdog_seconds: float = 900) -> dict:
    import os as _os
    import signal as _signal
    import subprocess
    import sys
    import threading as _threading
    from pathlib import Path

    _watchdog = _threading.Timer(
        float(_os.environ.get("PARITY_WATCHDOG_SECONDS", str(watchdog_seconds))),
        lambda: (_os.kill(_os.getpid(), _signal.SIGKILL)),
    )
    _watchdog.daemon = True
    _watchdog.start()

    import onnx

    onnx_path = str(Path(PARITY_OUT_PATH) / "groot" / "export" / "model.onnx")
    model = onnx.load(onnx_path, load_external_data=False)
    names = [(n.name, n.op_type) for n in model.graph.node]
    print(f"[place] graph nodes: {len(names)}", flush=True)

    build_src = (
        "import sys, onnxruntime as ort\n"
        f"opts = ort.SessionOptions()\n"
        "opts.log_severity_level = 0\n"
        f"sess = ort.InferenceSession({onnx_path!r}, sess_options=opts,\n"
        "    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])\n"
        "print('SESSION_BUILT providers=' + ','.join(sess.get_providers()), flush=True)\n"
    )
    with open("/tmp/build_sess.py", "w") as fh:
        fh.write(build_src)
    with open("/tmp/place.log", "w") as log:
        proc = subprocess.run(
            [sys.executable, "/tmp/build_sess.py"],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
    lines = Path("/tmp/place.log").read_text(errors="replace").splitlines()
    print(f"[place] captured {len(lines)} log lines, rc={proc.returncode}", flush=True)

    name_set = {n for n, _ in names if n}
    hits: dict = {}
    for line in lines:
        low = line.lower()
        if (
            "cpu" in low
            and ("assign" in low or "fallback" in low or "not supported" in low
                 or "unsupported" in low or "placed" in low or "partition" in low)
        ):
            key = line[:300]
            hits[key] = hits.get(key, 0) + 1
    print(f"[place] {len(hits)} distinct cpu-assignment lines", flush=True)
    for line, count in sorted(hits.items(), key=lambda kv: -kv[1])[:40]:
        print(f"[place] x{count} {line}", flush=True)

    # Which graph node names appear near cpu/fallback context?
    node_hits: dict = {}
    for i, line in enumerate(lines):
        for name, op in names:
            if name and name in line and (
                "cpu" in line.lower() or "CUDA" in line
            ):
                node_hits[(name, op)] = node_hits.get((name, op), 0) + 1
    top = sorted(node_hits.items(), key=lambda kv: -kv[1])[:30]
    for (name, op), count in top:
        print(f"[place] node x{count} {op} {name}", flush=True)

    _watchdog.cancel()
    return {
        "log_lines": len(lines),
        "cpu_assignment_lines": len(hits),
        "node_hits": [(n, o, c) for (n, o), c in top],
    }


@app.local_entrypoint()
def main() -> None:
    result = run_diag.remote()
    print("\n=== RESULT ===")
    print(f"  log_lines: {result['log_lines']}")
    print(f"  cpu_assignment_lines: {result['cpu_assignment_lines']}")
    for name, op, count in result["node_hits"]:
        print(f"  x{count} {op} {name}")
