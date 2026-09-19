"""Modal: pi0 CPU export + numerics pre-check (Tether Studio M4 #51).

Cheap CPU job that de-risks the A100 receipt run (same split as the
OpenVLA and pi0.5 flows):

1. clones the exact Tether implementation commit,
2. snapshots `lerobot/pi0_base` at the exact pinned revision,
3. runs `export_pi0_monolithic` to the shared output volume
   (`pi0/export/`), so the later A100 job reuses it,
4. prints an ONNX op census (Pad count must be 0 after the Pad-free fix),
5. runs `scripts/local_pi0_monolithic_parity.py` with the CPU provider
   as a numerics pre-check (verdict signal only — external acceptance
   stays `not-run` off CUDA by design; gates unmodified).

No GPU is used. A self-kill watchdog bounds wall time.

Usage:
    modal run scripts/modal_pi0_export_cpu_pinned.py
"""

import os

import modal


# Last `lerobot/pi0_base` revision before the relative-action processor
# steps landed (same hole as pi0.5's `a538eb27` pin).
PI0_BASE_REVISION = "26b99b9439acb1e352439e34ee9c67af0d76efa3"

# Exact Tether implementation commit under test. Override at the CLI for
# fix iterations; the final report pins the exact hash per job.
TETHER_REVISION = "3bd9507e0d17fc53311cc4dcc1ac47ab3fb68609"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-pi0-export-cpu-pinned")
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


# Same stack as the A100 vehicle minus the GPU (transformers pinned
# exactly: the monolithic exporter refuses anything else).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
    )
    .pip_install("transformers==5.3.0")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    timeout=5400,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_cpu(
    model_revision: str = PI0_BASE_REVISION,
    tether_revision: str = TETHER_REVISION,
    num_steps: int = 10,
    reuse_export: bool = False,
    run_harness: bool = True,
    watchdog_seconds: float = 5100,
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
            "container rather than burning time on a hung step.",
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
        "num_steps": num_steps,
    }

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

    from huggingface_hub import snapshot_download

    _progress("snapshot_download lerobot/pi0_base@" + model_revision[:12])
    snap = snapshot_download("lerobot/pi0_base", revision=model_revision)
    summary["snapshot_dir"] = snap

    from tether.exporters.monolithic import export_pi0_monolithic

    export_dir = Path(PARITY_OUT_PATH) / "pi0" / "export"
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
            "num_steps": num_steps,
        }
        _progress(f"reusing retained export ({export_result['size_mb']:.1f}MB)")
    else:
        _progress("export_pi0_monolithic from pinned snapshot (CPU)")
        export_result = export_pi0_monolithic(snap, export_dir, num_steps=num_steps)
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

    # ONNX census: the Pad-free fix must leave zero Pad nodes.
    import collections as _collections

    import onnx as _onnx

    _model = _onnx.load(str(export_dir / "model.onnx"), load_external_data=False)
    _census = _collections.Counter(n.op_type for n in _model.graph.node)
    print(f"[census] nodes={len(_model.graph.node)} Pad={int(_census.get('Pad', 0))}", flush=True)
    for _op, _count in _census.most_common(10):
        print(f"[census]   {_count:6d}  {_op}", flush=True)
    print(f"[census] inputs={[i.name for i in _model.graph.input]}", flush=True)
    summary["census"] = {
        "nodes": len(_model.graph.node),
        "pad": int(_census.get("Pad", 0)),
        "top_ops": dict(_census.most_common(10)),
    }

    if run_harness:
        receipt_path = Path(PARITY_OUT_PATH) / "pi0" / "pi0-cpu-precheck-receipt.json"
        cmd = [
            sys.executable,
            str(pin_dir / "scripts" / "local_pi0_monolithic_parity.py"),
            "--model-source", "lerobot/pi0_base",
            "--model-revision", model_revision,
            "--onnx-dir", str(export_dir),
            "--provider", "CPUExecutionProvider",
            "--num-steps", str(num_steps),
            "--receipt", str(receipt_path),
        ]
        _progress("running CPU numerics pre-check (acceptance stays not-run off CUDA)")
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
            summary["metrics"] = receipt.get("metrics")
            print("[receipt-metrics] " + _json.dumps(receipt.get("metrics", {}), sort_keys=True), flush=True)
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
    run_harness: bool = True,
    num_steps: int = 10,
) -> None:
    result = run_export_cpu.remote(
        tether_revision=tether_revision,
        reuse_export=reuse_export,
        run_harness=run_harness,
        num_steps=num_steps,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
