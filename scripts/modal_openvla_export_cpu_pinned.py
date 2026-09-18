"""Modal: OpenVLA CPU export + numerics pre-check (Tether Studio M4 #54).

Cheap CPU job that de-risks the A100 parity run:

1. clones the exact Tether implementation commit,
2. snapshots `openvla/openvla-7b` at the exact pinned revision (full weights),
3. runs `tether.exporters.openvla.export_openvla_monolithic` to the shared
   output volume (`openvla/export/`), so the later A100 job reuses it,
4. prints an ONNX op census (Pad count must be 0; Concat inventory),
5. runs `scripts/local_openvla_monolithic_parity.py` with the CPU provider
   as a numerics pre-check (verdict signal only — external acceptance stays
   `not-run` off CUDA by design; gates unmodified).

No GPU is used. A self-kill watchdog bounds wall time.

Usage:
    modal run scripts/modal_openvla_export_cpu_pinned.py
"""

import os

import modal


# `openvla/openvla-7b` HEAD (2026-02-17 README update; weights/code stable
# since 2024-09-16). Pinned exactly; floating HEAD is refused.
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# Exact Tether implementation commit under test (monolithic OpenVLA
# exporter + fused-vision pixel-channels fixture fix).
TETHER_REVISION = "cb643a0645f06700879488e50e8a340b24dfa154"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-openvla-export-cpu-pinned")
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


# modeling_prismatic.py (pinned snapshot) gates on exact contemporary deps:
#   timm in {0.9.10..0.9.16} (hard NotImplementedError otherwise),
#   transformers==4.40.1 / tokenizers==0.19.1 (warning otherwise).
# torch 2.4.x is the contemporary CUDA-capable build (export itself runs on
# CPU; the later parity job needs the same pinned source only).
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
    )
    .pip_install(
        "onnx>=1.16",
        "onnxruntime>=1.20,<1.24",
    )
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
    model_revision: str = OPENVLA_REVISION,
    tether_revision: str = TETHER_REVISION,
    reuse_export: bool = True,
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
        "model_source": "openvla/openvla-7b",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
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

    import torch as _torch

    _progress(f"torch {_torch.__version__} (cpu={not _torch.cuda.is_available()})")

    from huggingface_hub import snapshot_download

    _progress("snapshot_download openvla/openvla-7b@" + model_revision[:12])
    snap = snapshot_download("openvla/openvla-7b", revision=model_revision)
    summary["snapshot_dir"] = snap
    _progress("snapshot complete")

    from tether.exporters.openvla import export_openvla_monolithic

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
        _progress("export_openvla_monolithic from pinned snapshot (CPU)")
        export_result = export_openvla_monolithic(snap, export_dir)
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

    # ONNX census: Pad must be 0; inventory every Concat by input dtype.
    import collections as _collections

    import onnx as _onnx
    from onnx import TensorProto as _TP

    _model = _onnx.load(str(export_dir / "model.onnx"), load_external_data=False)
    _info = {}
    for _vi in list(_model.graph.input) + list(_model.graph.output) + list(
        _model.graph.value_info
    ):
        _t = _vi.type.tensor_type
        _info[_vi.name] = (
            _TP.DataType.Name(_t.elem_type),
            [_d.dim_value for _d in _t.shape.dim],
        )
    for _init in _model.graph.initializer:
        _info[_init.name] = (_TP.DataType.Name(_init.data_type), list(_init.dims))
    _census = _collections.Counter(n.op_type for n in _model.graph.node)
    _pads = int(_census.get("Pad", 0))
    _concats = [
        (n.name, sorted({_info.get(i, ("?", []))[0] for i in n.input}))
        for n in _model.graph.node
        if n.op_type == "Concat"
    ]
    print(f"[census] nodes={len(_model.graph.node)} Pad={_pads}", flush=True)
    for _op, _count in _census.most_common(12):
        print(f"[census]   {_count:6d}  {_op}", flush=True)
    print(f"[census] inputs={[i.name for i in _model.graph.input]}", flush=True)
    print(f"[census] outputs={[(o.name, _info.get(o.name)) for o in _model.graph.output]}", flush=True)
    print(f"[census] concat_count={len(_concats)}", flush=True)
    for _name, _dtypes in _concats[:20]:
        print(f"[census]   {_name} dtypes={_dtypes}", flush=True)
    summary["census"] = {
        "nodes": len(_model.graph.node),
        "pad": _pads,
        "concat_count": len(_concats),
        "top_ops": dict(_census.most_common(12)),
    }

    if run_harness:
        receipt_path = Path(PARITY_OUT_PATH) / "openvla" / "openvla-cpu-precheck-receipt.json"
        cmd = [
            sys.executable,
            str(pin_dir / "scripts" / "local_openvla_monolithic_parity.py"),
            "--model-source", "openvla/openvla-7b",
            "--model-revision", model_revision,
            "--onnx-dir", str(export_dir),
            "--provider", "CPUExecutionProvider",
            "--action-dim", "7",
            "--trust-remote-code",
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
            summary["action_token_agreement"] = (receipt.get("metrics") or {}).get(
                "action_token_agreement"
            )
            summary["full_cosine"] = (receipt.get("metrics") or {}).get("full_cosine")
            summary["full_max_abs"] = (receipt.get("metrics") or {}).get("full_max_abs")
            print("[receipt] " + _json.dumps(receipt.get("metrics", {}), sort_keys=True), flush=True)
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
    run_harness: bool = True,
) -> None:
    result = run_export_cpu.remote(
        tether_revision=tether_revision,
        reuse_export=reuse_export,
        run_harness=run_harness,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
