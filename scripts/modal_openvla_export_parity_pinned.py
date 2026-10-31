"""Modal: OpenVLA export-parity qualification run (Tether Studio M4 #54).

Runs the receipt-grade harness `scripts/local_openvla_monolithic_parity.py`
on Modal Linux + CUDA (A100-40GB), following the proven pi0.5 refix pattern:

- `openvla/openvla-7b` pinned to OPENVLA_REVISION (exact 40-hex).
- Tether implementation pinned to TETHER_REVISION (exact 40-hex commit
  carrying `export_openvla_monolithic` + the fused-vision fixture fix).
- This job never exports. It uses the export that
  `modal_openvla_export_cpu_pinned.py` built on cheap CPU under the same
  PARITY_OUT_PATH, and only if that export's provenance record
  (`openvla/export.provenance.json`: Tether commit, model revision, export
  library versions) equals this job's exactly. Anything else, including a
  missing record, refuses before the harness starts, with the CPU builder
  command to run first. (The old GPU-side rebuild traced on CPU inside an
  A100 container; that is GPU time spent on CPU work.)
- modeling_prismatic.py pins its own deps (timm 0.9.x, transformers
  4.40.1); the image below honors them. The reference runs on CPU
  (fp32); only the ORT session uses CUDA.
- GPU burn is bounded by a subprocess timeout on the harness, the Modal
  function timeout and retries=0, plus a restart guard keyed by --run-id
  (see PARITY_TIMEOUT_SECONDS).

Usage (CPU builder first, same --tether-revision):
    modal run scripts/modal_openvla_export_cpu_pinned.py \
        --tether-revision <40-hex> --run-id <id>
    modal run scripts/modal_openvla_export_parity_pinned.py \
        --tether-revision <40-hex> --run-id <id>
"""

import os

import modal


# `openvla/openvla-7b` HEAD (2026-02-17 README update; weights/code stable
# since 2024-09-16). Pinned exactly; floating HEAD is refused.
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# Exact Tether implementation commit under test: the 1733d30
# re-qualification. Override with --tether-revision; it must equal the
# commit the CPU builder exported at.
TETHER_REVISION = "1733d3045c98dfc1ed961763486acc5579a0b147"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-openvla-export-parity-pinned")
_hf_cache_volume = modal.Volume.from_name(
    "openvla-hf-cache", create_if_missing=True
)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out/requal-1733d30-20261030"

# Must equal PROVENANCE_PACKAGES in modal_openvla_export_cpu_pinned.py: the
# packages the export path imports. Both images pin these exactly.
PROVENANCE_PACKAGES = ("torch", "transformers", "tokenizers", "timm", "numpy")


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


# Harness budget from measured runs: the two round-2 A100-40GB OpenVLA
# parity jobs on a reused export took 10.3 and 3.9 GPU-min wall
# (docs/parity/parity-round2-report.md:113); the longer includes the
# reference snapshot download into the HF cache volume. 1800 s is about
# 2.9x the 618 s run. The setup margin covers container start, the CUDA EP
# smoke and the clone. The harness runs in a subprocess with this timeout,
# so an overrun ends the input as an ordinary failure. The previous
# in-container SIGKILL watchdog surfaced as a runner crash, which Modal's
# client treats as an internal failure and re-schedules regardless of the
# retry policy (modal 1.4.0 `_functions.py`, MAX_INTERNAL_FAILURE_COUNT = 8).
PARITY_TIMEOUT_SECONDS = 1800
SETUP_MARGIN_SECONDS = 600
MODAL_TIMEOUT_SECONDS = PARITY_TIMEOUT_SECONDS + SETUP_MARGIN_SECONDS


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=MODAL_TIMEOUT_SECONDS,
    retries=0,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        "/parity_out": _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_parity(
    model_revision: str = OPENVLA_REVISION,
    tether_revision: str = TETHER_REVISION,
    parity_timeout_seconds: float = PARITY_TIMEOUT_SECONDS,
    run_id: str = "",
) -> dict:
    import json as _json
    import os as _os
    import subprocess
    import sys
    import time
    from importlib.metadata import version as _v
    from pathlib import Path

    _started_at = time.monotonic()

    def _progress(stage: str) -> None:
        print(f"[progress] {time.monotonic() - _started_at:7.1f}s  {stage}", flush=True)

    _progress("container up")
    summary: dict = {
        "model_source": "openvla/openvla-7b",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "run_id": run_id,
        "provider": "CUDAExecutionProvider",
        "vehicle": "openvla-gpu-parity",
    }
    family_dir = Path(PARITY_OUT_PATH) / "openvla"
    export_dir = family_dir / "export"
    provenance_path = family_dir / "export.provenance.json"

    # Restart guard: if Modal re-schedules this input after a container
    # crash, the second attempt finds its own marker and exits at once.
    _parity_out_volume.reload()
    if run_id:
        marker = family_dir / "attempts" / f"gpu-{run_id}"
        if marker.exists():
            _progress("restarted attempt detected; refusing to run again")
            summary["verdict"] = "aborted-restarted-attempt"
            return summary
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()))
        _parity_out_volume.commit()

    # Provenance gate before any further GPU time: use the CPU-built export
    # only when it was built from exactly this Tether commit, model revision
    # and export library versions.
    provenance = {
        "tether_revision": tether_revision,
        "model_revision": model_revision,
        **{p: _v(p) for p in PROVENANCE_PACKAGES},
    }
    summary["export_provenance"] = provenance
    retained = (
        _json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    )
    if retained != provenance or not (export_dir / "model.onnx").is_file():
        _progress(f"refusing: retained provenance {retained} != {provenance}")
        summary["verdict"] = "refused-export-provenance-mismatch"
        summary["retained_provenance"] = retained
        summary["action"] = (
            "run the CPU builder first: modal run "
            "scripts/modal_openvla_export_cpu_pinned.py "
            f"--tether-revision {tether_revision} --run-id <id>; it writes "
            f"{export_dir} and {provenance_path}"
        )
        return summary

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

    # 2. The CPU-built export, provenance-matched above.
    export_bytes = sum(f.stat().st_size for f in export_dir.glob("*") if f.is_file())
    summary["export"] = {
        "status": "reused-provenance-matched",
        "onnx_path": str(export_dir / "model.onnx"),
        "size_mb": export_bytes / 1e6,
    }
    _progress(f"using CPU-built export ({export_bytes / 1e6:.1f}MB, provenance matches)")

    # 3. Receipt-grade parity with CUDA, CPU fallback disabled.
    receipt_path = family_dir / "openvla-export-parity-receipt.json"
    receipt_path.unlink(missing_ok=True)
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
    try:
        proc = subprocess.run(
            cmd, cwd=str(pin_dir), capture_output=True, text=True, env=harness_env,
            timeout=parity_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        _parity_out_volume.commit()
        raise RuntimeError(f"parity harness exceeded {parity_timeout_seconds:.0f}s; stopped") from exc
    print(proc.stdout[-6000:], flush=True)
    print(proc.stderr[-3000:], flush=True)
    summary["harness_returncode"] = proc.returncode

    if receipt_path.is_file():
        receipt = _json.loads(receipt_path.read_text())
        summary["verdict"] = receipt.get("verdict")
        summary["external_acceptance"] = receipt.get("external_acceptance")
        summary["metrics"] = receipt.get("metrics")
        print("[receipt] " + _json.dumps(receipt, sort_keys=True), flush=True)
    else:
        summary["verdict"] = "no-receipt"

    if run_id:
        run_record = family_dir / "runs" / f"gpu-{run_id}.json"
        run_record.parent.mkdir(parents=True, exist_ok=True)
        run_record.write_text(_json.dumps(summary, sort_keys=True, default=str) + "\n")
    _parity_out_volume.commit()
    _progress("done")
    return summary


@app.local_entrypoint()
def main(
    tether_revision: str = TETHER_REVISION,
    parity_timeout_seconds: float = PARITY_TIMEOUT_SECONDS,
    run_id: str = "",
) -> None:
    import uuid

    result = run_export_parity.remote(
        tether_revision=tether_revision,
        parity_timeout_seconds=parity_timeout_seconds,
        run_id=run_id or uuid.uuid4().hex,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
