"""Modal: GR00T export-parity qualification run (Tether Studio M4 #53).

Runs the receipt-grade harness `scripts/local_groot_monolithic_parity.py` on
Modal Linux + CUDA, following the pi0 vehicle pattern (exact 40-hex model
commit pin, no floating HEAD, bounded GPU time, provenance-checked reuse):

- `nvidia/GR00T-N1.6-3B` is pinned to GROOT_REVISION (the repo's single
  squashed main commit; floating HEAD is refused regardless).
- The export is built from a local snapshot of that exact revision, so the
  hashed artifact and the reference revision in the receipt agree by
  construction (the snapshot dir is passed as `model_id`, no repo edits).
- The Tether implementation under test is pinned to TETHER_REVISION (an
  exact 40-hex commit): the container clones the repo and checks out that
  commit, and the harness records `git rev-parse HEAD` from that checkout.
- GPU burn is bounded by per-phase subprocess timeouts, the Modal function
  timeout and retries=0, plus a restart guard keyed by --run-id (see
  EXPORT_TIMEOUT_SECONDS). No in-container SIGKILL watchdog.
- --reuse-export reuses the retained export only when its provenance
  record (`groot/export.provenance.json`: Tether commit, model revision,
  embodiment id, library versions) equals this job's exactly.

GR00T's gate (docs/groot-export-parity.md) is cosine >= 0.9999 and max
absolute error < 1e-3 on the per-step velocity graph; the harness enforces
it, this script only reports it. Under default ORT-CUDA TF32 math the gate
fails on full_max_abs (1.26e-03, docs/parity/parity-round2-report.md:43-79);
it passes only with NVIDIA_TF32_OVERRIDE=0, which is the one condition
Studio accepts (receipt `groot/groot-export-parity-receipt-strict-tf32.json`
with an `environment` block). --strict-fp32 runs the unmodified harness
under that env and writes that receipt; without it the default-env receipt
is written, as before.

Usage:
    modal run scripts/modal_groot_export_parity_pinned.py \
        --tether-revision <40-hex> --run-id <id> [--strict-fp32] [--reuse-export]
"""

import os

import modal


# `nvidia/GR00T-N1.6-3B` main is a single squashed commit (2025-12-15).
# Pinned exactly; floating HEAD is refused.
GROOT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"

# Exact Tether implementation commit under test: the 1733d30
# re-qualification. Override with --tether-revision.
TETHER_REVISION = "1733d3045c98dfc1ed961763486acc5579a0b147"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-groot-export-parity-pinned")
_hf_cache_volume = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out/requal-1733d30-20261030"


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
        # torch>=2.8: onnx-diagnostic 0.9.x torch_export_patches needs it (see pi0 vehicle).
        "torch>=2.8,<2.12",
        "safetensors>=0.4.0",
        "huggingface_hub",
        # transformers==5.5.4 EXACTLY: the monolithic exporter refuses
        # anything else.
        "transformers==5.5.4",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        # monolithic dep group (export_gr00t_monolithic gates on these):
        "lerobot==0.6.1",
        "num2words",
        "diffusers==0.39.0",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "accelerate",
        "draccus==0.11.6",
        "numpy==2.2.6",
        "Pillow",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
    )
    # onnxruntime-gpu for the CUDAExecutionProvider parity session (base
    # image carries CPU ORT only). Must include the CUDA runtime + cufft:
    # cudnn + cublas alone leaves ORT unable to create the CUDA EP
    # ("Require cuDNN 9.* and CUDA 12.*"), verified 2026-09-18.
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


# Phase budgets from measured runs on A100-40GB: GR00T export + parity
# together took 201 s (docs/parity/parity-agent-report.md:102) and
# "rebuild+parity 3.4" GPU-min, about 204 s
# (docs/parity/parity-round2-report.md:114). The split between export and
# harness is not recorded, so each phase gets a bound that alone exceeds
# the measured total: export 900 s (about 4.4x) and harness 600 s (about
# 2.9x). Those runs were at transformers 5.3.0; the 5.5.4 / torch 2.11 stack
# here is not measured for GR00T (pi0's export moved 1716.8 -> 1776.5 s
# across a similar change). The 900 s margin covers container start, the
# clone and the ~7 GB snapshot download, which the first agent run put at
# most of its ~10 min wall. Each phase runs in a subprocess with its
# timeout, so an overrun ends the input as an ordinary failure; the old
# in-container SIGKILL watchdog surfaced as a runner crash, which Modal's
# client re-schedules as an internal failure regardless of the retry policy
# (modal 1.4.0 `_functions.py`, MAX_INTERNAL_FAILURE_COUNT = 8).
EXPORT_TIMEOUT_SECONDS = 900
PARITY_TIMEOUT_SECONDS = 600
SETUP_MARGIN_SECONDS = 900
MODAL_TIMEOUT_SECONDS = EXPORT_TIMEOUT_SECONDS + PARITY_TIMEOUT_SECONDS + SETUP_MARGIN_SECONDS


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
    model_revision: str = GROOT_REVISION,
    tether_revision: str = TETHER_REVISION,
    embodiment_id: int = 0,
    reuse_export: bool = False,
    strict_fp32: bool = False,
    export_timeout_seconds: float = EXPORT_TIMEOUT_SECONDS,
    parity_timeout_seconds: float = PARITY_TIMEOUT_SECONDS,
    run_id: str = "",
) -> dict:
    import hashlib
    import json as _json
    import os as _os
    import shutil
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
        "model_source": "nvidia/GR00T-N1.6-3B",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "run_id": run_id,
        "provider": "CUDAExecutionProvider",
        "embodiment_id": embodiment_id,
        "strict_fp32": strict_fp32,
    }
    family_dir = Path(PARITY_OUT_PATH) / "groot"

    # Restart guard: if Modal re-schedules this input after a container
    # crash, the second attempt finds its own marker and exits at once
    # instead of starting another GPU export.
    if run_id:
        marker = family_dir / "attempts" / run_id
        _parity_out_volume.reload()
        if marker.exists():
            _progress("restarted attempt detected; refusing to run again")
            summary["verdict"] = "aborted-restarted-attempt"
            return summary
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()))
        _parity_out_volume.commit()

    # Fail fast if the CUDA EP cannot be created (a broken CUDA/cuDNN
    # image otherwise surfaces many minutes later as a no-fallback gate
    # failure after the reference has already run).
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
    _smoke = _onnx.helper.make_model(
        _graph, opset_imports=[_onnx.helper.make_opsetid("", 19)]
    )
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

    export_dir = family_dir / "export"
    # Beside the export, not inside it, so it never enters the artifact
    # identity the harness hashes over --onnx-dir.
    provenance_path = family_dir / "export.provenance.json"
    harness_env = dict(_os.environ)
    harness_env["PYTHONPATH"] = str(pin_dir / "src") + _os.pathsep + harness_env.get(
        "PYTHONPATH", ""
    )
    provenance = {
        "tether_revision": tether_revision,
        "model_revision": model_revision,
        "embodiment_id": embodiment_id,
        **{p: _v(p) for p in ("lerobot", "transformers", "torch", "onnx-diagnostic", "onnxscript")},
    }
    summary["export_provenance"] = provenance
    retained = (
        _json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    )
    if reuse_export and retained == provenance:
        export_result = {"status": "reused", "onnx_path": str(export_dir / "model.onnx")}
        _progress("reusing retained export (provenance matches)")
    else:
        if reuse_export:
            _progress(f"not reusing: retained provenance {retained} != {provenance}")
        # Drop the old provenance first and clear the directory, so a
        # failed or partial export can never be mistaken for a matching one.
        provenance_path.unlink(missing_ok=True)
        shutil.rmtree(export_dir, ignore_errors=True)
        _parity_out_volume.commit()
        # The export runs in its own process, so its CUDA allocations are
        # freed before the harness loads the reference (the in-process
        # export needed an explicit cache release: on pi0.5 the parent held
        # 22.03/22.06 GiB and the child got 16 MiB).
        _progress("export_gr00t_monolithic from pinned snapshot (subprocess)")
        code = (
            "import json, sys; from tether.exporters.monolithic import export_gr00t_monolithic; "
            "r = export_gr00t_monolithic(sys.argv[1], sys.argv[2], embodiment_id=int(sys.argv[3])); "
            "print('EXPORT_RESULT ' + json.dumps(r, default=str), flush=True)"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, snap, str(export_dir), str(embodiment_id)],
                cwd=str(pin_dir), env=harness_env, capture_output=True, text=True,
                timeout=export_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _parity_out_volume.commit()
            raise RuntimeError(
                f"export exceeded {export_timeout_seconds:.0f}s; stopped"
            ) from exc
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"export failed with return code {proc.returncode}")
        line = next(x for x in proc.stdout.splitlines() if x.startswith("EXPORT_RESULT "))
        export_result = _json.loads(line.split(" ", 1)[1])
        provenance_path.write_text(_json.dumps(provenance, sort_keys=True) + "\n")
        _parity_out_volume.commit()
        _progress(f"export ok: {export_result.get('size_mb', 0):.1f}MB")
    summary["export"] = export_result

    # 3. Receipt-grade parity with CUDA, CPU fallback disabled.
    if strict_fp32:
        # Same layout as the 2026-09-18 strict receipt Studio ingests: the
        # harness writes the raw receipt, and the qualifying file is that
        # receipt plus an `environment` block naming the condition and the
        # raw receipt's sha256.
        harness_env["NVIDIA_TF32_OVERRIDE"] = "0"
        receipt_path = family_dir / "groot-export-parity-receipt-strict-tf32.raw.json"
        qualifying_path = family_dir / "groot-export-parity-receipt-strict-tf32.json"
    else:
        receipt_path = family_dir / "groot-export-parity-receipt.json"
        qualifying_path = receipt_path
    for _stale in {receipt_path, qualifying_path}:
        _stale.unlink(missing_ok=True)
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
    _progress(
        "running receipt-grade parity (CUDA, no CPU fallback"
        + (", NVIDIA_TF32_OVERRIDE=0)" if strict_fp32 else ")")
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
        raw = receipt_path.read_bytes()
        receipt = _json.loads(raw)
        if strict_fp32:
            receipt["environment"] = {
                "NVIDIA_TF32_OVERRIDE": "0",
                "math_mode": "strict-fp32 (cuBLAS/cuDNN IEEE fp32, TF32 disabled)",
                "raw_harness_receipt_sha256": hashlib.sha256(raw).hexdigest(),
            }
            qualifying_path.write_text(_json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        summary["receipt_path"] = str(qualifying_path)
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

    if run_id:
        run_record = family_dir / "runs" / f"{run_id}.json"
        run_record.parent.mkdir(parents=True, exist_ok=True)
        run_record.write_text(_json.dumps(summary, sort_keys=True, default=str) + "\n")
    _parity_out_volume.commit()
    _progress("done")
    return summary


@app.local_entrypoint()
def main(
    tether_revision: str = TETHER_REVISION,
    reuse_export: bool = False,
    strict_fp32: bool = False,
    embodiment_id: int = 0,
    export_timeout_seconds: float = EXPORT_TIMEOUT_SECONDS,
    parity_timeout_seconds: float = PARITY_TIMEOUT_SECONDS,
    run_id: str = "",
) -> None:
    import uuid

    result = run_export_parity.remote(
        tether_revision=tether_revision,
        reuse_export=reuse_export,
        strict_fp32=strict_fp32,
        embodiment_id=embodiment_id,
        export_timeout_seconds=export_timeout_seconds,
        parity_timeout_seconds=parity_timeout_seconds,
        run_id=run_id or uuid.uuid4().hex,
    )
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
