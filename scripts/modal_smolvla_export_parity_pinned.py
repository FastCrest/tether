"""Modal: SmolVLA export-parity qualification run (Linux CUDA).

Runs ``scripts/local_smolvla_monolithic_parity.py`` on Modal Linux + CUDA on
the pattern of ``modal_pi0_export_parity_pinned.py``:

- ``lerobot/smolvla_base`` pinned to SMOLVLA_BASE_REVISION, the revision
  Studio qualifies (``tether_private/qualification/families.py``
  SMOLVLA_BASE_REVISION).
- The Tether implementation is the exact local commit, shipped as a
  ``git bundle`` and checked out in the container, so the receipt's
  ``tether_commit`` is a real commit even before it is pushed. The local
  checkout must be clean.
- Export runs in a subprocess from the pinned snapshot directory; it counts
  the shared repair pass's SplitToSequence rewrites (added after Studio's pin
  432923b) so the binding proof can show whether that change touched this graph.
- A10G: SmolVLA fp32 is about 1.8 GB of weights, far inside 24 GB.
- GPU burn is bounded by per-phase subprocess timeouts, the Modal function
  timeout, retries=0 and a restart guard.

Usage:
    modal run scripts/modal_smolvla_export_parity_pinned.py
"""

import os
import subprocess
from pathlib import Path

import modal

SMOLVLA_BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
CAMPAIGN = "smolvla-c83c3163-lang48-20261028"

app = modal.App("tether-smolvla-export-parity-pinned")
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
_parity_out_volume = modal.Volume.from_name("parity-gpu-receipts", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = f"/parity_out/{CAMPAIGN}"


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})  # smolvla_base and SmolVLM2 are public


# Same export stack as the pi0 harness: LeRobot 0.6.1 + transformers 5.5.4 is
# the qualified pair `_require_monolithic_deps` insists on.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.8,<2.12",
        "lerobot==0.6.1",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy==2.2.6",
        "accelerate",
        "draccus==0.11.6",
    )
    .pip_install("transformers==5.5.4")
    .env({"HF_HOME": HF_CACHE_PATH, "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers"})
)

# onnxruntime-gpu==1.23.2: the version the committed receipts record, so a rerun reproduces them.
# cudnn + cublas alone is not enough: ORT fails to create the CUDA EP without
# the CUDA runtime (verified 2026-09-18 on the pi0 harness).
gpu_image = (
    image.pip_install(
        "onnxruntime-gpu==1.23.2",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cufft-cu12>=11.0,<12.0",
        "nvidia-curand-cu12>=10.0,<11.0",
        extra_options="--no-deps",
    )
    .pip_install("onnxruntime-gpu==1.23.2")
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

# Budgets: Studio measured a 355 s CPU SmolVLA export
# (docs/research/verification/pre-retest-20261023); 10 reference cases on CPU
# plus ORT-CUDA are minutes. About 4x margin.
EXPORT_TIMEOUT_SECONDS = 1500
PARITY_TIMEOUT_SECONDS = 900
SETUP_MARGIN_SECONDS = 600
MODAL_TIMEOUT_SECONDS = EXPORT_TIMEOUT_SECONDS + PARITY_TIMEOUT_SECONDS + SETUP_MARGIN_SECONDS

_EXPORT_CODE = (
    "import json, sys; import tether.exporters.monolithic as m; calls = []; "
    "orig = m._split_to_sequence_to_split\n"
    "def counted(graph, shape_graph):\n"
    "    n = orig(graph, shape_graph); calls.append(n); return n\n"
    "m._split_to_sequence_to_split = counted\n"
    "r = m.export_smolvla_monolithic(sys.argv[1], sys.argv[2], num_steps=int(sys.argv[3]))\n"
    "r['split_to_sequence_rewrites'] = calls\n"
    "print('EXPORT_RESULT ' + json.dumps(r), flush=True)\n"
)


@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=MODAL_TIMEOUT_SECONDS,
    retries=0,
    volumes={HF_CACHE_PATH: _hf_cache_volume, "/parity_out": _parity_out_volume},
    secrets=[_hf_secret()],
)
def run_export_parity(
    source_bundle: bytes,
    tether_revision: str,
    model_revision: str = SMOLVLA_BASE_REVISION,
    num_steps: int = 10,
    run_id: str = "",
    tf32_override: str = "",
) -> dict:
    import json as _json
    import subprocess
    import sys
    import time
    from importlib.metadata import version as _v

    started = time.monotonic()

    def _progress(stage: str) -> None:
        print(f"[progress] {time.monotonic() - started:7.1f}s  {stage}", flush=True)

    summary: dict = {
        "model_source": "lerobot/smolvla_base",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
        "provider": "CUDAExecutionProvider",
        "num_steps": num_steps,
        "campaign": CAMPAIGN,
    }
    # tf32_override="0" is a diagnostic: it disables TF32 for the ORT-CUDA session
    # only, to attribute CUDA-vs-CPU error to TF32 math. It writes to its own
    # directory so the qualifying export and receipt are never overwritten.
    suffix = f"-tf32-override-{tf32_override}" if tf32_override else ""
    out_root = Path(PARITY_OUT_PATH) / f"smolvla{suffix}"
    if run_id:
        marker = out_root / "attempts" / run_id
        _parity_out_volume.reload()
        if marker.exists():
            _progress("restarted attempt detected; refusing to run again")
            summary["verdict"] = "aborted-restarted-attempt"
            return summary
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()))
        _parity_out_volume.commit()

    # Fail fast if the CUDA EP cannot be created with CPU fallback disabled.
    import numpy as _np
    import onnx as _onnx
    import onnxruntime as _ort

    node = _onnx.helper.make_node("Add", ["x", "y"], ["z"])
    graph = _onnx.helper.make_graph(
        [node], "smoke",
        [_onnx.helper.make_tensor_value_info(n, _onnx.TensorProto.FLOAT, [1]) for n in ("x", "y")],
        [_onnx.helper.make_tensor_value_info("z", _onnx.TensorProto.FLOAT, [1])],
    )
    smoke = _onnx.helper.make_model(graph, opset_imports=[_onnx.helper.make_opsetid("", 19)])
    smoke.ir_version = 10
    so = _ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = _ort.InferenceSession(smoke.SerializeToString(), sess_options=so, providers=["CUDAExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
    one = _np.ones((1,), dtype=_np.float32)
    assert float(sess.run(None, {"x": one, "y": one})[0][0]) == 2.0
    summary["cuda_ep_smoke"] = f"ok (ort {_ort.__version__})"
    _progress(summary["cuda_ep_smoke"])

    # 1. Pin the implementation from the shipped bundle.
    bundle = Path("/root/tether.bundle")
    bundle.write_bytes(source_bundle)
    pin_dir = Path("/root/tether-pin")
    subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(bundle), str(pin_dir)], check=True)
    subprocess.run(["git", "-C", str(pin_dir), "checkout", "--quiet", tether_revision], check=True)
    got = subprocess.check_output(["git", "-C", str(pin_dir), "rev-parse", "HEAD"], text=True).strip()
    assert got == tether_revision, f"tether pin mismatch: {got} != {tether_revision}"
    summary["tether_commit"] = got
    _progress(f"tether pinned at {got[:12]}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pin_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # 2. Pin the model and export from the local snapshot.
    from huggingface_hub import snapshot_download

    snap = snapshot_download("lerobot/smolvla_base", revision=model_revision)
    _hf_cache_volume.commit()
    export_dir = out_root / "export"
    subprocess.run(["rm", "-rf", str(export_dir)], check=True)  # identity must hash only this export
    provenance = {
        "tether_revision": tether_revision,
        "model_revision": model_revision,
        "num_steps": num_steps,
        **{p: _v(p) for p in ("lerobot", "transformers", "torch", "onnx-diagnostic", "onnxscript", "onnxruntime-gpu")},
    }
    summary["export_provenance"] = provenance
    _progress("export_smolvla_monolithic from pinned snapshot (subprocess)")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _EXPORT_CODE, snap, str(export_dir), str(num_steps)],
            check=False, cwd=str(pin_dir), env=env, capture_output=True, text=True,
            timeout=EXPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        _parity_out_volume.commit()
        raise RuntimeError(f"export exceeded {EXPORT_TIMEOUT_SECONDS}s; stopped") from exc
    print(proc.stdout[-4000:], flush=True)
    print(proc.stderr[-4000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"export failed with return code {proc.returncode}")
    line = next(x for x in proc.stdout.splitlines() if x.startswith("EXPORT_RESULT "))
    summary["export"] = _json.loads(line.split(" ", 1)[1])
    # Provenance lives beside, not inside, the export so it cannot change artifact identity.
    (out_root / "export-provenance.json").write_text(
        _json.dumps({**provenance, "export": summary["export"]}, sort_keys=True, indent=2) + "\n"
    )
    _parity_out_volume.commit()
    _progress(f"export ok: {summary['export'].get('size_mb', 0):.1f}MB")

    # 3. Receipt-grade parity with CUDA, CPU fallback disabled.
    receipt_path = out_root / "smolvla-export-parity-receipt.json"
    if tf32_override:
        env["NVIDIA_TF32_OVERRIDE"] = tf32_override
        summary["tf32_override"] = tf32_override
    cmd = [
        sys.executable, str(pin_dir / "scripts" / "local_smolvla_monolithic_parity.py"),
        "--model-source", "lerobot/smolvla_base",
        "--model-revision", model_revision,
        "--onnx-dir", str(export_dir),
        "--provider", "CUDAExecutionProvider",
        "--num-steps", str(num_steps),
        "--receipt", str(receipt_path),
    ]
    _progress("running receipt-grade parity (CUDA, no CPU fallback)")
    try:
        proc = subprocess.run(
            cmd, check=False, cwd=str(pin_dir), capture_output=True, text=True, env=env,
            timeout=PARITY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        _parity_out_volume.commit()
        raise RuntimeError(f"parity harness exceeded {PARITY_TIMEOUT_SECONDS}s; stopped") from exc
    print(proc.stdout[-8000:], flush=True)
    print(proc.stderr[-3000:], flush=True)
    (out_root / "parity-harness.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    summary["harness_returncode"] = proc.returncode
    if receipt_path.is_file():
        summary["receipt_bytes"] = receipt_path.read_text()
        receipt = _json.loads(summary["receipt_bytes"])
        summary["verdict"] = receipt["verdict"]
        summary["external_acceptance"] = receipt["external_acceptance"]
    else:
        summary["verdict"] = "no-receipt"
    _parity_out_volume.commit()
    _progress("done")
    return summary


def _local_bundle() -> tuple[bytes, str]:
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip()
    if dirty:
        raise SystemExit(f"Refusing to run: tracked changes are uncommitted:\n{dirty}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    bundle = root / ".git-smolvla-parity.bundle"
    try:
        subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=root, check=True,
                       capture_output=True)
        return bundle.read_bytes(), commit
    finally:
        bundle.unlink(missing_ok=True)


@app.local_entrypoint()
def main(num_steps: int = 10, run_id: str = "", receipt_out: str = "", tf32_override: str = "") -> None:
    import json
    import uuid

    data, commit = _local_bundle()
    print(f"Tether commit {commit} ({len(data) / 1e6:.1f} MB bundle)")
    result = run_export_parity.remote(
        source_bundle=data, tether_revision=commit, num_steps=num_steps, run_id=run_id or uuid.uuid4().hex,
        tf32_override=tf32_override,
    )
    receipt_bytes = result.pop("receipt_bytes", None)
    if receipt_bytes and receipt_out:
        Path(receipt_out).write_text(receipt_bytes)
        print(f"receipt written byte-for-byte to {receipt_out}")
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, sort_keys=True))
