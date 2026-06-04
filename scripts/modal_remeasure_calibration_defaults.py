"""Modal: re-measure auto-calibration baselines + write configs/calibration_defaults.json.

Per ADR/plan `2026-04-25-auto-calibration-architecture` Day 9:
runs the calibration measurement matrix on Modal A10G + A100-80GB +
writes the bundled JSON. Ships in `configs/calibration_defaults.json`
so the customer's first `--auto-calibrate` run gets instant cache-hit
on common hardware; uncommon hardware falls through to live
measurement.

Quarterly refresh (per ADR risk #6 cost-table audit): re-run this
script + check the diff against the previous bundle. Drift > 10%
flags a calibration regression to investigate.

Usage:
    modal run scripts/modal_remeasure_calibration_defaults.py
    # then commit the updated configs/calibration_defaults.json

Cost: ~$2-3 (A10G + A100 each ~10 min including image build amortized).

Output schema (configs/calibration_defaults.json):
{
  "schema_version": 1,
  "generated_at": "2026-04-25T...",
  "generator_app_id": "ap-...",
  "tiers": [
    {
      "fingerprint": {gpu_name, driver_version_major, ...},
      "measurements": [
        {
          "embodiment": "synthetic-numpy-stub",
          "model_hash": "v1",
          "chunk_size": 50, "nfe": 10,
          "latency_compensation_ms": 0.27,
          "provider": "CPUExecutionProvider",
          "variant": "synthetic-stub",
          "measurement_quality": {...},
          "measurement_context": {...},
          "timestamp": "2026-04-25T..."
        }
      ]
    }
  ]
}

Phase 1 limit: only the synthetic-numpy-stub measurement is bundled.
Real ONNX-predict measurements (per-embodiment, per-model) require
loading a real model -- a 100-task matrix run would be ~$50+. Filed
as Phase 2 expansion. The bundled stub-stub still gives customers a
fingerprint baseline (gpu_name + driver + cuda) so cache-validity
comparison works on first run.
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("tether-calibration-defaults-bundle")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUILD_BUST = _build_bust()


# Same lightweight image as modal_test_calibration_matrix.py -- no
# LIBERO, no lerobot. Calibration substrate is pure-Python after CUDA.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy", "torch",
        "onnx>=1.16", "onnxruntime-gpu>=1.20",
        "psutil", "pyyaml",
    )
    .run_commands(
        f'echo "build_bust={_BUILD_BUST}"',
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


def _measure_inner(tier: str) -> dict:
    """Run the calibration measurement on the current host + return a
    dict of (fingerprint, measurements) ready for inclusion in the
    bundle. Mirrors the structure used by modal_test_calibration_matrix.

    Phase 1 ships the synthetic-numpy-stub measurement. Phase 2 adds
    real ONNX-predict measurements per-embodiment.
    """
    from datetime import datetime, timezone
    import numpy as np

    from tether.runtime.calibration import (
        CalibrationEntry, HardwareFingerprint, MeasurementContext,
        measure_latency_profile,
    )

    print(f"[bundle] tier={tier} probing fingerprint...")
    fp = HardwareFingerprint.current()
    print(f"[bundle]   gpu: {fp.gpu_name!r}, driver: {fp.driver_version_major}.{fp.driver_version_minor}, cuda: {fp.cuda_version_major}.{fp.cuda_version_minor}")

    print(f"[bundle] tier={tier} measuring synthetic-numpy-stub latency...")

    def _stub_predict():
        a = np.random.rand(128, 128).astype(np.float32)
        b = np.random.rand(128, 128).astype(np.float32)
        c = a @ b
        return c.sum()

    quality = measure_latency_profile(
        _stub_predict, n_iters=200, warmup_iters=20,
    )
    print(f"[bundle]   median_ms: {quality.median_ms:.4f}, p99_ms: {quality.p99_ms:.4f}, quality_score: {quality.quality_score:.4f}")

    ctx = MeasurementContext.current()
    entry = CalibrationEntry(
        chunk_size=50,
        nfe=10,
        latency_compensation_ms=float(quality.median_ms),
        provider="CPUExecutionProvider",
        variant="synthetic-numpy-stub",
        measurement_quality=quality,
        measurement_context=ctx,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return {
        "fingerprint": fp.to_dict(),
        "measurements": [
            {
                "embodiment": "synthetic-numpy-stub",
                "model_hash": "v1",
                "entry": entry.to_dict(),
            }
        ],
    }


@app.function(
    image=image, gpu="A10G", timeout=600, secrets=[_hf_secret()],
)
def measure_a10g() -> dict:
    return _measure_inner("A10G")


@app.function(
    image=image, gpu="A100-80GB", timeout=600, secrets=[_hf_secret()],
)
def measure_a100() -> dict:
    return _measure_inner("A100-80GB")


@app.local_entrypoint()
def main(skip_a100: bool = False, output_path: str = ""):
    """Local entry: invoke both tiers + emit the bundle JSON.

    --skip-a100 runs only A10G (~$0.50). Default runs both (~$3).
    --output-path defaults to configs/calibration_defaults.json under
        the repo root.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    print("=" * 70)
    print("Auto-calibration defaults bundle -- A10G + A100-80GB")
    print("=" * 70)

    tiers: list[dict] = []
    print("\n[bundle] tier=A10G ...")
    tiers.append({"tier": "A10G", **measure_a10g.remote()})
    if not skip_a100:
        print("\n[bundle] tier=A100-80GB ...")
        tiers.append({"tier": "A100-80GB", **measure_a100.remote()})

    bundle = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        ),
        "generator": "modal_remeasure_calibration_defaults.py",
        "phase_1_limit": (
            "Only the synthetic-numpy-stub measurement is bundled in "
            "Phase 1. Real ONNX-predict measurements per-embodiment "
            "require loading a real model -- ~$50+ for the 100-task "
            "matrix. Filed as Phase 2 expansion. The bundled stub "
            "still gives customers a fingerprint baseline so cache-"
            "validity comparison works on first run."
        ),
        "tiers": tiers,
    }

    # Resolve output path: argument OR repo-relative default.
    if output_path:
        out = Path(output_path)
    else:
        # local entrypoint runs in the user's CWD; assume that's the repo root
        out = Path("configs") / "calibration_defaults.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, sort_keys=False))
    print(f"\n=== RESULT ===")
    print(f"  status:   ok")
    print(f"  tiers:    {[t['tier'] for t in tiers]}")
    print(f"  bundle:   {out}")
    print(f"  size:     {out.stat().st_size} bytes")
    print(f"\n[bundle] commit + push: git add {out} && git commit -m '...'")
