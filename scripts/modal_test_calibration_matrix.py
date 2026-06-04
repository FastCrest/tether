"""Modal cross-hardware calibration matrix smoke (auto-calibration Day 7).

Per ADR/plan `2026-04-25-auto-calibration-architecture` Day 7: validates
the calibration substrate on real Modal hardware. Phase 1 minimum: probe
HardwareFingerprint + run measure_latency_profile against a synthetic
predict callable on A10G. Asserts outputs are in expected ranges.

Larger-matrix expansion (A10G + A100 × franka + so100 + ur5) is the
follow-up for the full Day 7/9 cost-table refresh; this script ships
the cheapest meaningful entry first per CLAUDE.md "Maximum tests, not
minimum viable".

Usage:
    modal run scripts/modal_test_calibration_matrix.py

Cost: ~$0.30-$0.50 on A10G (image build cached after first run).
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("tether-calibration-matrix")


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


# Light image -- just CUDA + tether-vla. No LIBERO, no lerobot, no full
# inference stack. Calibration substrate is pure-Python after CUDA probe.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy",
        "torch",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20",
        "psutil",
        "pyyaml",
    )
    .run_commands(
        f'echo "build_bust={_BUILD_BUST}"',
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


def _probe_inner(tier: str) -> dict:
    """Shared body for both A10G + A100 probe functions.

    Probes HardwareFingerprint + runs measure_latency_profile against
    a synthetic numpy predict + validates CalibrationCache load_or_empty
    + is_stale + persist + reload (cache hit on second open).
    """
    import time
    import numpy as np
    from pathlib import Path

    print(f"[calibration] tier={tier} probing HardwareFingerprint...")
    from tether.runtime.calibration import (
        CalibrationCache,
        CalibrationEntry,
        HardwareFingerprint,
        MeasurementContext,
        measure_latency_profile,
    )

    fp = HardwareFingerprint.current()
    print(f"[calibration]   gpu_name: {fp.gpu_name!r}")
    print(f"[calibration]   gpu_uuid: {fp.gpu_uuid!r}")
    print(f"[calibration]   driver: {fp.driver_version_major}.{fp.driver_version_minor}")
    print(f"[calibration]   cuda: {fp.cuda_version_major}.{fp.cuda_version_minor}")
    print(f"[calibration]   kernel: {fp.kernel_release!r}")
    print(f"[calibration]   cpu_count: {fp.cpu_count}")
    print(f"[calibration]   ram_gb: {fp.ram_gb}")
    print(f"[calibration]   tether_version: {fp.tether_version}")

    print("[calibration] running measure_latency_profile against a synthetic predict...")

    def _stub_predict():
        a = np.random.rand(128, 128).astype(np.float32)
        b = np.random.rand(128, 128).astype(np.float32)
        c = a @ b
        return c.sum()

    quality = measure_latency_profile(
        _stub_predict, n_iters=100, warmup_iters=10,
    )
    print(f"[calibration]   median_ms: {quality.median_ms:.4f}")
    print(f"[calibration]   p99_ms: {quality.p99_ms:.4f}")
    print(f"[calibration]   quality_score: {quality.quality_score:.4f}")
    print(f"[calibration]   n_outliers_dropped: {quality.n_outliers_dropped}")

    # Substrate invariants
    assert quality.median_ms > 0, "median_ms must be > 0"
    assert quality.p99_ms >= quality.median_ms, "p99_ms must be >= median_ms"
    assert 0.0 <= quality.quality_score <= 1.0, (
        f"quality_score must be in [0, 1], got {quality.quality_score}"
    )
    assert quality.n_outliers_dropped >= 0, "n_outliers_dropped must be >= 0"

    # Day 7 expansion: validate CalibrationCache round-trip on Modal disk.
    # Real API per src/tether/runtime/calibration.py (not the names I
    # guessed earlier; lesson: read the dataclass first).
    print("[calibration] validating CalibrationCache round-trip...")
    cache_path = Path("/tmp/reflex_calibration_smoke.json")
    if cache_path.exists():
        cache_path.unlink()

    # First load -> empty + stale
    cache_v1 = CalibrationCache.load_or_empty(str(cache_path))
    assert len(cache_v1.entries) == 0, (
        f"fresh cache must be empty, got {len(cache_v1.entries)} entries"
    )
    assert cache_v1.is_stale(fp), "fresh cache must be stale"

    # Set hardware fingerprint + record one synthetic entry
    cache_v1.hardware_fingerprint = fp
    ctx = MeasurementContext.current()  # probes ort/torch/numpy/onnx versions
    entry = CalibrationEntry(
        chunk_size=50,
        nfe=10,
        latency_compensation_ms=float(quality.median_ms),
        provider="CPUExecutionProvider",
        variant=f"{tier}-synthetic-stub",
        measurement_quality=quality,
        measurement_context=ctx,
        timestamp=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    )
    cache_v1.record(
        embodiment="franka",
        model_hash="synthetic-stub-1234",
        entry=entry,
    )
    cache_v1.save(str(cache_path))

    # Reopen -> hit (1 entry; fingerprint matches -> not stale)
    cache_v2 = CalibrationCache.load_or_empty(str(cache_path))
    assert len(cache_v2.entries) == 1, (
        f"reloaded cache should have 1 entry, got {len(cache_v2.entries)}"
    )
    assert not cache_v2.is_stale(fp), (
        "reloaded cache with matching fingerprint must NOT be stale"
    )
    print(
        f"[calibration]   cache round-trip: persist + reload = "
        f"{len(cache_v2.entries)} entries (hit)"
    )

    # Lookup -> returns the entry
    looked_up = cache_v2.lookup(embodiment="franka", model_hash="synthetic-stub-1234")
    assert looked_up is not None, "lookup must return the recorded entry"
    assert looked_up.chunk_size == 50, "lookup must round-trip the entry exactly"

    # is_stale=True when fingerprint differs (simulate driver upgrade)
    fake_fp = HardwareFingerprint(
        gpu_uuid=fp.gpu_uuid, gpu_name=fp.gpu_name,
        driver_version_major=fp.driver_version_major + 1,  # bumped
        driver_version_minor=fp.driver_version_minor,
        cuda_version_major=fp.cuda_version_major,
        cuda_version_minor=fp.cuda_version_minor,
        kernel_release=fp.kernel_release, cpu_count=fp.cpu_count,
        ram_gb=fp.ram_gb, tether_version=fp.tether_version,
    )
    assert cache_v2.is_stale(fake_fp), (
        "cache must be stale when fingerprint differs (driver bump)"
    )
    print("[calibration]   cache invalidation: differing fingerprint -> stale OK")

    return {
        "status": "ok",
        "tier": tier,
        "fingerprint": fp.to_dict(),
        "quality": {
            "median_ms": quality.median_ms,
            "p99_ms": quality.p99_ms,
            "quality_score": quality.quality_score,
            "warmup_iters": quality.warmup_iters,
            "measurement_iters": quality.measurement_iters,
            "n_outliers_dropped": quality.n_outliers_dropped,
        },
        "cache": {
            "entries_persisted": 1,
            "reload_hit": True,
            "stale_on_fingerprint_change": True,
        },
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    secrets=[_hf_secret()],
)
def probe_calibration_a10g() -> dict:
    """Probe + cache round-trip on Modal A10G."""
    return _probe_inner("A10G")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    secrets=[_hf_secret()],
)
def probe_calibration_a100() -> dict:
    """Probe + cache round-trip on Modal A100-80GB.

    Validates that the calibration substrate produces SAME-SHAPE output
    on a different GPU tier (i.e., the median_ms differs but all the
    structural invariants still hold).
    """
    return _probe_inner("A100-80GB")


@app.local_entrypoint()
def main(skip_a100: bool = False):
    """Local entry: invoke both tiers + render the RESULT block.

    --skip-a100 runs only A10G (~$0.50). Default runs both (~$3 total).
    """
    print("=" * 70)
    print("Auto-calibration matrix -- A10G + A100-80GB cross-hardware")
    print("=" * 70)
    results: dict[str, dict] = {}
    print("\n[matrix] tier=A10G ...")
    results["A10G"] = probe_calibration_a10g.remote()
    if not skip_a100:
        print("\n[matrix] tier=A100-80GB ...")
        results["A100-80GB"] = probe_calibration_a100.remote()

    print("\n" + "=" * 70)
    print("=== MATRIX RESULT ===")
    print("=" * 70)
    for tier, r in results.items():
        print(f"\n[{tier}]")
        if r.get("status") == "fail":
            print(f"  status: FAIL")
            print(f"  reason: {r.get('reason', '(no reason)')}")
            continue
        fp = r["fingerprint"]
        q = r["quality"]
        cache = r.get("cache", {})
        print(f"  gpu:            {fp['gpu_name']!r}")
        print(f"  driver:         {fp['driver_version_major']}.{fp['driver_version_minor']}")
        print(f"  cuda:           {fp['cuda_version_major']}.{fp['cuda_version_minor']}")
        print(f"  ram:            {fp['ram_gb']}GB")
        print(f"  median_ms:      {q['median_ms']:.4f}")
        print(f"  p99_ms:         {q['p99_ms']:.4f}")
        print(f"  quality_score:  {q['quality_score']:.4f}")
        print(f"  cache_hit:      {cache.get('reload_hit', '?')}")
        print(f"  stale_on_diff:  {cache.get('stale_on_fingerprint_change', '?')}")

    # Cross-tier comparison: median_ms should differ across tiers
    # (A100 faster than A10G), but both should hold the substrate
    # invariants (positive, p99 >= median, quality in [0,1]).
    if "A10G" in results and "A100-80GB" in results:
        if (
            results["A10G"].get("status") == "ok"
            and results["A100-80GB"].get("status") == "ok"
        ):
            print("\n=== CROSS-TIER ===")
            a10g_med = results["A10G"]["quality"]["median_ms"]
            a100_med = results["A100-80GB"]["quality"]["median_ms"]
            print(f"  A10G median:      {a10g_med:.4f} ms")
            print(f"  A100 median:      {a100_med:.4f} ms")
            print(f"  ratio (A10G/A100): {a10g_med / a100_med:.2f}x")
            print(f"  (numpy stub is CPU-bound; ratio reflects host CPU diff)")
