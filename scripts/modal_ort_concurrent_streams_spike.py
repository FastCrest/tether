"""Modal: Q1 empirical spike — does ORT pipeline 2 concurrent session.run()
on different CUDA streams, and what's the actual speedup?

Day-3 spike for cross-request-pipelining (research_status: complete; Q1
flagged as load-bearing). Source-read of ORT (cuda_execution_provider.cc:
345-361, cuda_stream_handle.cc:261, inference_session.cc:3257-3260) found
that ORT creates per-session non-blocking CUDA streams by default + Python
releases GIL during Run() + is_concurrent_run_supported is true for CUDA
EP. So mechanism is TRULY PARALLEL by default.

But empirical question remains: SM occupancy may cap real-world speedup
when both sessions hit the same compute-saturated kernels. ActionFlow
paper claims 2.56x on AGX Orin (small GPU, big headroom); for pi0.5-on-
A100 the gain may be much smaller.

Decision the spike informs:

    For two concurrent session.run() on the same GPU at batch=1 with
    pi0.5-prefix-sized compute (~50-100ms per call on A100), what is
    the wall-clock speedup vs sequential?

    - If >= 1.5x: cross-request-pipelining is a real Phase 1.5 win
      (proceed with Option B reframe — A10G/AGX-only feature)
    - If 1.2-1.5x: marginal; consider Phase 2 deferral
    - If <= 1.2x: SM saturation kills the gain; defer entirely

Approach: synthetic model with matmul-heavy compute profile that mimics
pi0.5 prefix's ~80-100ms-on-A100 latency. Measure sequential vs
concurrent (2 threads) wall time; compute speedup ratio. Also run with
explicit cuda_graph=True provider option to confirm the source-read's
"forces unified stream" claim (should give 1.0x).

Usage:
    modal run scripts/modal_ort_concurrent_streams_spike.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-ort-concurrent-streams-spike")


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
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

# Reuse the reflex-vla image setup — it has the eager-dlopen-nvidia-libs
# loadchain debugged for ORT-CUDA EP (per 2026-04-30 per-step-overhead
# experiment, commit 462c191).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "reflex-vla[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
)
def spike_modal(
    n_warmup: int = 5,
    n_trials: int = 30,
    target_latency_ms: float = 80.0,
):
    """Build a synthetic conv-heavy ONNX of approximately the target
    latency, then measure sequential vs concurrent wall time."""
    import logging
    import time
    import threading
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("ort-spike")

    # CRITICAL: import reflex first to trigger _eager_dlopen_nvidia_libs
    # which puts the right CUDA-12-family libs (libcurand, libcublas,
    # libcudnn, libcusparse, etc) on the dlopen path for ORT-CUDA EP.
    # Without this, ORT silently falls back to CPU + we measure
    # CPU-thread parallelism, not GPU stream parallelism. Caught in v2.
    import reflex
    logger.info("[spike] imported reflex (eager-dlopen-nvidia-libs hook)")

    # ---- Build a synthetic model ~target_latency_ms ----
    # v3 was too small (2.4 ms / call on A100; launch-overhead-dominated).
    # Scale up until per-call latency reaches 50-100 ms, comparable to
    # pi0.5 vlm_prefix on A100 (~80 ms per Day-1 measurements).
    # Math: each Conv2d(C, C, 3) on (1, C, H, W) is ~9*C^2*H*W FLOPs.
    # Target ~50 ms on A100 (~310 TFLOP/s FP32 peak).
    # Going 64 layers × C=512 × H=W=128 → ~64 * 9 * 512^2 * 128^2 = 2.5 TFLOPs
    # → ~8 ms (overoptimistic; cudnn overhead pushes it ~50 ms in practice).
    n_layers = 64
    C = 512
    H = 128

    class Synth(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Conv2d(C, C, kernel_size=3, padding=1)
                for _ in range(n_layers)
            ])
            self.act = nn.ReLU()

        def forward(self, x):
            for layer in self.layers:
                x = self.act(layer(x))
            return x

    model = Synth().eval().to(torch.float32)
    dummy = torch.randn(1, C, H, H, dtype=torch.float32)

    out_dir = Path("/tmp/ort_spike")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "synth.onnx"
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["x"],
        output_names=["y"],
        opset_version=19,
    )
    logger.info("[spike] synthetic ONNX: %.1f MB",
                onnx_path.stat().st_size / 1024 / 1024)

    import onnxruntime as ort
    providers_default = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    providers_cuda_graph = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "enable_cuda_graph": "1",
        }),
        "CPUExecutionProvider",
    ]

    def make_sessions(providers, n=2):
        return [
            ort.InferenceSession(str(onnx_path), providers=providers)
            for _ in range(n)
        ]

    inp = np.random.randn(1, C, H, H).astype(np.float32)

    # ---- BASELINE CHECK: ORT CUDA EP actually loaded ----
    # Per feedback_validate_baseline_and_check_modal_midflight.md: validate
    # baseline > 0 BEFORE doing measurements. v2 of this spike silently fell
    # to CPU because of CUDA 12 vs 13 lib mismatch; all numbers were
    # invalid. Fail loud here instead of producing garbage measurements.
    sessions_single = make_sessions(providers_default, n=1)
    active_providers = sessions_single[0].get_providers()
    logger.info("[spike] active providers: %s", active_providers)
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            f"BASELINE FAIL — CUDAExecutionProvider not active. Active: "
            f"{active_providers}. ORT silently fell to CPU; measurements "
            f"would be invalid (CPU-thread parallelism, not GPU streams). "
            f"Fix CUDA loadchain before re-running."
        )
    logger.info("[spike] baseline OK: ORT-CUDA EP active")

    s0 = sessions_single[0]
    for _ in range(n_warmup):
        s0.run(None, {"x": inp})

    single_latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        s0.run(None, {"x": inp})
        single_latencies.append((time.perf_counter() - t0) * 1000)
    single_p50 = float(np.percentile(single_latencies, 50))
    single_p99 = float(np.percentile(single_latencies, 99))
    logger.info(
        "[spike] single-session latency: p50=%.1f ms  p99=%.1f ms",
        single_p50, single_p99,
    )

    # ---- Sequential 2-call baseline ----
    sessions_seq = make_sessions(providers_default, n=2)
    for _ in range(n_warmup):
        for s in sessions_seq:
            s.run(None, {"x": inp})

    seq_totals = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sessions_seq[0].run(None, {"x": inp})
        sessions_seq[1].run(None, {"x": inp})
        seq_totals.append((time.perf_counter() - t0) * 1000)
    seq_p50 = float(np.percentile(seq_totals, 50))
    seq_p99 = float(np.percentile(seq_totals, 99))
    logger.info(
        "[spike] sequential 2-call: p50=%.1f ms  p99=%.1f ms",
        seq_p50, seq_p99,
    )

    # ---- Concurrent 2-thread baseline (default providers) ----
    sessions_par = make_sessions(providers_default, n=2)
    for _ in range(n_warmup):
        threads = []
        for s in sessions_par:
            t = threading.Thread(target=s.run, args=(None, {"x": inp}))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    par_totals = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        threads = []
        for s in sessions_par:
            t = threading.Thread(target=s.run, args=(None, {"x": inp}))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        par_totals.append((time.perf_counter() - t0) * 1000)
    par_p50 = float(np.percentile(par_totals, 50))
    par_p99 = float(np.percentile(par_totals, 99))
    logger.info(
        "[spike] concurrent 2-thread (default): p50=%.1f ms  p99=%.1f ms",
        par_p50, par_p99,
    )

    # ---- Concurrent 2-thread with cuda_graph=True (should serialize) ----
    sessions_cg = make_sessions(providers_cuda_graph, n=2)
    cg_totals = []
    cg_skipped = False
    try:
        for _ in range(n_warmup):
            threads = []
            for s in sessions_cg:
                t = threading.Thread(target=s.run, args=(None, {"x": inp}))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        for _ in range(n_trials):
            t0 = time.perf_counter()
            threads = []
            for s in sessions_cg:
                t = threading.Thread(target=s.run, args=(None, {"x": inp}))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            cg_totals.append((time.perf_counter() - t0) * 1000)
        cg_p50 = float(np.percentile(cg_totals, 50))
        cg_p99 = float(np.percentile(cg_totals, 99))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[spike] cuda_graph trial failed: %s", exc)
        cg_skipped = True
        cg_p50 = float("nan")
        cg_p99 = float("nan")
    if not cg_skipped:
        logger.info(
            "[spike] concurrent 2-thread (cuda_graph=True): p50=%.1f ms  p99=%.1f ms",
            cg_p50, cg_p99,
        )

    # ---- Speedup analysis ----
    # speedup = sequential_time / concurrent_time
    # >= 1.8: nearly perfect overlap (rare; bounded by Amdahl)
    # 1.4-1.8: meaningful overlap; ActionFlow-style win achievable
    # 1.1-1.4: partial overlap; SM-saturation-bound; marginal Phase 1.5 win
    # ~1.0: ORT serializes OR full SM saturation; feature dead
    speedup_default = seq_p50 / par_p50 if par_p50 > 0 else 0
    speedup_cg = (
        seq_p50 / cg_p50 if (not cg_skipped and cg_p50 > 0) else float("nan")
    )

    print("\n" + "=" * 72)
    print("ORT CONCURRENT STREAMS SPIKE  (Q1 for cross-request-pipelining)")
    print("=" * 72)
    print(f"  GPU              : A100-80GB")
    print(f"  model            : synthetic ({n_layers} conv layers, C={C}, H=W={H})")
    print(f"  trials           : {n_trials} per mode")
    print()
    print(f"  single session   : p50 = {single_p50:6.1f} ms  p99 = {single_p99:6.1f} ms")
    print(f"  2 sequential     : p50 = {seq_p50:6.1f} ms  p99 = {seq_p99:6.1f} ms")
    print(f"  2 concurrent     : p50 = {par_p50:6.1f} ms  p99 = {par_p99:6.1f} ms")
    if not cg_skipped:
        print(f"  2 cuda-graph     : p50 = {cg_p50:6.1f} ms  p99 = {cg_p99:6.1f} ms")
    print()
    print(f"  speedup (default)   = {speedup_default:.2f}×  (sequential / concurrent)")
    if not cg_skipped:
        print(f"  speedup (cuda-graph) = {speedup_cg:.2f}×  (expected ≈ 1.0× per source-read)")
    print()
    if speedup_default >= 1.8:
        verdict = "EXCELLENT — near-perfect overlap; ActionFlow's 2.56x is plausible on this hardware"
    elif speedup_default >= 1.4:
        verdict = "GOOD — meaningful pipeline win; cross-request-pipelining is Phase 1.5 worthwhile"
    elif speedup_default >= 1.15:
        verdict = "MARGINAL — partial overlap; SM-saturation-bound; consider Phase 2 deferral"
    elif speedup_default >= 0.9:
        verdict = "BAD — feature dead. Either ORT serializes OR full SM saturation"
    else:
        verdict = "WORSE THAN SEQUENTIAL — overhead dominates; concurrent is harmful"
    print(f"  VERDICT (default): {verdict}")
    if not cg_skipped:
        if abs(speedup_cg - 1.0) < 0.15:
            cg_verdict = "CONFIRMED — cuda_graph forces unified stream (matches source-read)"
        else:
            cg_verdict = "UNEXPECTED — cuda_graph speedup diverges from source-read claim"
        print(f"  VERDICT (cuda_graph): {cg_verdict}")
    print("=" * 72)

    return {
        "single_p50_ms": single_p50,
        "single_p99_ms": single_p99,
        "seq_p50_ms": seq_p50,
        "seq_p99_ms": seq_p99,
        "par_p50_ms": par_p50,
        "par_p99_ms": par_p99,
        "cg_p50_ms": cg_p50,
        "cg_p99_ms": cg_p99,
        "cg_skipped": cg_skipped,
        "speedup_default": speedup_default,
        "speedup_cg": speedup_cg,
        "verdict_default": verdict,
    }


@app.local_entrypoint()
def main(
    n_warmup: int = 5,
    n_trials: int = 30,
):
    r = spike_modal.remote(n_warmup=n_warmup, n_trials=n_trials)
    print("\n=== RESULT ===")
    print(f"  speedup_default   : {r['speedup_default']:.2f}×")
    if not r["cg_skipped"]:
        print(f"  speedup_cuda_graph: {r['speedup_cg']:.2f}×")
    print(f"  verdict           : {r['verdict_default']}")
