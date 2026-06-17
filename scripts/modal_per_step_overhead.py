"""Modal A100: per-step ORT-call overhead bench (gate 4).

Question: does running 10 ORT calls in a Python Euler loop (per-step)
add unacceptable wall-clock overhead vs running 1 baked-loop ORT call?
The N=10 ORT call overhead = (Python loop overhead) + (10 × per-step
inference) - (1 × baked inference).

Reuses the ONNX exports already on the ``pi0-onnx-outputs`` volume from
the gate 3 parity run (``per_step_parity/pi05_teacher_n10_{baked,per_step}``).
No re-export needed.

Methodology:
  - Load both ORT sessions on CUDAExecutionProvider
  - Warm up each session 10 iterations
  - Measure N=100 chunks per session (one chunk = one full denoising pass)
  - For baked: 1 ORT call per chunk
  - For per-step: 10 ORT calls + Python Euler accumulation per chunk
  - Report: median, p95, p99 chunk latency for each path; absolute and
    relative per-step overhead

Acceptance criteria (per spec):
  - per-step chunk latency ≤ 1.20 × baked chunk latency (≤ 20% overhead)
  - per-step p99 ≤ 1.30 × baked p99 (jitter envelope)

Below those = ship-ready.
Above = investigate (likely from Python loop / numpy copy / ORT input
binding overhead). Common fixes: ORT IOBinding to avoid host-device
copies, fuse the Python accumulation, etc.

Spec:        features/03_export/per-step-expert-export.md
Research:    features/03_export/per-step-expert-export_research.md
Cost: ~$1.50 Modal (one A100-80GB invocation, ~5 min wall once cached).

Usage:
    modal profile activate suranjana-jain
    HF_TOKEN=<token> modal run --detach scripts/modal_per_step_overhead.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("tether-per-step-overhead")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


# Pin build_bust so the image cache from gate 3 stays warm.
_BUST = "20260501-per-step-gates-curand-eager-dlopen"

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20,<1.24",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        # The parity image picked these up transitively via torch 2.10.
        # Pinning explicitly so the overhead script's image build is
        # CUDA-EP-loadable regardless of pip's transitive resolution.
        "nvidia-curand-cu12>=10.0,<12.0",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cufft-cu12>=11.0,<12.0",
    )
    .add_local_dir(
        os.path.join(REPO_ROOT, "src"),
        remote_path="/root/tether-vla/src",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "pyproject.toml"),
        remote_path="/root/tether-vla/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "README.md"),
        remote_path="/root/tether-vla/README.md",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "LICENSE"),
        remote_path="/root/tether-vla/LICENSE",
        copy=True,
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        'pip install -e "/root/tether-vla[monolithic]"',
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


N_WARMUP = 10
N_BENCH = 100
NUM_STEPS = 10  # baked loop steps == per-step Python loop iters


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def overhead_bench() -> dict:
    import logging
    import time
    import numpy as np
    import onnxruntime as ort

    from tether.exporters.decomposed import (
        PI05_HEAD_DIM,
        PI05_KV_HEADS,
        PI05_PALIGEMMA_LAYERS,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    baked_path = Path(ONNX_OUT) / "per_step_parity/pi05_teacher_n10_baked/expert_denoise.onnx"
    per_step_path = Path(ONNX_OUT) / "per_step_parity/pi05_teacher_n10_per_step/expert_denoise.onnx"

    if not baked_path.exists():
        raise FileNotFoundError(f"Baked ONNX missing: {baked_path}. Re-run gate 3 first.")
    if not per_step_path.exists():
        raise FileNotFoundError(f"Per-step ONNX missing: {per_step_path}. Re-run gate 3 first.")

    log.info("Loading sessions on CUDAExecutionProvider")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_baked = ort.InferenceSession(str(baked_path), providers=providers)
    sess_per_step = ort.InferenceSession(str(per_step_path), providers=providers)
    actual_baked = sess_baked.get_providers()[0]
    actual_per_step = sess_per_step.get_providers()[0]
    log.info("providers: baked=%s, per_step=%s", actual_baked, actual_per_step)
    # Strict provider check — silent CPU fallback voids the bench
    if actual_baked != "CUDAExecutionProvider":
        raise RuntimeError(f"baked session fell back to {actual_baked}; CUDA EP load failed")
    if actual_per_step != "CUDAExecutionProvider":
        raise RuntimeError(f"per-step session fell back to {actual_per_step}; CUDA EP load failed")

    rng = np.random.default_rng(seed=42)
    prefix_seq_len = 968
    chunk = 50
    action_dim = 32
    B = 1

    past_kvs = [
        rng.standard_normal(
            (B, PI05_KV_HEADS, prefix_seq_len, PI05_HEAD_DIM),
            dtype=np.float64,
        ).astype(np.float32)
        for _ in range(PI05_PALIGEMMA_LAYERS * 2)
    ]
    prefix_pad_masks = np.ones((B, prefix_seq_len), dtype=bool)

    def _kv_names(num_layers):
        names = []
        for i in range(num_layers):
            names.append(f"past_k_{i}")
            names.append(f"past_v_{i}")
        return names

    kv_names = _kv_names(PI05_PALIGEMMA_LAYERS)

    # Pre-build the constant part of the per-step feed (past_kvs + prefix_pad_masks).
    # Mirrors production runtime's `expert_feed_base` pattern in
    # Pi05DecomposedInference.predict_action_chunk → _run_expert.
    feed_base = {n: past_kvs[i] for i, n in enumerate(kv_names)}
    feed_base["prefix_pad_masks"] = prefix_pad_masks

    def _run_baked(noise):
        feed = dict(feed_base)
        feed["noise"] = noise
        return sess_baked.run(["actions"], feed)[0]

    def _run_per_step_naive(noise):
        """Production-style: dict copy per iter, no IOBinding. ORT re-copies
        all 38 past_kv tensors host→device on every iter."""
        n = NUM_STEPS
        dt = -1.0 / n
        x_t = noise.copy()
        for step in range(n):
            time_val = 1.0 + step * dt
            t = np.full((B,), time_val, dtype=np.float32)
            feed = dict(feed_base)
            feed["x_t"] = x_t
            feed["t"] = t
            v_t = sess_per_step.run(["v_t"], feed)[0]
            x_t = x_t + dt * v_t
        return x_t

    def _run_per_step_iobinding(noise):
        """IOBinding: pin past_kvs + prefix_pad_masks to device ONCE per chunk,
        ORT skips the redundant host→device copies across the 10 Euler iters."""
        n = NUM_STEPS
        dt = -1.0 / n
        x_t = noise.copy()
        binding = sess_per_step.io_binding()
        # Hold the device OrtValues alive — bind_ortvalue_input doesn't
        # take ownership; if Python GC's them mid-loop, ORT segfaults.
        device_kept_alive = []
        for nm, arr in feed_base.items():
            ortval = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)
            binding.bind_ortvalue_input(nm, ortval)
            device_kept_alive.append(ortval)
        for step in range(n):
            time_val = 1.0 + step * dt
            t_arr = np.full((B,), time_val, dtype=np.float32)
            binding.bind_cpu_input("x_t", x_t)
            binding.bind_cpu_input("t", t_arr)
            binding.bind_output("v_t", "cpu")
            sess_per_step.run_with_iobinding(binding)
            v_t = binding.get_outputs()[0].numpy()
            x_t = x_t + dt * v_t
        return x_t

    log.info("Warming up baked (%d iters)", N_WARMUP)
    for _ in range(N_WARMUP):
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)
        _ = _run_baked(noise)

    log.info("Benching baked (N=%d)", N_BENCH)
    baked_times = []
    for i in range(N_BENCH):
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)
        t0 = time.perf_counter()
        _ = _run_baked(noise)
        baked_times.append((time.perf_counter() - t0) * 1000)  # ms

    log.info("Warming up per-step naive (%d iters × %d steps)", N_WARMUP, NUM_STEPS)
    for _ in range(N_WARMUP):
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)
        _ = _run_per_step_naive(noise)

    log.info("Benching per-step naive (N=%d × %d steps)", N_BENCH, NUM_STEPS)
    per_step_naive_times = []
    for i in range(N_BENCH):
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)
        t0 = time.perf_counter()
        _ = _run_per_step_naive(noise)
        per_step_naive_times.append((time.perf_counter() - t0) * 1000)  # ms

    log.info("Warming up per-step iobinding (%d iters × %d steps)", N_WARMUP, NUM_STEPS)
    for _ in range(N_WARMUP):
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)
        _ = _run_per_step_iobinding(noise)

    log.info("Benching per-step iobinding (N=%d × %d steps)", N_BENCH, NUM_STEPS)
    per_step_iob_times = []
    for i in range(N_BENCH):
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)
        t0 = time.perf_counter()
        _ = _run_per_step_iobinding(noise)
        per_step_iob_times.append((time.perf_counter() - t0) * 1000)  # ms

    def stats(arr):
        a = np.array(arr)
        return {
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(np.mean(a)),
            "std_ms": float(np.std(a)),
            "min_ms": float(np.min(a)),
            "max_ms": float(np.max(a)),
        }

    baked_stats = stats(baked_times)
    naive_stats = stats(per_step_naive_times)
    iob_stats = stats(per_step_iob_times)

    def gate_eval(per_step_stats):
        med_pct = (per_step_stats["median_ms"] / baked_stats["median_ms"]) - 1.0
        p99r = per_step_stats["p99_ms"] / baked_stats["p99_ms"]
        return {
            "median_overhead_ms": per_step_stats["median_ms"] - baked_stats["median_ms"],
            "median_overhead_pct": med_pct,
            "p99_ratio": p99r,
            "passes_median_gate": med_pct <= 0.20,
            "passes_p99_gate": p99r <= 1.30,
            "passes_overall": med_pct <= 0.20 and p99r <= 1.30,
        }

    naive_gate = gate_eval(naive_stats)
    iob_gate = gate_eval(iob_stats)

    log.info("BAKED                median=%.2fms p99=%.2fms", baked_stats["median_ms"], baked_stats["p99_ms"])
    log.info("PER-STEP NAIVE       median=%.2fms p99=%.2fms (overhead %+.1f%%, p99 %.2fx, gate %s)",
             naive_stats["median_ms"], naive_stats["p99_ms"],
             naive_gate["median_overhead_pct"] * 100, naive_gate["p99_ratio"],
             "PASS" if naive_gate["passes_overall"] else "FAIL")
    log.info("PER-STEP IOBINDING   median=%.2fms p99=%.2fms (overhead %+.1f%%, p99 %.2fx, gate %s)",
             iob_stats["median_ms"], iob_stats["p99_ms"],
             iob_gate["median_overhead_pct"] * 100, iob_gate["p99_ratio"],
             "PASS" if iob_gate["passes_overall"] else "FAIL")

    return {
        "n_warmup": N_WARMUP,
        "n_bench": N_BENCH,
        "num_steps": NUM_STEPS,
        "providers": {"baked": actual_baked, "per_step": actual_per_step},
        "baked": baked_stats,
        "per_step_naive": naive_stats,
        "per_step_iobinding": iob_stats,
        "naive_gate": naive_gate,
        "iobinding_gate": iob_gate,
        "thresholds": {"median_overhead_pct_max": 0.20, "p99_ratio_max": 1.30},
        # Top-level "passes_overall" reflects the IOBinding path since that's
        # what production runtime should adopt if it passes.
        "passes_overall": iob_gate["passes_overall"],
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Per-step ORT call overhead bench (gate 4)")
    print("=" * 60)
    result = overhead_bench.remote()

    receipt_path = Path(REPO_ROOT) / ".." / "reflex_context" / "per_step_overhead_last_run.json"
    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2))
    print(f"\nReceipt: {receipt_path}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    b, n_, i_ = result["baked"], result["per_step_naive"], result["per_step_iobinding"]
    print(f"  baked              median={b['median_ms']:.2f}ms  p99={b['p99_ms']:.2f}ms")
    print(f"  per-step naive     median={n_['median_ms']:.2f}ms  p99={n_['p99_ms']:.2f}ms  "
          f"({result['naive_gate']['median_overhead_pct'] * 100:+.1f}%, p99 {result['naive_gate']['p99_ratio']:.2f}x) "
          f"{'PASS' if result['naive_gate']['passes_overall'] else 'FAIL'}")
    print(f"  per-step IOBinding median={i_['median_ms']:.2f}ms  p99={i_['p99_ms']:.2f}ms  "
          f"({result['iobinding_gate']['median_overhead_pct'] * 100:+.1f}%, p99 {result['iobinding_gate']['p99_ratio']:.2f}x) "
          f"{'PASS' if result['iobinding_gate']['passes_overall'] else 'FAIL'}")
    print(f"\n  Production-relevant gate (IOBinding): "
          f"{'PASS' if result['passes_overall'] else 'FAIL'} (median≤+20%, p99≤1.30x)")
