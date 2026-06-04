"""Modal A100: per-step E2E chunk-latency bench (gate 5, v3 isolated invocations).

Question: through the FULL Pi05DecomposedInference pipeline (vlm_prefix
+ expert + Python glue), does the per-step path add unacceptable
chunk-latency overhead vs the baked path?

v1+v2 (single-process bench) measured +21-38% with the gap living in
vlm_prefix — but vlm_prefix.onnx should be IDENTICAL between baked and
per-step. v2 phase breakdown showed baked vlm bimodal (45ms / 91ms p99,
sometimes warm sometimes cold) and per-step vlm consistently 91ms — a
GPU L2 cache contention artifact from running both Pi05DecomposedInference
instances in the same Python process. Production never does this — a
customer loads ONE inference instance per process.

v3 fixes the methodology: ONE Pi05DecomposedInference per Modal function
invocation. Each function call is a fresh GPU rental, so the inference
instance gets a clean GPU L2 cache. Compares like-with-like.

Methodology:
  - bench_one(export_dir, label) is a Modal function: loads exactly ONE
    Pi05DecomposedInference, warms up, benches N=200 with phase breakdown
    (vlm_prefix + expert separately), returns stats
  - local entrypoint calls bench_one twice — once for baked, once for
    per-step — sequentially (so we don't pay 2x GPU rental at once)
  - computes overhead from the two stat dicts

Acceptance:
  - per-step median chunk latency ≤ 1.20 × baked
  - per-step p99 ≤ 1.30 × baked

Cost: ~$1.50-2 Modal (2 × A100-80GB ~5 min each).

Usage:
    HF_TOKEN=<token> modal run --detach scripts/modal_per_step_e2e_latency.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("tether-per-step-e2e-latency")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


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
    .add_local_file(os.path.join(REPO_ROOT, "pyproject.toml"), remote_path="/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file(os.path.join(REPO_ROOT, "README.md"), remote_path="/root/tether-vla/README.md", copy=True)
    .add_local_file(os.path.join(REPO_ROOT, "LICENSE"), remote_path="/root/tether-vla/LICENSE", copy=True)
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
N_BENCH = 200


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def bench_one(export_dir: str, label: str) -> dict:
    """Bench exactly ONE Pi05DecomposedInference instance. Each invocation
    is a fresh GPU rental → clean L2 cache → no cross-session contention."""
    import logging
    import time
    import numpy as np

    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    if not (Path(export_dir) / "expert_denoise.onnx").exists():
        raise FileNotFoundError(f"Export missing at {export_dir}. Run gate 3 first.")

    # Pin cudnn algo selection so both baked + per-step pick the same kernel
    # (v3 showed baked vlm bimodal between ~45ms / ~91ms, per-step stuck at
    # ~91ms — likely ORT's EXHAUSTIVE autotuner picking different cudnn
    # convolution algorithms for the two sessions despite identical graph
    # topology). HEURISTIC = deterministic, fast warmup, no autotune drift.
    log.info("[%s] Building Pi05DecomposedInference from %s", label, export_dir)
    providers = [
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
        "CPUExecutionProvider",
    ]
    inf = Pi05DecomposedInference(
        export_dir=export_dir, cache_level="none", providers=providers,
    )

    B = 1
    img_h, img_w = 224, 224
    chunk = 50
    action_dim = 32
    lang_len = 200

    def make_inputs(seed: int):
        r = np.random.default_rng(seed)
        return {
            "img_base": r.uniform(0.0, 1.0, size=(B, 3, img_h, img_w)).astype(np.float32),
            "img_wrist_l": r.uniform(0.0, 1.0, size=(B, 3, img_h, img_w)).astype(np.float32),
            "img_wrist_r": r.uniform(0.0, 1.0, size=(B, 3, img_h, img_w)).astype(np.float32),
            "mask_base": np.ones((B,), dtype=bool),
            "mask_wrist_l": np.ones((B,), dtype=bool),
            "mask_wrist_r": np.ones((B,), dtype=bool),
            "lang_tokens": r.integers(0, 100, size=(B, lang_len), dtype=np.int64),
            "lang_masks": np.ones((B, lang_len), dtype=bool),
            "noise": r.standard_normal(size=(B, chunk, action_dim)).astype(np.float32),
        }

    def time_one_chunk(inputs):
        t0 = time.perf_counter()
        past_kv, prefix_pad = inf._get_or_run_prefix(
            img_base=inputs["img_base"],
            img_wrist_l=inputs["img_wrist_l"],
            img_wrist_r=inputs["img_wrist_r"],
            mask_base=inputs["mask_base"],
            mask_wrist_l=inputs["mask_wrist_l"],
            mask_wrist_r=inputs["mask_wrist_r"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            image_phashes=(b"x" * 8, b"x" * 8, b"x" * 8),
            lang_hash="x",
        )
        t_vlm = (time.perf_counter() - t0) * 1000
        expert_feed_base = {nm: past_kv[i] for i, nm in enumerate(inf._past_kv_names)}
        expert_feed_base["prefix_pad_masks"] = prefix_pad
        t1 = time.perf_counter()
        _ = inf._run_expert(expert_feed_base, inputs["noise"])
        t_expert = (time.perf_counter() - t1) * 1000
        return t_vlm + t_expert, t_vlm, t_expert

    log.info("[%s] Warming up (%d iters)", label, N_WARMUP)
    for i in range(N_WARMUP):
        _ = inf.predict_action_chunk(**make_inputs(seed=i))

    log.info("[%s] Benching (N=%d) with phase breakdown", label, N_BENCH)
    totals, vlms, experts = [], [], []
    for i in range(N_BENCH):
        total, vlm, exp = time_one_chunk(make_inputs(seed=1000 + i))
        totals.append(total)
        vlms.append(vlm)
        experts.append(exp)

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

    total_stats = stats(totals)
    vlm_stats = stats(vlms)
    expert_stats = stats(experts)
    log.info("[%s] E2E    median=%.2fms p99=%.2fms", label, total_stats["median_ms"], total_stats["p99_ms"])
    log.info("[%s] vlm    median=%.2fms p99=%.2fms", label, vlm_stats["median_ms"], vlm_stats["p99_ms"])
    log.info("[%s] expert median=%.2fms p99=%.2fms", label, expert_stats["median_ms"], expert_stats["p99_ms"])

    return {
        "label": label,
        "n_warmup": N_WARMUP,
        "n_bench": N_BENCH,
        "total": total_stats,
        "vlm": vlm_stats,
        "expert": expert_stats,
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Per-step E2E chunk-latency bench (gate 5, isolated invocations)")
    print("=" * 60)

    baked_dir = f"{ONNX_OUT}/per_step_parity/pi05_teacher_n10_baked"
    per_step_dir = f"{ONNX_OUT}/per_step_parity/pi05_teacher_n10_per_step"

    print(f"\nFiring baked bench (sequential, isolated GPU rental)...")
    baked = bench_one.remote(baked_dir, "baked")
    print(f"\nFiring per-step bench (sequential, isolated GPU rental)...")
    per_step = bench_one.remote(per_step_dir, "per_step")

    median_pct = (per_step["total"]["median_ms"] / baked["total"]["median_ms"]) - 1.0
    p99_ratio = per_step["total"]["p99_ms"] / baked["total"]["p99_ms"]
    passes_overall = median_pct <= 0.20 and p99_ratio <= 1.30

    result = {
        "baked": baked,
        "per_step": per_step,
        "median_overhead_pct": median_pct,
        "p99_ratio": p99_ratio,
        "passes_overall": passes_overall,
        "thresholds": {"median_overhead_pct_max": 0.20, "p99_ratio_max": 1.30},
    }

    receipt_path = Path(REPO_ROOT) / ".." / "reflex_context" / "per_step_e2e_latency_last_run.json"
    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2))
    print(f"\nReceipt: {receipt_path}")

    print("\n" + "=" * 60)
    print("RESULTS (isolated invocations — production-relevant)")
    print("=" * 60)
    for label, r in [("baked", baked), ("per-step", per_step)]:
        print(f"  {label:9} E2E    median={r['total']['median_ms']:.2f}ms  p99={r['total']['p99_ms']:.2f}ms")
        print(f"  {label:9} vlm    median={r['vlm']['median_ms']:.2f}ms  p99={r['vlm']['p99_ms']:.2f}ms")
        print(f"  {label:9} expert median={r['expert']['median_ms']:.2f}ms  p99={r['expert']['p99_ms']:.2f}ms")
    print(f"\n  E2E overhead: median {median_pct * 100:+.1f}%  p99 {p99_ratio:.2f}x")
    print(f"  Overall: {'PASS' if passes_overall else 'FAIL'} (median≤+20%, p99≤1.30x)")
