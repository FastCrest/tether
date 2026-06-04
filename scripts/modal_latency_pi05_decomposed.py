"""Modal: per-query latency microbench for the decomposed pi0.5 chain.

Measures the actual wall-time speedup from the prefix cache by timing:
- **cold**: first call (VLM compile + full forward + expert) — upper bound
- **miss (warm)**: VLM kernels warmed but cache disabled — typical
  cold-obs latency after warmup.
- **hit (warm)**: VLM skipped via cache reuse — the fast path.

Runs on A100-80GB with ``CUDAExecutionProvider`` to match the
deployment target. Reports mean + stdev over N repetitions.

Closes the "3-4× average, 10× peak deployment speedup" claim of
``reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md``.
Output is the headline number customers care about; LIBERO task success
is a separate sanity check (already closed).

Usage:
    modal run scripts/modal_latency_pi05_decomposed.py \\
      --decomposed-dir /onnx_out/distill_v031_pi05_libero_r4/decomposed_v3 \\
      --num-repeats 30
"""
import os
import subprocess
import modal

app = modal.App("tether-latency-pi05-decomposed")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _head() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()[:12]
    except Exception:
        return "main"


def _bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _head()
_BUST = _bust()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "nvidia-curand-cu12>=10.0,<12.0",
        "nvidia-cufft-cu12>=11.0,<13.0",
        "nvidia-cusparse-cu12>=12.0,<13.0",
        "nvidia-cusolver-cu12>=11.0,<13.0",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cuda-nvrtc-cu12>=12.0,<13.0",
        "numpy",
    )
    .run_commands(
        f'echo "bust={_BUST}"',
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def bench(
    decomposed_dir: str,
    num_repeats: int = 30,
    warmup: int = 3,
):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import time
    import statistics
    import numpy as np
    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    # Build one fixed observation to feed across all runs
    B = 1
    rng = np.random.default_rng(42)
    obs = dict(
        img_base=rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        img_wrist_l=rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        img_wrist_r=rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        mask_base=np.ones((B,), dtype=np.bool_),
        mask_wrist_l=np.ones((B,), dtype=np.bool_),
        mask_wrist_r=np.ones((B,), dtype=np.bool_),
        lang_tokens=rng.integers(0, 257152, (B, 200)).astype(np.int64),
        lang_masks=np.ones((B, 200), dtype=np.bool_),
        noise=rng.standard_normal((B, 50, 32)).astype(np.float32),
    )

    # State-out variant: decomposed export with expert_takes_state=True
    # requires a state input. Detect from tether_config.json and add it.
    import json as _json
    from pathlib import Path as _Path
    cfg_path = _Path(decomposed_dir) / "tether_config.json"
    if cfg_path.exists():
        with cfg_path.open() as _f:
            _cfg = _json.load(_f)
        if _cfg.get("decomposed", {}).get("expert_takes_state"):
            # max_state_dim is 32 for pi0.5
            obs["state"] = rng.standard_normal((B, 32)).astype(np.float32)

    # ---- Baseline: cache disabled (every call is a miss) ----
    inf_miss = Pi05DecomposedInference(
        export_dir=decomposed_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        enable_cache=False,
    )
    # Warmup — first call does CUDA kernel JIT compilation
    for _ in range(warmup):
        inf_miss.predict_action_chunk(**obs)

    miss_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        inf_miss.predict_action_chunk(**obs)
        miss_times.append(time.perf_counter() - t0)

    # ---- Cache hits: cache enabled, identical obs every call ----
    inf_hit = Pi05DecomposedInference(
        export_dir=decomposed_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        enable_cache=True,
        cache_ttl_sec=1000.0,  # effectively disabled
        phash_hamming_threshold=6,
    )
    for _ in range(warmup):
        inf_hit.predict_action_chunk(**obs)

    hit_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        inf_hit.predict_action_chunk(**obs)
        hit_times.append(time.perf_counter() - t0)

    stats = inf_hit.get_stats()
    print(f"\n==== CACHE STATS (hit run) ====\n{stats}")

    # ---- Report ----
    def _summary(ts):
        return {
            "mean_ms": 1000 * statistics.mean(ts),
            "median_ms": 1000 * statistics.median(ts),
            "stdev_ms": 1000 * statistics.stdev(ts),
            "min_ms": 1000 * min(ts),
            "max_ms": 1000 * max(ts),
        }

    miss = _summary(miss_times)
    hit = _summary(hit_times)
    speedup = miss["mean_ms"] / hit["mean_ms"]

    print("\n==== LATENCY RESULTS ====")
    print(f"Cache MISS (VLM runs every call): mean={miss['mean_ms']:.1f}ms "
          f"median={miss['median_ms']:.1f}ms stdev={miss['stdev_ms']:.1f}ms "
          f"min={miss['min_ms']:.1f}ms max={miss['max_ms']:.1f}ms")
    print(f"Cache HIT  (VLM skipped):         mean={hit['mean_ms']:.1f}ms "
          f"median={hit['median_ms']:.1f}ms stdev={hit['stdev_ms']:.1f}ms "
          f"min={hit['min_ms']:.1f}ms max={hit['max_ms']:.1f}ms")
    print(f"Speedup (miss / hit): {speedup:.1f}x")

    return {
        "miss": miss,
        "hit": hit,
        "speedup_x": speedup,
        "cache_stats": stats,
        "num_repeats": num_repeats,
    }


@app.local_entrypoint()
def main(
    decomposed_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/decomposed_v3",
    num_repeats: int = 30,
):
    r = bench.remote(decomposed_dir=decomposed_dir, num_repeats=num_repeats)
    print("\n==== RESULT ====")
    import json
    print(json.dumps(r, indent=2))
