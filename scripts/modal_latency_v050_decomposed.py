"""Modal: per-stage latency microbench for the v0.5 state-out pi0.5 decomposed ONNX.

Closes `serve-latency-microbench-cache` (GOALS.yaml weight 7) / §1.3 of
serve_technical_plan_v3.md. Unlike scripts/modal_latency_pi05_decomposed.py
which measures miss-vs-hit (full forward vs expert-only via cache hit),
this script explicitly times each ONNX session in isolation so we can
report per-stage numbers AND the theoretical within-call cache speedup:

    theoretical_speedup = (vlm_time + expert_time) / expert_time

Target ≥3× (per design, VLM prefix should be ~80% of decomposed compute).
Independent of any cache-on/off LIBERO task-success test — this is a pure
latency upper-bound measurement.

Usage:
    modal run scripts/modal_latency_v050_decomposed.py \\
      --decomposed-dir /onnx_out/distill_v050_pi05_state_out_r3/decomposed_v3 \\
      --num-repeats 50

Expected cost: ~$1, 30 min A100-80GB. Writes no persistent artifacts —
copy the stdout table into reflex_context/03_experiments/YYYY-MM-DD-v050-
decomposed-per-stage-latency.md.
"""
import os
import subprocess

import modal

app = modal.App("tether-latency-v050-decomposed")


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
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
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
def bench_per_stage(
    decomposed_dir: str,
    num_repeats: int = 50,
    warmup: int = 5,
):
    import json
    import logging
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("bench_per_stage")

    # ---- Load tether_config + locate both ONNX files ----
    export_dir = Path(decomposed_dir)
    cfg_path = export_dir / "tether_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"tether_config.json missing in {export_dir}")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("export_kind") != "decomposed":
        raise ValueError(f"export_kind={cfg.get('export_kind')!r}; need 'decomposed'")

    vlm_path = export_dir / cfg["decomposed"]["vlm_prefix_onnx"]
    expert_path = export_dir / cfg["decomposed"]["expert_denoise_onnx"]
    past_kv_names = cfg["decomposed"]["past_kv_tensor_names"]
    expert_takes_state = cfg["decomposed"].get("expert_takes_state", False)

    log.info("vlm_prefix: %s", vlm_path)
    log.info("expert_denoise: %s", expert_path)
    log.info("expert_takes_state: %s (v0.5 state-out: expected True)", expert_takes_state)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_prefix = ort.InferenceSession(str(vlm_path), providers=providers)
    sess_expert = ort.InferenceSession(str(expert_path), providers=providers)
    log.info("vlm_prefix actual providers: %s", sess_prefix.get_providers())
    log.info("expert_denoise actual providers: %s", sess_expert.get_providers())

    prefix_input_names = [i.name for i in sess_prefix.get_inputs()]
    prefix_output_names = [o.name for o in sess_prefix.get_outputs()]
    expert_input_names = [i.name for i in sess_expert.get_inputs()]

    # ---- Fake observation (one fixed input for all runs) ----
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
    if expert_takes_state:
        obs["state"] = rng.standard_normal((B, 32)).astype(np.float32)

    # ---- STAGE 1: VLM prefix alone ----
    prefix_feed = {k: v for k, v in obs.items() if k in prefix_input_names}
    for _ in range(warmup):
        sess_prefix.run(prefix_output_names, prefix_feed)

    vlm_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        prefix_out = sess_prefix.run(prefix_output_names, prefix_feed)
        vlm_times.append(time.perf_counter() - t0)

    # Cache the VLM output once so expert timing reuses identical past_kv
    past_kv = [prefix_out[i] for i in range(len(past_kv_names))]
    prefix_pad = prefix_out[-1]

    # ---- STAGE 2: expert denoise alone (fresh noise each run) ----
    def _expert_feed():
        feed = {name: past_kv[i] for i, name in enumerate(past_kv_names)}
        feed["prefix_pad_masks"] = prefix_pad
        feed["noise"] = rng.standard_normal((B, 50, 32)).astype(np.float32)
        if expert_takes_state:
            feed["state"] = rng.standard_normal((B, 32)).astype(np.float32)
        return feed

    for _ in range(warmup):
        sess_expert.run(["actions"], _expert_feed())

    expert_times = []
    for _ in range(num_repeats):
        feed = _expert_feed()
        t0 = time.perf_counter()
        sess_expert.run(["actions"], feed)
        expert_times.append(time.perf_counter() - t0)

    # ---- Report ----
    def _pct(ts, p):
        return 1000 * float(np.percentile(ts, p))

    def _summary(name, ts):
        return {
            "stage": name,
            "mean_ms": 1000 * float(np.mean(ts)),
            "p50_ms": _pct(ts, 50),
            "p95_ms": _pct(ts, 95),
            "p99_ms": _pct(ts, 99),
            "min_ms": 1000 * float(np.min(ts)),
            "max_ms": 1000 * float(np.max(ts)),
            "stdev_ms": 1000 * float(np.std(ts, ddof=1)),
            "n": len(ts),
        }

    vlm = _summary("vlm_prefix", vlm_times)
    expert = _summary("expert_denoise", expert_times)
    theoretical_speedup = (vlm["mean_ms"] + expert["mean_ms"]) / expert["mean_ms"]
    vlm_share_pct = 100.0 * vlm["mean_ms"] / (vlm["mean_ms"] + expert["mean_ms"])

    print("\n==== V0.5 DECOMPOSED PER-STAGE LATENCY ====")
    print(f"Hardware: A100-80GB   Repeats: {num_repeats}   Warmup: {warmup}")
    print(f"Export:  {export_dir}")
    print(f"State-out: {expert_takes_state}")
    print()
    print(f"{'stage':<18} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8}")
    for row in (vlm, expert):
        print(
            f"{row['stage']:<18} "
            f"{row['mean_ms']:>7.2f}ms "
            f"{row['p50_ms']:>7.2f}ms "
            f"{row['p95_ms']:>7.2f}ms "
            f"{row['p99_ms']:>7.2f}ms "
            f"{row['min_ms']:>7.2f}ms "
            f"{row['max_ms']:>7.2f}ms"
        )
    print()
    print(f"VLM share:            {vlm_share_pct:.1f}%  (target ≥80% per design)")
    print(f"Theoretical speedup:  {theoretical_speedup:.2f}x  (target ≥3x)")
    print(f"  = (vlm={vlm['mean_ms']:.1f}ms + expert={expert['mean_ms']:.1f}ms) / expert={expert['mean_ms']:.1f}ms")

    if theoretical_speedup < 3.0:
        print(f"\n⚠ Theoretical speedup below 3x — investigate whether VLM prefix is "
              f"smaller than expected OR expert is unexpectedly cheap.")
    if vlm_share_pct < 70.0:
        print(f"\n⚠ VLM share below 70% — the cache moat is weaker than design. "
              f"A2C2 + RTC priorities may shift.")

    return {
        "vlm_prefix": vlm,
        "expert_denoise": expert,
        "theoretical_speedup_x": theoretical_speedup,
        "vlm_share_pct": vlm_share_pct,
        "export_dir": str(export_dir),
        "state_out": expert_takes_state,
    }


@app.local_entrypoint()
def main(
    decomposed_dir: str = "/onnx_out/distill_v050_pi05_state_out_r3/decomposed_v3",
    num_repeats: int = 50,
):
    import json
    r = bench_per_stage.remote(decomposed_dir=decomposed_dir, num_repeats=num_repeats)
    print("\n==== JSON RESULT ====")
    print(json.dumps(r, indent=2))
