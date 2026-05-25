"""Lift #5 Days 12-14 — Headline benchmark: ORT vs Triton+Graph latency.

Measures the customer-visible perf headline on A100:
  ARM A: lerobot PI05Policy.predict_action_chunk (PyTorch eager, fp32) — the baseline
  ARM B: Pi05FastKernelsInference (Triton + CUDA Graph, bf16) — the fast path

Reports median latency + speedup. This is the "14× faster" marketing number.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_headline_bench.py
"""
import os
import subprocess

import modal

app = modal.App("reflex-fast-kernels-headline-bench")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "lift/5-day1-2-vendor-triton-kernels"


_HEAD = _repo_head_sha()


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
    .apt_install(
        "git", "ninja-build", "clang", "build-essential",
        "gnupg", "wget",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb"
        " && apt-get update"
        " && apt-get install -y cuda-toolkit-12-4 --no-install-recommends"
        " && rm cuda-keyring_1.1-1_all.deb",
    )
    .pip_install(
        "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "triton>=3.1", "ninja",
        "lerobot==0.5.1", "num2words",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


@app.function(
    image=image, gpu="A100-40GB", timeout=3600,
    secrets=[_hf_secret()],
)
def run_headline_bench(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    n_warmup: int = 5,
    n_measure: int = 50,
) -> dict:
    """Headline latency benchmark: PyTorch eager vs Triton+Graph."""
    import os
    import time

    import torch

    os.environ["TORCHINDUCTOR_DISABLE"] = "1"

    print(f"[bench] Headline benchmark — model={model_id}", flush=True)
    print(f"[bench] CUDA: {torch.cuda.get_device_name(0)}, sm {torch.cuda.get_device_capability(0)}", flush=True)
    print(f"[bench] warmup={n_warmup}, measure={n_measure}", flush=True)
    t_total = time.time()

    import gc
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    # ══════════════════════════════════════════════════════════════════
    # Sequential ARMs — only ONE model on CUDA at a time (avoids OOM
    # on 40GB and device-mismatch from shared weights).
    # ══════════════════════════════════════════════════════════════════

    num_views = 3
    img_size = 224

    # ── ARM A: PyTorch baseline (predict_action_chunk) ────────────────
    print(f"\n[bench] ARM A: PyTorch baseline (predict_action_chunk, fp32)", flush=True)

    t0 = time.time()
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).to("cuda")
    policy.eval()
    print(f"[bench] [{time.time()-t0:.1f}s] PI05Policy loaded (CUDA)", flush=True)

    # Preprocessor for building batch_pp
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(repo_id=model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
    )

    dummy_img = torch.randn(1, 3, img_size, img_size, device="cuda")
    dummy_batch = {
        "observation.images.image": dummy_img,
        "observation.images.image2": dummy_img.clone(),
        "observation.state": torch.zeros(1, 8, device="cuda"),
        "task": ["pick up the red cup"],
    }
    batch_pp = preprocessor(dummy_batch)
    batch_pp = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch_pp.items()}

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = policy.predict_action_chunk(batch_pp)
    torch.cuda.synchronize()

    # Measure
    baseline_times = []
    for i in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = policy.predict_action_chunk(batch_pp)
        torch.cuda.synchronize()
        baseline_times.append((time.time() - t0) * 1000)

    baseline_median = sorted(baseline_times)[len(baseline_times) // 2]
    baseline_p95 = sorted(baseline_times)[int(len(baseline_times) * 0.95)]
    print(f"[bench] Baseline: median={baseline_median:.1f}ms, p95={baseline_p95:.1f}ms", flush=True)

    # Full teardown — free ALL CUDA memory before Triton ARM
    del policy, batch_pp, dummy_batch, dummy_img, preprocessor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[bench] Baseline ARM torn down, CUDA freed", flush=True)

    # ── ARM B: Triton + CUDA Graph ────────────────────────────────────
    print(f"\n[bench] ARM B: Triton + CUDA Graph (bf16)", flush=True)

    t0 = time.time()
    policy_b = PI05Policy.from_pretrained(model_id)
    policy_b = policy_b.to(dtype=torch.float32).cpu()

    from reflex.models.vlas.pi05 import Pi05VLA
    vla = Pi05VLA.from_lerobot_policy(policy_b)
    del policy_b
    gc.collect()
    vla.vision_backbone.to("cuda")
    vla.llm_backbone.to("cuda")
    vla.vla_head.to("cuda")

    from reflex.runtime.fast_inference.pi05 import Pi05FastKernelsInference
    triton_rt = Pi05FastKernelsInference(vla, capture=True, num_views=num_views)
    triton_rt.prepare_triton_inference()
    print(f"[bench] [{time.time()-t0:.1f}s] Triton runtime ready", flush=True)

    # Triton inputs
    images_concat = torch.randn(1, num_views * 3, img_size, img_size, device="cuda")
    lang_tokens = torch.randint(0, 256000, (1, 16), dtype=torch.int64, device="cuda")
    lang_masks = torch.ones(1, 16, dtype=torch.bool, device="cuda")
    states = torch.zeros(1, 32, device="cuda")
    noise = torch.randn(1, 50, 32, device="cuda")

    # Warmup (includes JIT compile + graph build on first call)
    for _ in range(n_warmup):
        _ = triton_rt.predict_action(
            images=images_concat, lang_tokens=lang_tokens,
            states=states, lang_masks=lang_masks, noise=noise,
        )
    torch.cuda.synchronize()

    # Measure
    triton_times = []
    for i in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = triton_rt.predict_action(
            images=images_concat, lang_tokens=lang_tokens,
            states=states, lang_masks=lang_masks, noise=noise,
        )
        torch.cuda.synchronize()
        triton_times.append((time.time() - t0) * 1000)

    triton_median = sorted(triton_times)[len(triton_times) // 2]
    triton_p95 = sorted(triton_times)[int(len(triton_times) * 0.95)]
    print(f"[bench] Triton: median={triton_median:.1f}ms, p95={triton_p95:.1f}ms", flush=True)

    # ── Results ───────────────────────────────────────────────────────
    speedup = baseline_median / triton_median if triton_median > 0 else float("inf")

    print(f"\n[bench] {'='*60}", flush=True)
    print(f"[bench] HEADLINE BENCHMARK — Pi0.5 on A100", flush=True)
    print(f"[bench] {'='*60}", flush=True)
    print(f"[bench] Baseline (PyTorch fp32):  {baseline_median:.1f}ms median, {baseline_p95:.1f}ms p95", flush=True)
    print(f"[bench] Triton+Graph (bf16):      {triton_median:.1f}ms median, {triton_p95:.1f}ms p95", flush=True)
    print(f"[bench] SPEEDUP:                  {speedup:.1f}×", flush=True)
    print(f"[bench] {'='*60}", flush=True)
    print(f"[bench] Total time: {time.time()-t_total:.1f}s", flush=True)

    return {
        "baseline_median_ms": baseline_median,
        "baseline_p95_ms": baseline_p95,
        "triton_median_ms": triton_median,
        "triton_p95_ms": triton_p95,
        "speedup_x": speedup,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "device": torch.cuda.get_device_name(0),
        "head_sha": _HEAD,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Lift #5 Headline Benchmark — PyTorch baseline vs Triton+Graph")
    print("=" * 70)
    result = run_headline_bench.remote()
    print("\n" + "=" * 70)
    for k, v in result.items():
        print(f"  {k}={v}")
    print("=" * 70)
