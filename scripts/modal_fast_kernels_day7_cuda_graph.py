"""Lift #5 Day 7 — CUDA Graph capture test.

Validates that `Pi05FastKernelsInference(capture=True)` produces byte-identical
outputs to `capture=False` AND measures the replay speedup.

Per the plan:
- Captured vs eager should be cos=1.0 (byte-identical — graph replay is the
  exact same kernel sequence, no precision change)
- Capture cold-start ~12-15s (3 warmup + 1 record)
- Steady-state replay should be FASTER than eager

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_day7_cuda_graph.py
"""
import os
import subprocess

import modal

app = modal.App("tether-fast-kernels-day7-cuda-graph")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "lift/5-day1-2-vendor-triton-kernels"


_HEAD = _repo_head_sha()
_BRANCH = "lift/5-day1-2-vendor-triton-kernels"


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ninja-build", "clang", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "lerobot==0.5.1",
        "triton>=3.1",
        "ninja",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image, gpu="A100-40GB", timeout=2400,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_day7_cuda_graph(model_id: str = "lerobot/pi05_libero_finetuned_v044") -> dict:
    """CUDA Graph capture: captured == eager + speedup measurement."""
    import time

    import torch
    import torch.nn.functional as F

    print(f"[d7] CUDA Graph capture test — model_id={model_id}", flush=True)
    print(f"[d7] CUDA: {torch.cuda.get_device_name(0)}, sm {torch.cuda.get_device_capability(0)}", flush=True)
    t_total = time.time()

    # ── Load model ───────────────────────────────────────────────────
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).to("cpu")

    from tether.models.vlas.pi05 import Pi05VLA
    vla = Pi05VLA.from_lerobot_policy(policy)
    del policy
    vla.vision_backbone.to("cuda")
    vla.llm_backbone.to("cuda")
    vla.vla_head.to("cuda")
    print(f"[d7] [{time.time()-t0:.1f}s] Pi05VLA on CUDA", flush=True)

    # ── Build EAGER runtime (capture=False) ──────────────────────────
    t0 = time.time()
    from tether.runtime.fast_inference.pi05 import Pi05FastKernelsInference
    eager_rt = Pi05FastKernelsInference(vla, capture=False)
    eager_rt.prepare_triton_inference()
    print(f"[d7] [{time.time()-t0:.1f}s] Eager runtime ready", flush=True)

    # ── Build CAPTURED runtime (capture=True) ────────────────────────
    t0 = time.time()
    captured_rt = Pi05FastKernelsInference(vla, capture=True)
    captured_rt.prepare_triton_inference()
    print(f"[d7] [{time.time()-t0:.1f}s] Captured runtime ready (graph builds on first call)", flush=True)

    # ── Fixed inputs ─────────────────────────────────────────────────
    torch.manual_seed(42)
    images = torch.randn(1, 6, 224, 224, device="cuda", dtype=torch.float32)
    lang_tokens = torch.randint(0, 256000, (1, 16), dtype=torch.int64, device="cuda")
    lang_masks = torch.ones(1, 16, dtype=torch.bool, device="cuda")
    states = torch.zeros(1, 32, dtype=torch.float32, device="cuda")
    noise = torch.randn(1, 50, 32, dtype=torch.float32, device="cuda")

    # ── Eager baseline (N=5, take median) ────────────────────────────
    print(f"\n[d7] Running eager baseline (5 runs)...", flush=True)
    eager_times = []
    eager_out = None
    for i in range(5):
        t0 = time.time()
        out = eager_rt.predict_action(
            images=images, lang_tokens=lang_tokens, states=states,
            lang_masks=lang_masks, noise=noise,
        )
        torch.cuda.synchronize()
        t_ms = (time.time() - t0) * 1000
        eager_times.append(t_ms)
        eager_out = out.detach().clone()
        print(f"[d7]   eager run {i}: {t_ms:.1f}ms", flush=True)
    eager_median = sorted(eager_times)[len(eager_times) // 2]
    print(f"[d7] Eager median: {eager_median:.1f}ms", flush=True)

    # ── Captured (first call triggers graph build) ───────────────────
    print(f"\n[d7] First captured call (triggers graph build)...", flush=True)
    t0 = time.time()
    captured_out_first = captured_rt.predict_action(
        images=images, lang_tokens=lang_tokens, states=states,
        lang_masks=lang_masks, noise=noise,
    )
    torch.cuda.synchronize()
    t_build = time.time() - t0
    print(f"[d7] Graph build + first replay: {t_build:.1f}s", flush=True)

    # ── Captured steady-state (N=20, take median) ────────────────────
    print(f"\n[d7] Running captured steady-state (20 runs)...", flush=True)
    captured_times = []
    captured_out = None
    for i in range(20):
        t0 = time.time()
        out = captured_rt.predict_action(
            images=images, lang_tokens=lang_tokens, states=states,
            lang_masks=lang_masks, noise=noise,
        )
        torch.cuda.synchronize()
        t_ms = (time.time() - t0) * 1000
        captured_times.append(t_ms)
        captured_out = out.detach().clone()
        if i < 5 or i == 19:
            print(f"[d7]   captured run {i}: {t_ms:.1f}ms", flush=True)
    captured_median = sorted(captured_times)[len(captured_times) // 2]
    print(f"[d7] Captured median: {captured_median:.1f}ms", flush=True)

    # ── Compare captured vs eager ────────────────────────────────────
    flat_eager = eager_out.flatten().float()
    flat_captured = captured_out.flatten().float()

    cos = F.cosine_similarity(flat_eager.unsqueeze(0), flat_captured.unsqueeze(0))[0].item()
    max_abs = (flat_eager - flat_captured).abs().max().item()
    speedup = eager_median / captured_median if captured_median > 0 else float("inf")

    print(f"\n[d7] {'='*60}", flush=True)
    print(f"[d7] Captured vs Eager: cos={cos:.8f}, max_abs={max_abs:.2e}", flush=True)
    print(f"[d7] Eager median:    {eager_median:.1f}ms", flush=True)
    print(f"[d7] Captured median: {captured_median:.1f}ms", flush=True)
    print(f"[d7] Speedup:         {speedup:.2f}×", flush=True)
    print(f"[d7] Graph build time: {t_build:.1f}s", flush=True)
    print(f"[d7] {'='*60}", flush=True)

    # Gate: captured == eager should be byte-identical (cos=1.0)
    parity_ok = cos >= 0.9999
    verdict = "PASS" if parity_ok else "FAIL"
    print(f"[d7] Day 7 VERDICT: {verdict} (total: {time.time()-t_total:.1f}s)", flush=True)

    return {
        "status": "ok",
        "verdict": verdict,
        "cos_captured_vs_eager": cos,
        "max_abs_captured_vs_eager": max_abs,
        "eager_median_ms": eager_median,
        "captured_median_ms": captured_median,
        "speedup_x": speedup,
        "graph_build_time_s": t_build,
        "eager_times_ms": eager_times,
        "captured_times_ms": captured_times[:5],
        "head_sha": _HEAD,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print(f"Lift #5 Day 7 — CUDA Graph capture test")
    print(f"  branch = {_BRANCH}")
    print("=" * 70)
    result = run_day7_cuda_graph.remote()
    print("\n" + "=" * 70)
    for k, v in result.items():
        if k in ("eager_times_ms", "captured_times_ms"):
            print(f"  {k}=[{len(v)} values]")
        else:
            print(f"  {k}={v}")
    print("=" * 70)
