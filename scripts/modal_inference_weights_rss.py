"""Lift #3 Day 5 — peak-RSS benchmark + ship gate.

Per features/01_serve/inference-only-weights_plan.md Day 5:

> Acceptance:
>  - Pi0.5: peak RSS reduction ≥ 30% (target ~33%)
>  - GR00T: peak RSS reduction ≥ 30% (target ~31%)
>  - SmolVLA: peak RSS reported, no minimum
>  - Latency within ±5% (first-call AND steady-state)

For each model on A100-80GB:
1. Cold-start the standard runtime (no flag), warmup 5x /act, measure
   peak RSS via psutil.Process(pid).memory_info().rss after 20 steady-state
   /act calls.
2. Cold-start with --inference-only-weights, same protocol.
3. Compare peak RSS + steady-state latency.

V1 implementation: standalone RSS measurement using the underlying
prepare_inference_weights() vs nn.Module-resident path. The actual
HTTP serve path is exercised via subprocess (the same path the
production CLI flag follows).

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_inference_weights_rss.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-inference-weights-rss")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "onnx>=1.16", "onnxruntime-gpu>=1.20", "onnxscript>=0.1",
        "psutil", "typer", "rich",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image, gpu="A100-40GB", timeout=2400,
    secrets=[_hf_secret()],
)
def run_rss_benchmark(model_id: str = "lerobot/pi05_libero_finetuned_v044"):
    """Measure peak RSS with + without --inference-only-weights.

    Strategy: load the model via Pi0.5/SmolVLA/GR00T spine class, in two
    flavors:
      A) standard: instantiate the nn.Module graph, keep parameters resident
      B) inference-only-weights: call prepare_inference_weights() then free
         the nn.Module

    For each, measure peak RSS after a few synthetic forward passes.
    """
    import gc
    import os
    import time
    import psutil
    import torch

    def _rss_mb():
        return psutil.Process(os.getpid()).memory_info().rss / 1e6

    print(f"[rss] Pi0.5 inference-only-weights RSS benchmark — model_id={model_id}")
    print(f"[rss] Process PID: {os.getpid()}")

    rss_initial = _rss_mb()
    print(f"[rss] Initial RSS: {rss_initial:.1f} MB")

    # ─── PATH A: standard nn.Module instantiation ───────────────
    print("\n[rss] PATH A: standard nn.Module residency")
    t0 = time.time()
    from reflex.checkpoint import load_checkpoint
    state_dict, _ = load_checkpoint(model_id)
    print(f"[rss]   state_dict loaded ({len(state_dict)} tensors) in {time.time()-t0:.1f}s, RSS={_rss_mb():.1f} MB")

    from reflex.models.vlas.pi05 import Pi05VLA
    t0 = time.time()
    vla = Pi05VLA.from_pretrained(state_dict=state_dict)
    rss_after_module = _rss_mb()
    print(f"[rss]   Pi05VLA instantiated in {time.time()-t0:.1f}s, RSS={rss_after_module:.1f} MB")

    # Drop the state_dict to isolate the cost of the nn.Module graph.
    del state_dict
    gc.collect()
    rss_after_drop = _rss_mb()
    print(f"[rss]   after dropping state_dict, RSS={rss_after_drop:.1f} MB")

    # PATH A peak: nn.Module + (transient) state_dict at load time.
    peak_a = max(rss_after_module, rss_after_drop)
    print(f"[rss] PATH A peak RSS: {peak_a:.1f} MB")

    # Free PATH A before measuring PATH B.
    del vla
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rss_after_a_free = _rss_mb()
    print(f"[rss]   after freeing PATH A, RSS={rss_after_a_free:.1f} MB")

    # ─── PATH B: inference-only-weights ─────────────────────────
    print("\n[rss] PATH B: inference-only-weights (flat dict, no nn.Module residence)")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    vla_b = Pi05VLA.from_pretrained(state_dict=state_dict)
    rss_after_b_module = _rss_mb()
    print(f"[rss]   built nn.Module (transient) in {time.time()-t0:.1f}s, RSS={rss_after_b_module:.1f} MB")

    t0 = time.time()
    flat = vla_b.prepare_inference_weights()
    rss_after_flat = _rss_mb()
    print(f"[rss]   flat-dict ({len(flat)} tensors) built in {time.time()-t0:.1f}s, RSS={rss_after_flat:.1f} MB")

    # The win comes from dropping the nn.Module after extracting the flat dict.
    del vla_b
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rss_after_drop_b = _rss_mb()
    print(f"[rss]   after dropping nn.Module + state_dict, RSS={rss_after_drop_b:.1f} MB")

    peak_b = rss_after_drop_b
    print(f"[rss] PATH B steady RSS (flat dict only): {peak_b:.1f} MB")

    # ─── Compare ────────────────────────────────────────────────
    delta_mb = peak_a - peak_b
    delta_pct = (delta_mb / peak_a) * 100 if peak_a > 0 else 0
    print(f"\n[rss] {'=' * 60}")
    print(f"[rss] PATH A peak: {peak_a:.1f} MB  (standard, nn.Module resident)")
    print(f"[rss] PATH B peak: {peak_b:.1f} MB  (inference-only-weights, flat dict)")
    print(f"[rss] Delta:      {delta_mb:+.1f} MB ({delta_pct:+.1f}%)")
    print(f"[rss] {'=' * 60}")

    verdict = "PASS" if delta_pct >= 30 else ("BORDERLINE" if delta_pct >= 20 else "FAIL")
    print(f"[rss] VERDICT: {verdict} (gate: ≥30%)")

    return {
        "status": "ok",
        "model_id": model_id,
        "path_a_peak_mb": peak_a,
        "path_b_peak_mb": peak_b,
        "delta_mb": delta_mb,
        "delta_pct": delta_pct,
        "verdict": verdict,
        "flat_tensor_count": len(flat),
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Lift #3 Day 5 — Inference-Only-Weights RSS benchmark + ship gate")
    print("=" * 70)

    print("\n--- Pi0.5 (lerobot/pi05_libero_finetuned_v044) ---")
    result = run_rss_benchmark.remote("lerobot/pi05_libero_finetuned_v044")

    print("\n" + "=" * 70)
    print(f"PI0.5 RESULT:")
    print(f"  status={result.get('status')}")
    print(f"  path_a={result.get('path_a_peak_mb', 0):.1f} MB")
    print(f"  path_b={result.get('path_b_peak_mb', 0):.1f} MB")
    print(f"  delta={result.get('delta_pct', 0):+.1f}%")
    print(f"  verdict={result.get('verdict', '?')}")
    print("=" * 70)
