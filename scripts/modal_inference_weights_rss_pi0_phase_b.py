"""Lift #3 Phase B — Pi0 safetensors-direct RSS validation gate.

Validates the Phase B claim: `Pi0VLA.flat_dict_from_safetensors(path)` produces
a flat dict on CUDA with peak RSS ≥30% lower than the Phase A path that
extracts via `from_lerobot_policy(policy).prepare_inference_weights()`.

Compared paths on the same `lerobot/pi0_base` checkpoint:

- **PATH A** (Phase A): `PI0Policy.from_pretrained` → `Pi0VLA.from_lerobot_policy(policy)`
  → `prepare_inference_weights()`. Holds both source nn.Module + cloned flat dict
  at peak. This is the path the failed PR #174 bench measured at -15.7%.

- **PATH C** (Phase B): `Pi0VLA.flat_dict_from_safetensors(safetensors_path)`.
  Reads safetensors header → allocates bf16 CUDA tensors directly. No nn.Module,
  no nn.Parameter wrapping, no clone.

Acceptance: PATH C peak RSS ≤ PATH A peak RSS × 0.70 (≥30% reduction).

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_inference_weights_rss_pi0_phase_b.py
"""
import os
import subprocess
import time

import modal

app = modal.App("tether-inference-weights-rss-pi0-phase-b")


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "lift/3-phase-b-safetensors-direct-loader"


_HEAD = _repo_head_sha()
_BRANCH = "lift/3-phase-b-safetensors-direct-loader"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "lerobot==0.5.1",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_BRANCH}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image, gpu="A100-40GB", timeout=2400,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_rss_phase_b_pi0(model_id: str = "lerobot/pi0_base") -> dict:
    """Measure peak RSS for Phase A (extract from loaded nn.Module) vs
    Phase B (safetensors → flat dict direct) on Pi0VLA.
    """
    import gc
    import os
    import time

    import psutil
    import torch

    def _rss_mb():
        return psutil.Process(os.getpid()).memory_info().rss / 1e6

    print(f"[rss] Pi0 Phase B safetensors-direct RSS benchmark — model_id={model_id}", flush=True)
    print(f"[rss] Process PID: {os.getpid()}", flush=True)
    print(f"[rss] CUDA available: {torch.cuda.is_available()}", flush=True)

    rss_initial = _rss_mb()
    print(f"[rss] Initial RSS: {rss_initial:.1f} MB", flush=True)

    # ── Pre-fetch the safetensors file so the cache is warm for both paths ──
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    safetensors_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    print(f"[rss] [{time.time()-t0:.1f}s] safetensors downloaded to {safetensors_path}", flush=True)

    # ── PATH A: Phase A (extract from loaded nn.Module) ──────────────────
    print(f"\n[rss] PATH A: Phase A path (Pi0VLA.from_lerobot_policy → prepare_inference_weights)", flush=True)
    t_a_start = time.time()

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).to("cpu")
    rss_after_policy = _rss_mb()
    print(f"[rss]   [{time.time()-t_a_start:.1f}s] policy loaded, RSS={rss_after_policy:.1f} MB", flush=True)

    from tether.models.vlas.pi0 import Pi0VLA
    vla_a = Pi0VLA.from_lerobot_policy(policy)
    rss_after_vla = _rss_mb()
    print(f"[rss]   [{time.time()-t_a_start:.1f}s] Pi0VLA built, RSS={rss_after_vla:.1f} MB", flush=True)

    flat_a = vla_a.prepare_inference_weights(prefix="")
    rss_after_flat_a = _rss_mb()
    print(f"[rss]   [{time.time()-t_a_start:.1f}s] flat_a built ({len(flat_a)} tensors), RSS={rss_after_flat_a:.1f} MB", flush=True)

    # Peak A is the max RSS observed during PATH A — at the moment the flat dict exists alongside nn.Module
    peak_a = max(rss_after_policy, rss_after_vla, rss_after_flat_a)
    print(f"[rss] PATH A peak RSS: {peak_a:.1f} MB ({len(flat_a)} tensors in flat dict)", flush=True)

    # Free PATH A before PATH C
    del flat_a, vla_a, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rss_after_a_free = _rss_mb()
    print(f"[rss]   after freeing PATH A, RSS={rss_after_a_free:.1f} MB", flush=True)

    # ── PATH C: Phase B (safetensors → flat dict direct) ─────────────────
    print(f"\n[rss] PATH C: Phase B path (Pi0VLA.flat_dict_from_safetensors)", flush=True)
    t_c_start = time.time()

    # Use cuda device so tensors land on GPU (the actual Phase B deploy target)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flat_c = Pi0VLA.flat_dict_from_safetensors(
        safetensors_path,
        dtype=torch.bfloat16,
        device=device,
        device_id=0,
    )
    rss_after_flat_c = _rss_mb()
    print(f"[rss]   [{time.time()-t_c_start:.1f}s] flat_c built ({len(flat_c)} tensors), RSS={rss_after_flat_c:.1f} MB", flush=True)

    peak_c = rss_after_flat_c
    print(f"[rss] PATH C peak RSS: {peak_c:.1f} MB ({len(flat_c)} tensors in flat dict)", flush=True)

    # ── Compare ──────────────────────────────────────────────────────────
    delta_mb = peak_a - peak_c
    delta_pct = (delta_mb / peak_a) * 100 if peak_a > 0 else 0

    print(f"\n[rss] {'=' * 60}", flush=True)
    print(f"[rss] PATH A (Phase A) peak RSS: {peak_a:.1f} MB  [nn.Module + flat dict resident]", flush=True)
    print(f"[rss] PATH C (Phase B) peak RSS: {peak_c:.1f} MB  [flat dict only, no nn.Module]", flush=True)
    print(f"[rss] Delta:                     {delta_mb:+.1f} MB ({delta_pct:+.1f}%)", flush=True)
    print(f"[rss] {'=' * 60}", flush=True)

    verdict = "PASS" if delta_pct >= 30 else ("BORDERLINE" if delta_pct >= 20 else "FAIL")
    print(f"[rss] VERDICT: {verdict} (gate: ≥30%)", flush=True)

    return {
        "status": "ok",
        "model_id": model_id,
        "path_a_peak_mb": peak_a,
        "path_c_peak_mb": peak_c,
        "delta_mb": delta_mb,
        "delta_pct": delta_pct,
        "verdict": verdict,
        "flat_tensor_count_a": -1,  # freed before C measurement
        "flat_tensor_count_c": len(flat_c),
        "head_sha": _HEAD,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("Lift #3 Phase B — Pi0 safetensors-direct RSS validation gate")
    print(f"  HEAD = {_HEAD}")
    print(f"  branch = {_BRANCH}")
    print("=" * 70)

    print("\n--- Pi0 (lerobot/pi0_base) ---")
    result = run_rss_phase_b_pi0.remote("lerobot/pi0_base")

    print("\n" + "=" * 70)
    print(f"PI0 PHASE B RESULT:")
    for k, v in result.items():
        print(f"  {k}={v}")
    print("=" * 70)
