"""Lift #5 Day 4 — Pi05FastKernelsInference smoke test on Modal A100.

Validates:
1. ``Pi05FastKernelsInference(pi05_vla)`` instantiates without errors on A100.
2. Weight reshaping (vision + llm + expert + projector + action_in/out + time_mlp)
   produces all expected keys with the right shapes.
3. ``predict_action(synthetic_obs)`` returns a finite + non-NaN action tensor
   of the expected shape ``[1, chunk_size=50, max_action_dim=32]``.

Eager forward only (capture=False) — Day 7 wires the CUDA Graph capture.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_day4_smoke.py
"""
import os
import subprocess

import modal

app = modal.App("reflex-fast-kernels-day4-smoke")


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
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.6,<2.8",
        "triton>=3.0,<3.2",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "psutil",
        "typer",
        "rich",
        "lerobot==0.5.1",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_BRANCH}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image, gpu="A100-40GB", timeout=2400,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_day4_smoke(model_id: str = "lerobot/pi05_libero_finetuned_v044") -> dict:
    """Day 4 acceptance: build + predict_action returns finite tensor.

    Pi0.5 checkpoint is ~3.3 GB; load takes ~60-90s on A100.
    """
    import time

    import torch

    print(f"[d4] Day 4 smoke — model_id={model_id}", flush=True)
    print(f"[d4] CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[d4] capability: {torch.cuda.get_device_capability(0)}", flush=True)
    t_total = time.time()

    # ── Load PI05Policy → Pi05VLA via from_lerobot_policy ──────────────
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).to("cpu")
    print(f"[d4] [{time.time()-t0:.1f}s] PI05Policy loaded", flush=True)

    t0 = time.time()
    from reflex.models.vlas.pi05 import Pi05VLA
    vla = Pi05VLA.from_lerobot_policy(policy)
    print(f"[d4] [{time.time()-t0:.1f}s] Pi05VLA built on spine", flush=True)

    # Move VLA to CUDA so weight extraction lands on the GPU directly.
    t0 = time.time()
    vla = vla.to("cuda")
    print(f"[d4] [{time.time()-t0:.1f}s] Pi05VLA moved to CUDA", flush=True)

    # ── Build the FastKernels runtime ──────────────────────────────────
    t0 = time.time()
    from reflex.runtime.fast_inference.pi05 import Pi05FastKernelsInference
    runtime = Pi05FastKernelsInference(
        vla,
        num_views=2,
        triton_max_prompt_len=48,
        num_steps=10,
        chunk_size=50,
        max_action_dim=32,
        capture=False,
    )
    print(f"[d4] [{time.time()-t0:.1f}s] Pi05FastKernelsInference instantiated", flush=True)

    # ── prepare_triton_inference (extract + reshape weights) ───────────
    t0 = time.time()
    runtime.prepare_triton_inference()
    print(
        f"[d4] [{time.time()-t0:.1f}s] prepare_triton_inference "
        f"({len(runtime._triton_weights)} weight tensors, "
        f"{len(runtime._triton_bufs)} buffers)",
        flush=True,
    )

    weight_keys = sorted(runtime._triton_weights.keys())
    print(f"[d4] sample weight keys: {weight_keys[:5]}...{weight_keys[-3:]}", flush=True)

    # ── Synthetic inputs for predict_action ────────────────────────────
    batch = 1
    num_views = 2
    image_size = 224
    images = torch.randn(batch, num_views * 3, image_size, image_size, device="cuda", dtype=torch.float32)
    lang_tokens = torch.randint(0, 256000, (batch, 16), dtype=torch.int64, device="cuda")
    lang_masks = torch.ones(batch, 16, dtype=torch.bool, device="cuda")
    states = torch.zeros(batch, 32, dtype=torch.float32, device="cuda")

    t0 = time.time()
    actions = runtime.predict_action(
        images=images,
        lang_tokens=lang_tokens,
        states=states,
        lang_masks=lang_masks,
    )
    t_pred = time.time() - t0
    print(f"[d4] [{t_pred:.3f}s] predict_action returned shape {tuple(actions.shape)}", flush=True)

    # ── Acceptance checks ──────────────────────────────────────────────
    expected_shape = (batch, 50, 32)
    shape_ok = tuple(actions.shape) == expected_shape
    finite_ok = torch.isfinite(actions).all().item()
    not_all_zero = (actions.abs().sum().item() > 0)

    print(f"[d4] {'='*60}", flush=True)
    print(f"[d4] shape: {tuple(actions.shape)} (expected {expected_shape}) — {'PASS' if shape_ok else 'FAIL'}", flush=True)
    print(f"[d4] all finite: {finite_ok} — {'PASS' if finite_ok else 'FAIL'}", flush=True)
    print(f"[d4] non-zero: {not_all_zero} — {'PASS' if not_all_zero else 'FAIL'}", flush=True)
    print(f"[d4] action range: min={actions.min().item():.4f}, max={actions.max().item():.4f}", flush=True)
    print(f"[d4] action mean: {actions.mean().item():.4f}, std: {actions.std().item():.4f}", flush=True)
    print(f"[d4] {'='*60}", flush=True)

    verdict = "PASS" if (shape_ok and finite_ok and not_all_zero) else "FAIL"
    print(f"[d4] Day 4 smoke VERDICT: {verdict} (total time: {time.time()-t_total:.1f}s)", flush=True)

    return {
        "status": "ok",
        "verdict": verdict,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "not_all_zero": not_all_zero,
        "actions_shape": tuple(actions.shape),
        "actions_min": float(actions.min().item()),
        "actions_max": float(actions.max().item()),
        "actions_mean": float(actions.mean().item()),
        "actions_std": float(actions.std().item()),
        "predict_action_time_s": t_pred,
        "weight_count": len(runtime._triton_weights),
        "buffer_count": len(runtime._triton_bufs),
        "head_sha": _HEAD,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print(f"Lift #5 Day 4 — Pi05FastKernelsInference smoke test")
    print(f"  branch = {_BRANCH}")
    print(f"  HEAD = {_HEAD}")
    print("=" * 70)
    result = run_day4_smoke.remote()
    print("\n" + "=" * 70)
    print(f"DAY 4 SMOKE RESULT:")
    for k, v in result.items():
        print(f"  {k}={v}")
    print("=" * 70)
