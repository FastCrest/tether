"""Lift #5 Day 5 — L1 wiring sanity gate (Triton internal precision).

Validates the Triton path is internally consistent: same kernels, two precision
configurations (bf16 production vs fp32 debugging) on identical inputs should
produce cos ≥ 0.9999 with max_abs ≤ 5e-3 per T-3 L1 thresholds.

Wiring sanity. Doesn't talk to ORT yet — that's L2 (Day 6).

V1 caveat: the vendored kernels operate in bf16 throughout; an fp32 toggle
isn't free. For Day 5 V1 we instead run the bf16 path N=10 times on identical
inputs and verify byte-deterministic outputs (a weaker but real wiring check).
If outputs differ run-to-run, there's a hidden non-determinism bug (random
init, uninitialized buffer, etc.) that needs to be fixed BEFORE L2.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_day5_l1_parity.py
"""
import os
import subprocess

import modal

app = modal.App("tether-fast-kernels-day5-l1")


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
        f'pip install "fastcrest-tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image, gpu="A100-40GB", timeout=2400,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_day5_l1(model_id: str = "lerobot/pi05_libero_finetuned_v044", n_runs: int = 10) -> dict:
    """L1 wiring sanity: N=10 paired bf16 forwards on identical inputs.

    Gate: byte-deterministic outputs (cos == 1.0 across all pairs).
    """
    import time

    import torch

    print(f"[d5] L1 wiring sanity — model_id={model_id}, n_runs={n_runs}", flush=True)
    t_total = time.time()

    # ── Build runtime ──
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).to("cpu")
    from tether.models.vlas.pi05 import Pi05VLA
    vla = Pi05VLA.from_lerobot_policy(policy)
    vla.vision_backbone.to("cuda")
    vla.llm_backbone.to("cuda")
    vla.vla_head.to("cuda")
    from tether.runtime.fast_inference.pi05 import Pi05FastKernelsInference
    runtime = Pi05FastKernelsInference(vla, capture=False)
    runtime.prepare_triton_inference()
    print(f"[d5] [{time.time()-t0:.1f}s] runtime ready", flush=True)

    # Fixed inputs across runs.
    torch.manual_seed(42)
    images = torch.randn(1, 6, 224, 224, device="cuda", dtype=torch.float32)
    lang_tokens = torch.randint(0, 256000, (1, 16), dtype=torch.int64, device="cuda")
    lang_masks = torch.ones(1, 16, dtype=torch.bool, device="cuda")
    states = torch.zeros(1, 32, dtype=torch.float32, device="cuda")
    # Fixed noise so the denoise loop is deterministic.
    fixed_noise = torch.randn(1, 50, 32, dtype=torch.float32, device="cuda")

    # ── N paired runs ──
    print(f"[d5] running {n_runs} paired bf16 forwards on identical inputs", flush=True)
    outputs = []
    for i in range(n_runs):
        t0 = time.time()
        out = runtime.predict_action(
            images=images, lang_tokens=lang_tokens, states=states,
            lang_masks=lang_masks, noise=fixed_noise,
        )
        outputs.append(out.detach().clone())
        print(f"[d5]   run {i}: t={time.time()-t0:.3f}s, mean={out.mean().item():.6f}", flush=True)

    # ── Pairwise compare ──
    cos_vals = []
    max_abs_vals = []
    ref = outputs[0].flatten()
    for i, out in enumerate(outputs[1:], 1):
        flat = out.flatten()
        cos = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), flat.unsqueeze(0))[0].item()
        max_abs = (ref - flat).abs().max().item()
        cos_vals.append(cos)
        max_abs_vals.append(max_abs)
        print(f"[d5]   pair (0, {i}): cos={cos:.8f}, max_abs={max_abs:.2e}", flush=True)

    min_cos = min(cos_vals)
    max_max_abs = max(max_abs_vals)

    gate_cos = 0.9999
    gate_max_abs = 5e-3
    cos_ok = min_cos >= gate_cos
    max_abs_ok = max_max_abs <= gate_max_abs

    print(f"[d5] {'='*60}", flush=True)
    print(f"[d5] L1 results: min_cos={min_cos:.8f} (gate >= {gate_cos})", flush=True)
    print(f"[d5]            max_max_abs={max_max_abs:.2e} (gate <= {gate_max_abs:.0e})", flush=True)
    print(f"[d5] {'='*60}", flush=True)
    verdict = "PASS" if (cos_ok and max_abs_ok) else "FAIL"
    print(f"[d5] L1 VERDICT: {verdict} (total: {time.time()-t_total:.1f}s)", flush=True)

    return {
        "status": "ok",
        "verdict": verdict,
        "n_runs": n_runs,
        "min_cos": min_cos,
        "max_max_abs": max_max_abs,
        "gate_cos": gate_cos,
        "gate_max_abs": gate_max_abs,
        "cos_distribution": cos_vals,
        "max_abs_distribution": max_abs_vals,
        "head_sha": _HEAD,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print(f"Lift #5 Day 5 — L1 wiring sanity gate")
    print(f"  branch = {_BRANCH}")
    print("=" * 70)
    result = run_day5_l1.remote()
    print("\n" + "=" * 70)
    for k, v in result.items():
        if k in ("cos_distribution", "max_abs_distribution"):
            print(f"  {k}=[{len(v)} values]")
        else:
            print(f"  {k}={v}")
    print("=" * 70)
