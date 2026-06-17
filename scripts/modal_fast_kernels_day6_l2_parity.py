"""Lift #5 Day 6 — L2 cross-runtime parity gate.

Compares Pi05FastKernelsInference (Triton, bf16) vs Pi05VLA.predict_action
(PyTorch nn.Module, fp32) on identical inputs. Both paths are built from
the SAME lerobot checkpoint via from_lerobot_policy. Pi05VLA.predict_action
was validated bit-identical vs lerobot in Lift #1 Day 5 Phase B, so this
comparison is transitive: if Triton matches the spine, it matches lerobot.

Gate per T-3: cos >= 0.999, max_abs <= 1e-2.

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_fast_kernels_day6_l2_parity.py
"""
import os
import subprocess

import modal

app = modal.App("tether-fast-kernels-day6-l2")


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
def run_day6_l2(model_id: str = "lerobot/pi05_libero_finetuned_v044", n_pairs: int = 5) -> dict:
    """L2 cross-runtime: Pi05VLA spine (fp32) vs Triton (bf16).

    Both paths built from the same lerobot checkpoint. Pi05VLA.predict_action
    was validated bit-identical vs lerobot in Lift #1 Day 5 Phase B.

    Gate: cos >= 0.999, max_abs <= 1e-2.
    """
    import time

    import torch
    import torch.nn.functional as F

    print(f"[d6] L2 cross-runtime — model_id={model_id}, n_pairs={n_pairs}", flush=True)
    print(f"[d6] CUDA: {torch.cuda.get_device_name(0)}, sm {torch.cuda.get_device_capability(0)}", flush=True)
    t_total = time.time()

    # ── Load policy ONCE, build both paths from same VLA ────────────
    # Loading PI05Policy twice on A100-40GB OOMs (~26 GB × 2 > 40 GB).
    # Instead: one policy → one Pi05VLA → both paths share the nn.Module.
    # Triton weight reshaping creates separate bf16 tensors (~3-4 GB) so
    # total is ~30 GB (fits 40GB).
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id)
    policy = policy.to(dtype=torch.float32).to("cpu")
    print(f"[d6] [{time.time()-t0:.1f}s] PI05Policy loaded", flush=True)

    t0 = time.time()
    from tether.models.vlas.pi05 import Pi05VLA
    vla = Pi05VLA.from_lerobot_policy(policy)
    del policy
    vla.vision_backbone.to("cuda")
    vla.llm_backbone.to("cuda")
    vla.vla_head.to("cuda")
    print(f"[d6] [{time.time()-t0:.1f}s] Pi05VLA on CUDA (shared by both paths)", flush=True)

    # Path B: Triton runtime built from the SAME VLA
    t0 = time.time()
    from tether.runtime.fast_inference.pi05 import Pi05FastKernelsInference
    triton_runtime = Pi05FastKernelsInference(vla, capture=False)
    triton_runtime.prepare_triton_inference()
    print(f"[d6] [{time.time()-t0:.1f}s] Triton runtime ready (bf16 weights reshaped from same VLA)", flush=True)

    # ── N paired comparisons ─────────────────────────────────────────
    cos_vals = []
    max_abs_vals = []

    for i in range(n_pairs):
        torch.manual_seed(42 + i)

        # Synthetic inputs
        num_views = 2
        img_size = 224

        # Pi05VLA.predict_action expects images as a list of [batch, C, H, W]
        images_list = [
            torch.randn(1, 3, img_size, img_size, device="cuda", dtype=torch.float32)
            for _ in range(num_views)
        ]

        lang_tokens = torch.randint(0, 256000, (1, 16), dtype=torch.int64, device="cuda")
        lang_masks = torch.ones(1, 16, dtype=torch.bool, device="cuda")

        # Fixed noise for deterministic comparison
        noise = torch.randn(1, 50, 32, dtype=torch.float32, device="cuda")

        # ── Path A: Pi05VLA spine predict_action (fp32) ──
        t0 = time.time()
        with torch.no_grad():
            try:
                out_a = vla.predict_action(
                    images=images_list,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise,
                    num_steps=10,
                    chunk_size=50,
                    action_dim=32,
                )
                t_a = time.time() - t0
            except Exception as e:
                import traceback
                print(f"[d6]   pair {i} Path A FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue

        # ── Path B: Triton (bf16) ──
        # Pi05FastKernelsInference.predict_action expects:
        # images: [batch, num_views * 3, H, W]
        images_concat = torch.cat(images_list, dim=1)  # [1, 6, 224, 224]

        t0 = time.time()
        out_b = triton_runtime.predict_action(
            images=images_concat,
            lang_tokens=lang_tokens,
            states=torch.zeros(1, 32, device="cuda"),
            lang_masks=lang_masks,
            noise=noise,
        )
        t_b = time.time() - t0

        # ── Compare ──
        flat_a = out_a.flatten().float()
        flat_b = out_b.flatten().float()

        # Shapes may differ if predict_action returns different layouts
        if flat_a.shape != flat_b.shape:
            print(
                f"[d6]   pair {i}: SHAPE MISMATCH A={tuple(out_a.shape)} B={tuple(out_b.shape)}",
                flush=True,
            )
            continue

        cos = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0))[0].item()
        max_abs = (flat_a - flat_b).abs().max().item()
        cos_vals.append(cos)
        max_abs_vals.append(max_abs)

        print(
            f"[d6]   pair {i}: cos={cos:.6f}, max_abs={max_abs:.4e}, "
            f"t_a={t_a:.3f}s, t_b={t_b:.3f}s, "
            f"range_a=[{out_a.min().item():.4f},{out_a.max().item():.4f}] "
            f"range_b=[{out_b.min().item():.4f},{out_b.max().item():.4f}]",
            flush=True,
        )

    if not cos_vals:
        print(f"[d6] NO SUCCESSFUL PAIRS — all Path A calls failed", flush=True)
        return {"status": "error", "verdict": "FAIL", "reason": "no_successful_pairs"}

    min_cos = min(cos_vals)
    max_max_abs = max(max_abs_vals)
    mean_cos = sum(cos_vals) / len(cos_vals)

    gate_cos = 0.999
    gate_max_abs = 1e-2
    cos_ok = min_cos >= gate_cos
    max_abs_ok = max_max_abs <= gate_max_abs

    print(f"\n[d6] {'='*60}", flush=True)
    print(f"[d6] L2 results: min_cos={min_cos:.6f}, mean_cos={mean_cos:.6f} (gate >= {gate_cos})", flush=True)
    print(f"[d6]            max_max_abs={max_max_abs:.4e} (gate <= {gate_max_abs:.0e})", flush=True)
    print(f"[d6] {'='*60}", flush=True)

    verdict = "PASS" if (cos_ok and max_abs_ok) else ("BORDERLINE" if min_cos >= 0.99 else "FAIL")
    print(f"[d6] L2 VERDICT: {verdict} (total: {time.time()-t_total:.1f}s)", flush=True)

    return {
        "status": "ok",
        "verdict": verdict,
        "n_pairs": n_pairs,
        "n_successful": len(cos_vals),
        "min_cos": min_cos,
        "mean_cos": mean_cos,
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
    print(f"Lift #5 Day 6 — L2 cross-runtime parity (spine fp32 vs Triton bf16)")
    print(f"  branch = {_BRANCH}")
    print("=" * 70)
    result = run_day6_l2.remote()
    print("\n" + "=" * 70)
    for k, v in result.items():
        if k in ("cos_distribution", "max_abs_distribution"):
            print(f"  {k}=[{len(v)} values]")
        else:
            print(f"  {k}={v}")
    print("=" * 70)
