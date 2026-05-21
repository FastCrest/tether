"""Lift #1 Day 4h — Pi0VLA.predict_action vs lerobot PI0Policy parity gate.

Validates that Pi0VLA.predict_action (the new BaseVLA-spine path, shipped
Day 4g Phase B) produces bit-identical actions to lerobot's PI0Policy
upstream reference on the same inputs + same noise seed.

Pass criteria:
    max abs error  <  1e-4   (bit-identical for fp32 inference)
    p95 abs error  <  1e-5

If divergence is observed, investigate root cause (per CLAUDE.md "no
band-aids") — do NOT widen the tolerance.

Usage:
    modal run scripts/modal_pi0_predict_action_parity.py
    modal run scripts/modal_pi0_predict_action_parity.py --num-steps 10 --chunk-size 50

Hardware:    A10G (~$1.10/hr)
Cold start:  ~2 min (PaliGemma 3B + lerobot + LIBERO-less deps)
Wall clock:  ~3-5 min total
Spend:       ~$1.50 per run
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import modal


def _hf_secret():
    """HF token secret (PaliGemma + lerobot/pi0_base are gated)."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
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


# Image: lerobot==0.5.1 (upstream PI0Policy) + reflex-vla (Pi0VLA spine).
# No LIBERO / MuJoCo needed — we're doing one forward pass, not a rollout.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",  # match modal_libero_lerobot_native.py pin
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "lerobot==0.5.1",
        "num2words",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


app = modal.App("reflex-pi0-spine-parity")
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[_hf_secret()],
    volumes={"/root/.cache/huggingface": _hf_cache_volume},
)
def run_parity(
    num_steps: int = 10,
    chunk_size: int = 50,
    noise_seed: int = 99,
    input_seed: int = 42,
) -> dict:
    """Single-shot parity: PI0Policy.predict_action_chunk vs Pi0VLA.predict_action.

    Both pipelines fed identical preprocessed inputs + identical noise.
    Returns the error distribution + verdict.
    """
    import time

    import numpy as np
    import torch

    results: dict = {"num_steps": num_steps, "chunk_size": chunk_size, "steps": []}

    def step(name: str, status: str, detail: str = ""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "INFO")
        print(f"[{tag}] {name} — {detail}", flush=True)

    # ─── transformers-version patches for lerobot PI0Policy ────────────
    # Mirror local_pi0_monolithic_parity.py:11-58 — required for lerobot
    # to load on transformers 4.51+.
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        _stub = types.ModuleType(_mod)
        _stub.GrootPolicy = None
        _stub.GR00TN15 = None
        sys.modules[_mod] = _stub

    def _patch_pi0_for_transformers_457():
        from lerobot.policies.pi0 import modeling_pi0

        def patched_embed_image(self, image):
            out_dtype = image.dtype
            if image.dtype != torch.float32:
                image = image.to(torch.float32)
            image_outputs = self.paligemma.model.get_image_features(image)
            if hasattr(image_outputs, "pooler_output"):
                features = image_outputs.pooler_output
            else:
                features = image_outputs
            features = features * self.paligemma.config.text_config.hidden_size ** 0.5
            if features.dtype != out_dtype:
                features = features.to(out_dtype)
            return features

        modeling_pi0.PaliGemmaWithExpertModel.embed_image = patched_embed_image

    def _patch_create_causal_mask_kwarg():
        from transformers import masking_utils
        original = masking_utils.create_causal_mask

        def shim(*args, **kwargs):
            if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            return original(*args, **kwargs)

        masking_utils.create_causal_mask = shim
        try:
            from lerobot.policies import pi_gemma
            if hasattr(pi_gemma, "create_causal_mask"):
                pi_gemma.create_causal_mask = shim
        except ImportError:
            pass

    _patch_pi0_for_transformers_457()
    _patch_create_causal_mask_kwarg()
    step("patches", "pass", "lerobot patched for transformers 4.51+")

    # ─── 1. Load lerobot PI0Policy (the oracle) ────────────────────────
    print("\n=== Step 1: Load lerobot PI0Policy ===", flush=True)
    start = time.time()
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download

    policy = PI0Policy.from_pretrained("lerobot/pi0_base").eval()
    policy = policy.to(dtype=torch.float32).to("cpu")  # CPU + fp32 for max determinism
    step("load_lerobot", "pass", f"{time.time() - start:.1f}s, params={sum(p.numel() for p in policy.parameters())/1e9:.2f}B")

    # ─── 2. Build the input batch + preprocess ─────────────────────────
    print("\n=== Step 2: Build deterministic input batch ===", flush=True)
    rng = np.random.RandomState(input_seed)
    img_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0  # [-1, 1] SigLIP normalization
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)

    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": ["pick up the red bowl"],
    }
    repo = snapshot_download("lerobot/pi0_base")
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch_raw)
    step("preprocess", "pass", f"seed={input_seed}, state.shape={state.shape}")

    # ─── 3. Generate shared noise (the only stochastic input) ──────────
    cfg = policy.config
    action_dim = cfg.max_action_dim  # pi0 padded action dim = 32
    noise_np = np.random.RandomState(noise_seed).randn(1, chunk_size, action_dim).astype(np.float32)
    noise = torch.from_numpy(noise_np)
    step("noise", "pass", f"seed={noise_seed}, shape={tuple(noise.shape)}")

    # ─── 4. Run lerobot PI0Policy (oracle) ─────────────────────────────
    print("\n=== Step 4: lerobot PI0Policy.predict_action_chunk (oracle) ===", flush=True)
    start = time.time()
    with torch.no_grad():
        oracle_actions = policy.predict_action_chunk(batch_pp, noise=noise.clone())
    oracle_actions = oracle_actions.cpu().numpy() if hasattr(oracle_actions, "cpu") else np.asarray(oracle_actions)
    step("lerobot_forward", "pass", f"{time.time() - start:.1f}s, shape={oracle_actions.shape}, first={oracle_actions[0, 0, :5]}")

    # Extract the SAME tensors PI0Policy uses internally — we feed these to Pi0VLA.
    images, img_masks = policy._preprocess_images(batch_pp)
    lang_tokens = batch_pp["observation.language.tokens"]
    lang_masks = batch_pp["observation.language.attention_mask"]
    state_tensor = policy.prepare_state(batch_pp)
    step("extract_inputs", "pass",
         f"images={[tuple(i.shape) for i in images]}, lang_tokens={tuple(lang_tokens.shape)}, "
         f"state={tuple(state_tensor.shape)}")

    # Free lerobot to make room for Pi0VLA
    del policy
    import gc
    gc.collect()

    # ─── 5. Build Pi0VLA via from_pretrained ───────────────────────────
    print("\n=== Step 5: Build Pi0VLA (BaseVLA spine) ===", flush=True)
    start = time.time()
    from reflex.models.vlas.pi0 import Pi0VLA

    vla = Pi0VLA.from_pretrained("lerobot/pi0_base")
    # Ensure same dtype + device as oracle
    for module in [vla.vision_backbone, vla.llm_backbone, vla.projector, vla.vla_head]:
        module.to(dtype=torch.float32).to("cpu")
    step("build_vla", "pass",
         f"{time.time() - start:.1f}s, slots wired: vision/llm/projector/head")

    # ─── 6. Run Pi0VLA.predict_action ──────────────────────────────────
    print("\n=== Step 6: Pi0VLA.predict_action ===", flush=True)
    start = time.time()
    with torch.no_grad():
        vla_actions = vla.predict_action(
            images=images,
            image_masks=img_masks,
            state=state_tensor,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise.clone(),
            num_steps=num_steps,
            chunk_size=chunk_size,
        )
    vla_actions = vla_actions.cpu().numpy()
    step("vla_forward", "pass", f"{time.time() - start:.1f}s, shape={vla_actions.shape}, first={vla_actions[0, 0, :5]}")

    # ─── 7. Compare ────────────────────────────────────────────────────
    print("\n=== Step 7: Parity comparison ===", flush=True)
    if oracle_actions.shape != vla_actions.shape:
        step("compare", "fail", f"shape mismatch: oracle={oracle_actions.shape}, vla={vla_actions.shape}")
        return results

    diff = oracle_actions - vla_actions
    abs_diff = np.abs(diff).flatten()
    err_mean = float(abs_diff.mean())
    err_p50 = float(np.percentile(abs_diff, 50))
    err_p95 = float(np.percentile(abs_diff, 95))
    err_p99 = float(np.percentile(abs_diff, 99))
    err_max = float(abs_diff.max())

    # Cosine similarity on first action (for sanity)
    first_oracle = oracle_actions[0, 0]
    first_vla = vla_actions[0, 0]
    cos = float(
        np.dot(first_oracle, first_vla)
        / (np.linalg.norm(first_oracle) * np.linalg.norm(first_vla) + 1e-8)
    )

    metrics = {
        "err_mean": err_mean, "err_p50": err_p50, "err_p95": err_p95,
        "err_p99": err_p99, "err_max": err_max, "first_action_cos": cos,
    }
    results["metrics"] = metrics

    print(f"\n  err_mean = {err_mean:.4e}", flush=True)
    print(f"  err_p50  = {err_p50:.4e}", flush=True)
    print(f"  err_p95  = {err_p95:.4e}", flush=True)
    print(f"  err_p99  = {err_p99:.4e}", flush=True)
    print(f"  err_max  = {err_max:.4e}", flush=True)
    print(f"  first_action_cos = {cos:+.6f}", flush=True)

    # ─── 8. Verdict ────────────────────────────────────────────────────
    passed = (err_max < 1e-4) and (err_p95 < 1e-5)
    results["passed"] = passed
    if passed:
        step("VERDICT", "pass", f"max={err_max:.2e} < 1e-4 ✓, p95={err_p95:.2e} < 1e-5 ✓")
    else:
        step("VERDICT", "fail",
             f"max={err_max:.2e} (need < 1e-4), p95={err_p95:.2e} (need < 1e-5) — "
             f"investigate root cause per CLAUDE.md")

    return results


@app.local_entrypoint()
def main(
    num_steps: int = 10,
    chunk_size: int = 50,
    noise_seed: int = 99,
    input_seed: int = 42,
):
    print(f"=== Pi0VLA vs lerobot PI0Policy parity gate ===")
    print(f"num_steps={num_steps}, chunk_size={chunk_size}")
    print(f"noise_seed={noise_seed}, input_seed={input_seed}")
    print()

    results = run_parity.remote(
        num_steps=num_steps, chunk_size=chunk_size,
        noise_seed=noise_seed, input_seed=input_seed,
    )

    print("\n========== FINAL ==========")
    for s in results["steps"]:
        tag = "PASS" if s["status"] == "pass" else ("FAIL" if s["status"] == "fail" else "INFO")
        print(f"  [{tag}] {s['step']} — {s['detail']}")

    if "metrics" in results:
        m = results["metrics"]
        print(f"\nError distribution:")
        print(f"  max  = {m['err_max']:.4e}    (gate: < 1e-4)")
        print(f"  p95  = {m['err_p95']:.4e}    (gate: < 1e-5)")
        print(f"  p99  = {m['err_p99']:.4e}")
        print(f"  cos  = {m['first_action_cos']:+.6f}")

    verdict = "PASS" if results.get("passed", False) else "FAIL"
    print(f"\nVerdict: {verdict}")
    sys.exit(0 if results.get("passed", False) else 1)
