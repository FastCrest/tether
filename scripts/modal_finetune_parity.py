"""Parity check: fine-tuned SmolVLA PyTorch vs its exported ONNX.

Closes the caveat on the `2026-04-20-finetune-e2e` measured row:
v10 proved the chain is structurally sound, but didn't measure
whether the exported ONNX still matches the fine-tuned PyTorch at
cos=+1.0. This script adds that number.

Usage:
    modal run scripts/modal_finetune_parity.py \\
        --subdir finetune_smolvla_pusht_v10 \\
        --checkpoint-step 000200

Reads from the same `pi0-onnx-outputs` volume where `tether finetune`
writes. Uses shared seeded inputs matching the SmolVLA monolithic
wrapper signature. Returns cos + max_abs + verdict.
"""
import os
import subprocess
import modal

app = modal.App("tether-finetune-parity")


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

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "clang")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "lerobot==0.5.1",
        "peft>=0.10",
        "num2words",
    )
    .run_commands(
        # GITHUB_TOKEN from modal secret `github-token` (repo now private).
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def parity_modal(
    subdir: str = "finetune_smolvla_pusht_v10",
    checkpoint_step: str = "000200",
    num_steps: int = 10,
    seed: int = 42,
):
    """Load fine-tuned SmolVLA + its ONNX, compare on shared seeded inputs."""
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    import torch

    output_root = Path(ONNX_OUTPUT_PATH) / subdir
    merged_ckpt = (
        output_root / "training" / "checkpoints" / checkpoint_step
        / "pretrained_model" / "merged"
    )
    onnx_path = output_root / "export" / "model.onnx"

    if not merged_ckpt.exists():
        return {"status": "fail", "reason": f"merged checkpoint not found: {merged_ckpt}"}
    if not onnx_path.exists():
        return {"status": "fail", "reason": f"ONNX not found: {onnx_path}"}

    print(f"[parity] merged checkpoint: {merged_ckpt}")
    print(f"[parity] onnx:               {onnx_path}")

    # reuse the monolithic export's patch stack so PyTorch and ONNX are
    # on identical code paths (eager attention, no DynamicCache quirks).
    from tether.exporters.monolithic import apply_export_patches
    apply_export_patches()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("[parity] Loading fine-tuned SmolVLA...")
    policy = SmolVLAPolicy.from_pretrained(str(merged_ckpt))
    policy.eval().to(torch.float32).to("cpu")
    policy.model.config.num_steps = num_steps

    # Force eager attention on every sub-module (same as monolithic
    # export path). Without this, sdpa kernel can differ from ORT's
    # default attention implementation.
    def _force_eager(m):
        for mod in m.modules():
            if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                mod.config._attn_implementation = "eager"
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    _force_eager(policy.model)

    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    torch.manual_seed(seed)
    img = torch.randn(B, 3, 512, 512, dtype=torch.float32)
    mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, 49152, (B, 16), dtype=torch.long)
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, state_dim, dtype=torch.float32)
    noise = torch.randn(B, chunk, action_dim, dtype=torch.float32)

    images = [img, img, img]
    img_masks = [mask, mask, mask]

    print(f"[parity] Running PyTorch ref (num_steps={num_steps})...")
    with torch.no_grad():
        pt_actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )
    pt_np = pt_actions.cpu().numpy()
    print(f"[parity]   pt shape={pt_np.shape} first={pt_np[0, 0, :5]}")

    print(f"[parity] Running ONNX...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "img_cam1": img.numpy(),
        "img_cam2": img.numpy(),
        "img_cam3": img.numpy(),
        "mask_cam1": mask.numpy(),
        "mask_cam2": mask.numpy(),
        "mask_cam3": mask.numpy(),
        "lang_tokens": lang_tokens.numpy().astype(np.int64),
        "lang_masks": lang_masks.numpy(),
        "state": state.numpy(),
        "noise": noise.numpy(),
    }
    ort_out = sess.run(None, ort_inputs)[0]
    print(f"[parity]   onnx shape={ort_out.shape} first={ort_out[0, 0, :5]}")

    pt0 = pt_np[0, 0]
    on0 = ort_out[0, 0]
    first_max_abs = float(np.abs(pt0 - on0).max())
    first_cos = float(
        np.dot(pt0, on0) / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-8)
    )
    full_max_abs = float(np.abs(pt_np - ort_out).max())
    full_cos = float(
        np.dot(pt_np.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt_np) * np.linalg.norm(ort_out) + 1e-8)
    )

    # Fine-tune preserves parity iff cos stays >= 0.999 AND max_abs is
    # within normal fp32 rounding. The pre-fine-tune smolvla_base number
    # is full-chunk max_abs 3.70e-06 at num_steps=10 (measured_numbers
    # row 2026-04-18). A fine-tuned export should hit similar orders.
    passed = full_cos >= 0.999 and full_max_abs < 1e-3

    print(f"\n====== FINETUNE PARITY ({subdir}, num_steps={num_steps}) ======")
    print(f"  first-action max_abs: {first_max_abs:.4e}")
    print(f"  first-action cos:     {first_cos:+.6f}")
    print(f"  full-chunk  max_abs:  {full_max_abs:.4e}")
    print(f"  full-chunk  cos:      {full_cos:+.6f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    return {
        "status": "ok",
        "subdir": subdir,
        "num_steps": num_steps,
        "first_cos": first_cos,
        "first_max_abs": first_max_abs,
        "full_cos": full_cos,
        "full_max_abs": full_max_abs,
        "passed": passed,
    }


@app.local_entrypoint()
def main(
    subdir: str = "finetune_smolvla_pusht_v10",
    checkpoint_step: str = "000200",
    num_steps: int = 10,
):
    r = parity_modal.remote(
        subdir=subdir,
        checkpoint_step=checkpoint_step,
        num_steps=num_steps,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k}: {v}")
