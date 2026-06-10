"""Modal: parity verification — SnapFlow student ONNX vs PyTorch
`sample_actions_1step`. Feeds the same seeded inputs through both paths
and reports cos-sim + max_abs + mean_abs on the action output.

This closes the "cos=1.0 translates to task success" assumption for the
exported student: we separately measured LIBERO task success = 29/30 on
the PyTorch 1-NFE path; this verifies the ONNX reproduces the PyTorch
output numerically, so the LIBERO number transfers.

Usage:
    modal run scripts/modal_verify_snapflow_student_onnx.py \\
      --checkpoint /onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model \\
      --onnx-dir  /onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe
"""
import os
import subprocess
import modal

app = modal.App("tether-verify-snapflow-onnx")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def verify_modal(checkpoint: str, onnx_dir: str, seed: int = 7):
    """Compare PyTorch sample_actions_1step vs ONNX on seeded inputs."""
    import logging
    from pathlib import Path

    import numpy as np
    import torch
    import onnxruntime as ort

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("verify")

    ckpt_path = Path(checkpoint)
    onnx_path = Path(onnx_dir) / "model.onnx"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX missing: {onnx_path}")

    log.info("loading PyTorch student via load_snapflow_student ...")
    from tether.distill.snapflow_pi0_model import load_snapflow_student
    policy = load_snapflow_student(ckpt_path)
    policy.eval().to("cpu").to(torch.float32)

    # Force eager attn everywhere (same as export) so PyTorch matches the
    # export trace path exactly.
    for m in policy.model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"

    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size        # 50
    action_dim = cfg.max_action_dim  # 32

    # Seeded inputs — identical bytes fed to both paths.
    torch.manual_seed(seed)
    np.random.seed(seed)
    dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    log.info("running PyTorch sample_actions_1step ...")
    with torch.no_grad():
        actions_torch = policy.model.sample_actions_1step(
            [dummy["img_base"], dummy["img_wrist_l"], dummy["img_wrist_r"]],
            [dummy["mask_base"], dummy["mask_wrist_l"], dummy["mask_wrist_r"]],
            dummy["lang_tokens"],
            dummy["lang_masks"],
            noise=dummy["noise"],
        )
    actions_torch_np = actions_torch.cpu().numpy()
    log.info("PyTorch output shape: %s dtype: %s", actions_torch_np.shape, actions_torch_np.dtype)

    # Free PyTorch model before loading ONNX — 12.99GB ONNX + pi0.5 weights
    # together would bust even 80GB RAM.
    del policy, actions_torch
    import gc; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log.info("loading ONNX via onnxruntime (CPU EP to match PyTorch CPU run) ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    log.info("ONNX session ready. inputs: %s", [i.name for i in sess.get_inputs()])

    ort_inputs = {
        "img_base": dummy["img_base"].numpy(),
        "img_wrist_l": dummy["img_wrist_l"].numpy(),
        "img_wrist_r": dummy["img_wrist_r"].numpy(),
        "mask_base": dummy["mask_base"].numpy(),
        "mask_wrist_l": dummy["mask_wrist_l"].numpy(),
        "mask_wrist_r": dummy["mask_wrist_r"].numpy(),
        "lang_tokens": dummy["lang_tokens"].numpy(),
        "lang_masks": dummy["lang_masks"].numpy(),
        "noise": dummy["noise"].numpy(),
    }
    log.info("running ONNX forward ...")
    actions_onnx_np = sess.run(["actions"], ort_inputs)[0]
    log.info("ONNX output shape: %s dtype: %s", actions_onnx_np.shape, actions_onnx_np.dtype)

    # Parity
    assert actions_torch_np.shape == actions_onnx_np.shape, (
        f"shape mismatch: torch {actions_torch_np.shape} vs onnx {actions_onnx_np.shape}"
    )
    diff = actions_torch_np.astype(np.float64) - actions_onnx_np.astype(np.float64)
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())

    tf = actions_torch_np.reshape(-1).astype(np.float64)
    onx = actions_onnx_np.reshape(-1).astype(np.float64)
    cos = float(np.dot(tf, onx) / (np.linalg.norm(tf) * np.linalg.norm(onx) + 1e-12))

    log.info("==== PARITY ====")
    log.info("  shape:    %s", actions_torch_np.shape)
    log.info("  cos_sim:  %.10f", cos)
    log.info("  max_abs:  %.6e", max_abs)
    log.info("  mean_abs: %.6e", mean_abs)
    log.info("  torch sample values: %s", actions_torch_np.flatten()[:5])
    log.info("  onnx  sample values: %s", actions_onnx_np.flatten()[:5])

    return {
        "status": "ok",
        "shape": list(actions_torch_np.shape),
        "cos_sim": cos,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "pytorch_first_values": actions_torch_np.flatten()[:5].tolist(),
        "onnx_first_values": actions_onnx_np.flatten()[:5].tolist(),
    }


@app.local_entrypoint()
def main(
    checkpoint: str = "/onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model",
    onnx_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe",
    seed: int = 7,
):
    """
    --checkpoint    Path on the pi0-onnx-outputs volume to the distilled
                    student checkpoint dir (used by load_snapflow_student).
    --onnx-dir      Path on the volume containing model.onnx + model.onnx.data
                    from export_snapflow_student_monolithic.
    --seed          Seed for torch.manual_seed + np.random.seed so both
                    paths get identical input bytes. Default 7.
    """
    r = verify_modal.remote(checkpoint=checkpoint, onnx_dir=onnx_dir, seed=seed)
    print("\n=== VERIFICATION ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
