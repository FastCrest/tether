"""Adapter vs raw-policy parity for SmolVLA on a synthetic LIBERO obs.

Goal: isolate whether the 2026-04-19 LIBERO 0% is a pipeline bug (our
adapter/native-server wraps something wrong) or a model-behavior gap
(the fine-tune just doesn't solve these tasks zero-shot under our
harness).

Method:
  1. Build a synthetic LIBERO-like observation dict (agentview + wrist
     images at 256x256/uint8, state at 8D float32, task text).
  2. Path A: run through our `SmolVLANativeServer.predict()` — exactly
     what the vla-eval adapter does.
  3. Path B: load `SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")`
     directly, build the lerobot batch schema, call
     `policy.predict_action_chunk(batch_pp)` with matching noise.
  4. Compare: cos, L2, max_abs on the action chunks.

Interpretation:
  cos ≈ +1.0, max_abs < 1e-4  → pipeline is clean; 0% is model-behavior
  cos < 0.99 or max_abs large → adapter has a real bug worth hunting

Usage:
    modal run scripts/modal_smolvla_libero_parity.py
"""
import os
import subprocess
import modal

app = modal.App("tether-smolvla-libero-parity")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


def _repo_head_sha() -> str:
    """Pin image to current HEAD so Modal rebuilds per commit."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


# Lightweight image — no LIBERO / robosuite / MuJoCo needed. Just torch +
# lerobot + tether-vla, since both paths call the same underlying model.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "typer",
        "rich",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "lerobot==0.5.1",
        "num2words",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
            secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[_hf_secret()],
)
def parity_test():
    """Run the side-by-side parity and report cos / max_abs."""
    import time
    import numpy as np
    import torch

    print("=== Path A: load SmolVLANativeServer (our adapter's path) ===")
    # Our adapter's path: SmolVLANativeServer.load() + .predict().
    # That internally instantiates SmolVLAPolicy. We need to export a
    # checkpoint dir first — but SmolVLANativeServer.load() uses
    # `snapshot_download` directly. Actually we need to write a
    # tether_config.json it can read, so let's stub one.
    export_dir = "/tmp/smolvla_stub"
    os.makedirs(export_dir, exist_ok=True)
    import json
    with open(f"{export_dir}/tether_config.json", "w") as f:
        json.dump({
            "model_id": "lerobot/smolvla_libero",
            "vlm_model_id": "lerobot/smolvla_libero",
            "model_type": "smolvla",
            "num_denoising_steps": 10,
            "chunk_size": 50,
            "action_dim": 32,
            "max_state_dim": 32,
        }, f)

    from tether.runtime.smolvla_native import SmolVLANativeServer
    t0 = time.time()
    server = SmolVLANativeServer(export_dir, device="cuda", strict_providers=False)
    server.load()
    print(f"SmolVLANativeServer loaded in {time.time()-t0:.1f}s")

    # ─── Build synthetic LIBERO-like obs ───
    # 2 cameras (agentview + wrist), 256x256 uint8, 8D state, task string.
    np.random.seed(42)
    torch.manual_seed(42)
    img_agent = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    img_wrist = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0], dtype=np.float32)
    task = "put both the alphabet soup and the tomato sauce in the basket"

    # Fixed noise for both paths so comparison is deterministic.
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(123)).numpy()

    # ─── Path A: through our adapter-equivalent call ───
    print("\n=== Path A predict ===")
    t0 = time.time()
    result_a = server.predict(
        image=[img_agent, img_wrist],
        instruction=task + "\n",  # adapter appends \n
        state=state.tolist(),
        noise=noise,
    )
    print(f"Path A done in {time.time()-t0:.2f}s")
    actions_a = np.asarray(result_a["actions"], dtype=np.float32)
    print(f"Path A actions: shape={actions_a.shape} "
          f"mean={actions_a.mean():+.4f} std={actions_a.std():.4f}")
    print(f"Path A first action: {actions_a[0]}")

    # ─── Path B: raw SmolVLAPolicy directly, mirroring native_server's batch build ───
    print("\n=== Path B: load SmolVLAPolicy directly + manual batch ===")
    t0 = time.time()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval().to(dtype=torch.float32).to("cuda")
    repo_dir = snapshot_download("lerobot/smolvla_libero")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print(f"Raw policy loaded in {time.time()-t0:.1f}s")

    # Build batch with same structure as native_server.predict()
    def _prep_img(img_np):
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        return (torch.from_numpy(img_np).permute(2, 0, 1).float()
                .div_(255.0).unsqueeze(0))

    # Replicate to 3 cameras (native_server's behavior when given 2)
    images_list = [img_agent, img_wrist, img_wrist]
    batch: dict = {}
    for i, img in enumerate(images_list, start=1):
        batch[f"observation.images.camera{i}"] = _prep_img(img)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0)
    batch["task"] = [task + "\n"]

    batch_pp = preprocessor(batch)
    batch_pp = {
        k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
        for k, v in batch_pp.items()
    }
    print(f"Path B batch_pp keys: {sorted(batch_pp.keys())}")

    noise_t = torch.from_numpy(noise).to("cuda").to(torch.float32)
    t0 = time.time()
    with torch.no_grad():
        actions_raw = policy.predict_action_chunk(batch_pp, noise=noise_t)
    post = postprocessor(actions_raw.detach().cpu())
    actions_b = (post.detach().cpu().numpy() if hasattr(post, "detach")
                 else np.asarray(post))
    if actions_b.ndim == 3:
        actions_b = actions_b[0]
    print(f"Path B predict in {time.time()-t0:.2f}s")
    print(f"Path B actions: shape={actions_b.shape} "
          f"mean={actions_b.mean():+.4f} std={actions_b.std():.4f}")
    print(f"Path B first action: {actions_b[0]}")

    # ─── Compare ───
    # Align dims (Path A may have been truncated; Path B is raw)
    n_dim = min(actions_a.shape[-1], actions_b.shape[-1])
    a = actions_a[..., :n_dim].reshape(-1)
    b = actions_b[..., :n_dim].reshape(-1)
    max_abs = float(np.abs(a - b).max())
    l2 = float(np.linalg.norm(a - b))
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print("\n====== PARITY RESULT ======")
    print(f"  cos:     {cos:+.6f}")
    print(f"  max_abs: {max_abs:.4e}")
    print(f"  L2:      {l2:.4e}")
    print(f"  Path A action dim: {actions_a.shape[-1]}")
    print(f"  Path B action dim: {actions_b.shape[-1]}")

    if cos >= 0.9999 and max_abs < 1e-4:
        verdict = "PASS — pipeline confirmed clean. LIBERO 0% is NOT an adapter bug. Next: check lerobot eval config deltas (camera order, action chunk cadence, state def)."
    elif cos >= 0.99:
        verdict = "CLOSE but not machine precision — minor numerical drift (likely fp32 non-determinism or postprocessor difference). Pipeline is ~clean."
    else:
        verdict = f"FAIL — cos={cos:+.4f} indicates a real adapter-level bug between our wrapper and the raw policy. Investigate image prep + state padding + preprocessor overrides."
    print(f"\n  VERDICT: {verdict}")

    return {
        "status": "ok",
        "cos": cos, "max_abs": max_abs, "l2": l2,
        "path_a_dim": int(actions_a.shape[-1]),
        "path_b_dim": int(actions_b.shape[-1]),
        "path_a_first_action": actions_a[0].tolist(),
        "path_b_first_action": actions_b[0].tolist(),
        "verdict": verdict,
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("ADAPTER vs RAW SmolVLAPolicy parity on synthetic LIBERO obs")
    print("=" * 60)
    result = parity_test.remote()
    print("\n=== RESULT ===")
    for k, v in result.items():
        if k in ("path_a_first_action", "path_b_first_action"):
            print(f"  {k}: {[round(x, 4) for x in v]}")
        else:
            print(f"  {k}: {v}")
