"""Step 5 — Eagle + DiT end-to-end chain test for GR00T N1.6.

Verifies the two exported ONNXes compose correctly:

    pixel_values + input_ids → eagle_vlm.onnx → hidden [B, T, 2048]
                                      │
                                      ↓ (as vlm_kv)
    noisy_actions + t + pos + state + vlm_kv → expert_stack_with_vlm.onnx → velocity

Two assertions:
  1. Parity:   cos(PyTorch_chain, ONNX_chain) > 0.9999 for each image.
  2. Sensitivity: image A vs B produces meaningfully different actions
     (max_abs > 0.01), in BOTH the PyTorch and ONNX paths. This is the
     "VLM conditioning is actually alive" check — with a zero-KV stub,
     actions don't change when the image changes. With real Eagle KV,
     they must.

Usage:
    modal run scripts/modal_gr00t_e2e_chain_test.py
"""
import os
import subprocess
import modal

app = modal.App("tether-gr00t-e2e-chain")


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

hf_cache = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("gr00t-onnx-outputs", create_if_missing=True)
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
        "typer",
        "rich",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
            secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def chain_test(model_id: str = "nvidia/GR00T-N1.6-3B"):
    """End-to-end chain: Eagle ONNX → DiT ONNX, compare vs PyTorch."""
    import math as _math
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    import torch

    from tether.checkpoint import load_checkpoint
    from tether.exporters.eagle_export_stack import build_eagle_export_stack
    from tether.exporters.gr00t import build_gr00t_full_stack

    eagle_onnx = Path(ONNX_OUTPUT_PATH) / "eagle_vlm" / "eagle_vlm.onnx"
    dit_onnx = Path(ONNX_OUTPUT_PATH) / "monolithic_with_vlm" / "expert_stack_with_vlm.onnx"
    if not eagle_onnx.exists():
        return {"status": "fail", "reason": f"{eagle_onnx} not found"}
    if not dit_onnx.exists():
        return {"status": "fail", "reason": f"{dit_onnx} not found"}

    print(f"[e2e] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    print(f"[e2e] {len(state_dict)} tensors in {time.time()-t0:.1f}s")

    # Eagle reference (PyTorch)
    print(f"[e2e] Building Eagle PyTorch ref...")
    eagle_stack, eagle_meta = build_eagle_export_stack(state_dict)
    eagle_stack = eagle_stack.to("cuda").eval()
    print(f"[e2e] Eagle: {eagle_meta['total_params_m']:.1f}M")

    # DiT+full reference (PyTorch)
    print(f"[e2e] Building DiT PyTorch ref...")
    dit_stack, dit_meta = build_gr00t_full_stack(state_dict)
    dit_stack = dit_stack.to("cuda").eval()
    print(f"[e2e] DiT: {dit_meta.get('total_params_m', '?')}M  "
          f"has_state_encoder={dit_stack.state_encoder is not None}")

    # Derive image dims
    pos_embed_key = (
        "backbone.model.vision_model.vision_model.embeddings.position_embedding.weight"
    )
    num_positions = state_dict[pos_embed_key].shape[0]
    patches_per_side = int(_math.sqrt(num_positions))
    patch_size = 14
    H = W = patches_per_side * patch_size
    n_image_tokens = num_positions // 4
    B = 1
    seq = n_image_tokens + 16
    img_tok = eagle_meta["image_token_index"]
    vocab = eagle_meta["vocab_size"]

    # Shared inputs (same across images A & B)
    torch.manual_seed(0)
    input_ids = torch.zeros(B, seq, dtype=torch.long, device="cuda")
    input_ids[:, :n_image_tokens] = img_tok
    input_ids[:, n_image_tokens:] = torch.randint(
        low=1, high=min(32000, vocab), size=(B, seq - n_image_tokens), device="cuda",
    )
    attention_mask = torch.ones(B, seq, dtype=torch.long, device="cuda")
    image_flags = torch.tensor([1], device="cuda", dtype=torch.long)

    # DiT ONNX was exported with chunk=50 FIXED (only batch is dynamic).
    # Position_ids shape is [1, chunk + 1] = [1, 51] (state + 50 actions).
    chunk = 50
    raw_action_dim = dit_stack.action_encoder.raw_action_dim
    raw_state_dim = dit_stack.state_encoder.raw_state_dim if dit_stack.state_encoder else 128
    noisy_actions = torch.randn(B, chunk, raw_action_dim, device="cuda", dtype=torch.float32)
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    position_ids = torch.arange(chunk + 1, device="cuda", dtype=torch.long).unsqueeze(0)
    state = torch.randn(B, raw_state_dim, device="cuda", dtype=torch.float32)

    # Image A and B — different seeds, same shape
    torch.manual_seed(1)
    pixel_values_A = torch.randn(B, 3, H, W, device="cuda", dtype=torch.float32)
    torch.manual_seed(2)
    pixel_values_B = torch.randn(B, 3, H, W, device="cuda", dtype=torch.float32)

    # ---- PyTorch chain ----
    def pt_chain(pixel_values):
        with torch.no_grad():
            hidden = eagle_stack(pixel_values, input_ids, attention_mask, image_flags)
            actions = dit_stack(noisy_actions, timestep, position_ids, state=state, vlm_kv=hidden)
        return hidden.detach().cpu().numpy().astype(np.float32), \
               actions.detach().cpu().numpy().astype(np.float32)

    print(f"[e2e] PyTorch chain (A)...")
    t0 = time.time()
    hidden_A_pt, actions_A_pt = pt_chain(pixel_values_A)
    print(f"[e2e]   hidden={hidden_A_pt.shape}  actions={actions_A_pt.shape}  "
          f"[{time.time()-t0:.2f}s]")
    print(f"[e2e] PyTorch chain (B)...")
    t0 = time.time()
    hidden_B_pt, actions_B_pt = pt_chain(pixel_values_B)
    print(f"[e2e]   [{time.time()-t0:.2f}s]")

    # Free PyTorch to make room for ONNX sessions
    del eagle_stack, dit_stack
    torch.cuda.empty_cache()

    # ---- ONNX chain ----
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    print(f"[e2e] Loading eagle_vlm.onnx ({eagle_onnx.stat().st_size / 1e6:.0f}MB)...")
    t0 = time.time()
    eagle_sess = ort.InferenceSession(str(eagle_onnx), providers=providers)
    print(f"[e2e]   loaded in {time.time()-t0:.1f}s")

    print(f"[e2e] Loading expert_stack_with_vlm.onnx ({dit_onnx.stat().st_size / 1e6:.0f}MB)...")
    t0 = time.time()
    dit_sess = ort.InferenceSession(str(dit_onnx), providers=providers)
    print(f"[e2e]   loaded in {time.time()-t0:.1f}s")

    dit_input_names = [i.name for i in dit_sess.get_inputs()]
    print(f"[e2e]   DiT ONNX inputs: {dit_input_names}")

    def onnx_chain(pixel_values_np):
        hidden = eagle_sess.run(
            ["hidden_states"],
            {
                "pixel_values": pixel_values_np,
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
                "image_flags": image_flags.cpu().numpy().astype(np.int64),
            },
        )[0].astype(np.float32)

        feed = {
            "noisy_actions": noisy_actions.cpu().numpy().astype(np.float32),
            "timestep": timestep.cpu().numpy().astype(np.float32),
            "position_ids": position_ids.cpu().numpy().astype(np.int64),
            "state": state.cpu().numpy().astype(np.float32),
            "vlm_kv": hidden,
        }
        # Only pass inputs the ONNX expects (handles optional state/vlm_kv
        # if the export used different names).
        feed = {k: v for k, v in feed.items() if k in dit_input_names}
        actions = dit_sess.run(None, feed)[0].astype(np.float32)
        return hidden, actions

    print(f"[e2e] ONNX chain (A)...")
    t0 = time.time()
    hidden_A_on, actions_A_on = onnx_chain(pixel_values_A.cpu().numpy().astype(np.float32))
    print(f"[e2e]   hidden={hidden_A_on.shape}  actions={actions_A_on.shape}  "
          f"[{time.time()-t0:.2f}s]")
    print(f"[e2e] ONNX chain (B)...")
    t0 = time.time()
    hidden_B_on, actions_B_on = onnx_chain(pixel_values_B.cpu().numpy().astype(np.float32))
    print(f"[e2e]   [{time.time()-t0:.2f}s]")

    # ---- Metrics ----
    def _stats(a, b, label):
        a_f = a.flatten().astype(np.float64)
        b_f = b.flatten().astype(np.float64)
        diff = np.abs(a_f - b_f)
        cos = float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))
        return {
            "label": label,
            "cos_sim": cos,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        }

    # Parity: PyTorch vs ONNX for each image
    parity_A_h = _stats(hidden_A_pt, hidden_A_on, "parity_A_hidden")
    parity_A_a = _stats(actions_A_pt, actions_A_on, "parity_A_actions")
    parity_B_h = _stats(hidden_B_pt, hidden_B_on, "parity_B_hidden")
    parity_B_a = _stats(actions_B_pt, actions_B_on, "parity_B_actions")

    # Sensitivity: A vs B within each path
    sens_pt = _stats(actions_A_pt, actions_B_pt, "sens_pytorch")
    sens_on = _stats(actions_A_on, actions_B_on, "sens_onnx")

    print(f"\n[e2e] === RESULTS ===")
    for m in [parity_A_h, parity_A_a, parity_B_h, parity_B_a, sens_pt, sens_on]:
        print(f"[e2e] {m['label']:24s}  "
              f"cos={m['cos_sim']:+.6f}  "
              f"max_abs={m['max_abs']:.4e}  "
              f"mean_abs={m['mean_abs']:.4e}")

    # Verdict
    parity_ok = all(
        m["cos_sim"] > 0.9999 for m in [parity_A_a, parity_B_a]
    )
    sens_pt_ok = sens_pt["max_abs"] > 0.01
    sens_on_ok = sens_on["max_abs"] > 0.01
    verdict = "PASS" if (parity_ok and sens_pt_ok and sens_on_ok) else "FAIL"
    print(f"\n[e2e] parity_ok={parity_ok}  sens_pt_ok={sens_pt_ok}  sens_on_ok={sens_on_ok}")
    print(f"[e2e] VERDICT: {verdict}")

    return {
        "status": "ok",
        "verdict": verdict,
        "parity_A_actions_cos": parity_A_a["cos_sim"],
        "parity_B_actions_cos": parity_B_a["cos_sim"],
        "parity_A_actions_maxabs": parity_A_a["max_abs"],
        "parity_B_actions_maxabs": parity_B_a["max_abs"],
        "sens_pytorch_maxabs": sens_pt["max_abs"],
        "sens_onnx_maxabs": sens_on["max_abs"],
        "sens_pytorch_cos": sens_pt["cos_sim"],
        "sens_onnx_cos": sens_on["cos_sim"],
    }


@app.local_entrypoint()
def main():
    r = chain_test.remote()
    print(f"\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
