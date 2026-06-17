"""Stage-by-stage PyTorch-vs-ONNX diff to localise the divergence.

Runs one Modal call. Compares intermediate outputs between the real lerobot
SmolVLAPolicy forward and our ONNX pipeline. First stage where L2 diverges
is where the bug lives.

Stages:
  1. Vision encoder:   our vision_encoder.onnx vs policy.vlm.vision_tower + connector
  2. Text embedder:    our text_embedder.onnx vs vlm.text_model.embed_tokens
  3. State projection: our state_proj_weight @ state vs policy.state_proj(state)
  4. Decoder prefill:  our decoder_prefill.onnx vs policy running decoder layer-by-layer

Usage:
    modal run scripts/modal_stage_diff.py
"""
import json
import subprocess
import time

import modal

app = modal.App("tether-stage-diff")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa",
                 "libglvnd0", "ffmpeg", "cmake", "build-essential")
    .pip_install(
        "torch", "safetensors", "huggingface_hub", "transformers>=4.51",
        "onnx", "onnxruntime", "onnxscript", "numpy", "Pillow",
        "pydantic>=2.0", "typer", "rich", "pyyaml", "einops",
    )
    .pip_install("lerobot", "num2words")
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e .")
)


def _diff(name, torch_arr, onnx_arr):
    import numpy as np
    t = np.asarray(torch_arr, dtype=np.float64).flatten()
    o = np.asarray(onnx_arr, dtype=np.float64).flatten()
    if t.shape != o.shape:
        print(f"  {name}: SHAPE MISMATCH torch={t.shape} onnx={o.shape}")
        return
    max_abs = float(np.abs(t - o).max())
    l2 = float(np.linalg.norm(t - o))
    cos = float(np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o) + 1e-8))
    print(f"  {name}: max_abs={max_abs:.4e}  L2={l2:.4e}  cos={cos:+.4f}  "
          f"torch||={np.linalg.norm(t):.3e}  onnx||={np.linalg.norm(o):.3e}")


@app.function(image=image, gpu="A10G", timeout=1200, scaledown_window=60)
def stage_diff():
    import os
    os.environ.setdefault("HF_HOME", "/tmp/hf")

    import numpy as np
    import torch

    # ── Export ONNX ─────────────────────────────────────────────────
    print("=" * 60); print("Step 1: export"); print("=" * 60)
    export_dir = "/tmp/tether_libero_export"
    r = subprocess.run(
        ["tether", "export", "lerobot/smolvla_libero",
         "--target", "desktop", "--output", export_dir],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        print("EXPORT FAILED"); print(r.stdout[-1500:]); return {"error": "export"}
    # Dump export stdout for vlm-weights diagnostics
    print(r.stdout[-3000:])
    print("exported")

    # ── Load PyTorch policy ────────────────────────────────────────
    print("=" * 60); print("Step 2: load policy"); print("=" * 60)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval()
    # Our ONNX is fp32 on CPU — force the policy to match or the input-dtype
    # checks inside torch blow up with "Input FloatTensor vs weight BFloat16".
    policy.to(dtype=torch.float32)
    device = "cpu"
    policy.to(device)

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    vision = vlm.vision_tower if hasattr(vlm, "vision_tower") else vlm.vision_model
    connector = vlm.connector if hasattr(vlm, "connector") else None
    text_model = vlm.text_model

    # ── Build deterministic input ──────────────────────────────────
    rng = np.random.RandomState(42)
    H, W = 512, 512   # match our export vision input size
    img_uint8 = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    state8 = rng.randn(8).astype(np.float32) * 0.1
    task = "put the red bowl on the plate"

    # ───────────────────────────────────────────────────────────────
    # STAGE 1: Vision encoder
    # ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Vision encoder (pixel_values → image_embeds)")
    print("=" * 60)

    # Build pixel_values in SigLIP format: [B, 3, 512, 512], range [-1, 1].
    img_f = img_uint8.astype(np.float32) / 255.0
    img_f = img_f * 2.0 - 1.0
    pix_np = np.transpose(img_f, (2, 0, 1))[None, ...].astype(np.float32)   # [1,3,512,512]

    # PyTorch side
    pix_t = torch.from_numpy(pix_np).to(device)
    with torch.no_grad():
        vout = vision(pix_t)
        vfeats = vout.last_hidden_state if hasattr(vout, "last_hidden_state") else vout
        if connector is not None:
            vfeats = connector(vfeats)
    torch_vision = vfeats.cpu().numpy()
    print(f"  torch vision shape: {torch_vision.shape}")

    # Our ONNX side
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{export_dir}/vision_encoder.onnx",
                                providers=["CPUExecutionProvider"])
    onnx_vision = sess.run(None, {"pixel_values": pix_np})[0]
    print(f"  onnx vision shape: {onnx_vision.shape}")
    _diff("vision_embeds", torch_vision, onnx_vision)

    # ───────────────────────────────────────────────────────────────
    # STAGE 2: Text embeddings
    # ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Text embedder (tokens → text_embeds)")
    print("=" * 60)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    lang = task + "\n"
    ids = tok(lang, return_tensors="pt", padding="max_length",
              truncation=True, max_length=48)["input_ids"].to(device)

    with torch.no_grad():
        emb_layer = text_model.get_input_embeddings()
        torch_text = emb_layer(ids).cpu().numpy()

    sess_t = ort.InferenceSession(f"{export_dir}/text_embedder.onnx",
                                  providers=["CPUExecutionProvider"])
    onnx_text = sess_t.run(None, {"input_ids": ids.cpu().numpy().astype(np.int64)})[0]
    _diff("text_embeds", torch_text, onnx_text)

    # ───────────────────────────────────────────────────────────────
    # STAGE 3: State projection
    # ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3: State projection")
    print("=" * 60)
    state8_t = torch.from_numpy(state8).unsqueeze(0).to(device)
    # policy.model.state_proj takes max_state_dim (32) so pad 8 -> 32
    padded = torch.zeros(1, 32, dtype=torch.float32, device=device)
    padded[:, :8] = state8_t

    with torch.no_grad():
        torch_state = policy.model.state_proj(padded).cpu().numpy()

    # Our side: load saved weights and apply
    w = np.load(f"{export_dir}/state_proj_weight.npy")
    b = np.load(f"{export_dir}/state_proj_bias.npy")
    onnx_state = padded.cpu().numpy() @ w.T + b
    _diff("state_proj", torch_state, onnx_state)

    # ───────────────────────────────────────────────────────────────
    # STAGE 4: Decoder prefill (per-layer k and v)
    # ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 4: Decoder prefill (per-layer vlm_k, vlm_v)")
    print("=" * 60)

    # Build a simple prefix: just take image_embeds as inputs_embeds
    # (~64 tokens at 960-dim). Same for both paths.
    prefix_seq = int(torch_vision.shape[1])   # 64
    embs = torch.from_numpy(torch_vision).to(device)   # [1, seq, 960]
    mask = torch.ones(1, prefix_seq, dtype=torch.long, device=device)

    # PyTorch path: replicate DecoderPrefillForONNX logic against the same
    # text_model inside the loaded policy — this is our reference.
    with torch.no_grad():
        tm = policy.model.vlm_with_expert.get_vlm_model().text_model
        # Use only the truncated layers (SmolVLA uses first 16)
        num_keep = min(16, len(tm.layers))

        outs = tm(inputs_embeds=embs, attention_mask=mask, output_hidden_states=True)
        hidden_states = list(outs.hidden_states)   # tuple of [B, seq, 960]

        pos_ids = mask.long().cumsum(-1) - 1
        pos_ids = pos_ids.clamp(min=0)
        rotary = getattr(tm, "rotary_emb", None) or tm.layers[0].self_attn.rotary_emb
        cos_r, sin_r = rotary(embs, pos_ids)
        cos_r = cos_r.unsqueeze(1); sin_r = sin_r.unsqueeze(1)

        def rotate_half(x):
            h = x.shape[-1] // 2
            return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

        torch_ks, torch_vs = [], []
        for i in range(num_keep):
            layer = tm.layers[i]
            h_in = hidden_states[i]
            h_norm = layer.input_layernorm(h_in)
            k_flat = layer.self_attn.k_proj(h_norm)
            v_flat = layer.self_attn.v_proj(h_norm)
            b_, s_, _ = k_flat.shape
            nkv = layer.self_attn.config.num_key_value_heads
            hd = layer.self_attn.head_dim
            k_heads = k_flat.view(b_, s_, nkv, hd).transpose(1, 2)
            k_roped = (k_heads * cos_r) + (rotate_half(k_heads) * sin_r)
            k_out = k_roped.transpose(1, 2).contiguous().view(b_, s_, nkv * hd)
            torch_ks.append(k_out.cpu().numpy())
            torch_vs.append(v_flat.cpu().numpy())

    # Our ONNX path
    sess_d = ort.InferenceSession(f"{export_dir}/decoder_prefill.onnx",
                                  providers=["CPUExecutionProvider"])
    onnx_k, onnx_v = sess_d.run(
        ["vlm_k", "vlm_v"],
        {
            "inputs_embeds": embs.cpu().numpy().astype(np.float32),
            "attention_mask": mask.cpu().numpy().astype(np.int64),
        },
    )
    for i in [0, num_keep // 2, num_keep - 1]:
        _diff(f"layer_{i}_k", torch_ks[i], onnx_k[i])
        _diff(f"layer_{i}_v", torch_vs[i], onnx_v[i])

    # ───────────────────────────────────────────────────────────────
    # STAGE 5: Expert one-step velocity (single denoising step)
    # ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 5: Expert velocity (one flow-matching step)")
    print("=" * 60)

    chunk = policy.config.chunk_size
    max_a = policy.config.max_action_dim
    noise_t = torch.from_numpy(
        np.random.RandomState(7).randn(1, chunk, max_a).astype(np.float32)
    ).to(device)
    t_scalar = torch.tensor([0.5], device=device)

    # Build the position_ids that the real expert gets: start from 0, length chunk.
    pos_ids_exp = torch.arange(chunk, device=device).unsqueeze(0)

    # PyTorch: invoke one iteration of flow-matching via the model's own
    # helper, or simpler: run embed_suffix + full forward with fixed noise.
    # Shortcut: call sample_actions with only 1 denoising step by monkey-patch.
    # Easier: just directly call model.forward on full prefix + suffix.
    # For now skip this — the decoder diffs already show per-layer kv is fine,
    # so the divergence must be in expert_stack.onnx itself. Compare by feeding
    # identical vlm_k/vlm_v to our expert ONNX and to an in-place PyTorch expert.
    try:
        sess_e = ort.InferenceSession(f"{export_dir}/expert_stack.onnx",
                                      providers=["CPUExecutionProvider"])
        # All inputs: noisy_actions, timestep, position_ids, vlm_k, vlm_v
        feed_onnx = {
            "noisy_actions": noise_t.cpu().numpy(),
            "timestep": t_scalar.cpu().numpy(),
            "position_ids": pos_ids_exp.cpu().numpy().astype(np.int64),
            "vlm_k": np.asarray(onnx_k, dtype=np.float32),
            "vlm_v": np.asarray(onnx_v, dtype=np.float32),
        }
        onnx_vel = sess_e.run(None, feed_onnx)[0]
        print(f"  onnx velocity shape: {onnx_vel.shape}")
        print(f"  onnx velocity first 7 dims: {np.round(onnx_vel[0, 0, :7], 3).tolist()}")
        print(f"  onnx velocity norm: {float(np.linalg.norm(onnx_vel)):.3e}")
    except Exception as e:
        print(f"  expert ONNX failed: {e}")

    return {"done": True}


@app.local_entrypoint()
def main():
    result = stage_diff.remote()
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)
