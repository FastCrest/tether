"""Isolate the expert: feed the SAME pre-computed VLM k/v to both PyTorch and
our expert_stack.onnx, compare velocity output.

If cos < ~0.95, the expert_stack ONNX export has a bug — the per-layer k/v
we've verified are correct aren't being used correctly by the expert.
"""
import numpy as np
import torch


def main():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("Loading policy ...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval().to(dtype=torch.float32).to("cpu")
    cfg = policy.config

    # Build a synthetic fully-assembled prefix via the real embed_prefix path.
    rng = np.random.RandomState(42)
    img = rng.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = (img_t * 2.0 - 1.0).unsqueeze(0)  # [1, 3, 512, 512]
    state8_t = torch.from_numpy(rng.randn(8).astype(np.float32) * 0.1).unsqueeze(0)

    images = [img_t, img_t, img_t]  # 3 cameras
    img_masks = [torch.ones(1, dtype=torch.bool)] * 3

    # Fake lang tokens (doesn't matter much for this test — same input both paths)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ids = tok("put the red bowl on the plate\n", return_tensors="pt",
              padding="max_length", truncation=True, max_length=48)
    lang_tokens = ids["input_ids"]
    lang_masks = ids["attention_mask"].bool()

    # Pad state to 32
    state_padded = torch.zeros(1, 32)
    state_padded[:, :8] = state8_t

    # ── Run PyTorch embed_prefix + populate KV cache ────────────────
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state_padded
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_kv = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
    print(f"  prefix_seq_len: {prefix_embs.shape[1]}")
    print(f"  num layers in past_kv: {len(past_kv)}")
    # past_kv layout: dict mapping layer_idx -> {"key_states": [...], "value_states": [...]}
    sample_k = past_kv[0]["key_states"]
    sample_v = past_kv[0]["value_states"]
    print(f"  per-layer key_states shape: {tuple(sample_k.shape)}")
    print(f"  per-layer value_states shape: {tuple(sample_v.shape)}")

    # Build per-layer stacked [L, B, seq, kv_dim] for our expert
    num_layers = policy.model.vlm_with_expert.num_vlm_layers
    # past_kv[i]["key_states"] shape: [B, num_kv_heads, seq, head_dim] → flatten to [B, seq, num_kv_heads*head_dim]
    def flatten_kv(t):
        # Real past_kv has layout [B, seq, nkv, head_dim] already.
        b, s, nkv, hd = t.shape
        return t.contiguous().view(b, s, nkv * hd)

    stacked_k = torch.stack(
        [flatten_kv(past_kv[i]["key_states"]) for i in range(num_layers)], dim=0
    )   # [L, B, seq, kv_dim]
    stacked_v = torch.stack(
        [flatten_kv(past_kv[i]["value_states"]) for i in range(num_layers)], dim=0
    )
    print(f"  stacked_k shape: {tuple(stacked_k.shape)}")

    # ── Run ONE denoising step in PyTorch ──────────────────────────
    chunk = cfg.chunk_size
    max_action = cfg.max_action_dim
    noise = torch.from_numpy(np.random.RandomState(7).randn(1, chunk, max_action).astype(np.float32))
    t = torch.tensor([0.5])

    with torch.no_grad():
        # denoise_step returns velocity
        v_torch = policy.model.denoise_step(
            x_t=noise,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_kv,
            timestep=t.expand(1),
        )
    print(f"\n  torch velocity shape: {tuple(v_torch.shape)}")
    print(f"  torch first velocity[0, :7]: {np.round(v_torch[0, 0, :7].numpy(), 3).tolist()}")

    # ── Run our expert ONNX with stacked k/v ────────────────────────
    import onnxruntime as ort
    sess = ort.InferenceSession("/tmp/tether_libero_export3/expert_stack.onnx",
                                providers=["CPUExecutionProvider"])
    pos_ids = torch.arange(chunk).unsqueeze(0).numpy().astype(np.int64)
    prefix_off = np.array([[prefix_embs.shape[1]]], dtype=np.int64)  # [B, 1]
    feed = {
        "noisy_actions": noise.numpy().astype(np.float32),
        "timestep": t.numpy().astype(np.float32),
        "position_ids": pos_ids,
        "vlm_k": stacked_k.numpy().astype(np.float32),
        "vlm_v": stacked_v.numpy().astype(np.float32),
        "prefix_offset": prefix_off,
    }
    v_onnx = sess.run(None, feed)[0]  # [B, chunk, action_dim]
    print(f"\n  onnx velocity shape: {v_onnx.shape}")
    print(f"  onnx first velocity[0, :7]: {np.round(v_onnx[0, 0, :7], 3).tolist()}")

    # ── Compare ─────────────────────────────────────────────────────
    t_flat = v_torch[0].numpy().flatten().astype(np.float64)
    o_flat = v_onnx[0][:, :max_action].flatten().astype(np.float64)
    if len(o_flat) > len(t_flat):
        o_flat = o_flat[:len(t_flat)]
    cos = float(np.dot(t_flat, o_flat) / (np.linalg.norm(t_flat) * np.linalg.norm(o_flat) + 1e-8))
    l2 = float(np.linalg.norm(t_flat - o_flat))
    max_abs = float(np.abs(t_flat - o_flat).max())
    print(f"\n  velocity cos={cos:+.4f}  L2={l2:.3e}  max_abs={max_abs:.3e}")
    print(f"  torch||={np.linalg.norm(t_flat):.3e}  onnx||={np.linalg.norm(o_flat):.3e}")


if __name__ == "__main__":
    main()
