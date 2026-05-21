"""Compare ONE expert layer forward pass: ours vs real. Reference for bisecting.

Steps:
  A. Load real expert layer 0
  B. Copy its weights into our ExpertGQALayer
  C. Feed identical inputs
  D. Compare outputs at each sub-stage
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from reflex.models.heads.expert_stack import ExpertGQALayer, _DecomposedRoPE

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval().to(dtype=torch.float32).to("cpu")

    import sys
    layer_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # layer 1 is cross
    real_layer = policy.model.vlm_with_expert.lm_expert.layers[layer_idx]
    print(f"Testing layer {layer_idx}")
    print(f"real q_proj.weight: {real_layer.self_attn.q_proj.weight.shape}")
    print(f"real k_proj.weight: {real_layer.self_attn.k_proj.weight.shape}")
    print(f"real v_proj.weight: {real_layer.self_attn.v_proj.weight.shape}")
    print(f"real o_proj.weight: {real_layer.self_attn.o_proj.weight.shape}")
    expert_hidden = real_layer.self_attn.q_proj.weight.shape[1]
    qout = real_layer.self_attn.q_proj.weight.shape[0]
    kout = real_layer.self_attn.k_proj.weight.shape[0]
    kin = real_layer.self_attn.k_proj.weight.shape[1]

    # Check attention config
    conf = real_layer.self_attn.config
    print(f"config: nq={conf.num_attention_heads} nkv={conf.num_key_value_heads} hd={conf.head_dim}")
    nq = conf.num_attention_heads
    nkv = conf.num_key_value_heads
    hd = conf.head_dim

    is_cross = (kin != expert_hidden)
    print(f"layer 0: expert_hidden={expert_hidden}, is_cross={is_cross}")

    # Build our layer and copy weights
    our = ExpertGQALayer(
        expert_hidden, nq, nkv, hd,
        inter=real_layer.mlp.gate_proj.weight.shape[0],
        kv_in=kin if is_cross else None,
        rope_theta=conf.rope_theta,
    )
    our.eval()
    our.input_layernorm.weight.data.copy_(real_layer.input_layernorm.weight)
    our.post_attention_layernorm.weight.data.copy_(real_layer.post_attention_layernorm.weight)
    our.q_proj.weight.data.copy_(real_layer.self_attn.q_proj.weight)
    our.k_proj.weight.data.copy_(real_layer.self_attn.k_proj.weight)
    our.v_proj.weight.data.copy_(real_layer.self_attn.v_proj.weight)
    our.o_proj.weight.data.copy_(real_layer.self_attn.o_proj.weight)
    our.gate_proj.weight.data.copy_(real_layer.mlp.gate_proj.weight)
    our.up_proj.weight.data.copy_(real_layer.mlp.up_proj.weight)
    our.down_proj.weight.data.copy_(real_layer.mlp.down_proj.weight)

    # Build input
    rng = np.random.RandomState(42)
    x = torch.from_numpy(rng.randn(1, 5, expert_hidden).astype(np.float32))
    pos_ids = torch.arange(5).unsqueeze(0)

    # Generate a synthetic cross_k / cross_v for if is_cross
    cross_k = torch.from_numpy(rng.randn(1, 10, kin).astype(np.float32)) if is_cross else None
    cross_v = torch.from_numpy(rng.randn(1, 10, kin).astype(np.float32)) if is_cross else None

    # ── Run our layer ───────────────────────────────────────────────
    with torch.no_grad():
        if is_cross:
            our_out = our(x, pos_ids, cross_k=cross_k, cross_v=cross_v)
        else:
            our_out = our(x, pos_ids)
    print(f"\nOur layer output: shape={tuple(our_out.shape)} norm={our_out.norm():.4f}")

    # ── Run real layer ──────────────────────────────────────────────
    # HF Llama layer.forward takes hidden_states, attention_mask, position_ids, etc.
    # But for our isolated test with cross-attn, we can't use its forward directly
    # because cross-attn uses kv from outside. Let's reconstruct the forward manually.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    try:
        rotary_emb = policy.model.vlm_with_expert.lm_expert.rotary_emb
    except AttributeError:
        rotary_emb = real_layer.self_attn.rotary_emb

    with torch.no_grad():
        # For self-attn layer reconstruction (no cross)
        res = x.clone()
        h_norm = real_layer.input_layernorm(x)
        q = real_layer.self_attn.q_proj(h_norm).view(1, 5, nq, hd).transpose(1, 2)

        if is_cross:
            k_src = cross_k
            v_src = cross_v
        else:
            k_src = h_norm
            v_src = h_norm
        kv_len = k_src.shape[1]
        k = real_layer.self_attn.k_proj(k_src).view(1, kv_len, nkv, hd).transpose(1, 2)
        v = real_layer.self_attn.v_proj(v_src).view(1, kv_len, nkv, hd).transpose(1, 2)

        cos, sin = rotary_emb(x, pos_ids)
        q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
        if not is_cross:
            _, k = apply_rotary_pos_emb(torch.zeros_like(k), k, cos, sin)

        # GQA: repeat k, v
        k = k.repeat_interleave(nq // nkv, dim=1)
        v = v.repeat_interleave(nq // nkv, dim=1)

        att = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / hd**0.5, dim=-1)
        att_out = torch.matmul(att, v).transpose(1, 2).reshape(1, 5, nq * hd)
        att_out = real_layer.self_attn.o_proj(att_out)
        x_after_attn = res + att_out

        res2 = x_after_attn
        post_norm = real_layer.post_attention_layernorm(x_after_attn)
        gate = real_layer.mlp.gate_proj(post_norm)
        up = real_layer.mlp.up_proj(post_norm)
        mlp_out = real_layer.mlp.down_proj(F.silu(gate) * up)
        real_out = res2 + mlp_out
    print(f"Real layer output: shape={tuple(real_out.shape)} norm={real_out.norm():.4f}")

    diff = (real_out - our_out).abs()
    print(f"\nDiff: max_abs={diff.max():.4e}  mean={diff.mean():.4e}")
    cos = float(
        (real_out.flatten() @ our_out.flatten())
        / (real_out.norm() * our_out.norm() + 1e-8)
    )
    print(f"cos: {cos:+.4f}")


if __name__ == "__main__":
    main()
