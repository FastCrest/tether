"""ExpertStack — shared flow-matching action expert for pi0 / pi05 / SmolVLA.

Moved here in Day 4g cleanup (lift #1 basevla-spine) so the spine's
`FlowMatchingHead` doesn't reach into `reflex.exporters/*` to find these
primitives. The exporters package now imports these from here.

The 4 classes / helpers in this file:

- `_sinusoidal_pos_embedding(t, dim)` — flow-matching time embedding
  (matches lerobot's `create_sinusoidal_pos_embedding` with [sin, cos]
  order + 2π scaling factor).
- `_DecomposedRoPE` — ONNX-friendly rotary embedding (cached cos/sin).
- `ExpertGQALayer` — single GQA layer with three attention modes:
  self-attention, cross-attention (SmolVLA), and block-causal prefix
  concat (pi0's PaliGemmaWithExpertModel).
- `ExpertStack` — full expert stack wrapping N layers + suffix + action
  projection + final norm. The thing wrapped by `FlowMatchingHead`.

Backwards compat: `reflex.exporters.smolvla_exporter` re-exports all 4
from this module so any external code that still imports from the old
path keeps working.

Family-specific variants stay in their exporter files for now:

- `Pi05ExpertStack` (AdaRMSNorm variant) → `exporters/pi0_exporter.py`
- `Pi0ExpertStackWithPrefix` → `exporters/pi0_prefix.py`
- `GR00TExpertStack` → `exporters/gr00t_exporter.py`

They'll migrate here when their families get spine-decomposed in
Days 5, 7, 9.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from reflex.decompose import DecomposedRMSNorm


def _sinusoidal_pos_embedding(t, dim, min_p=4e-3, max_p=4.0):
    """Matches lerobot's ``create_sinusoidal_pos_embedding`` exactly.

    Earlier version was missing the ``2π`` scaling factor AND used [cos, sin]
    order instead of [sin, cos]. Both wrong. Time signal to the expert was
    therefore completely mis-phased, making every denoising step operate at
    the wrong "time," which cascaded into flow-matching catastrophic drift.

    Lerobot computes `fraction` / `period` / `scaling_factor` in **float64**
    (modeling_pi0.py:90-95), then `sin_input` in fp64 (because scaling is
    fp64), then sin/cos of fp64, finally `.type_as(timestep.dtype)`. Computing
    in fp32 throughout introduces a tiny per-element drift in time_emb that
    compounds through 18 AdaRMSNorm layers and shows up in pi0.5 v_t parity.
    """
    assert dim % 2 == 0
    # Use fp64 for fraction/period/scaling to match lerobot precision exactly.
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float64)
    period = min_p * (max_p / min_p) ** fraction
    scaling = (1.0 / period) * 2 * math.pi  # [dim/2] in fp64
    angle = t.unsqueeze(-1).double() * scaling.unsqueeze(0)  # [B, dim/2] in fp64
    return torch.cat([angle.sin(), angle.cos()], dim=-1).to(t.dtype)


class _DecomposedRoPE(nn.Module):
    # max_seq_len=2048 covers pi0 full inference (3 images × 256 + lang ~256 +
    # state 1 + chunk_size 50 ≈ 1100 absolute positions; was 512 which OOB'd
    # at position ~775 during the Day 4h parity run).
    def __init__(self, dim, max_seq_len=2048, base=10000.0, bf16_precision: bool = False):
        super().__init__()
        self.dim = dim
        self.base = base
        # Match stock GemmaRotaryEmbedding's inv_freq EXACTLY (incl. int64 arange).
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        # For models loaded in bf16 (e.g., lerobot's pi0.5 default precision),
        # stock's inv_freq buffer goes through bf16 cast on policy.to(bf16) and
        # is then cast back to fp32 — losing precision irreversibly. To match
        # lerobot's effective inv_freq for parity, simulate the same roundtrip.
        if bf16_precision:
            inv_freq = inv_freq.to(torch.bfloat16).to(torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Cached cos/sin (used by SmolVLA / pi0 path). Computed from the
        # (possibly bf16-rounded) inv_freq above for consistency.
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq.float())
        self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
        self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

    def apply(self, x, position_ids):
        # Compute cos/sin AT RUNTIME via stock's matmul path. Even though
        # mathematically equivalent to the pre-cached lookup, BLAS GEMM
        # rounding can differ subtly from outer-product+cos. Day 5 Phase B
        # parity run-11 showed element-specific divergence in Q post-RoPE
        # despite local test confirming bit-identical math — suspect runtime
        # context (numerical precision via stock's matmul-based freq compute)
        # vs cached lookup explains it.
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype=x.dtype)
            sin = emb.sin().to(dtype=x.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class ExpertGQALayer(nn.Module):
    """Single expert transformer layer with decomposed ops for ONNX export."""

    def __init__(self, hidden, nq, nkv, hd, inter, kv_in=None, rope_theta=100000.0):
        super().__init__()
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.kv_groups = nq // nkv
        self.input_layernorm = DecomposedRMSNorm(torch.ones(hidden))
        self.post_attention_layernorm = DecomposedRMSNorm(torch.ones(hidden))
        self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
        self.k_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
        self.v_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        # SmolLM2 / SmolVLM2 uses rope_theta=100000 (not the Llama default 10000).
        # Wrong base → wrong frequency per position → corrupts attention.
        self.rope = _DecomposedRoPE(hd, base=rope_theta)

    def forward(
        self,
        x,
        pos_ids,
        cross_k=None,
        cross_v=None,
        kv_mask=None,
        prefix_k_concat=None,
        prefix_v_concat=None,
        attn_mask=None,
    ):
        """Run one transformer layer.

        Three attention modes, mutually exclusive:
        1. Self-attention (default): k, v come from x (action tokens).
        2. Cross-attention: `cross_k`/`cross_v` REPLACE action k/v entirely;
           used by SmolVLA's forward_cross_attn_layer pattern.
        3. Block-causal prefix concat: `prefix_k_concat`/`prefix_v_concat`
           are prepended onto action-side k/v AFTER RoPE; attention spans
           prefix+action tokens. Used by pi0's PaliGemmaWithExpertModel where
           the VLM's per-layer past_key_values form the prefix.

        For cross-attn, ``kv_mask`` is an optional ``[B, kv_len]`` boolean
        tensor marking valid KV tokens. Padded positions get -inf logits.

        For block-causal prefix, the prefix tensors are already RoPE'd by the
        backbone (their absolute position is fixed during the denoise loop).
        """
        b, s, _ = x.shape
        res = x
        x = self.input_layernorm(x)
        q = self.q_proj(x).view(b, s, self.nq, self.hd).transpose(1, 2)

        is_cross = cross_k is not None
        use_prefix_concat = prefix_k_concat is not None
        k_src = cross_k if is_cross else x
        v_src = cross_v if is_cross else x
        action_kv_len = k_src.shape[1]

        k = self.k_proj(k_src).view(b, action_kv_len, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(v_src).view(b, action_kv_len, self.nkv, self.hd).transpose(1, 2)
        q = self.rope.apply(q, pos_ids)
        if not is_cross:
            k = self.rope.apply(k, pos_ids)

        # Block-causal prefix concat: prepend prefix_kv onto action k/v.
        # Expected shapes: prefix_k_concat [B, nkv, prefix_len, hd]
        # (already in post-transpose layout, RoPE-applied by backbone).
        if use_prefix_concat:
            # Accept either [B, nkv, prefix_len, hd] (post-transpose) or
            # [B, prefix_len, nkv, hd] (pre-transpose) shape.
            pk = prefix_k_concat
            pv = prefix_v_concat
            if pk.ndim == 4 and pk.shape[1] != self.nkv:
                pk = pk.transpose(1, 2)
                pv = pv.transpose(1, 2)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        kv_len = k.shape[2]
        k = k.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)
        v = v.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)  # [B, nq, s, kv_len]
        if is_cross and kv_mask is not None:
            # Cross-attn padded KV mask: set padded scores to large negative.
            mask = kv_mask[:, None, None, :]
            scores = scores.masked_fill(~mask, -1e9)
        elif use_prefix_concat and attn_mask is not None:
            # Prefix-concat path bool mask: [B, 1, s, kv_len]. True = attendable.
            # Used by pi0's block-attention pattern (state isolated from actions,
            # actions mutually visible, both see prefix bidirectionally).
            scores = scores.masked_fill(~attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        x = res + self.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1))
        res = x
        x = self.post_attention_layernorm(x)
        # MLP: GeMM uses `gelu_pytorch_tanh` (Gemma's default hidden_act per
        # transformers/models/gemma/configuration_gemma.py:119). SmolVLA / SmolLM2
        # uses silu but that's a different family. Verified via parity diff:
        # gate_proj, up_proj outputs match lerobot bit-identically; the composition
        # step (gate's activation * up) was the divergence.
        return res + self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class ExpertStack(nn.Module):
    """Full expert stack for ONNX export (single denoising step)."""

    def __init__(self, layers, expert_hidden, action_dim, cross_indices, vlm_kv_dim,
                 suffix_weights, action_proj_weights, final_norm_weight):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.expert_hidden = expert_hidden
        self.cross_indices = set(cross_indices)
        self.vlm_kv_dim = vlm_kv_dim

        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        self.action_in_proj.weight = nn.Parameter(suffix_weights["in_w"])
        self.action_in_proj.bias = nn.Parameter(suffix_weights["in_b"])
        self.action_time_mlp_in.weight = nn.Parameter(suffix_weights["t_in_w"])
        self.action_time_mlp_in.bias = nn.Parameter(suffix_weights["t_in_b"])
        self.action_time_mlp_out.weight = nn.Parameter(suffix_weights["t_out_w"])
        self.action_time_mlp_out.bias = nn.Parameter(suffix_weights["t_out_b"])

        self.action_out_proj = nn.Linear(expert_hidden, action_dim)
        self.action_out_proj.weight = nn.Parameter(action_proj_weights["w"])
        self.action_out_proj.bias = nn.Parameter(action_proj_weights["b"])

        self.final_norm = DecomposedRMSNorm(final_norm_weight)

    def forward(
        self,
        noisy_actions,
        timestep,
        position_ids,
        vlm_k: torch.Tensor | None = None,
        vlm_v: torch.Tensor | None = None,
        prefix_offset: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ):
        """Run one denoising step.

        ``vlm_k`` and ``vlm_v`` are PER-LAYER tensors of shape
        ``[L, B, seq, kv_dim]`` where ``L`` equals the number of expert
        layers. For each cross-attn layer ``i``:
            - ``vlm_k[i]`` = VLM's layer-i k_proj output, RoPE-applied.
            - ``vlm_v[i]`` = VLM's layer-i v_proj output, no RoPE.

        Expert's k_proj/v_proj further project these into expert-head space.
        Matches real SmolVLA (smolvlm_with_expert.py::forward_cross_attn_layer).
        """
        b, c, _ = noisy_actions.shape
        act = self.action_in_proj(noisy_actions)
        t_emb = _sinusoidal_pos_embedding(timestep, self.expert_hidden)
        t_emb = t_emb.unsqueeze(1).expand(-1, c, -1)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([act, t_emb], dim=-1))))

        # Self-attention layers use position_ids OFFSET by the prefix length —
        # this matches denoise_step in real SmolVLA which does
        # `prefix_offsets + cumsum(suffix_pad_masks) - 1`. Cross-attention
        # layers keep position_ids [0..chunk-1] (matching the renormalisation
        # real code does in forward_cross_attn_layer).
        self_pos_ids = position_ids
        if prefix_offset is not None:
            self_pos_ids = position_ids + prefix_offset

        for i, layer in enumerate(self.layers):
            if i in self.cross_indices:
                if vlm_k is None or vlm_v is None:
                    layer_k = torch.zeros(
                        b, 1, self.vlm_kv_dim, device=x.device, dtype=x.dtype
                    )
                    layer_v = layer_k
                else:
                    layer_k = vlm_k[i]
                    layer_v = vlm_v[i]
                x = layer(x, position_ids, cross_k=layer_k, cross_v=layer_v, kv_mask=kv_mask)
            else:
                x = layer(x, self_pos_ids)

        x = self.final_norm(x)
        return self.action_out_proj(x)


class Pi05ExpertGQALayer(nn.Module):
    """pi0.5 expert layer — AdaRMSNorm + prefix-concat + AdaLN gating.

    Mirrors `ExpertGQALayer` (the pi0/SmolVLA layer) but with the three
    pi0.5-specific changes:

    1. **AdaRMSNorm** (input + post_attention) — per-layer normalization
       conditioned on the time embedding. Returns `(normed, gate)` where
       `gate` modulates the residual SIDE (AdaLN pattern).
    2. **AdaLN gating** — residual = res + block_out * gate (NOT plain
       `res + block_out`). The gate comes from the SAME norm call as the
       pre-block normalization. Matches lerobot `_gated_residual` at
       `pi_gemma.py:54-66` and `_PiGemmaDecoderLayerBase.forward:158-188`.
    3. **MLP activation** — `gelu(approximate="tanh")` (Gemma default per
       `transformers/models/gemma/configuration_gemma.py:119`), NOT silu.
       Day 4h surfaced this bug in pi0's ExpertGQALayer; pi0.5 inherits the
       fix.

    Attention modes: SAME as ExpertGQALayer — self-attn (default),
    cross-attn (SmolVLA), or block-causal prefix concat (pi0.5).
    """

    def __init__(self, hidden, nq, nkv, hd, inter, kv_in=None, rope_theta=10000.0, bf16_inv_freq: bool = True):
        super().__init__()
        from reflex.decompose import DecomposedAdaRMSNorm
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.kv_groups = nq // nkv
        self.input_layernorm = DecomposedAdaRMSNorm(hidden, time_dim=hidden)
        self.post_attention_layernorm = DecomposedAdaRMSNorm(hidden, time_dim=hidden)
        self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
        self.k_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
        self.v_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        # bf16_inv_freq=True: cast inv_freq through bf16 to match lerobot's pi0.5
        # checkpoint precision (loaded in bf16 by default per
        # PaliGemmaWithExpertModel.__init__ precision arg).
        self.rope = _DecomposedRoPE(hd, base=rope_theta, bf16_precision=bf16_inv_freq)

    # Debug: set to a dict at class level to enable intra-attention capture.
    # Each layer writes to debug_captures[f"{layer_id}_{stage}"]. Set
    # to None (default) to disable.
    debug_captures: dict | None = None
    debug_layer_id: int = 0

    def forward(
        self,
        x,
        pos_ids,
        time_emb,
        cross_k=None,
        cross_v=None,
        kv_mask=None,
        prefix_k_concat=None,
        prefix_v_concat=None,
        attn_mask=None,
    ):
        """One pi0.5 transformer layer with AdaLN gating.

        Args:
            x: `[B, suffix_len, hidden]` action-only suffix (no state token
                — pi0.5 uses state-in-language).
            pos_ids: `[B, suffix_len]` absolute position ids (offset by prefix_len).
            time_emb: `[B, hidden]` time-conditioned embedding for AdaRMSNorm.
            prefix_k_concat / prefix_v_concat: per-layer K/V from PaliGemma prefill.
            attn_mask: optional bool `[B, 1, S, K]` for block attention.

        Other kwargs mirror ExpertGQALayer.
        """
        b, s, _ = x.shape
        # Input layernorm — returns (normed, gate) for AdaLN gating
        res = x
        x_norm, gate_attn = self.input_layernorm(x, time_emb, return_gate=True)

        q = self.q_proj(x_norm).view(b, s, self.nq, self.hd).transpose(1, 2)
        if Pi05ExpertGQALayer.debug_captures is not None:
            lid = Pi05ExpertGQALayer.debug_layer_id
            Pi05ExpertGQALayer.debug_captures[f"L{lid}_q_pre_rope"] = q.detach().clone()

        is_cross = cross_k is not None
        use_prefix_concat = prefix_k_concat is not None
        k_src = cross_k if is_cross else x_norm
        v_src = cross_v if is_cross else x_norm
        action_kv_len = k_src.shape[1]

        k = self.k_proj(k_src).view(b, action_kv_len, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(v_src).view(b, action_kv_len, self.nkv, self.hd).transpose(1, 2)
        q = self.rope.apply(q, pos_ids)
        if not is_cross:
            k = self.rope.apply(k, pos_ids)
        if Pi05ExpertGQALayer.debug_captures is not None:
            lid = Pi05ExpertGQALayer.debug_layer_id
            Pi05ExpertGQALayer.debug_captures[f"L{lid}_q_post_rope"] = q.detach().clone()
            Pi05ExpertGQALayer.debug_captures[f"L{lid}_k_post_rope"] = k.detach().clone()
            Pi05ExpertGQALayer.debug_captures[f"L{lid}_v"] = v.detach().clone()

        if use_prefix_concat:
            pk = prefix_k_concat
            pv = prefix_v_concat
            if pk.ndim == 4 and pk.shape[1] != self.nkv:
                pk = pk.transpose(1, 2)
                pv = pv.transpose(1, 2)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        if Pi05ExpertGQALayer.debug_captures is not None:
            lid = Pi05ExpertGQALayer.debug_layer_id
            Pi05ExpertGQALayer.debug_captures[f"L{lid}_k_concat"] = k.detach().clone()

        kv_len = k.shape[2]
        k = k.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)
        v = v.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)

        # Match stock GemmaAttention exactly: multiply by scaling, additive
        # mask, explicit fp32 softmax. These should be mathematically equivalent
        # for fp32 inputs.
        scaling = self.hd ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        if is_cross and kv_mask is not None:
            mask = kv_mask[:, None, None, :]
            scores = scores.masked_fill(~mask, -1e9)
        elif use_prefix_concat and attn_mask is not None:
            additive_mask = torch.zeros_like(scores)
            additive_mask = additive_mask.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            scores = scores + additive_mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        if Pi05ExpertGQALayer.debug_captures is not None:
            lid = Pi05ExpertGQALayer.debug_layer_id
            Pi05ExpertGQALayer.debug_captures[f"L{lid}_attn_weights"] = attn.detach().clone()

        attn_out = self.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1))
        # AdaLN gating: res + attn_out * gate (NOT plain residual)
        x = res + attn_out * gate_attn

        # MLP block — same pattern: post-layernorm returns (normed, gate_mlp)
        res = x
        x_norm, gate_mlp = self.post_attention_layernorm(x, time_emb, return_gate=True)
        mlp_out = self.down_proj(F.gelu(self.gate_proj(x_norm), approximate="tanh") * self.up_proj(x_norm))
        return res + mlp_out * gate_mlp


__all__ = [
    "_sinusoidal_pos_embedding",
    "_DecomposedRoPE",
    "ExpertGQALayer",
    "Pi05ExpertGQALayer",
    "ExpertStack",
]
