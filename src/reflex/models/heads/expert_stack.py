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
- `Pi0ExpertStackWithPrefix` → `exporters/pi0_prefix_exporter.py`
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
    """
    assert dim % 2 == 0
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=t.dtype)
    period = min_p * (max_p / min_p) ** fraction
    scaling = (1.0 / period) * 2 * math.pi  # [dim/2]
    angle = t.unsqueeze(-1) * scaling.unsqueeze(0)  # [B, dim/2]
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class _DecomposedRoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
        self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

    def apply(self, x, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
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
        attn = F.softmax(scores, dim=-1)
        x = res + self.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1))
        res = x
        x = self.post_attention_layernorm(x)
        return res + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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


__all__ = [
    "_sinusoidal_pos_embedding",
    "_DecomposedRoPE",
    "ExpertGQALayer",
    "ExpertStack",
]
