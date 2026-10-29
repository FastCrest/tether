# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Prepare the GR00T N1.7 action head (vlln + VLSA + AlternateVLDiT + encoders/decoder).

Follows upstream ``Gr00tN1d7ActionHead``: a 4-block 2048-wide self-attention stack over
the backbone features, then a 32-block 1536-wide DiT whose odd blocks self-attend over
[state, 40 actions] and whose even blocks cross-attend to text (idx % 4 == 0) or image
(idx % 4 == 2) rows. Everything that depends only on the flow step (timestep embedding,
AdaLN modulations, output modulation, the time half of the action encoder) is
precomputed for the four upstream steps ``t_disc = 0, 250, 500, 750``.
Embodiment-specific linears are sliced to the served embodiment at load.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from tether.accel.kernels.fp8 import quantize_weight

P = "action_head."
DIT_BLOCKS, VLSA_BLOCKS = 32, 4


@dataclass
class VlsaBlock:
    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    qkv: object
    o: object
    ln3_w: torch.Tensor
    ln3_b: torch.Tensor
    ff1: object
    ff2: object


@dataclass
class DitBlock:
    cross: bool
    text: bool  # cross blocks: text rows (else image rows)
    qkv: object  # self: [4608, 1536]; cross: q only [1536, 1536]
    kv: object  # cross: [3072, 2048] projecting encoder rows; None for self blocks
    o: object
    ff1: object
    ff2: object


@dataclass
class GrootWeights:
    vlln_w: torch.Tensor
    vlln_b: torch.Tensor
    vlsa: list
    dit: list
    mod: torch.Tensor  # [steps, 32, 2, 1536] (scale, shift)
    out_mod: torch.Tensor  # [steps, 2, 1536] (shift, scale)
    proj_out_w: torch.Tensor  # [1024, 1536]
    proj_out_b: torch.Tensor
    pos: torch.Tensor  # [40, 1536]
    state_w1: torch.Tensor  # [132, 1024] (x @ W layout)
    state_b1: torch.Tensor
    state_w2: torch.Tensor
    state_b2: torch.Tensor
    act_w: torch.Tensor  # W1 @ W2[:1536]: [132, 1536]
    act_c: torch.Tensor  # [steps, 1536] time half + folded biases
    act_w3: torch.Tensor
    act_b3: torch.Tensor
    dec_w1: torch.Tensor
    dec_b1: torch.Tensor
    dec_w2: torch.Tensor
    dec_b2: torch.Tensor
    t_disc: list
    dt: float
    embodiment: int


def _timestep_embedding(t, dim=256, max_period=10000.0, shift=1.0):
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32) / (half - shift)
    emb = torch.tensor(t, dtype=torch.float32)[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], 1)  # flip_sin_to_cos


def _action_sinusoid(t, dim=1536):
    half = dim // 2
    g = torch.exp(-torch.arange(half, dtype=torch.float32) * math.log(10000.0) / half)
    a = torch.tensor(t, dtype=torch.float32)[:, None] * g[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


def embodiment_id(checkpoint_dir, tag):
    """The checkpoint's own slot for ``tag``; no built-in mapping is ever substituted."""
    mapping = json.loads((Path(checkpoint_dir) / "embodiment_id.json").read_text())
    if tag not in mapping:
        raise ValueError(f"embodiment {tag!r} is not declared by the checkpoint")
    slot = mapping[tag]
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError(f"invalid embodiment slot {slot!r} for {tag!r}")
    return slot


def load_groot_weights(checkpoint_dir, device, embodiment: int, steps: int = 4) -> GrootWeights:
    from safetensors import safe_open

    root = Path(checkpoint_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    files = {f: safe_open(str(root / f), "pt") for f in set(index.values())}
    dev = torch.device(device)

    def get(name):
        return files[index[P + name]].get_tensor(P + name).to(device=dev, dtype=torch.float32)

    def cat_w(name):
        w, b = get(name + ".W"), get(name + ".b")
        if not 0 <= embodiment < w.shape[0]:
            raise ValueError(f"embodiment slot {embodiment} is outside {name}'s {w.shape[0]} slots")
        return w[embodiment].contiguous(), b[embodiment].contiguous()

    vlsa = []
    for i in range(VLSA_BLOCKS):
        p = f"vl_self_attention.transformer_blocks.{i}."
        qkv = torch.cat([get(p + f"attn1.to_{n}.weight") for n in "qkv"])
        qkv_b = torch.cat([get(p + f"attn1.to_{n}.bias") for n in "qkv"])
        vlsa.append(
            VlsaBlock(
                get(p + "norm1.weight"),
                get(p + "norm1.bias"),
                quantize_weight(qkv, qkv_b),
                quantize_weight(get(p + "attn1.to_out.0.weight"), get(p + "attn1.to_out.0.bias")),
                get(p + "norm3.weight"),
                get(p + "norm3.bias"),
                quantize_weight(get(p + "ff.net.0.proj.weight"), get(p + "ff.net.0.proj.bias")),
                quantize_weight(get(p + "ff.net.2.weight"), get(p + "ff.net.2.bias")),
            )
        )

    t_disc = [int(k / steps * 1000) for k in range(steps)]
    temb_in = _timestep_embedding(t_disc).to(torch.bfloat16).float().to(dev)
    h = torch.nn.functional.silu(
        temb_in @ get("model.timestep_encoder.timestep_embedder.linear_1.weight").t()
        + get("model.timestep_encoder.timestep_embedder.linear_1.bias")
    )
    temb = h @ get("model.timestep_encoder.timestep_embedder.linear_2.weight").t() + get(
        "model.timestep_encoder.timestep_embedder.linear_2.bias"
    )
    stemb = torch.nn.functional.silu(temb)

    dit, mods = [], []
    for i in range(DIT_BLOCKS):
        p = f"model.transformer_blocks.{i}."
        cross = i % 2 == 0
        if cross:
            qkv = quantize_weight(get(p + "attn1.to_q.weight"), get(p + "attn1.to_q.bias"))
            kv = quantize_weight(
                torch.cat([get(p + "attn1.to_k.weight"), get(p + "attn1.to_v.weight")]),
                torch.cat([get(p + "attn1.to_k.bias"), get(p + "attn1.to_v.bias")]),
            )
        else:
            qkv = quantize_weight(
                torch.cat([get(p + f"attn1.to_{n}.weight") for n in "qkv"]),
                torch.cat([get(p + f"attn1.to_{n}.bias") for n in "qkv"]),
            )
            kv = None
        dit.append(
            DitBlock(
                cross,
                i % 4 == 0,
                qkv,
                kv,
                quantize_weight(get(p + "attn1.to_out.0.weight"), get(p + "attn1.to_out.0.bias")),
                quantize_weight(get(p + "ff.net.0.proj.weight"), get(p + "ff.net.0.proj.bias")),
                quantize_weight(get(p + "ff.net.2.weight"), get(p + "ff.net.2.bias")),
            )
        )
        m = stemb @ get(p + "norm1.linear.weight").t() + get(
            p + "norm1.linear.bias"
        )  # [steps, 3072]
        mods.append(m.view(steps, 2, -1))  # scale, shift
    out = stemb @ get("model.proj_out_1.weight").t() + get("model.proj_out_1.bias")
    s_w1, s_b1 = cat_w("state_encoder.layer1")
    s_w2, s_b2 = cat_w("state_encoder.layer2")
    a_w1, a_b1 = cat_w("action_encoder.W1")
    a_w2, a_b2 = cat_w("action_encoder.W2")
    a_w3, a_b3 = cat_w("action_encoder.W3")
    d_w1, d_b1 = cat_w("action_decoder.layer1")
    d_w2, d_b2 = cat_w("action_decoder.layer2")
    width = a_w1.shape[1]
    tau = _action_sinusoid(t_disc, width).to(torch.bfloat16).float().to(dev)
    w2a, w2t = a_w2[:width], a_w2[width:]
    weights = GrootWeights(
        vlln_w=get("vlln.weight"),
        vlln_b=get("vlln.bias"),
        vlsa=vlsa,
        dit=dit,
        mod=torch.stack(mods, 1).contiguous(),
        out_mod=out.view(steps, 2, -1).contiguous(),
        proj_out_w=get("model.proj_out_2.weight").t().contiguous(),
        proj_out_b=get("model.proj_out_2.bias"),
        pos=get("position_embedding.weight")[:40].contiguous(),
        state_w1=s_w1,
        state_b1=s_b1,
        state_w2=s_w2,
        state_b2=s_b2,
        act_w=(a_w1 @ w2a).contiguous(),
        act_c=(tau @ w2t + a_b1 @ w2a + a_b2).contiguous(),
        act_w3=a_w3,
        act_b3=a_b3,
        dec_w1=d_w1,
        dec_b1=d_b1,
        dec_w2=d_w2,
        dec_b2=d_b2,
        t_disc=t_disc,
        dt=1.0 / steps,
        embodiment=embodiment,
    )
    torch.cuda.empty_cache()
    return weights
