# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Prepare a pi05 (PaliGemma + Gemma action expert) checkpoint for the accel engine.

Every large projection becomes a per-channel FP8 weight. Learned RMSNorm
weights of the language model are folded into the following projections in
FP32 (``W * (1 + w)``), so the runtime norm has no affine term. The action
expert's adaptive norms depend only on the flow time, so their
``(scale, shift, gate)`` rows are precomputed for every Euler step.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tether.accel.kernels.fp8 import Fp8Weight, quantize_weight

ROOT = "model.paligemma_with_expert."
VIS = ROOT + "paligemma.model.vision_tower.vision_model."
LM = ROOT + "paligemma.model.language_model."
EXPERT = ROOT + "gemma_expert.model."


@dataclass
class VisionLayer:
    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    qkv: Fp8Weight
    o: Fp8Weight
    ln2_w: torch.Tensor
    ln2_b: torch.Tensor
    fc1: Fp8Weight
    fc2: Fp8Weight


@dataclass
class DecoderLayer:
    qkv: Fp8Weight
    o: Fp8Weight | None
    gate_up: Fp8Weight | None
    down: Fp8Weight | None


@dataclass
class Pi05Weights:
    patch_w: torch.Tensor
    patch_b: torch.Tensor
    pos_emb: torch.Tensor
    vision: list[VisionLayer]
    post_ln_w: torch.Tensor
    post_ln_b: torch.Tensor
    projector: Fp8Weight
    prefix: list[DecoderLayer]
    expert: list[DecoderLayer]
    embed: torch.Tensor  # CPU [vocab, 2048] BF16 token table
    action_in_w: torch.Tensor
    action_in_b: torch.Tensor
    action_out_w: torch.Tensor
    action_out_b: torch.Tensor
    mod_attn: torch.Tensor  # [steps, L, 3, D] scale, shift, gate
    mod_ffn: torch.Tensor  # [steps, L, 3, D]
    mod_final: torch.Tensor  # [steps, 2, D] scale, shift
    steps: int
    config: dict = field(default_factory=dict)


def _sinusoid(t: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    angle = (1.0 / period * 2 * math.pi)[None, :] * t.double()[:, None]
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1).float()


def load_pi05_weights(checkpoint_dir, device, steps: int = 10) -> Pi05Weights:
    from safetensors import safe_open

    path = Path(checkpoint_dir)
    config = json.loads((path / "config.json").read_text())
    dev = torch.device(device)
    with safe_open(str(path / "model.safetensors"), "pt") as f:

        def get(name, dtype=torch.float32):
            return f.get_tensor(name).to(device=dev, dtype=dtype)

        vision = []
        for i in range(27):
            p = f"{VIS}encoder.layers.{i}."
            qkv_w = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
            qkv_b = torch.cat([get(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
            vision.append(
                VisionLayer(
                    ln1_w=get(p + "layer_norm1.weight"),
                    ln1_b=get(p + "layer_norm1.bias"),
                    qkv=quantize_weight(qkv_w, qkv_b),
                    o=quantize_weight(
                        get(p + "self_attn.out_proj.weight"), get(p + "self_attn.out_proj.bias")
                    ),
                    ln2_w=get(p + "layer_norm2.weight"),
                    ln2_b=get(p + "layer_norm2.bias"),
                    fc1=quantize_weight(get(p + "mlp.fc1.weight"), get(p + "mlp.fc1.bias")),
                    fc2=quantize_weight(get(p + "mlp.fc2.weight"), get(p + "mlp.fc2.bias")),
                )
            )
        projector = quantize_weight(
            get(ROOT + "paligemma.model.multi_modal_projector.linear.weight"),
            get(ROOT + "paligemma.model.multi_modal_projector.linear.bias"),
        )

        prefix = []
        for i in range(18):
            p = f"{LM}layers.{i}."
            g_in = 1.0 + get(p + "input_layernorm.weight")
            qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"]) * g_in[None, :]
            if i == 17:
                # Only the last layer's keys/values are consumed (by the action expert).
                prefix.append(DecoderLayer(quantize_weight(qkv), None, None, None))
                continue
            g_post = 1.0 + get(p + "post_attention_layernorm.weight")
            gate_up = (
                torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")])
                * g_post[None, :]
            )
            prefix.append(
                DecoderLayer(
                    qkv=quantize_weight(qkv),
                    o=quantize_weight(get(p + "self_attn.o_proj.weight")),
                    gate_up=quantize_weight(gate_up),
                    down=quantize_weight(get(p + "mlp.down_proj.weight")),
                )
            )
            del qkv, gate_up

        expert, mod_attn_w, mod_ffn_w = [], [], []
        for i in range(18):
            p = f"{EXPERT}layers.{i}."
            qkv = torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
            gate_up = torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")])
            expert.append(
                DecoderLayer(
                    qkv=quantize_weight(qkv),
                    o=quantize_weight(get(p + "self_attn.o_proj.weight")),
                    gate_up=quantize_weight(gate_up),
                    down=quantize_weight(get(p + "mlp.down_proj.weight")),
                )
            )
            mod_attn_w.append(
                (get(p + "input_layernorm.dense.weight"), get(p + "input_layernorm.dense.bias"))
            )
            mod_ffn_w.append(
                (
                    get(p + "post_attention_layernorm.dense.weight"),
                    get(p + "post_attention_layernorm.dense.bias"),
                )
            )

        # Flow-time conditioning for the fixed Euler schedule t_s = 1 - s / steps.
        t = torch.tensor([1.0 - s / steps for s in range(steps)], dtype=torch.float32)
        emb = _sinusoid(t, 1024, config["min_period"], config["max_period"]).to(dev)
        h = torch.nn.functional.silu(
            emb @ get("model.time_mlp_in.weight").t() + get("model.time_mlp_in.bias")
        )
        cond = torch.nn.functional.silu(
            h @ get("model.time_mlp_out.weight").t() + get("model.time_mlp_out.bias")
        )
        mod_attn = torch.stack([cond @ w.t() + b for w, b in mod_attn_w], 1).view(
            steps, 18, 3, 1024
        )
        mod_ffn = torch.stack([cond @ w.t() + b for w, b in mod_ffn_w], 1).view(steps, 18, 3, 1024)
        final = cond @ get(EXPERT + "norm.dense.weight").t() + get(EXPERT + "norm.dense.bias")
        mod_final = final.view(steps, 3, 1024)[:, :2].contiguous()

        weights = Pi05Weights(
            patch_w=get(VIS + "embeddings.patch_embedding.weight"),
            patch_b=get(VIS + "embeddings.patch_embedding.bias"),
            pos_emb=get(VIS + "embeddings.position_embedding.weight"),
            vision=vision,
            post_ln_w=get(VIS + "post_layernorm.weight"),
            post_ln_b=get(VIS + "post_layernorm.bias"),
            projector=projector,
            prefix=prefix,
            expert=expert,
            embed=f.get_tensor(ROOT + "paligemma.lm_head.weight").to(torch.bfloat16),
            action_in_w=get("model.action_in_proj.weight"),
            action_in_b=get("model.action_in_proj.bias"),
            action_out_w=get("model.action_out_proj.weight"),
            action_out_b=get("model.action_out_proj.bias"),
            mod_attn=mod_attn.contiguous(),
            mod_ffn=mod_ffn.contiguous(),
            mod_final=mod_final,
            steps=steps,
            config=config,
        )
    torch.cuda.empty_cache()
    return weights
