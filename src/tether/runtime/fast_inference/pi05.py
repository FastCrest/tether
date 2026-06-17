"""Pi0.5 Triton fast-kernels inference path (Lift #5 Day 4-7).

Ported from ``reference/FluxVLA/fluxvla/models/vlas/pi05_flowmatching_inference.py``
(656 LOC) + the three FluxVLA backbone ``prepare_triton`` methods that produce
the kernel-specific weight layout:

- ``backbones/visions/siglip_vit_inference.py`` (101 LOC) → vision triton layout
- ``backbones/llms/condition_gemma_inference.py`` (170 LOC) → llm + expert layouts
- ``projectors/linear_projector_inference.py`` (14 LOC) → projector layout

Adapted for the tether BaseVLA spine: instead of subclassing
``PI05FlowMatching``, this module consumes a ``tether.models.vlas.pi05.Pi05VLA``
instance and walks its slot components (``vision_backbone``, ``llm_backbone``,
``vla_head.expert_stack``) to extract the kernel weights.

V1 scope: Pi0.5 only (T-2). The procedural forward functions
(``vision_encoder``, ``transformer_encoder``, ``transformer_decoder``,
``pi05_model``) port the FluxVLA reference verbatim — they're tied to the
kernel signatures and would require kernel changes to alter.

Day 4 (this file): eager forward path (no graph capture). Days 5-6 add the
L1/L2 parity gates. Day 7 wraps ``torch.cuda.CUDAGraph()`` capture.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

# These vendored kernel imports trigger Triton + CUDA at import time. Lazy
# import inside the module via a deferred import — pi05.py itself can be
# imported on a CPU host for unit testing the class scaffold (the actual
# inference path obviously requires CUDA).
if TYPE_CHECKING:
    from tether.models.vlas.pi05 import Pi05VLA


# ─────────────────────────────────────────────────────────────────────────
# Procedural forward — ports of pi05_flowmatching_inference.py:22-326.
# Each function takes (weights, buffers) plus shape metadata; tied to the
# kernel signatures. Re-vendoring the kernels requires updating these calls.
# ─────────────────────────────────────────────────────────────────────────


def vision_encoder(weights: dict, buffers: dict, num_views: int, num_vit_layers: int = 27) -> None:
    """SigLIP-base vision encoder via Triton kernels.

    Mirrors ``pi05_flowmatching_inference.py:22-66``. Hard-codes pi0.5 +
    PaliGemma SigLIP-base shapes (num_patches=256, vit_hidden=1152,
    vit_intermediate=4304, vit_num_heads=16, vit_head_dim=72, grid_size=16,
    patch_size=14). The Lift #5 shape whitelist (``_shape_whitelist.py``)
    enforces this at runtime construction.
    """
    from tether.kernels.atomic_ops import (
        AttnMultiKey,
        conv2d_embed_res,
        layer_norm_QKV_matmul_bias,
        layer_norm_matmul_bias_gelu,
        matmul_bias_res,
        matmul_split_k_bias_res,
    )

    num_patches = 256
    vit_hidden = 1152
    vit_intermediate = 4304
    vit_num_heads = 16
    vit_head_dim = 72
    vit_qkv_hidden = 3 * vit_hidden
    grid_size = 16
    patch_size = 14

    conv2d_embed_res(
        buffers["observation_images_normalized"],
        weights["vision_patch_embedding_w"],
        weights["vision_patch_embedding_b"],
        weights["vision_position_embedding"],
        buffers["vision_x"],
        grid_size, patch_size, num_patches, vit_hidden,
    )

    for i in range(num_vit_layers):
        layer_norm_QKV_matmul_bias(
            buffers["vision_x"], weights["vision_pre_attn_norm_w"][i],
            weights["vision_pre_attn_norm_b"][i],
            weights["vision_attn_qkv_w"][i], weights["vision_attn_qkv_b"][i],
            buffers["vision_QKV"], buffers["vision_x_norm"], num_patches,
            vit_hidden, vit_qkv_hidden,
        )

        attn = AttnMultiKey(buffers["vision_QKV"], num_patches, vit_num_heads, vit_head_dim, vit_hidden)

        matmul_bias_res(
            attn, weights["vision_attn_o_w"][i], weights["vision_attn_o_b"][i],
            buffers["vision_x"], buffers["vision_x"], buffers["vision_x_split_k_buf"],
            num_patches, vit_hidden,
        )

        layer_norm_matmul_bias_gelu(
            buffers["vision_x"], weights["vision_pre_ffn_norm_w"][i],
            weights["vision_pre_ffn_norm_b"][i], weights["vision_ffn_up_w"][i],
            weights["vision_ffn_up_b"][i], buffers["vision_hidden"],
            buffers["vision_x_norm"], num_patches, vit_hidden, vit_intermediate,
        )

        matmul_split_k_bias_res(
            buffers["vision_hidden"], weights["vision_ffn_down_w"][i],
            weights["vision_ffn_down_b"][i], buffers["vision_x"], buffers["vision_x"],
            buffers["vision_x_split_k_buf"], num_patches, vit_intermediate, vit_hidden,
        )


def transformer_encoder(
    weights: dict, buffers: dict, encoder_seq_len: int, num_encoder_layers: int = 18,
) -> None:
    """Gemma encoder (PaliGemma language model) — Pi0.5 prefix path.

    Mirrors ``pi05_flowmatching_inference.py:69-153``. Includes the
    final-layer skip (no FFN on the last layer) which matches FluxVLA's
    pi0.5 reference pattern.
    """
    from tether.kernels.atomic_ops import (
        layer_norm_matmul_bias,
        matmul_attn_v,
        matmul_res,
        rms_matmul_gate,
        rms_matmul_qkv_rope,
    )
    from tether.kernels.triton.attention import (
        matmul_abT_scale,
        softmax_kernel_masklen,
    )

    layer_norm_matmul_bias(
        buffers["vision_x"], weights["vision_final_norm_w"], weights["vision_final_norm_b"],
        weights["encoder_multi_modal_projector_w"], weights["encoder_multi_modal_projector_b"],
        buffers["encoder_x"], buffers["vision_x_norm"],
        num_patches=256, in_features=1152, out_features=2048, eps=1e-5,
    )

    for i in range(num_encoder_layers):
        rms_matmul_qkv_rope(
            buffers["encoder_x"], weights["encoder_attn_qkv_w"][i],
            buffers["encoder_rope_weights"],
            buffers["encoder_Q"], buffers["encoder_K"][i, :encoder_seq_len],
            buffers["encoder_V"][i, :encoder_seq_len], buffers["encoder_x_norm"],
            hidden_dim=2048, head_dim=256, num_kv_heads=8,
        )

        if i != num_encoder_layers - 1:
            scale = 1.0 / (256 ** 0.5)
            total_queries = buffers["encoder_Q"].shape[0]
            total_keys = encoder_seq_len

            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers["encoder_Q"], buffers["encoder_K"][i, :encoder_seq_len],
                buffers["encoder_logits_buf"], total_queries, total_keys, 256, scale,
                BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
            )

            softmax_kernel_masklen[((total_queries + 3) // 4,)](
                buffers["encoder_logits_buf"], total_queries, total_keys,
                buffers["valid_encoder_len"], buffers["encoder_attn_buf"],
                BLOCK_SIZE_M=4, BLOCK_SIZE=1024,
            )

            matmul_attn_v(
                buffers["encoder_attn_buf"], buffers["encoder_V"][i, :encoder_seq_len],
                buffers["encoder_ctx_buf"], head_dim=256,
            )

            matmul_res(
                buffers["encoder_ctx_buf"].view(-1, 2048),
                weights["encoder_attn_o_w"][i], buffers["encoder_x"],
                in_features=2048, out_features=2048,
            )

            rms_matmul_gate(
                buffers["encoder_x"], weights["encoder_ffn_gate_w"][i],
                weights["encoder_ffn_up_w"][i], buffers["encoder_hidden"],
                buffers["encoder_x_norm"], hidden_dim=2048, intermediate_dim=16384,
            )

            matmul_res(
                buffers["encoder_hidden"], weights["encoder_ffn_down_w"][i], buffers["encoder_x"],
                in_features=16384, out_features=2048,
            )


def transformer_decoder(
    weights: dict, buffers: dict, encoder_seq_len: int,
    num_decoder_layers: int = 18, num_steps: int = 10,
) -> None:
    """Gemma expert (Pi0.5 action denoising) — AdaRMSNorm + AdaLN gating.

    Mirrors ``pi05_flowmatching_inference.py:155-312``. Runs the full
    flow-matching denoise loop (default 10 Euler steps) UNROLLED inside this
    function — no control flow visible to ``torch.cuda.CUDAGraph()``.
    """
    from tether.kernels.atomic_ops import (
        adarms_norm_style_proj,
        matmul_attn_v,
        matmul_bias_silu,
        matmul_bias_small,
        matmul_gate,
        matmul_qkv_rope,
        matmul_res_gate,
    )
    from tether.kernels.triton.attention import (
        matmul_abT_scale,
        softmax_kernel_prefix_suffix,
    )

    for step in range(num_steps):
        matmul_bias_silu(
            weights["decoder_time_embeds"][step].view(1, -1),
            weights["decoder_time_mlp_in_w"], weights["decoder_time_mlp_in_b"],
            buffers["decoder_x_buf"], in_features=1024, out_features=1024,
        )
        matmul_bias_silu(
            buffers["decoder_x_buf"], weights["decoder_time_mlp_out_w"],
            weights["decoder_time_mlp_out_b"], buffers["decoder_time_emb"],
            in_features=1024, out_features=1024,
        )
        matmul_bias_small(
            buffers["diffusion_noise"], weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"], buffers["decoder_x"],
            in_features=32, out_features=1024,
            BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=32,
        )
        seq_len = buffers["decoder_x"].shape[0]

        for i in range(num_decoder_layers):
            adarms_norm_style_proj(
                buffers["decoder_x"], buffers["decoder_time_emb"],
                weights["decoder_pre_attn_norm_mod_w"][i],
                weights["decoder_pre_attn_norm_mod_b"][i],
                buffers["x_normed_buf"], buffers["gate_buf"], buffers["decoder_style"],
                hidden_dim=1024, style_dim=3072,
            )

            matmul_qkv_rope(
                buffers["x_normed_buf"], weights["decoder_attn_qkv_w"][i],
                buffers["decoder_rope_weights"], buffers["decoder_q_buf"],
                buffers["encoder_K"][i, encoder_seq_len:encoder_seq_len + seq_len],
                buffers["encoder_V"][i, encoder_seq_len:encoder_seq_len + seq_len],
                hidden_dim=1024, head_dim=256, num_kv_heads=8,
            )

            total_queries = buffers["decoder_q_buf"].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, :encoder_seq_len + seq_len],
                buffers["decoder_logits_buf"], total_queries, total_keys, 256,
                256 ** -0.5,
                BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
            )

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                buffers["decoder_logits_buf"], total_queries,
                prefix_keys, suffix_keys, buffers["valid_encoder_len"],
                buffers["decoder_attn_buf"],
                BLOCK_SIZE_M=4, BLOCK_SIZE=1024,
            )

            matmul_attn_v(
                buffers["decoder_attn_buf"],
                buffers["encoder_V"][i, :encoder_seq_len + seq_len],
                buffers["decoder_q_buf"], head_dim=256,
            )

            matmul_res_gate(
                buffers["decoder_q_buf"].view(-1, 2048),
                weights["decoder_attn_o_w"][i], buffers["decoder_x"], buffers["gate_buf"],
                in_features=2048, out_features=1024,
                BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=128,
            )

            adarms_norm_style_proj(
                buffers["decoder_x"], buffers["decoder_time_emb"],
                weights["decoder_pre_ffn_norm_mod_w"][i],
                weights["decoder_pre_ffn_norm_mod_b"][i],
                buffers["x_normed_buf"], buffers["gate_buf"], buffers["decoder_style"],
                hidden_dim=1024, style_dim=3072,
            )

            matmul_gate(
                buffers["x_normed_buf"],
                weights["decoder_ffn_gate_w"][i], weights["decoder_ffn_up_w"][i],
                buffers["decoder_hidden"], in_features=1024, intermediate_dim=4096,
            )

            matmul_res_gate(
                buffers["decoder_hidden"],
                weights["decoder_ffn_down_w"][i], buffers["decoder_x"], buffers["gate_buf"],
                in_features=4096, out_features=1024,
                BLOCK_SIZE_N=16, BLOCK_SIZE_M=32, BLOCK_SIZE_K=256,
            )

        seq_len = buffers["decoder_x"].shape[0]
        adarms_norm_style_proj(
            buffers["decoder_x"], buffers["decoder_time_emb"],
            weights["decoder_final_norm_mod_w"], weights["decoder_final_norm_mod_b"],
            buffers["x_normed_buf"], buffers["gate_buf"], buffers["decoder_style"],
            hidden_dim=1024, style_dim=3072,
        )

        matmul_bias_small(
            buffers["x_normed_buf"], weights["decoder_action_out_proj_w"],
            weights["decoder_action_out_proj_b"], buffers["decoder_action_buf"],
            in_features=1024, out_features=32,
            BLOCK_SIZE_N=16, BLOCK_SIZE_M=16, BLOCK_SIZE_K=256,
        )

        buffers["diffusion_noise"].add_(buffers["decoder_action_buf"], alpha=-1.0 / num_steps)


def pi05_model(
    weights: dict, buffers: dict,
    num_views: int, encoder_seq_len: int,
    num_vit_layers: int = 27, num_encoder_layers: int = 18,
    num_decoder_layers: int = 18, num_steps: int = 10,
) -> None:
    """Compose the three procedural forwards. Single entry point for CUDA Graph capture."""
    vision_encoder(weights, buffers, num_views, num_vit_layers)
    transformer_encoder(weights, buffers, encoder_seq_len, num_encoder_layers)
    transformer_decoder(weights, buffers, encoder_seq_len, num_decoder_layers, num_steps)


# ─────────────────────────────────────────────────────────────────────────
# Weight reshaping — convert tether Pi05VLA spine to kernel-specific layout
# ─────────────────────────────────────────────────────────────────────────


def _rope_format_conversion(
    w_q: torch.Tensor, w_k: torch.Tensor,
    head_dim: int = 256, num_heads_q: int = 8, num_heads_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Q/K weights from training format to Triton RoPE format.

    Ported from ``condition_gemma_inference.py:46-63``.
    """
    in_dim = w_q.shape[1]
    out_dim_q = num_heads_q * head_dim

    w_q = w_q.view(num_heads_q, head_dim, in_dim)
    w_q = w_q.view(num_heads_q, 2, head_dim // 2, in_dim)
    w_q = w_q.permute(0, 2, 1, 3).reshape(out_dim_q, in_dim)

    w_k = w_k.view(2, head_dim // 2, in_dim)
    w_k = w_k.permute(1, 0, 2).reshape(head_dim, in_dim)

    return w_q, w_k


def _vision_prepare_triton(vision_backbone: Any) -> dict:
    """Port of ``siglip_vit_inference.py::prepare_triton`` adapted for tether.

    Tether's ``SigLIPBackbone`` wraps ``transformers`` ``SiglipVisionModel`` at
    ``self.model``. Walks the same attribute tree as FluxVLA's reference, just
    via ``vision_backbone.model.*`` instead of ``self.vision.vision_model.*``.
    """
    vm = vision_backbone.model.vision_model  # SiglipVisionTransformer
    embed = vm.embeddings
    encoder = vm.encoder

    weights: dict = {}

    # Patch Embedding: [out, in, kH, kW] → [kH, kW, in, out]
    patch_w = embed.patch_embedding.weight.data
    weights["vision_patch_embedding_w"] = (
        patch_w.permute(2, 3, 1, 0).contiguous().bfloat16().cuda()
    )
    if embed.patch_embedding.bias is not None:
        weights["vision_patch_embedding_b"] = embed.patch_embedding.bias.data.bfloat16().cuda()

    # Position Embedding
    weights["vision_position_embedding"] = embed.position_embedding.weight.data.bfloat16().cuda()

    attn_qkv_w, attn_qkv_b = [], []
    attn_o_w, attn_o_b = [], []
    ffn_up_w, ffn_up_b = [], []
    ffn_down_w, ffn_down_b = [], []
    pre_attn_norm_w, pre_attn_norm_b = [], []
    pre_ffn_norm_w, pre_ffn_norm_b = [], []

    for layer in encoder.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        q_w = attn.q_proj.weight.data
        k_w = attn.k_proj.weight.data
        v_w = attn.v_proj.weight.data
        qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1)
        attn_qkv_w.append(qkv_w)

        q_b = attn.q_proj.bias.data if attn.q_proj.bias is not None else torch.zeros(1152)
        k_b = attn.k_proj.bias.data if attn.k_proj.bias is not None else torch.zeros(1152)
        v_b = attn.v_proj.bias.data if attn.v_proj.bias is not None else torch.zeros(1152)
        attn_qkv_b.append(torch.cat([q_b, k_b, v_b], dim=0))

        attn_o_w.append(attn.out_proj.weight.data.T)
        attn_o_b.append(
            attn.out_proj.bias.data if attn.out_proj.bias is not None else torch.zeros(1152)
        )

        ffn_up_w.append(mlp.fc1.weight.data.T)
        ffn_up_b.append(mlp.fc1.bias.data)
        ffn_down_w.append(mlp.fc2.weight.data.T)
        ffn_down_b.append(mlp.fc2.bias.data)

        pre_attn_norm_w.append(layer.layer_norm1.weight.data)
        pre_attn_norm_b.append(layer.layer_norm1.bias.data)
        pre_ffn_norm_w.append(layer.layer_norm2.weight.data)
        pre_ffn_norm_b.append(layer.layer_norm2.bias.data)

    weights["vision_attn_qkv_w"] = torch.stack(attn_qkv_w).bfloat16().cuda()
    weights["vision_attn_qkv_b"] = torch.stack(attn_qkv_b).bfloat16().cuda()
    weights["vision_attn_o_w"] = torch.stack(attn_o_w).bfloat16().cuda()
    weights["vision_attn_o_b"] = torch.stack(attn_o_b).bfloat16().cuda()
    weights["vision_ffn_up_w"] = torch.stack(ffn_up_w).bfloat16().cuda()
    weights["vision_ffn_up_b"] = torch.stack(ffn_up_b).bfloat16().cuda()
    weights["vision_ffn_down_w"] = torch.stack(ffn_down_w).bfloat16().cuda()
    weights["vision_ffn_down_b"] = torch.stack(ffn_down_b).bfloat16().cuda()
    weights["vision_pre_attn_norm_w"] = torch.stack(pre_attn_norm_w).bfloat16().cuda()
    weights["vision_pre_attn_norm_b"] = torch.stack(pre_attn_norm_b).bfloat16().cuda()
    weights["vision_pre_ffn_norm_w"] = torch.stack(pre_ffn_norm_w).bfloat16().cuda()
    weights["vision_pre_ffn_norm_b"] = torch.stack(pre_ffn_norm_b).bfloat16().cuda()

    weights["vision_final_norm_w"] = vm.post_layernorm.weight.data.bfloat16().cuda()
    weights["vision_final_norm_b"] = vm.post_layernorm.bias.data.bfloat16().cuda()

    return weights


def _llm_prepare_triton(language_model: Any, role: str = "llm") -> dict:
    """Port of ``condition_gemma_inference.py::prepare_triton`` adapted for tether.

    ``role='llm'`` fuses ``(1 + pre_norm)`` into Q/K/V/gate/up matmuls (encoder
    pattern). ``role='expert'`` keeps the AdaRMS norm modulation separate
    (decoder pattern).

    Args:
        language_model: A module exposing ``.layers`` (a ``ModuleList`` of
            transformer blocks) and ``.norm`` (the final RMS norm). For Pi05's
            paligemma_with_expert.paligemma, this is the language_model submodule.
            For the gemma_expert, this is the expert top-level module.
        role: ``'llm'`` or ``'expert'``.
    """
    weights: dict = {}
    layers = language_model.layers
    n = len(layers)

    if role == "llm":
        attn_qkv_w, attn_o_w = [], []
        ffn_gate_w, ffn_up_w, ffn_down_w = [], [], []

        for i in range(n):
            layer = layers[i]
            pre_attn_norm = layer.input_layernorm.weight.data.float()
            pre_ffn_norm = layer.post_attention_layernorm.weight.data.float()

            q_w = layer.self_attn.q_proj.weight.data.float()
            k_w = layer.self_attn.k_proj.weight.data.float()
            v_w = layer.self_attn.v_proj.weight.data.float()
            o_w = layer.self_attn.o_proj.weight.data.float()

            scale = (1 + pre_attn_norm).unsqueeze(0)
            q_w = q_w * scale
            k_w = k_w * scale
            v_w = v_w * scale

            q_w, k_w = _rope_format_conversion(q_w, k_w)
            qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1)
            attn_qkv_w.append(qkv_w.bfloat16().cuda())
            attn_o_w.append(o_w.T.contiguous().bfloat16().cuda())

            gate_w = layer.mlp.gate_proj.weight.data.float()
            up_w = layer.mlp.up_proj.weight.data.float()
            down_w = layer.mlp.down_proj.weight.data.float()

            ffn_scale = (1 + pre_ffn_norm).unsqueeze(0)
            gate_w = gate_w * ffn_scale
            up_w = up_w * ffn_scale

            ffn_gate_w.append(gate_w.T.contiguous().bfloat16().cuda())
            ffn_up_w.append(up_w.T.contiguous().bfloat16().cuda())
            ffn_down_w.append(down_w.T.contiguous().bfloat16().cuda())

        weights["encoder_attn_qkv_w"] = torch.stack(attn_qkv_w)
        weights["encoder_attn_o_w"] = torch.stack(attn_o_w)
        weights["encoder_ffn_gate_w"] = torch.stack(ffn_gate_w)
        weights["encoder_ffn_up_w"] = torch.stack(ffn_up_w)
        weights["encoder_ffn_down_w"] = torch.stack(ffn_down_w)

    elif role == "expert":
        # Tether's Pi05ExpertGQALayer has FLAT attribute names — no .self_attn / .mlp
        # parents (unlike HF Gemma layers used in the 'llm' role above). Walk
        # the projections directly off each layer.
        attn_qkv_w, attn_o_w = [], []
        ffn_gate_w, ffn_up_w, ffn_down_w = [], [], []
        pre_attn_mod_w, pre_attn_mod_b = [], []
        pre_ffn_mod_w, pre_ffn_mod_b = [], []

        for i in range(n):
            layer = layers[i]

            pre_attn_mod_w.append(
                layer.input_layernorm.dense.weight.data.T.contiguous().bfloat16().cuda()
            )
            pre_attn_mod_b.append(layer.input_layernorm.dense.bias.data.bfloat16().cuda())
            pre_ffn_mod_w.append(
                layer.post_attention_layernorm.dense.weight.data.T.contiguous().bfloat16().cuda()
            )
            pre_ffn_mod_b.append(layer.post_attention_layernorm.dense.bias.data.bfloat16().cuda())

            q_w = layer.q_proj.weight.data.float()
            k_w = layer.k_proj.weight.data.float()
            v_w = layer.v_proj.weight.data.float()
            o_w = layer.o_proj.weight.data.float()

            q_w, k_w = _rope_format_conversion(q_w, k_w)
            qkv_w = torch.cat([q_w.T, k_w.T, v_w.T], dim=1)
            attn_qkv_w.append(qkv_w.bfloat16().cuda())
            attn_o_w.append(o_w.T.contiguous().bfloat16().cuda())

            ffn_gate_w.append(layer.gate_proj.weight.data.T.contiguous().bfloat16().cuda())
            ffn_up_w.append(layer.up_proj.weight.data.T.contiguous().bfloat16().cuda())
            ffn_down_w.append(layer.down_proj.weight.data.T.contiguous().bfloat16().cuda())

        weights["decoder_attn_qkv_w"] = torch.stack(attn_qkv_w)
        weights["decoder_attn_o_w"] = torch.stack(attn_o_w)
        weights["decoder_ffn_gate_w"] = torch.stack(ffn_gate_w)
        weights["decoder_ffn_up_w"] = torch.stack(ffn_up_w)
        weights["decoder_ffn_down_w"] = torch.stack(ffn_down_w)
        weights["decoder_pre_attn_norm_mod_w"] = torch.stack(pre_attn_mod_w)
        weights["decoder_pre_attn_norm_mod_b"] = torch.stack(pre_attn_mod_b)
        weights["decoder_pre_ffn_norm_mod_w"] = torch.stack(pre_ffn_mod_w)
        weights["decoder_pre_ffn_norm_mod_b"] = torch.stack(pre_ffn_mod_b)

        # Final norm — tether's expert_stack has a `norm` attribute that's a
        # DecomposedAdaRMSNorm (matches FluxVLA's pattern). FluxVLA's
        # condition_gemma_inference.py reads `language_model.norm.dense.*`.
        # Tether stores this on the expert_stack itself (not nested).
        # Falls back to `final_norm` if `norm` isn't present (some expert
        # builds use a different attribute name).
        final_norm = getattr(language_model, "norm", None) or getattr(
            language_model, "final_norm", None
        )
        if final_norm is None:
            raise AttributeError(
                "expert stack must expose `.norm` or `.final_norm` "
                "(DecomposedAdaRMSNorm); got neither"
            )
        weights["decoder_final_norm_mod_w"] = (
            final_norm.dense.weight.data.T.contiguous().bfloat16().cuda()
        )
        weights["decoder_final_norm_mod_b"] = final_norm.dense.bias.data.bfloat16().cuda()
    else:
        raise ValueError(f"role must be 'llm' or 'expert'; got {role!r}")
    return weights


def _projector_prepare_triton(projector: Any, prefix: str = "encoder_multi_modal_projector") -> dict:
    """Port of ``linear_projector_inference.py::prepare_triton``.

    Tether's ``LinearProjector`` exposes its underlying ``nn.Linear`` at
    ``self.linear``; FluxVLA's exposes ``self.projector``. Translate.
    """
    layer = projector.linear  # tether.models.projectors.linear_projector.LinearProjector.linear
    return {
        f"{prefix}_w": layer.weight.data.T.contiguous().bfloat16().cuda(),
        f"{prefix}_b": layer.bias.data.bfloat16().cuda(),
    }


def _prepare_adarms_cond(num_steps: int) -> torch.Tensor:
    """Pre-compute sinusoidal time embeddings for each Euler step.

    Ported from ``pi05_flowmatching_inference.py:564-583``.
    """
    dt = -1.0 / num_steps
    time_val = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    min_period = 4e-3
    max_period = 4.0
    embedding_dim = 1024
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device="cuda")
    period = min_period * (max_period / min_period) ** fraction

    time_embs = []
    for _ in range(num_steps):
        sinusoid_input = (
            time_val.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        )
        emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)
        time_embs.append(emb.to(torch.bfloat16))
        time_val = time_val + dt
    return torch.cat(time_embs, dim=0)


# ─────────────────────────────────────────────────────────────────────────
# Pi05FastKernelsInference — the second runtime alongside ORT.
# Ported from PI05FlowMatchingInference (pi05_flowmatching_inference.py:329-656).
# ─────────────────────────────────────────────────────────────────────────


class Pi05FastKernelsInference:
    """Triton fast-kernels inference path for Pi0.5 on the tether spine.

    V1 scope per T-2: Pi0.5 only. Constructor takes a built ``Pi05VLA``
    instance (from ``Pi05VLA.from_lerobot_policy`` or any other spine builder)
    and walks the slot components to extract the kernel weights.

    Hardware: A100 (sm 8.0) is the V1 primary target. The ``_hardware_gate``
    refuses on sm 8.6 / 8.7 / pre-Ampere. The ``_shape_whitelist`` refuses on
    non-PaliGemma-SigLIP-base configs.

    Args:
        vla: A built ``Pi05VLA`` instance. The ``vision_backbone``,
            ``llm_backbone``, ``vla_head.expert_stack`` slots are walked to
            extract kernel weights.
        num_views: Camera view count (default 2 for Pi0.5).
        triton_max_prompt_len: Pre-allocate buffer for prompts up to this many
            tokens (default 48). Prompts longer than this will fail at runtime.
        num_steps: Number of Euler denoise steps (default 10). Burned into the
            captured graph in Day 7+.
        chunk_size: Number of actions per chunk (default 50 for Pi0.5).
        max_action_dim: Padded action dimension (default 32 for Pi0.5).
        capture: If True, build the CUDA Graph at the first ``predict_action``
            call (Day 7 feature). Day 4 leaves capture=False (eager path) for
            parity testing before adding the capture complexity.

    Raises:
        RuntimeError: Hardware gate refuses (sm 8.6 / 8.7 / pre-Ampere / no CUDA).
        RuntimeError: Shape whitelist refuses (non-PaliGemma-SigLIP-base config).
    """

    def __init__(
        self,
        vla: "Pi05VLA",
        *,
        num_views: int = 2,
        triton_max_prompt_len: int = 48,
        num_steps: int = 10,
        chunk_size: int = 50,
        max_action_dim: int = 32,
        capture: bool = False,
        _skip_hardware_gate: bool = False,  # for unit tests
        _skip_shape_whitelist: bool = False,  # for unit tests
    ) -> None:
        # ── Hardware gate ──
        if not _skip_hardware_gate:
            from tether.kernels._hardware_gate import is_fast_kernels_hardware_compatible
            ok, msg = is_fast_kernels_hardware_compatible()
            if not ok:
                raise RuntimeError(f"Pi05FastKernelsInference refused: {msg}")

        # ── Shape whitelist ──
        if not _skip_shape_whitelist:
            from tether.kernels._shape_whitelist import validate_shape_signature
            vit_cfg = vla.vision_backbone.model.vision_model.config
            shape_config = {
                "vit_hidden": vit_cfg.hidden_size,
                "vit_intermediate": vit_cfg.intermediate_size,
                "vit_num_heads": vit_cfg.num_attention_heads,
                "image_size": vit_cfg.image_size,
                "patch_size": vit_cfg.patch_size,
            }
            ok, msg = validate_shape_signature(shape_config)
            if not ok:
                raise RuntimeError(f"Pi05FastKernelsInference refused: {msg}")

        # ── Save references ──
        self._vla = vla
        self.num_views = num_views
        self.triton_max_prompt_len = triton_max_prompt_len
        self.num_steps = num_steps
        self.n_action_steps = chunk_size
        self.max_action_dim = max_action_dim
        self._capture = capture

        # Pi0.5 expert is wrapped inside vla_head.expert_stack — extract the
        # paligemma_with_expert.gemma_expert submodule for the role='expert'
        # walk. The tether spine names this slightly differently than FluxVLA;
        # the expert stack itself holds layers + final norm.
        self._expert_stack = vla.vla_head.expert_stack

        # ── Build weights, buffers, RoPE table ──
        self._triton_weights: dict = {}
        self._triton_bufs: dict = {}
        self._rope_table: torch.Tensor | None = None
        self._triton_ready = False
        self._cuda_graph = None
        self._cuda_graph_ready = False

    # ─── Public API ─────────────────────────────────────────────────────

    def prepare_triton_inference(self) -> None:
        """Collect weights from the Pi05VLA spine + build buffers + RoPE table.

        Idempotent; can be re-called to re-extract (e.g. after a state_dict
        reload). Lazy — called on first ``predict_action`` if not already
        prepped.
        """
        self._triton_weights = {}
        self._triton_weights.update(_vision_prepare_triton(self._vla.vision_backbone))
        # PaliGemma's text model is under llm_backbone.model.model.language_model.
        # For Pi0.5 the encoder is the paligemma text stack.
        text_model = self._vla.llm_backbone.model.model.language_model
        self._triton_weights.update(_llm_prepare_triton(text_model, role="llm"))
        # Expert is the gemma_expert under vla_head.expert_stack.
        # Its layers + norm follow the same module attribute pattern.
        self._triton_weights.update(_llm_prepare_triton(self._expert_stack, role="expert"))
        # Projector — Pi0.5's state_proj lives on vla_head (state-in-language).
        # But the kernel path expects an encoder_multi_modal_projector_w/b — port
        # from PaliGemma's multi_modal_projector when present, else from state_proj.
        mm_projector = self._vla.llm_backbone.model.model.multi_modal_projector
        self._triton_weights.update(_projector_prepare_triton(_LinearWrapper(mm_projector.linear)))
        # Action-time MLP and projections from the expert stack.
        self._triton_weights.update(self._prepare_action_time_triton())
        # Pre-computed time-step embeddings (constants — burnable into graph).
        self._triton_weights["decoder_time_embeds"] = _prepare_adarms_cond(self.num_steps)

        # ── Cache shape metadata ──
        vit_cfg = self._vla.vision_backbone.model.vision_model.config
        self._vit_image_size = vit_cfg.image_size
        self._vit_num_patches = (vit_cfg.image_size // vit_cfg.patch_size) ** 2
        self._vit_hidden = vit_cfg.hidden_size
        self._vit_intermediate = vit_cfg.intermediate_size
        self._num_vit_layers = vit_cfg.num_hidden_layers

        enc_cfg = text_model.config
        self._enc_hidden = enc_cfg.hidden_size
        self._enc_intermediate = enc_cfg.intermediate_size
        self._num_encoder_layers = len(text_model.layers)
        self._head_dim = getattr(enc_cfg, "head_dim", 256)
        self._num_kv_heads = enc_cfg.num_attention_heads

        # Expert dimensions
        self._dec_hidden = 1024
        self._dec_intermediate = 4096
        self._dec_style_dim = 3 * self._dec_hidden
        self._num_decoder_layers = len(self._expert_stack.layers)
        self._action_dim = self.max_action_dim

        self._encoder_seq_len = self.num_views * self._vit_num_patches + self.triton_max_prompt_len
        self._decoder_seq_len = self.n_action_steps

        self._init_buffers()
        self._init_rope_table()
        self._triton_ready = True

    def predict_action(
        self,
        *,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        states: torch.Tensor,
        img_masks: torch.Tensor | None = None,
        lang_masks: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Pi0.5 inference via Triton kernels.

        V1 inputs match the existing ``Pi05VLA.predict_action`` signature so
        the FastKernelsRuntime is drop-in. Internally, all tensors get cast to
        bf16 on cuda before kernel calls.

        Args:
            images: ``[batch, num_views * 3, H, W]`` float images.
            lang_tokens: ``[batch, max_prompt_len]`` int64 prompt tokens.
            states: ``[batch, max_state_dim]`` float robot state (state-in-language
                means this is unused at the kernel level for Pi0.5; preserved
                for API compatibility).
            img_masks: Optional ``[batch, num_views]`` mask.
            lang_masks: Optional ``[batch, max_prompt_len]`` mask. Used to
                determine ``prompt_len`` if provided; else ``lang_tokens.shape[1]``.
            noise: Optional initial noise ``[batch, chunk_size, max_action_dim]``.
                If None, random bf16 noise is drawn.

        Returns:
            ``[batch, chunk_size, max_action_dim]`` float actions.
        """
        if not self._triton_ready:
            self.prepare_triton_inference()

        # Vision input: take view 0 (FluxVLA convention) and convert to NHWC bf16.
        pixel_values = images.unflatten(1, (-1, 3))[0]
        images_nhwc = pixel_values.permute(0, 2, 3, 1).contiguous().bfloat16()

        prompt_len = (
            int(lang_masks[0].sum().item())
            if lang_masks is not None
            else lang_tokens.shape[1]
        )
        # Embed via the paligemma text model's embed_tokens.
        # Pi0.5 scales the embedding by sqrt(hidden_dim) — matches the FluxVLA reference.
        text_model = self._vla.llm_backbone.model.model.language_model
        lang_emb = text_model.embed_tokens(lang_tokens[0, :prompt_len])
        lang_emb = (lang_emb * math.sqrt(lang_emb.shape[-1])).bfloat16()

        chunk_size = self.n_action_steps
        if noise is None:
            noise_t = torch.randn(
                chunk_size, self.max_action_dim,
                dtype=torch.bfloat16, device=states.device,
            )
        else:
            noise_t = noise[0].to(dtype=torch.bfloat16)
        if noise_t.shape[-1] < 32:
            pad = torch.zeros(
                noise_t.shape[0], 32 - noise_t.shape[-1],
                dtype=torch.bfloat16, device=noise_t.device,
            )
            noise_t = torch.cat([noise_t, pad], dim=-1)

        denoised = self._triton_forward(images_nhwc, lang_emb, prompt_len, noise_t)
        return denoised[:, : self.max_action_dim].unsqueeze(0).float()

    # ─── Internal ───────────────────────────────────────────────────────

    def _prepare_action_time_triton(self) -> dict:
        """Collect action-time projection weights from the expert stack.

        Tether's ``Pi05ExpertStackWithPrefix`` names the time MLP as
        ``time_mlp_in/out`` (no ``action_`` prefix) — distinct from pi0's
        ``Pi0ExpertStackWithPrefix`` which uses ``action_time_mlp_in/out``.
        Detect which is present + use the right name.
        """
        es = self._expert_stack
        weights: dict = {}
        weights.update(_projector_prepare_triton(_LinearWrapper(es.action_in_proj), "decoder_action_in_proj"))
        weights.update(_projector_prepare_triton(_LinearWrapper(es.action_out_proj), "decoder_action_out_proj"))

        # Time MLP attribute name differs between pi0 and pi0.5 expert stacks.
        time_in = getattr(es, "time_mlp_in", None) or getattr(es, "action_time_mlp_in", None)
        time_out = getattr(es, "time_mlp_out", None) or getattr(es, "action_time_mlp_out", None)
        if time_in is None or time_out is None:
            raise AttributeError(
                "expert stack must expose `time_mlp_in/out` (pi0.5) or "
                "`action_time_mlp_in/out` (pi0); got neither"
            )
        weights.update(_projector_prepare_triton(_LinearWrapper(time_in), "decoder_time_mlp_in"))
        weights.update(_projector_prepare_triton(_LinearWrapper(time_out), "decoder_time_mlp_out"))
        return weights

    def _init_buffers(self) -> None:
        """Allocate the bf16 CUDA buffers used by the procedural forward."""
        nv = self.num_views
        enc = self._encoder_seq_len
        dec = self._decoder_seq_len
        num_kv_layers = max(self._num_encoder_layers, self._num_decoder_layers)
        bf = torch.bfloat16
        dev = "cuda"

        img = self._vit_image_size
        np_ = self._vit_num_patches
        vh = self._vit_hidden
        vi = self._vit_intermediate
        eh = self._enc_hidden
        ei = self._enc_intermediate
        dh = self._dec_hidden
        di = self._dec_intermediate
        ds = self._dec_style_dim
        hd = self._head_dim
        nkv = self._num_kv_heads
        ad = self._action_dim

        self._triton_bufs = {
            "observation_images_normalized": torch.zeros(nv, img, img, 3, dtype=bf, device=dev),
            "vision_x": torch.zeros(nv, np_, vh, dtype=bf, device=dev),
            "vision_x_norm": torch.zeros(nv, np_, vh, dtype=bf, device=dev),
            "vision_QKV": torch.zeros(nv, np_, 3 * vh, dtype=bf, device=dev),
            "vision_hidden": torch.zeros(nv, np_, vi, dtype=bf, device=dev),
            "vision_x_split_k_buf": torch.zeros(nv * np_ * vh * 4, dtype=torch.float32, device=dev),
            "encoder_rope_weights": torch.zeros(enc, hd, dtype=bf, device=dev),
            "encoder_x": torch.zeros(enc, eh, dtype=bf, device=dev),
            "encoder_x_norm": torch.zeros(enc, eh, dtype=bf, device=dev),
            "encoder_K": torch.zeros(num_kv_layers, enc + dec, hd, dtype=bf, device=dev),
            "encoder_V": torch.zeros(num_kv_layers, enc + dec, hd, dtype=bf, device=dev),
            "encoder_Q": torch.zeros(enc * nkv, hd, dtype=bf, device=dev),
            "encoder_hidden": torch.zeros(enc, ei, dtype=bf, device=dev),
            "valid_encoder_len": torch.zeros((1,), dtype=torch.int32, device=dev),
            "encoder_logits_buf": torch.zeros(enc * nkv, enc, dtype=torch.float32, device=dev),
            "encoder_attn_buf": torch.zeros(enc * nkv, enc, dtype=bf, device=dev),
            "encoder_ctx_buf": torch.zeros(enc * nkv, hd, dtype=bf, device=dev),
            "decoder_rope_weights": torch.zeros(dec, hd, dtype=bf, device=dev),
            "decoder_x": torch.zeros(dec, dh, dtype=bf, device=dev),
            "decoder_x_buf": torch.zeros(dec, dh, dtype=bf, device=dev),
            "decoder_action_buf": torch.zeros(dec, ad, dtype=bf, device=dev),
            "decoder_time_emb": torch.zeros(dec, dh, dtype=bf, device=dev),
            "decoder_style": torch.zeros(dec, ds, dtype=bf, device=dev),
            "decoder_norm_factor_buf": torch.zeros(dec, dtype=bf, device=dev),
            "decoder_q_buf": torch.zeros(dec * nkv, hd, dtype=bf, device=dev),
            "decoder_logits_buf": torch.zeros(dec * nkv, enc + dec, dtype=torch.float32, device=dev),
            "decoder_attn_buf": torch.zeros(dec * nkv, enc + dec, dtype=bf, device=dev),
            "decoder_hidden": torch.zeros(dec, di, dtype=bf, device=dev),
            "decode_split_k_buf": torch.zeros(2, dec, dh, dtype=torch.float32, device=dev),
            "x_normed_buf": torch.zeros(dec, dh, dtype=bf, device=dev),
            "gate_buf": torch.zeros(dec, dh, dtype=bf, device=dev),
            "diffusion_noise": torch.zeros(dec, ad, dtype=bf, device=dev),
        }

    def _init_rope_table(self) -> None:
        """Pre-compute the RoPE cos/sin table (interleaved layout)."""
        prefix_alloc = self.num_views * 256 + self.triton_max_prompt_len
        max_pos = prefix_alloc - 1 + self._decoder_seq_len
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (
            torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256
        ))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat(
            [k_cos[:, :, None], k_sin[:, :, None]], 2,
        ).view(-1, 256)
        self._triton_bufs["encoder_rope_weights"].copy_(self._rope_table[:prefix_alloc])

    def _get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len - 1
        end = start + self._decoder_seq_len
        return self._rope_table[start:end]

    def _run_forward(self) -> None:
        pi05_model(
            self._triton_weights, self._triton_bufs, self.num_views,
            self._encoder_seq_len, self._num_vit_layers,
            self._num_encoder_layers, self._num_decoder_layers, self.num_steps,
        )

    def _triton_forward(
        self,
        images_nhwc: torch.Tensor, prompt_embeds: torch.Tensor,
        prompt_len: int, diffusion_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Copy inputs into pre-allocated buffers, run forward, return outputs."""
        self._triton_bufs["observation_images_normalized"].copy_(images_nhwc)
        start = self.num_views * 256
        self._triton_bufs["encoder_x"][start:start + prompt_len].copy_(prompt_embeds)
        self._triton_bufs["valid_encoder_len"].fill_(start + prompt_len)
        self._triton_bufs["decoder_rope_weights"].copy_(self._get_decoder_rope_weights(prompt_len))
        self._triton_bufs["diffusion_noise"].copy_(diffusion_noise)

        if self._capture:
            # Lazy build the graph on first call. Day 7 implementation.
            if not self._cuda_graph_ready:
                self._build_cuda_graph()
            self._cuda_graph.replay()
        else:
            self._run_forward()

        return self._triton_bufs["diffusion_noise"]

    def _build_cuda_graph(self) -> None:
        """Day 7: capture the procedural forward in a CUDA Graph for replay."""
        for _ in range(3):
            self._run_forward()
        torch.cuda.synchronize()

        self._cuda_graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._cuda_graph.capture_begin()
            self._run_forward()
            self._cuda_graph.capture_end()
        torch.cuda.synchronize()
        self._cuda_graph_ready = True


class _LinearWrapper:
    """Adapter for ``_projector_prepare_triton`` so it can accept either a
    ``LinearProjector`` (with ``.linear``) or a bare ``nn.Linear`` directly.

    Bare ``nn.Linear`` is wrapped so ``.linear`` resolves to itself, matching the
    FluxVLA reference's ``projector.weight`` access path.
    """

    def __init__(self, layer: Any) -> None:
        self.linear = layer


__all__ = [
    "Pi05FastKernelsInference",
    "pi05_model",
    "vision_encoder",
    "transformer_encoder",
    "transformer_decoder",
]
