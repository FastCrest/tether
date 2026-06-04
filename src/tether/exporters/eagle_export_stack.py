"""EagleExportStack — export-friendly wrapper for GR00T's Eagle VLM backbone.

Mirrors `lerobot.policies.groot.groot_n1.EagleBackbone.forward_eagle` for
a single-shot encode (no autoregressive decode). Takes:

    pixel_values   [B, 3, H, W]        — SigLIP input image
    input_ids      [B, seq]             — Qwen2 token ids (with image_token placeholders)
    attention_mask [B, seq]             — 1 for real tokens, 0 for padding
    image_flags    [B]                  — 1 if image is real (kept), 0 to drop

Produces:

    hidden_states  [B, seq, 2048]       — Qwen2 hidden at select_layer (last by default)

This is what GR00T's DiT cross-attn consumes as vlm_kv. The corresponding
ONNX export feeds these hidden_states as the `vlm_kv` input of
`expert_stack_with_vlm.onnx` (Step 4a).

Weight loading: state_dict keys come from `nvidia/GR00T-N1.6-3B` under the
`backbone.model.*` prefix (NOT `backbone.eagle_model.*` as older research
guessed — verified by scripts/modal_gr00t_keys_dump.py).

Export-relevant tweaks:
  - `_attn_implementation="eager"` forced in the vendored config default
    (FA2 not consumable by ONNX/TRT)
  - `use_cache=False` in the language_model forward to avoid DynamicCache
  - No autoregressive loop → no cache growth, no need for the 3-patch
    stack that pi0/pi0.5 required
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EagleExportStack(nn.Module):
    """Single-shot encode: (pixel_values, input_ids, attention_mask, image_flags)
    → hidden_states at select_layer.
    """

    def __init__(self, eagle_model: nn.Module, select_layer: int = -1):
        super().__init__()
        self.eagle = eagle_model
        self.select_layer = select_layer
        # Cache useful sub-modules for readability
        self.vision_model = eagle_model.vision_model
        self.language_model = eagle_model.language_model
        self.mlp1 = eagle_model.mlp1
        self.image_token_index = eagle_model.image_token_index

    def _splice_vision_into_text(
        self,
        input_embeds: torch.Tensor,     # [B, seq, C]
        input_ids: torch.Tensor,        # [B, seq]
        vit_embeds: torch.Tensor,       # [B, vision_seq, C]
    ) -> torch.Tensor:
        """Replace image-token positions in input_embeds with vit_embeds.

        Export contract: image tokens are packed at the FRONT of each
        sequence. The caller constructs input_ids as
        `[img_tok]*vision_seq + [text tokens]*(seq - vision_seq)`.

        Eagle's original forward uses `input_embeds[selected] = vit_flat`
        (boolean-index assignment). That lowers to an index_put → Where
        in ONNX that ORT can't broadcast when vit length (64) differs
        from seq length (80). We bypass it with a dtype-safe cat:
        replace the first vision_seq rows with vit_embeds, keep the
        tail as text embeds.
        """
        vs = vit_embeds.shape[1]
        text_embeds = input_embeds[:, vs:, :]
        return torch.cat([vit_embeds, text_embeds], dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_flags: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Vision path: SigLIP → pixel_shuffle → mlp1
        vit_embeds = self.eagle.extract_feature(pixel_values)  # [B, vision_seq, C]

        # Filter by image_flags (drop images marked absent)
        if image_flags is not None:
            image_flags_flat = image_flags.view(-1)
            vit_embeds = vit_embeds[image_flags_flat == 1]

        # 2. Text embed lookup
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 3. Splice vision into text at image_token positions
        input_embeds = self._splice_vision_into_text(
            input_embeds, input_ids, vit_embeds,
        )

        # 4. Qwen2 forward, with output_hidden_states to get the select_layer
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states is a tuple of (num_layers + 1) tensors. select_layer
        # is typically -1 (last layer).
        hidden = outputs.hidden_states[self.select_layer]  # [B, seq, 2048]
        return hidden


def build_eagle_export_stack(
    state_dict: dict[str, torch.Tensor],
    select_layer: int = -1,
) -> tuple[EagleExportStack, dict]:
    """Instantiate Eagle from vendored source + load state dict.

    Returns (stack, meta). Meta includes:
      - vit_hidden: vision hidden dim (1152 for SigLIP-so400m)
      - llm_hidden: text hidden dim (2048 for Qwen2-0.5B)
      - shuffled_hidden: pixel-shuffled vision dim before mlp1 (4608 = 1152*4)
      - vocab_size: Qwen2 tokenizer vocab (151680)
      - image_token_index: sentinel ID for image placeholder in input_ids
      - total_params_m: total parameter count in millions

    Requires the `action_head.state_encoder.*` and `backbone.model.*` keys
    to be present (the N1.6 pattern verified by modal_gr00t_keys_dump.py).
    """
    # Lazy-import the vendor so downstream files don't always pull it.
    from tether.exporters.eagle_vendor.modeling_eagle2_5_vl import (
        Eagle25VLForConditionalGeneration,
    )
    from tether.exporters.eagle_vendor.configuration_eagle2_5_vl import (
        Eagle25VLConfig,
    )

    # Derive config from state_dict shapes (Eagle's default dims are for
    # larger variants; N1.6 uses a specific Qwen2-0.5B + SigLIP-so400m).
    vocab_size, llm_hidden = state_dict[
        "backbone.model.language_model.model.embed_tokens.weight"
    ].shape
    # Qwen2 MLP: gate_proj is [intermediate, hidden]. Count layers.
    llm_intermediate = state_dict[
        "backbone.model.language_model.model.layers.0.mlp.gate_proj.weight"
    ].shape[0]
    # num_layers = count distinct layers.X prefixes
    llm_num_layers = 0
    for k in state_dict.keys():
        if k.startswith("backbone.model.language_model.model.layers."):
            layer_idx = int(k.split(".")[5])
            llm_num_layers = max(llm_num_layers, layer_idx + 1)
    # Attention dims: q_proj is [hidden, hidden] for standard, [head_dim*num_q_heads, hidden] for GQA
    q_proj_out, _ = state_dict[
        "backbone.model.language_model.model.layers.0.self_attn.q_proj.weight"
    ].shape
    k_proj_out, _ = state_dict[
        "backbone.model.language_model.model.layers.0.self_attn.k_proj.weight"
    ].shape
    # For Qwen2: num_key_value_heads = k_proj_out / head_dim; assume head_dim = 128 (Qwen2 standard).
    # Actually derive: num_attention_heads × head_dim = q_proj_out.
    # N1.6 Qwen2-0.5B: hidden=2048, q_proj_out=2048 (16 heads × 128 head_dim),
    # k_proj_out=256 (2 kv_heads × 128 head_dim via GQA).
    head_dim = 128  # Qwen2 default; override if shape implies otherwise
    num_attention_heads = q_proj_out // head_dim
    num_key_value_heads = k_proj_out // head_dim

    # SigLIP hidden derivation: patch_embedding weight is [hidden, 3, patch, patch]
    # for siglip_vision_model. Find it.
    vit_patch_key = "backbone.model.vision_model.vision_model.embeddings.patch_embedding.weight"
    if vit_patch_key in state_dict:
        vit_hidden = state_dict[vit_patch_key].shape[0]
    else:
        # Fallback: look for any key with vision + "weight" and reasonable shape
        vit_hidden = 1152  # SigLIP-so400m default
    # SigLIP intermediate: mlp.fc1 weight [intermediate, hidden]
    vit_fc1_key = (
        "backbone.model.vision_model.vision_model.encoder.layers.0.mlp.fc1.weight"
    )
    if vit_fc1_key in state_dict:
        vit_intermediate = state_dict[vit_fc1_key].shape[0]
    else:
        vit_intermediate = 4304  # SigLIP-so400m default (1152 × 3.74)
    # SigLIP num layers
    vit_num_layers = 0
    vit_layer_prefix = "backbone.model.vision_model.vision_model.encoder.layers."
    for k in state_dict.keys():
        if k.startswith(vit_layer_prefix):
            layer_idx = int(k[len(vit_layer_prefix):].split(".")[0])
            vit_num_layers = max(vit_num_layers, layer_idx + 1)
    # Image size: derive from position_embedding (SigLIP: num_positions=(size/patch)^2)
    pos_embed_key = (
        "backbone.model.vision_model.vision_model.embeddings.position_embedding.weight"
    )
    if pos_embed_key in state_dict:
        num_positions = state_dict[pos_embed_key].shape[0]
        # image_size = patch_size * sqrt(num_positions). Assume patch_size=14 (SigLIP-so400m default).
        import math
        image_size = 14 * int(math.sqrt(num_positions))
    else:
        image_size = 448

    logger.info(
        "[eagle-export] derived config: Qwen2 hidden=%d, intermediate=%d, "
        "layers=%d, heads=%d/%d, vocab=%d",
        llm_hidden, llm_intermediate, llm_num_layers,
        num_attention_heads, num_key_value_heads, vocab_size,
    )
    logger.info(
        "[eagle-export] derived config: SigLIP hidden=%d, intermediate=%d, "
        "layers=%d, image_size=%d",
        vit_hidden, vit_intermediate, vit_num_layers, image_size,
    )

    # Build explicit sub-configs with the derived dims.
    vision_config_dict = {
        "model_type": "siglip_vision_model",
        "hidden_size": int(vit_hidden),
        "intermediate_size": int(vit_intermediate),
        "num_hidden_layers": int(vit_num_layers),
        "num_attention_heads": 16,  # SigLIP-so400m default
        "num_channels": 3,
        "image_size": int(image_size),
        "patch_size": 14,
    }
    # N1.6 uses Qwen3-style layers (q_norm + k_norm inside self-attn,
    # no bias on q/k/v projections). Confirmed by smoke run: loading with
    # Qwen2 config produced 48 missing keys (q/k/v.bias for each layer)
    # and 32 unexpected keys (k_norm/q_norm each layer). Switch to Qwen3.
    text_config_dict = {
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": int(vocab_size),
        "hidden_size": int(llm_hidden),
        "intermediate_size": int(llm_intermediate),
        "num_hidden_layers": int(llm_num_layers),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    }

    cfg = Eagle25VLConfig(
        vision_config=vision_config_dict,
        text_config=text_config_dict,
        image_token_index=151655,  # Qwen2-VL's default; actual may differ per-ckpt
    )

    # Instantiate. The Eagle25VL vendor handles submodule construction.
    model = Eagle25VLForConditionalGeneration(cfg)

    # Load weights from state_dict — prefix is `backbone.model.*` so strip it.
    # Also reshape patch_embedding from flat (1152, 588) to conv-style
    # (1152, 3, 14, 14). N1.6 stores it flattened (why?); standard SigLIP
    # uses Conv2d weights. 588 = 3 channels × 14 × 14 patch.
    remapped = {}
    patch_key = "vision_model.vision_model.embeddings.patch_embedding.weight"
    for k, v in state_dict.items():
        if k.startswith("backbone.model."):
            sub = k[len("backbone.model."):]
            v = v.float()
            if sub == patch_key and v.ndim == 2:
                # Reshape (hidden, C*H*W) → (hidden, C, H, W)
                hidden, flat = v.shape
                # Assume C=3, square patch
                patch = int((flat // 3) ** 0.5)
                v = v.reshape(hidden, 3, patch, patch).contiguous()
                logger.info("[eagle-export] reshaped patch_embedding to %s", tuple(v.shape))
            remapped[sub] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.warning("[eagle-export] %d missing keys (first 3: %s)",
                       len(missing), list(missing)[:3])
    if unexpected:
        logger.warning("[eagle-export] %d unexpected keys (first 3: %s)",
                       len(unexpected), list(unexpected)[:3])

    stack = EagleExportStack(model, select_layer=select_layer).float().eval()

    vit_hidden = cfg.vision_config.hidden_size
    meta = {
        "vit_hidden": vit_hidden,
        "llm_hidden": int(llm_hidden),
        "shuffled_hidden": int(vit_hidden * 4),  # pixel_shuffle concat of 2×2
        "vocab_size": int(vocab_size),
        "image_token_index": stack.image_token_index,
        "total_params_m": sum(p.numel() for p in stack.parameters()) / 1e6,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
    }
    return stack, meta


__all__ = ["EagleExportStack", "build_eagle_export_stack"]
