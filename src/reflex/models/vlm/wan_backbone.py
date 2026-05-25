"""WanBackbone — Wan2.1 encoder for DreamZero on the BaseVLA spine.

Fills the ``vlm_backbone`` slot. Contains T5 text encoder + CLIP image
encoder + Video VAE — all frozen. Ported from FluxVLA
``backbones/vlms/wan_backbone.py`` (Apache-2.0, LimX Dynamics).

The reflex version rewrites imports to point at
``reflex.models.third_party.dreamzero.modules`` instead of
``fluxvla.models.third_party_models.dreamzero.modules``.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_file(path: str | None, hf_filename: str) -> str:
    if path is not None and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="Wan-AI/Wan2.1-I2V-14B-480P", filename=hf_filename)


def _import_wan_modules():
    from reflex.models.third_party.dreamzero.modules.wan_video_text_encoder import WanTextEncoder
    from reflex.models.third_party.dreamzero.modules.wan_video_image_encoder import WanImageEncoder
    from reflex.models.third_party.dreamzero.modules.wan_video_vae import WanVideoVAE
    return WanTextEncoder, WanImageEncoder, WanVideoVAE


class WanBackbone(nn.Module):
    """Wan 2.1 encoder backbone for DreamZero.

    Contains T5 (text), CLIP (image), VAE (video latents). All frozen.

    Args:
        text_encoder_path: Path to T5 weights (.pth). Downloads from HF if None.
        image_encoder_path: Path to CLIP weights (.pth).
        vae_path: Path to VAE weights (.pth).
        text_len: Max text token length (default 512).
        dtype: Compute dtype (default bfloat16).
    """

    def __init__(
        self,
        text_encoder_path: str | None = None,
        image_encoder_path: str | None = None,
        vae_path: str | None = None,
        text_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        WanTextEncoder, WanImageEncoder, WanVideoVAE = _import_wan_modules()

        text_encoder_path = _ensure_file(text_encoder_path, "models_t5_umt5-xxl-enc-bf16.pth")
        image_encoder_path = _ensure_file(image_encoder_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        vae_path = _ensure_file(vae_path, "Wan2.1_VAE.pth")

        self.text_encoder = WanTextEncoder(
            text_len=text_len,
            dtype=dtype,
            pretrained_model_path=text_encoder_path,
        )
        self.image_encoder = WanImageEncoder(
            dtype=dtype,
            pretrained_model_path=image_encoder_path,
        )
        self.vae = WanVideoVAE(
            pretrained_model_path=vae_path,
        )

        # Freeze all encoders
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Encode video + text + image for the DreamZero head.

        Args:
            video: [B, C, T, H, W] float tensor (normalized).
            input_ids: [B, seq_len] int64 T5 token ids.
            attention_mask: [B, seq_len] int64 attention mask.

        Returns:
            Dict with keys: prompt_embs, latents, clip_feas, image_cond.
        """
        # T5 text encoding
        prompt_embs = self.text_encoder(input_ids, attention_mask)

        # CLIP image features (from first frame)
        clip_feas = self.image_encoder(video[:, :, 0])  # first frame

        # VAE latent encoding
        with torch.no_grad():
            latents = self.vae.encode(video)

        # Image conditioning (first frame latent)
        image_cond = latents[:, :, :1]

        return {
            "prompt_embs": prompt_embs,
            "latents": latents,
            "clip_feas": clip_feas,
            "image_cond": image_cond,
        }

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}{name}": param.detach().clone()
            for name, param in self.named_parameters()
        }


__all__ = ["WanBackbone"]
