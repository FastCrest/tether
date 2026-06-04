"""SmolVLA model loader and component splitter for export.

Reconstructs the SmolVLA architecture from a raw safetensors checkpoint
without requiring lerobot as a dependency. Maps directly to the
lerobot/policies/smolvla/ source code structure.

Architecture (450M total):
  model.vlm_with_expert.vlm       (350.2M) — SmolVLM2-500M (SigLIP + SmolLM2 16-layer decoder)
  model.vlm_with_expert.lm_expert  (98.2M) — Action expert (cross-attn to VLM KV, 0.75x width)
  model.action_in_proj             — nn.Linear(32, expert_hidden)
  model.action_out_proj            — nn.Linear(expert_hidden, 32)
  model.action_time_mlp_in         — nn.Linear(expert_hidden*2, expert_hidden)
  model.action_time_mlp_out        — nn.Linear(expert_hidden, expert_hidden)
  model.state_proj                 — nn.Linear(32, vlm_hidden)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# SmolVLA default config values (from configuration_smolvla.py)
SMOLVLA_CONFIG = {
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "chunk_size": 50,
    "n_action_steps": 50,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "num_steps": 10,
    "num_vlm_layers": 16,
    "self_attn_every_n_layers": 2,
    "expert_width_multiplier": 0.75,
    "image_size": 512,
    "min_period": 4e-3,
    "max_period": 4.0,
}


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sinusoidal timestep embedding for flow matching."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")
    d_half = dimension // 2
    exponent = torch.linspace(0, 1, d_half, device=time.device, dtype=time.dtype)
    freq = torch.exp(exponent * (math.log(min_period) - math.log(max_period))) / min_period
    angle = time.unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([angle.cos(), angle.sin()], dim=-1)


class SmolVLASuffixEncoder(nn.Module):
    """Encodes noisy actions + timestep into suffix embeddings for the action expert.

    This is the part that runs on every denoising step.
    Extracted from VLAFlowMatching.embed_suffix() for standalone ONNX export.
    """

    def __init__(
        self,
        action_dim: int = 32,
        expert_hidden_size: int = 432,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.expert_hidden_size = expert_hidden_size
        self.min_period = min_period
        self.max_period = max_period

        self.action_in_proj = nn.Linear(action_dim, expert_hidden_size)
        self.action_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)

    def forward(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (batch, chunk_size, action_dim)
            timestep: (batch,) or (1,) scalar timestep in [0, 1]

        Returns:
            suffix_embs: (batch, chunk_size, expert_hidden_size)
        """
        batch_size, chunk_size, _ = noisy_actions.shape

        # Project actions
        action_embs = self.action_in_proj(noisy_actions)

        # Sinusoidal timestep embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.expert_hidden_size,
            min_period=self.min_period, max_period=self.max_period,
        )
        # Expand to match action sequence
        time_emb = time_emb.unsqueeze(1).expand(-1, chunk_size, -1)

        # Fuse action + time
        fused = torch.cat([action_embs, time_emb], dim=-1)
        suffix_embs = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(fused))
        )
        return suffix_embs


class SmolVLAActionProjection(nn.Module):
    """Projects expert output to predicted velocity. Extracted for ONNX export."""

    def __init__(self, expert_hidden_size: int = 432, action_dim: int = 32):
        super().__init__()
        self.action_out_proj = nn.Linear(expert_hidden_size, action_dim)

    def forward(self, expert_output: torch.Tensor) -> torch.Tensor:
        return self.action_out_proj(expert_output)


def load_smolvla_components(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, nn.Module]:
    """Extract SmolVLA into exportable components from raw state_dict.

    Returns dict with keys: 'suffix_encoder', 'action_projection',
    and metadata about the VLM backbone.
    """
    # Determine hidden sizes from weight shapes
    action_in_weight = state_dict.get("model.action_in_proj.weight")
    action_out_weight = state_dict.get("model.action_out_proj.weight")
    time_mlp_in_weight = state_dict.get("model.action_time_mlp_in.weight")

    if action_in_weight is None:
        raise ValueError("Cannot find model.action_in_proj.weight in state_dict")

    expert_hidden_size = action_in_weight.shape[0]
    action_dim = action_in_weight.shape[1]

    logger.info(
        "SmolVLA dimensions: expert_hidden=%d, action_dim=%d",
        expert_hidden_size, action_dim,
    )

    # Build suffix encoder
    suffix_encoder = SmolVLASuffixEncoder(
        action_dim=action_dim,
        expert_hidden_size=expert_hidden_size,
    )
    suffix_sd = {
        "action_in_proj.weight": state_dict["model.action_in_proj.weight"],
        "action_in_proj.bias": state_dict["model.action_in_proj.bias"],
        "action_time_mlp_in.weight": state_dict["model.action_time_mlp_in.weight"],
        "action_time_mlp_in.bias": state_dict["model.action_time_mlp_in.bias"],
        "action_time_mlp_out.weight": state_dict["model.action_time_mlp_out.weight"],
        "action_time_mlp_out.bias": state_dict["model.action_time_mlp_out.bias"],
    }
    suffix_encoder.load_state_dict(suffix_sd)
    suffix_encoder.eval()

    # Build action projection
    action_projection = SmolVLAActionProjection(
        expert_hidden_size=expert_hidden_size,
        action_dim=action_dim,
    )
    proj_sd = {
        "action_out_proj.weight": state_dict["model.action_out_proj.weight"],
        "action_out_proj.bias": state_dict["model.action_out_proj.bias"],
    }
    action_projection.load_state_dict(proj_sd)
    action_projection.eval()

    # Count VLM and expert params
    vlm_params = sum(
        v.numel() for k, v in state_dict.items()
        if k.startswith("model.vlm_with_expert.vlm")
    )
    expert_params = sum(
        v.numel() for k, v in state_dict.items()
        if k.startswith("model.vlm_with_expert.lm_expert")
    )

    return {
        "suffix_encoder": suffix_encoder,
        "action_projection": action_projection,
        "metadata": {
            "expert_hidden_size": expert_hidden_size,
            "action_dim": action_dim,
            "vlm_params_m": vlm_params / 1e6,
            "expert_params_m": expert_params / 1e6,
            "total_params_m": sum(v.numel() for v in state_dict.values()) / 1e6,
            "vlm_model_name": SMOLVLA_CONFIG["vlm_model_name"],
            "num_denoising_steps": SMOLVLA_CONFIG["num_steps"],
            "chunk_size": SMOLVLA_CONFIG["chunk_size"],
        },
    }
