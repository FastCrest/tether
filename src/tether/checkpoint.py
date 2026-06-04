"""Checkpoint loading and validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

SMOLVLA_EXPECTED_PREFIXES = [
    "model.vision_encoder",
    "model.language_model",
    "model.action_expert",
    "model.vlm_with_expert",
]

PI0_EXPECTED_PREFIXES = [
    "paligemma_with_expert",
]

GR00T_EXPECTED_PREFIXES = [
    "action_head.model.transformer_blocks.",
]

OPENVLA_EXPECTED_PREFIXES = [
    "vision_backbone.featurizer.",
    "projector.fc1.",
]

PI05_MARKERS = [
    "input_layernorm.dense.weight",  # AdaRMSNorm — pi0.5 only
]

SUPPORTED_MODELS = {
    "smolvla": {
        "hf_id": "lerobot/smolvla_base",
        "params_m": 450,
        "vision_encoder": "siglip",
        "backbone": "smollm2",
        "action_head": "flow_matching",
        "num_denoising_steps": 10,
        "action_chunk_size": 50,
        "action_dim": 6,
    },
    "pi0": {
        "hf_id": "lerobot/pi0_base",
        "params_m": 3500,
        "vision_encoder": "siglip",
        "backbone": "paligemma_gemma2b",
        "action_head": "flow_matching",
        "num_denoising_steps": 10,
        "action_chunk_size": 50,
        "action_dim": 32,  # max_action_dim
    },
    "pi05": {
        "hf_id": "lerobot/pi05_base",
        "params_m": 3620,
        "vision_encoder": "siglip",
        "backbone": "paligemma_gemma2b",
        "action_head": "flow_matching_adarms",
        "num_denoising_steps": 10,
        "action_chunk_size": 50,
        "action_dim": 32,
    },
    "gr00t": {
        "hf_id": "nvidia/GR00T-N1.6-3B",
        "params_m": 3290,
        "vision_encoder": "siglip2",
        "backbone": "qwen3",
        "action_head": "flow_matching_dit_adaln",
        "num_denoising_steps": 4,
        "action_chunk_size": 50,
        "action_dim": 128,  # max_action_dim
    },
    "openvla": {
        "hf_id": "openvla/openvla-7b",
        "params_m": 7541,
        "vision_encoder": "dino_siglip",
        "backbone": "llama2_7b",
        "action_head": "tokenized_bins",  # argmax + 256-bin lookup, no flow matching
        "num_denoising_steps": 1,  # single forward pass
        "action_chunk_size": 1,
        "action_dim": 7,
    },
}


def detect_model_type(state_dict: dict[str, torch.Tensor]) -> str | None:
    keys = set(state_dict.keys())
    # GR00T check first (most specific prefix)
    if any(any(k.startswith(p) for k in keys) for p in GR00T_EXPECTED_PREFIXES):
        return "gr00t"
    # OpenVLA (vision_backbone + projector)
    if any(any(k.startswith(p) for k in keys) for p in OPENVLA_EXPECTED_PREFIXES):
        return "openvla"
    # pi0 / pi0.5
    if any(any(k.startswith(p) for k in keys) for p in PI0_EXPECTED_PREFIXES):
        # Differentiate pi0 vs pi0.5 via AdaRMSNorm marker
        if any(any(marker in k for k in keys) for marker in PI05_MARKERS):
            return "pi05"
        return "pi0"
    # SmolVLA
    if any(any(k.startswith(p) for k in keys) for p in SMOLVLA_EXPECTED_PREFIXES):
        return "smolvla"
    return None


def load_checkpoint(
    path_or_id: str,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a VLA checkpoint from local path or HuggingFace Hub.

    Returns (state_dict, config_dict).
    """
    path = Path(path_or_id)

    if path.exists() and path.suffix == ".safetensors":
        logger.info("Loading local safetensors: %s", path)
        state_dict = load_file(str(path), device=device)
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return state_dict, config

    if path.is_dir():
        safetensors_files = list(path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files in {path}")
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(str(f), device=device))
        config_path = path / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return state_dict, config

    # Assume HuggingFace Hub ID
    logger.info("Downloading from HuggingFace Hub: %s", path_or_id)
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(path_or_id)
        return load_checkpoint(local_dir, device=device)
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")


def validate_checkpoint(
    state_dict: dict[str, torch.Tensor],
    model_type: str,
) -> list[str]:
    """Validate checkpoint keys against expected structure. Returns list of warnings."""
    warnings = []
    model_info = SUPPORTED_MODELS.get(model_type)
    if model_info is None:
        warnings.append(f"Unknown model type: {model_type}")
        return warnings

    total_params = sum(p.numel() for p in state_dict.values())
    expected_params = model_info["params_m"] * 1_000_000
    ratio = total_params / expected_params
    if ratio < 0.8 or ratio > 1.2:
        warnings.append(
            f"Parameter count {total_params / 1e6:.1f}M differs from expected "
            f"{model_info['params_m']}M by {abs(1 - ratio) * 100:.0f}%"
        )

    key_prefixes = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) >= 2:
            key_prefixes.add(f"{parts[0]}.{parts[1]}")

    logger.info(
        "Loaded %d tensors, %.1fM params, %d key prefixes",
        len(state_dict),
        total_params / 1e6,
        len(key_prefixes),
    )
    return warnings
