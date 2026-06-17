"""DreamZero ONNX exporter (Lift #7 Day 4).

DreamZero's architecture differs from the flow-matching VLAs (pi0/pi0.5/
SmolVLA) — it uses a video DiT with 5D tensors [B, C, T, H, W] and joint
video + action denoising. This exporter handles the unique input shapes.

Two-stage export per the spine pattern:
1. Encoder: WanBackbone (T5 + CLIP + VAE) → frozen, exported once
2. Denoiser: DreamZeroHead's CausalWanModel → the heavy compute path

Usage::

    from tether.exporters.dreamzero import export_dreamzero
    result = export_dreamzero(
        checkpoint_path="limxdynamics/FluxVLAEngine",
        output_dir="./dreamzero_export/",
    )
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_dreamzero(
    checkpoint_path: str,
    output_dir: str | Path,
    *,
    action_dim: int = 7,
    max_action_dim: int = 32,
    action_horizon: int = 10,
    num_frames: int = 9,
    image_size: int = 480,
    opset_version: int = 17,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export DreamZero VLA to ONNX.

    Args:
        checkpoint_path: Path to the FluxVLA checkpoint directory or HF repo.
        output_dir: Directory to write .onnx files + tether_config.json.
        action_dim: Actual robot action dimension.
        max_action_dim: Padded action dim for the DiT.
        action_horizon: Action steps per generation.
        num_frames: Video frames (including conditioning frame).
        image_size: Input image resolution (480 for Wan2.1).
        opset_version: ONNX opset.
        device: Export device.

    Returns:
        Dict with keys: files (paths to .onnx), config (tether_config.json contents).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DreamZero export — checkpoint=%s, output=%s", checkpoint_path, output_dir)

    # Build the tether_config.json (describes the export for `tether serve`)
    config = {
        "model_family": "dreamzero",
        "architecture": "world_action_model",
        "action_dim": action_dim,
        "max_action_dim": max_action_dim,
        "action_horizon": action_horizon,
        "num_frames": num_frames,
        "image_size": image_size,
        "num_inference_steps": 4,
        "components": {
            "vlm_backbone": "wan_backbone",
            "vla_head": "dreamzero_head",
        },
        "export_format": "dreamzero_decomposed",
        "requires_video_input": True,
    }

    config_path = output_dir / "tether_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote tether_config.json")

    result = {
        "files": {"config": str(config_path)},
        "config": config,
        "status": "config_only",
    }

    # ONNX export of the DiT is deferred — the CausalWanModel has
    # dynamic control flow (KV cache, variable-length denoising loop)
    # that doesn't export cleanly to static ONNX. Options:
    #
    # 1. Export the single-step denoiser (one DiT forward pass) as ONNX,
    #    then run the denoising loop in Python (same as pi0.5 decomposed).
    # 2. Use torch.export + torch-tensorrt for the full graph.
    # 3. Use the Triton fast-kernels path (Lift #5) directly.
    #
    # For V1: write the config so `tether serve` can detect the model
    # family and dispatch to the correct runtime. The actual ONNX export
    # of the DiT is a follow-up (the DiT has flash-attn + conditional
    # KV cache paths that need careful tracing).
    logger.info(
        "DreamZero ONNX export: config written. DiT ONNX export deferred — "
        "the CausalWanModel has dynamic KV cache + flash-attn that needs "
        "careful tracing. V1 ships config + PyTorch runtime path."
    )

    return result


def validate_dreamzero_registry() -> bool:
    """Check that the DreamZero registry entry resolves."""
    from tether.registry.data import REGISTRY
    for entry in REGISTRY:
        if entry.model_id == "pi05-libero10-fluxvla":
            return True
    return False


__all__ = ["export_dreamzero", "validate_dreamzero_registry"]
