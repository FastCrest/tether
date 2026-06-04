"""ONNX export for VLA model components."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tether.config import ExportConfig, HardwareProfile

logger = logging.getLogger(__name__)


def export_module_to_onnx(
    module: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 19,
) -> Path:
    """Export a PyTorch module to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            opset_version=opset_version,
            do_constant_folding=True,
        )

    logger.info("Exported ONNX: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def optimize_onnx(onnx_path: Path, num_steps: int = 10) -> Path:
    """Run weight fusion + onnxsim for constant folding and optimization.

    Weight fusion (lifted from dexmal/realtime-vla MIT pattern):
    fuses RMSNorm scales into adjacent MatMul weights and folds Euler
    step dt into output projections. Then runs onnxsim for remaining
    constant folding.
    """
    try:
        from tether.exporters.weight_fusion import fuse_weights
        fuse_weights(onnx_path, num_steps=num_steps)
    except Exception as e:
        logger.warning("Weight fusion pass failed (non-fatal): %s", e)

    try:
        import onnxsim
        import onnx

        model = onnx.load(str(onnx_path))
        model_opt, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_opt, str(onnx_path))
            logger.info("Optimized ONNX: %s", onnx_path)
        else:
            logger.warning("ONNX simplification check failed, keeping original")
    except ImportError:
        logger.warning("onnxsim not installed, skipping ONNX optimization")
    return onnx_path


class VLAOnnxExporter:
    """Export VLA model components to ONNX."""

    def __init__(self, config: ExportConfig, hardware: HardwareProfile):
        self.config = config
        self.hardware = hardware
        self.output_dir = Path(config.output_dir)

    def export_vision_encoder(
        self, vision_encoder: nn.Module, image_size: int = 512
    ) -> Path:
        """Export vision encoder (SigLIP) to ONNX."""
        dummy_image = torch.randn(1, 3, image_size, image_size)
        path = self.output_dir / "vision_encoder.onnx"
        export_module_to_onnx(
            vision_encoder,
            (dummy_image,),
            path,
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={"pixel_values": {0: "batch"}},
            opset_version=self.config.opset,
        )
        return optimize_onnx(path)

    def export_backbone(
        self, backbone: nn.Module, hidden_size: int = 576, max_seq_len: int = 1024
    ) -> Path:
        """Export VLM backbone (SmolLM2) to ONNX."""
        seq_len = min(256, max_seq_len)
        dummy_hidden = torch.randn(1, seq_len, hidden_size)
        dummy_mask = torch.ones(1, seq_len, dtype=torch.long)
        path = self.output_dir / "backbone.onnx"
        export_module_to_onnx(
            backbone,
            (dummy_hidden, dummy_mask),
            path,
            input_names=["hidden_states", "attention_mask"],
            output_names=["backbone_output"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
            },
            opset_version=self.config.opset,
        )
        return optimize_onnx(path)

    def export_denoising_step(
        self,
        action_expert: nn.Module,
        hidden_size: int = 432,
        action_dim: int = 6,
        chunk_size: int = 50,
    ) -> Path:
        """Export a single denoising step of the action head."""
        dummy_noisy_actions = torch.randn(1, chunk_size, action_dim)
        dummy_timestep = torch.tensor([0.5])
        dummy_conditioning = torch.randn(1, 256, hidden_size)
        path = self.output_dir / "denoising_step.onnx"
        export_module_to_onnx(
            action_expert,
            (dummy_noisy_actions, dummy_timestep, dummy_conditioning),
            path,
            input_names=["noisy_actions", "timestep", "conditioning"],
            output_names=["velocity"],
            dynamic_axes={"conditioning": {1: "cond_seq_len"}},
            opset_version=self.config.opset,
        )
        return optimize_onnx(path)

    def export_all(self, model_components: dict[str, nn.Module]) -> dict[str, Path]:
        """Export all VLA components to ONNX."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        if "vision_encoder" in model_components:
            logger.info("Exporting vision encoder...")
            paths["vision_encoder"] = self.export_vision_encoder(
                model_components["vision_encoder"]
            )

        if "backbone" in model_components:
            logger.info("Exporting VLM backbone...")
            paths["backbone"] = self.export_backbone(model_components["backbone"])

        if "action_expert" in model_components:
            logger.info("Exporting denoising step...")
            paths["denoising_step"] = self.export_denoising_step(
                model_components["action_expert"]
            )

        return paths
