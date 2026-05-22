"""SmolVLA export pipeline — refactored composition via the BaseVLA spine.

Mirrors the legacy ``smolvla_exporter.export_smolvla`` shape (output_dir
layout, file names, ``reflex_config.json``), but builds via the
``SmolVLA(BaseVLA)`` composition class instead of directly assembling the
``ExpertStack``. Same checkpoint → same ONNX bytes (bit-identical numerics
guaranteed by reusing ``build_expert_stack`` under the hood).

The legacy module ``reflex.exporters.smolvla_exporter`` stays available for
backward compatibility per the lift #1 plan (Day 11 sunsets it together
with pi0_exporter / pi05_exporter after all callers migrate).

Per the user's 2026-05-22 Day 6 scope choice (Phase A + B bundled), this
file lands the export refactor in the same PR as the spine composition.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from reflex.checkpoint import load_checkpoint
from reflex.config import ExportConfig, get_hardware_profile
from reflex.exporters.onnx_export import export_module_to_onnx, optimize_onnx
from reflex.exporters.trt_build import build_engine, check_trtexec

logger = logging.getLogger(__name__)


def export_smolvla(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Full SmolVLA export — spine-based composition.

    Behaves identically to ``smolvla_exporter.export_smolvla`` but routes
    through ``SmolVLA(BaseVLA).from_pretrained`` for the build step. The
    final ONNX bytes are bit-identical (same ``build_expert_stack`` call,
    same dummy inputs, same opset).

    Args:
        config: ``ExportConfig`` (output_dir, opset, action_chunk_size, ...)
        state_dict: optional pre-loaded SmolVLA checkpoint to skip the
            ``load_checkpoint`` step.

    Returns:
        ``{"status": "ok", "files": {...}, "metadata": {...}}``
    """
    from reflex.models.vlas.smolvla import SmolVLA

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = get_hardware_profile(config.target)
    result: dict[str, Any] = {"status": "ok", "files": {}, "metadata": {}}

    # 1. Load checkpoint (if not provided)
    if state_dict is None:
        logger.info("Loading checkpoint: %s", config.model_id)
        state_dict, _ = load_checkpoint(config.model_id)
    total_params = sum(v.numel() for v in state_dict.values())
    logger.info("Loaded %d tensors, %.1fM params", len(state_dict), total_params / 1e6)

    # 2. Build via the BaseVLA spine. SmolVLA.from_pretrained internally:
    #    - Loads SmolVLM2 (vision_backbone + llm_backbone slots)
    #    - Builds the cross-attn expert via build_expert_stack
    #    - Wraps state_proj as a LinearProjector
    logger.info("Building SmolVLA via the BaseVLA spine...")
    vla = SmolVLA.from_pretrained(state_dict=state_dict)
    expert_stack = vla.vla_head.expert_stack
    expert_meta = {
        "expert_hidden": expert_stack.expert_hidden,
        "action_dim": expert_stack.action_in_proj.in_features,
        "num_layers": len(expert_stack.layers),
        "cross_attn_layers": sorted(expert_stack.cross_indices),
        "vlm_kv_dim": expert_stack.vlm_kv_dim,
        "total_params_m": sum(p.numel() for p in expert_stack.parameters()) / 1e6,
    }
    result["metadata"]["expert"] = expert_meta
    logger.info(
        "Expert: %d layers, %.1fM params, cross_attn=%s",
        expert_meta["num_layers"], expert_meta["total_params_m"], expert_meta["cross_attn_layers"],
    )

    # 3. Export expert stack to ONNX — identical shape to the legacy path.
    action_dim = expert_meta["action_dim"]
    chunk_size = config.action_chunk_size
    vlm_kv_dim = expert_meta["vlm_kv_dim"]
    num_layers = expert_meta["num_layers"]

    dummy_actions = torch.randn(1, chunk_size, action_dim)
    dummy_time = torch.tensor([0.5])
    dummy_pos = torch.arange(chunk_size).unsqueeze(0)
    dummy_vlm_k = torch.zeros(num_layers, 1, 1, vlm_kv_dim)
    dummy_vlm_v = torch.zeros(num_layers, 1, 1, vlm_kv_dim)
    dummy_prefix_offset = torch.tensor([[241]], dtype=torch.int64)
    dummy_kv_mask = torch.ones(1, 1, dtype=torch.bool)

    expert_onnx = output_dir / "expert_stack.onnx"
    logger.info("Exporting expert stack to ONNX: %s", expert_onnx)
    export_module_to_onnx(
        expert_stack,
        (dummy_actions, dummy_time, dummy_pos, dummy_vlm_k, dummy_vlm_v,
         dummy_prefix_offset, dummy_kv_mask),
        expert_onnx,
        input_names=["noisy_actions", "timestep", "position_ids", "vlm_k", "vlm_v",
                     "prefix_offset", "kv_mask"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"}, "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
            "vlm_k": {1: "batch", 2: "seq"},
            "vlm_v": {1: "batch", 2: "seq"},
            "prefix_offset": {0: "batch"},
            "kv_mask": {0: "batch", 1: "seq"},
        },
        opset_version=config.opset,
    )
    optimize_onnx(expert_onnx)
    result["files"]["expert_onnx"] = str(expert_onnx)

    # 4. Validate ONNX (optional — same as legacy)
    if config.validate:
        logger.info("Validating ONNX export...")
        try:
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(str(expert_onnx))
            ort_out = sess.run(None, {
                "noisy_actions": dummy_actions.numpy(),
                "timestep": dummy_time.numpy(),
                "position_ids": dummy_pos.numpy().astype(np.int64),
                "vlm_k": dummy_vlm_k.numpy(),
                "vlm_v": dummy_vlm_v.numpy(),
                "prefix_offset": dummy_prefix_offset.numpy(),
                "kv_mask": dummy_kv_mask.numpy(),
            })[0]
            torch_out = expert_stack(
                dummy_actions, dummy_time, dummy_pos,
                dummy_vlm_k, dummy_vlm_v, dummy_prefix_offset, dummy_kv_mask
            ).detach().numpy()
            max_diff = float(np.abs(ort_out - torch_out).max())
            result["metadata"]["onnx_validation"] = {"max_diff": max_diff, "passed": max_diff < 0.01}
            logger.info("ONNX validation: max_diff=%.2e (%s)",
                        max_diff, "PASS" if max_diff < 0.01 else "FAIL")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")

    # 5. Build TRT engine if available — same as legacy
    if check_trtexec():
        expert_trt = output_dir / "expert_stack.trt"
        try:
            build_engine(expert_onnx, expert_trt, hardware)
            result["files"]["expert_trt"] = str(expert_trt)
        except RuntimeError as e:
            logger.warning("TRT build failed: %s", e)

    # 6. Save reflex_config.json — same schema as legacy
    export_config = {
        "model_id": config.model_id,
        "target": config.target,
        "precision": config.precision,
        "opset": config.opset,
        "num_denoising_steps": config.num_denoising_steps,
        "action_chunk_size": config.action_chunk_size,
        "action_dim": action_dim,
        "hardware": {
            "name": hardware.name,
            "memory_gb": hardware.memory_gb,
            "fp8": hardware.fp8_support,
            "precision": hardware.trt_precision,
        },
        "expert": expert_meta,
        "vlm_kv_input": True,
        "vlm_kv_dim": vlm_kv_dim,
        "spine_path": True,
    }
    config_path = output_dir / "reflex_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)

    logger.info("Export complete: %s", output_dir)
    return result


__all__ = ["export_smolvla"]
