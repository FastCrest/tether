"""GR00T export pipeline — refactored composition via the BaseVLA spine.

Mirrors the legacy ``gr00t_exporter.{export_gr00t, export_gr00t_full}`` but
builds via the ``GR00TVLA(BaseVLA)`` composition class instead of directly
assembling the DiT stack. Same checkpoint → same ONNX bytes (bit-identical
numerics guaranteed by reusing ``build_gr00t_expert_stack`` /
``build_gr00t_full_stack`` under the hood).

The legacy module ``reflex.exporters.gr00t_exporter`` stays available for
backward compatibility per the lift #1 plan (Day 11 sunsets it together
with ``pi0_exporter`` / ``smolvla_exporter`` / ``decomposed`` after all
callers migrate).

Per the user's 2026-05-22 Day 7 scope choice (full bundle), this file
lands the export refactor in the same PR as the spine composition.
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


def export_gr00t(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Spine-based GR00T expert-only export (zero VLM-KV placeholder).

    Output is the [b, chunk, output_dim] action-token velocity — consumers
    need to run the per-embodiment action_decoder downstream to recover
    actions in the native DoF space (same as the legacy export_gr00t).
    """
    from reflex.models.vlas.gr00t import GR00TVLA

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = get_hardware_profile(config.target)
    result: dict[str, Any] = {"status": "ok", "files": {}, "metadata": {}}

    if state_dict is None:
        logger.info("Loading GR00T checkpoint: %s", config.model_id)
        state_dict, _ = load_checkpoint(config.model_id)
    total_params = sum(v.numel() for v in state_dict.values())
    logger.info("Loaded %d tensors, %.1fM params", len(state_dict), total_params / 1e6)

    logger.info("Building GR00T via the BaseVLA spine...")
    vla = GR00TVLA.from_pretrained(state_dict=state_dict, embodiment_id=0)
    # Expert-only export uses just the DiT stack (no encoders/decoders).
    dit_stack = vla.vla_head.dit_stack
    meta = vla.vla_head.metadata
    result["metadata"]["expert"] = meta

    chunk_size = meta["chunk_size"]
    hidden = meta["hidden"]
    dummy_action_tokens = torch.randn(1, chunk_size, hidden)
    dummy_time = torch.tensor([0.5])
    dummy_pos = torch.arange(chunk_size).unsqueeze(0)

    expert_onnx = output_dir / "expert_stack.onnx"
    logger.info("Exporting DiT expert stack to ONNX: %s", expert_onnx)
    export_module_to_onnx(
        dit_stack,
        (dummy_action_tokens, dummy_time, dummy_pos),
        expert_onnx,
        input_names=["noisy_actions", "timestep", "position_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
        },
        opset_version=config.opset,
    )
    optimize_onnx(expert_onnx)
    result["files"]["expert_onnx"] = str(expert_onnx)

    if config.validate:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(expert_onnx))
            ort_out = sess.run(None, {
                "noisy_actions": dummy_action_tokens.numpy(),
                "timestep": dummy_time.numpy(),
                "position_ids": dummy_pos.numpy().astype(np.int64),
            })[0]
            torch_out = dit_stack(dummy_action_tokens, dummy_time, dummy_pos).detach().numpy()
            max_diff = float(np.abs(ort_out - torch_out).max())
            result["metadata"]["onnx_validation"] = {"max_diff": max_diff, "passed": max_diff < 0.01}
            logger.info("ONNX validation: max_diff=%.2e (%s)",
                        max_diff, "PASS" if max_diff < 0.01 else "FAIL")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")

    if check_trtexec():
        expert_trt = output_dir / "expert_stack.trt"
        try:
            build_engine(expert_onnx, expert_trt, hardware)
            result["files"]["expert_trt"] = str(expert_trt)
        except RuntimeError as e:
            logger.warning("TRT build failed: %s", e)

    meta_with_action_dim = dict(meta)
    meta_with_action_dim["action_dim"] = hidden
    export_config = {
        "model_id": config.model_id,
        "model_type": "gr00t",
        "target": config.target,
        "precision": config.precision,
        "opset": config.opset,
        "num_denoising_steps": 4,
        "action_chunk_size": chunk_size,
        "action_dim": hidden,
        "hidden": hidden,
        "output_dim": meta["output_dim"],
        "note": "expert accepts action tokens (hidden-dim), emits velocity tokens (output_dim). "
                "action_decoder (per-embodiment) needed downstream to recover native actions.",
        "hardware": {
            "name": hardware.name,
            "memory_gb": hardware.memory_gb,
            "fp8": hardware.fp8_support,
            "precision": hardware.trt_precision,
        },
        "expert": meta_with_action_dim,
        "spine_path": True,
    }
    config_path = output_dir / "reflex_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)
    return result


def export_gr00t_full(
    config: ExportConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
    embodiment_id: int = 0,
) -> dict[str, Any]:
    """Spine-based GR00T full-stack export — raw actions in, raw actions out.

    Includes per-embodiment action_encoder + DiT + action_decoder. Pinned
    to one embodiment_id (default 0).
    """
    from reflex.models.vlas.gr00t import GR00TVLA

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = get_hardware_profile(config.target)
    result: dict[str, Any] = {"status": "ok", "files": {}, "metadata": {}}

    if state_dict is None:
        logger.info("Loading GR00T checkpoint: %s", config.model_id)
        state_dict, _ = load_checkpoint(config.model_id)

    logger.info("Building GR00T full stack via the spine (embodiment=%d)...", embodiment_id)
    vla = GR00TVLA.from_pretrained(state_dict=state_dict, embodiment_id=embodiment_id)
    full = vla.vla_head.full_stack
    meta = vla.vla_head.metadata
    result["metadata"]["expert"] = meta

    chunk_size = 50
    raw_action_dim = meta["raw_action_dim"]
    dummy_actions = torch.randn(1, chunk_size, raw_action_dim)
    dummy_time = torch.tensor([0.5])
    dummy_pos = torch.arange(chunk_size).unsqueeze(0)

    expert_onnx = output_dir / "expert_stack.onnx"
    logger.info("Exporting full stack to ONNX: %s", expert_onnx)
    export_module_to_onnx(
        full,
        (dummy_actions, dummy_time, dummy_pos),
        expert_onnx,
        input_names=["noisy_actions", "timestep", "position_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
        },
        opset_version=config.opset,
    )
    optimize_onnx(expert_onnx)
    result["files"]["expert_onnx"] = str(expert_onnx)

    if config.validate:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(expert_onnx))
            ort_out = sess.run(None, {
                "noisy_actions": dummy_actions.numpy(),
                "timestep": dummy_time.numpy(),
                "position_ids": dummy_pos.numpy().astype(np.int64),
            })[0]
            torch_out = full(dummy_actions, dummy_time, dummy_pos).detach().numpy()
            max_diff = float(np.abs(ort_out - torch_out).max())
            result["metadata"]["onnx_validation"] = {"max_diff": max_diff, "passed": max_diff < 0.01}
            logger.info("ONNX validation: max_diff=%.2e (%s)",
                        max_diff, "PASS" if max_diff < 0.01 else "FAIL")
        except ImportError:
            logger.warning("onnxruntime not installed, skipping validation")

    if check_trtexec():
        expert_trt = output_dir / "expert_stack.trt"
        try:
            build_engine(expert_onnx, expert_trt, hardware)
            result["files"]["expert_trt"] = str(expert_trt)
        except RuntimeError as e:
            logger.warning("TRT build failed: %s", e)

    meta_with_action_dim = dict(meta)
    meta_with_action_dim["action_dim"] = raw_action_dim
    export_config = {
        "model_id": config.model_id,
        "model_type": "gr00t",
        "full_stack": True,
        "embodiment_id": embodiment_id,
        "target": config.target,
        "precision": config.precision,
        "opset": config.opset,
        "num_denoising_steps": 4,
        "action_chunk_size": chunk_size,
        "action_dim": raw_action_dim,
        "hidden": meta["hidden"],
        "output_dim": raw_action_dim,
        "hardware": {
            "name": hardware.name,
            "memory_gb": hardware.memory_gb,
            "fp8": hardware.fp8_support,
            "precision": hardware.trt_precision,
        },
        "expert": meta_with_action_dim,
        "spine_path": True,
    }
    config_path = output_dir / "reflex_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)
    return result


__all__ = ["export_gr00t", "export_gr00t_full"]
