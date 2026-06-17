"""SmolVLA export pipeline — refactored composition via the BaseVLA spine.

Mirrors the legacy ``smolvla_exporter.export_smolvla`` shape (output_dir
layout, file names, ``tether_config.json``), but builds via the
``SmolVLA(BaseVLA)`` composition class instead of directly assembling the
``ExpertStack``. Same checkpoint → same ONNX bytes (bit-identical numerics
guaranteed by reusing ``build_expert_stack`` under the hood).

The legacy module ``tether.exporters.smolvla_exporter`` stays available for
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

from tether.checkpoint import load_checkpoint
from tether.config import ExportConfig, get_hardware_profile
from tether.exporters.onnx_export import export_module_to_onnx, optimize_onnx
from tether.exporters.trt_build import build_engine, check_trtexec

# Re-exports — moved here from the deleted ``smolvla_exporter.py`` at Day 11
# sunset. External callers still doing ``from tether.exporters.smolvla import
# ExpertGQALayer`` keep working through this surface.
from tether.models.heads.expert_stack import (  # noqa: F401
    ExpertGQALayer, ExpertStack, _DecomposedRoPE, _sinusoidal_pos_embedding,
)

logger = logging.getLogger(__name__)


def build_expert_stack(
    state_dict: dict[str, torch.Tensor], head_dim: int
) -> tuple[ExpertStack, dict]:
    """Build the full expert stack from SmolVLA state_dict.

    Moved here from ``smolvla_exporter.py`` at Day 11 sunset (PR #...).
    Behavior identical — same builder ``SmolVLA.from_pretrained``,
    ``Pi05VLA.from_pretrained``, etc all depend on.

    Note: ``head_dim`` is the **VLM's** head_dim (e.g. 64 for SmolLM2). The
    expert's head_dim is DIFFERENT (expert_hidden / num_heads, typically 48
    for SmolVLA's 0.75× width multiplier). We recover it from the q_proj shape.
    """
    expert_hidden = state_dict["model.action_in_proj.weight"].shape[0]
    action_dim = state_dict["model.action_in_proj.weight"].shape[1]

    all_expert_keys = [k for k in state_dict.keys() if "lm_expert" in k]
    base_prefix = all_expert_keys[0][: all_expert_keys[0].index("layers.")]

    layer_indices = set()
    for k in all_expert_keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_indices.add(int(parts[i + 1]))
    num_layers = max(layer_indices) + 1

    q_shape = state_dict[f"{base_prefix}layers.0.self_attn.q_proj.weight"].shape
    k_shape = state_dict[f"{base_prefix}layers.0.self_attn.k_proj.weight"].shape
    gate_shape = state_dict[f"{base_prefix}layers.0.mlp.gate_proj.weight"].shape

    try:
        from transformers import AutoConfig
        vlm_cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        nq = int(vlm_cfg.text_config.num_attention_heads)
        nkv = int(vlm_cfg.text_config.num_key_value_heads)
    except Exception:
        nq = q_shape[0] // head_dim
        nkv = k_shape[0] // head_dim

    expert_head_dim = q_shape[0] // nq
    inter = gate_shape[0]
    logger.info(
        "[expert-stack] expert_hidden=%d, num_q_heads=%d, num_kv_heads=%d, "
        "expert_head_dim=%d (vlm head_dim=%d), intermediate=%d",
        expert_hidden, nq, nkv, expert_head_dim, head_dim, inter,
    )
    head_dim = expert_head_dim

    layers = []
    cross_indices = []
    vlm_kv_dim = 0
    for i in range(num_layers):
        prefix = f"{base_prefix}layers.{i}"
        kv_in = state_dict[f"{prefix}.self_attn.k_proj.weight"].shape[1]
        is_cross = kv_in != expert_hidden
        if is_cross:
            cross_indices.append(i)
            vlm_kv_dim = kv_in

        layer = ExpertGQALayer(
            expert_hidden, nq, nkv, head_dim, inter,
            kv_in=kv_in if is_cross else None,
        )
        layer_sd = {
            "input_layernorm.weight": state_dict[f"{prefix}.input_layernorm.weight"],
            "post_attention_layernorm.weight": state_dict[f"{prefix}.post_attention_layernorm.weight"],
            "q_proj.weight": state_dict[f"{prefix}.self_attn.q_proj.weight"],
            "k_proj.weight": state_dict[f"{prefix}.self_attn.k_proj.weight"],
            "v_proj.weight": state_dict[f"{prefix}.self_attn.v_proj.weight"],
            "o_proj.weight": state_dict[f"{prefix}.self_attn.o_proj.weight"],
            "gate_proj.weight": state_dict[f"{prefix}.mlp.gate_proj.weight"],
            "up_proj.weight": state_dict[f"{prefix}.mlp.up_proj.weight"],
            "down_proj.weight": state_dict[f"{prefix}.mlp.down_proj.weight"],
        }
        layer.load_state_dict(layer_sd, strict=False)
        layers.append(layer)

    final_norm_w = torch.ones(expert_hidden)
    for candidate in [
        f"{base_prefix}norm.weight",
        "model.vlm_with_expert.lm_expert.norm.weight",
    ]:
        if candidate in state_dict:
            final_norm_w = state_dict[candidate]
            break

    stack = ExpertStack(
        layers=layers,
        expert_hidden=expert_hidden,
        action_dim=action_dim,
        cross_indices=cross_indices,
        vlm_kv_dim=vlm_kv_dim,
        suffix_weights={
            "in_w": state_dict["model.action_in_proj.weight"],
            "in_b": state_dict["model.action_in_proj.bias"],
            "t_in_w": state_dict["model.action_time_mlp_in.weight"],
            "t_in_b": state_dict["model.action_time_mlp_in.bias"],
            "t_out_w": state_dict["model.action_time_mlp_out.weight"],
            "t_out_b": state_dict["model.action_time_mlp_out.bias"],
        },
        action_proj_weights={
            "w": state_dict["model.action_out_proj.weight"],
            "b": state_dict["model.action_out_proj.bias"],
        },
        final_norm_weight=final_norm_w,
    )
    stack.eval()

    metadata = {
        "expert_hidden": expert_hidden,
        "action_dim": action_dim,
        "num_layers": num_layers,
        "n_q_heads": nq,
        "n_kv_heads": nkv,
        "head_dim": head_dim,
        "intermediate": inter,
        "cross_attn_layers": cross_indices,
        "vlm_kv_dim": vlm_kv_dim,
        "total_params_m": sum(p.numel() for p in stack.parameters()) / 1e6,
    }
    return stack, metadata


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
    from tether.models.vlas.smolvla import SmolVLA

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

    # 6. Save tether_config.json — same schema as legacy
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
    config_path = output_dir / "tether_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))
    result["files"]["config"] = str(config_path)

    logger.info("Export complete: %s", output_dir)
    return result


__all__ = [
    "export_smolvla",
    "build_expert_stack",
    # Re-exports moved here at Day 11 sunset:
    "ExpertGQALayer",
    "ExpertStack",
]
