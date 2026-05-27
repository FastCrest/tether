"""MolmoAct2VLA — Allen AI's MolmoAct2 on the BaseVLA spine.

Architecture: SigLIP2 ViT (vision) + Qwen3-based Molmo2-ER (LLM) +
flow-matching continuous action expert. Uses per-layer KV-cache
conditioning from VLM to action expert.

Spine mapping:
    MolmoAct2VLA = BaseVLA(
        vision_backbone = SigLIP2 ViT (model.vision_backbone.*),
        vlm_backbone    = Molmo2-ER transformer (model.transformer.*),
        vla_head        = Flow-matching action expert (model.action_expert.*),
        # Unused slots:
        llm_backbone = None,
        projector    = None,
        text_encoder = None,
    )

Source: huggingface.co/allenai/MolmoAct2 (Apache-2.0).
"""
from __future__ import annotations

from typing import Any, ClassVar

import torch

from reflex.models.base_vla import BaseVLA
from reflex.registry.components import VLAS


@VLAS.register
class MolmoAct2VLA(BaseVLA):
    """MolmoAct2 spine composition — SigLIP2 vision + Molmo2-ER VLM + flow-matching action expert."""

    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = (
        "vision_backbone",
        "vlm_backbone",
        "vla_head",
    )
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str = "allenai/MolmoAct2",
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "MolmoAct2VLA":
        """Build MolmoAct2VLA from a HuggingFace checkpoint.

        MolmoAct2 uses custom model code (auto_map in config.json), so we
        load via transformers' trust_remote_code path and wrap the components.
        """
        if state_dict is None:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id, trust_remote_code=True,
            )
            return cls(
                vision_backbone=model.model.vision_backbone,
                vlm_backbone=model.model.transformer,
                vla_head=model.model.action_expert if hasattr(model.model, "action_expert") else None,
            )

        raise NotImplementedError(
            "MolmoAct2VLA.from_pretrained with raw state_dict not yet supported. "
            "Use hf_id= to load via transformers."
        )

    # ── Phase B safetensors-direct loader ─────────────────────────────

    @classmethod
    def flat_dict_from_safetensors(
        cls,
        safetensors_path: str,
        *,
        dtype: torch.dtype | None = torch.bfloat16,
        device: str = "cuda",
        device_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Load MolmoAct2 safetensors checkpoint into a flat dict.

        Supports sharded checkpoints (directory with index.json).
        """
        from pathlib import Path

        from reflex.models.vlas._molmoact2_safetensors_mapping import (
            molmoact2_safetensors_to_flat,
        )

        path = Path(safetensors_path)
        if path.is_dir():
            from reflex.runtime.inference_weights.safetensors_direct import (
                load_flat_dict_from_safetensors_dir,
            )
            raw = load_flat_dict_from_safetensors_dir(
                path, dtype=dtype, device=device, device_id=device_id,
            )
        else:
            from safetensors import safe_open
            device_str = f"{device}:{device_id}" if device == "cuda" else device
            raw = {}
            with safe_open(str(path), framework="pt", device=device_str) as f:
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype=dtype)
                    raw[k] = tensor

        flat: dict[str, torch.Tensor] = {}
        for src_key, tensor in raw.items():
            target_key = molmoact2_safetensors_to_flat(src_key)
            if target_key is None:
                continue
            flat[target_key] = tensor

        return flat

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError(
            "MolmoAct2VLA.forward() requires the full MolmoAct2 inference pipeline. "
            "Use from_pretrained() + the model's generate/act methods instead."
        )

    def predict_action(self, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError(
            "MolmoAct2VLA.predict_action() requires the full MolmoAct2 inference pipeline. "
            "Use from_pretrained() + the model's generate/act methods instead."
        )


__all__ = ["MolmoAct2VLA"]
