"""LIBERO eval adapter for Pi05FastKernelsInference (Lift #5 L3 gate).

Bridges the lerobot PI05Policy's preprocessor pipeline with the Triton
runtime's predict_action interface. This lets the existing LIBERO eval
loop call the Triton path with the SAME observation preprocessing as
the native lerobot path — isolating the Triton-vs-PyTorch action quality
difference from any preprocessing drift.

Usage in the LIBERO eval script::

    from reflex.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter

    # Build from the same lerobot policy used for the baseline ARM
    adapter = TritonLIBEROAdapter.from_policy(policy)

    # In the eval loop, replace policy.select_action(batch_pp) with:
    chunk = adapter.predict_chunk(batch_pp)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass


class TritonLIBEROAdapter:
    """Wraps Pi05FastKernelsInference for use in LIBERO eval loops.

    Handles the format bridging:
    - Extracts images + masks via ``policy._preprocess_images(batch_pp)``
    - Extracts lang_tokens + lang_masks from the batch
    - Concatenates multi-view images into ``[B, num_views*3, H, W]``
    - Calls ``Pi05FastKernelsInference.predict_action``
    - Returns the raw action chunk (caller handles denormalization via
      the same postprocessor as the native path)
    """

    def __init__(
        self,
        triton_runtime: Any,
        policy: Any,
        *,
        chunk_size: int = 50,
        max_action_dim: int = 32,
    ) -> None:
        self._runtime = triton_runtime
        self._policy = policy
        self._chunk_size = chunk_size
        self._max_action_dim = max_action_dim

    @classmethod
    def from_policy(
        cls,
        policy: Any,
        *,
        capture: bool = True,
    ) -> "TritonLIBEROAdapter":
        """Build the Triton runtime from a loaded PI05Policy.

        Shares the same checkpoint weights (via from_lerobot_policy).
        """
        from reflex.models.vlas.pi05 import Pi05VLA
        from reflex.runtime.fast_inference.pi05 import Pi05FastKernelsInference

        vla = Pi05VLA.from_lerobot_policy(policy)
        vla.vision_backbone.to("cuda")
        vla.llm_backbone.to("cuda")
        vla.vla_head.to("cuda")

        # Detect num_views from the policy config's image keys.
        cfg = policy.config
        image_keys = [k for k in cfg.input_features if "image" in k.lower()]
        num_views = max(len(image_keys), 2)  # at least 2; pi0.5 typically 3

        runtime = Pi05FastKernelsInference(
            vla, capture=capture, num_views=num_views,
            triton_max_prompt_len=128,  # LIBERO task descriptions tokenize to 50-60 tokens; 48 is too small
        )
        runtime.prepare_triton_inference()

        return cls(
            triton_runtime=runtime,
            policy=policy,
            chunk_size=getattr(cfg, "chunk_size", 50),
            max_action_dim=getattr(cfg, "max_action_dim", 32),
        )

    def reset(self) -> None:
        """Match lerobot policy.reset() interface for the eval loop."""
        pass

    def predict_chunk(self, batch_pp: dict[str, Any]) -> torch.Tensor:
        """Predict an action chunk from a preprocessed LIBERO batch.

        Args:
            batch_pp: The output of ``preprocessor(batch)`` — same dict
                that ``policy.select_action`` receives. Must contain the
                image + language keys that ``policy._preprocess_images``
                expects.

        Returns:
            ``[B, chunk_size, action_dim]`` raw action chunk (NOT
            denormalized — caller applies postprocessor, matching the
            native ARM's denormalization path).
        """
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        # Extract images + masks via the policy's own preprocessor.
        # Policy may be on CPU (to avoid OOM on A100-40GB when both the
        # reference policy + Triton VLA are loaded); move batch tensors
        # to the policy's device for preprocessing, then back to CUDA.
        policy_device = next(self._policy.parameters()).device
        batch_for_pp = {
            k: (v.to(policy_device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch_pp.items()
        }
        images, img_masks = self._policy._preprocess_images(batch_for_pp)
        # Move images back to CUDA for the Triton runtime
        images = [img.to("cuda") for img in images]
        lang_tokens = batch_pp[OBS_LANGUAGE_TOKENS]
        lang_masks = batch_pp[OBS_LANGUAGE_ATTENTION_MASK]

        # Concat multi-view images: list of [B, C, H, W] → [B, N*C, H, W]
        images_concat = torch.cat(images, dim=1)

        # Generate noise (deterministic per call if needed — caller seeds)
        bsize = images[0].shape[0]
        noise = torch.randn(
            bsize, self._chunk_size, self._max_action_dim,
            device=images[0].device, dtype=torch.float32,
        )

        # States: Pi0.5 uses state-in-language (no explicit state input);
        # pass zeros for API compatibility
        states = torch.zeros(bsize, 32, device=images[0].device, dtype=torch.float32)

        # Call the Triton runtime
        with torch.no_grad():
            actions = self._runtime.predict_action(
                images=images_concat,
                lang_tokens=lang_tokens,
                states=states,
                lang_masks=lang_masks,
                noise=noise,
            )

        # actions: [B, chunk_size, max_action_dim]
        # Trim to actual action_dim (matches native path)
        cfg = self._policy.config
        orig_dim = cfg.output_features.get("action", None)
        if orig_dim is not None and hasattr(orig_dim, "shape"):
            actions = actions[:, :, : orig_dim.shape[0]]

        return actions


__all__ = ["TritonLIBEROAdapter"]
