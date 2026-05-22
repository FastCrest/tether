"""SigLIPBackbone — SigLIP vision tower wrapped as a spine VisionBackbone.

Per Romir's 2026-05-20 fork decision in the lift #1 Day 4 design:
**SigLIP is split out as a separate `vision_backbone` slot**, NOT folded into
PaliGemma's `llm_backbone`. Cleaner spine taxonomy; downstream lifts (#3
inference-only-weights, #5 fast-kernels) bind to a discrete component.

Loads via either:
- A HuggingFace model id (the standard `transformers.SiglipVisionModel.from_pretrained`)
- A pre-built `SiglipVisionModel` instance (for tests + Day 4c which extracts
  the vision tower from a PaliGemmaForConditionalGeneration via
  `paligemma_model.vision_tower`)

Output: per-image patch embeddings `[batch, num_patches, hidden_dim]`. For
pi0's PaliGemma-3B-pt-224 ([224×224 input, 14×14 patch] → 256 patches,
1152 hidden), this is `[B, 256, 1152]`.

Registered under `VISION_BACKBONES` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from reflex.models.vision import VisionBackbone
from reflex.registry.components import VISION_BACKBONES

if TYPE_CHECKING:
    pass


@VISION_BACKBONES.register
class SigLIPBackbone(VisionBackbone, nn.Module):
    """SigLIP vision tower wrapper.

    Args (exactly one of model_id / model required):
        model_id: HF repo id to load via `SiglipVisionModel.from_pretrained`
            (e.g. `"google/paligemma-3b-pt-224"` will extract the vision
            tower from PaliGemma; pure SigLIP repos like
            `"google/siglip-base-patch16-224"` work too).
        model: A pre-built `transformers.SiglipVisionModel` instance. Used by
            Day 4c to wrap an existing `paligemma.vision_tower`.
        output_hidden_states: Whether to return all layer hidden states
            (pi0 uses only `last_hidden_state`; defaults False).

    Raises:
        ValueError: if neither or both of `model_id` / `model` are provided.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model: Any = None,
        output_hidden_states: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if (model_id is None) == (model is None):
            raise ValueError(
                "Provide exactly one of `model_id` or `model` "
                "(got model_id=%r, model=%r)." % (model_id, model)
            )

        if model is not None:
            # Pre-built instance — typically extracted from a PaliGemma model
            # via paligemma.vision_tower (see Day 4c).
            self.model = model
        else:
            # Lazy import — transformers is an existing reflex dep but the
            # SiglipVisionModel auto-loader hits HuggingFace, which isn't
            # needed for tests that pass a stub model.
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(
                model_id,
                output_hidden_states=output_hidden_states,
            )

        self.output_hidden_states = output_hidden_states

    def forward(
        self,
        images: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Images → patch embeddings.

        Args:
            images: `[batch, channels, height, width]` pixel tensor.
                Standard SigLIP input — float32 [-1, 1] range after the
                upstream image processor's normalization.

        Returns:
            `last_hidden_state` of shape `[batch, num_patches, hidden_dim]`.
            For pi0's PaliGemma-3B (224×224 input, 14×14 patches): `[B, 256, 1152]`.
        """
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for `--inference-only-weights` mode (lift #3).

        Returns every parameter under the given prefix. For a SigLIP-Base-Patch16-224
        wrapped here, that's ~100M params (~400 MB FP32 / ~200 MB BF16).
        """
        return {
            f"{prefix}{name}": param.detach().clone()
            for name, param in self.named_parameters()
        }


__all__ = ["SigLIPBackbone"]
