"""DreamZeroVLA — DreamZero World-Action Model on the BaseVLA spine.

Lift #7 per ``features/03_export/dreamzero-exporter.md``. Fills slots:

    DreamZeroVLA = BaseVLA(
        vlm_backbone = WanBackbone (T5 + CLIP + VAE — the 6th slot)
        vla_head     = DreamZeroHead (DiT diffusion + flow matching)
        vision_backbone = None (Wan uses integrated video encoding)
        llm_backbone    = None (no autoregressive LLM)
        projector       = None (encoding handled by vlm_backbone)
        text_encoder    = None (T5 is inside vlm_backbone)
    )

This is the 6th model family on the spine (after pi0, pi0.5, SmolVLA,
GR00T, OpenVLA). Validates the ``vlm_backbone`` slot for a non-LLM
architecture.

Ported from FluxVLA ``dreamzero_vla.py`` (Apache-2.0, LimX Dynamics).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from reflex.models.base_vla import BaseVLA
from reflex.registry.components import VLAS

if TYPE_CHECKING:
    pass


@VLAS.register
class DreamZeroVLA(BaseVLA):
    """DreamZero World-Action Model on the spine.

    Uses ``vlm_backbone`` (WanBackbone) for encoding and ``vla_head``
    (DreamZeroHead) for the DiT diffusion + flow-matching action head.

    Slots:
    - vlm_backbone: WanBackbone (REQUIRED) — T5 + CLIP + VAE
    - vla_head: DreamZeroHead (REQUIRED) — DiT + flow matching
    - vision_backbone: None (integrated in vlm_backbone)
    - llm_backbone: None (no autoregressive LLM)
    - projector: None
    - text_encoder: None (T5 is inside vlm_backbone)
    """

    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = (
        "vlm_backbone",
        "vla_head",
    )
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        *,
        vlm_backbone: Any = None,
        vla_head: Any = None,
        num_views: int = 2,
        frame_window_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vlm_backbone=vlm_backbone,
            vla_head=vla_head,
            **kwargs,
        )
        self.num_views = num_views
        self.frame_window_size = frame_window_size

    def _prepare_states(self, states: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.shape[1] < num_tokens:
            repeats = (num_tokens + states.shape[1] - 1) // states.shape[1]
            states = states.repeat(1, repeats, 1)[:, :num_tokens]
        return states

    def forward(self, batch: dict[str, Any]) -> Any:
        """Training forward — passes encoded video + text to the DiT head."""
        images = batch["images"]
        lang_tokens = batch["lang_tokens"]
        lang_masks = batch["lang_masks"]
        states = batch["states"]
        actions = batch["actions"]
        action_masks = batch.get("action_masks")
        embodiment_ids = batch.get("embodiment_ids")

        vlm_out = self.vlm_backbone(
            video=images,
            input_ids=lang_tokens.long(),
            attention_mask=lang_masks.long(),
        )

        return self.vla_head(
            prompt_embs=vlm_out["prompt_embs"],
            latents=vlm_out["latents"],
            clip_feas=vlm_out["clip_feas"],
            ys=vlm_out["image_cond"],
            states=states,
            actions=actions,
            action_masks=action_masks,
            embodiment_ids=embodiment_ids,
        )

    def predict_action(
        self,
        *,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Inference — encode + denoise to produce action chunk."""
        device = images.device

        if embodiment_ids is None:
            embodiment_ids = torch.zeros(images.shape[0], dtype=torch.long, device=device)

        # Pad video to frame_window_size
        b, c, t_obs, h, w = images.shape
        if t_obs < self.frame_window_size:
            pad = images.new_zeros(b, c, self.frame_window_size - t_obs, h, w)
            images = torch.cat([images, pad], dim=2)

        vlm_out = self.vlm_backbone(
            video=images,
            input_ids=lang_tokens.long().to(device),
            attention_mask=lang_masks.long().to(device),
        )

        # Prepare states
        t_video = images.shape[2]
        latent_frames = 1 + (t_video - 1) // 4
        num_blocks = max(1, (latent_frames - 1) // self.vla_head.num_frame_per_block)
        num_state_tokens = num_blocks * self.vla_head.num_state_per_block
        states = self._prepare_states(states, num_state_tokens)

        max_state_dim = self.vla_head.max_state_dim
        if states.shape[-1] < max_state_dim:
            states = torch.nn.functional.pad(states, (0, max_state_dim - states.shape[-1]))

        latents = vlm_out["latents"].transpose(1, 2)

        return self.vla_head.predict_action(
            prompt_embs=vlm_out["prompt_embs"],
            latents=latents,
            clip_feas=vlm_out["clip_feas"],
            ys=vlm_out["image_cond"],
            states=states,
            embodiment_ids=embodiment_ids,
        )


__all__ = ["DreamZeroVLA"]
