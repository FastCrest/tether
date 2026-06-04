"""SmolVLA — SmolVLM2 + cross-attention action expert on the BaseVLA spine.

Per lerobot ``policies/smolvla/`` (``modeling_smolvla.py``,
``smolvlm_with_expert.py``), SmolVLA is::

    SmolVLA = BaseVLA(
        vision_backbone = SmolVLM2 vision tower (SigLIP-like, 512×512 input),
        llm_backbone    = SmolVLM2 connector + text_model (SmolLM2 16-layer),
        projector       = state_proj (Linear 32 → vlm_hidden) — state IS a
                          separate input embedding, NOT in language (unlike
                          pi0.5's knowledge insulation).
        vla_head        = FlowMatchingHead wrapping the action expert. The
                          expert uses **cross-attention** to per-layer VLM KV
                          at specific layer indices, NOT prefix-concat
                          (unlike pi0.5).
        vlm_backbone    = not used (vision + text are separate, not fused),
        text_encoder    = not used (decoder-only LLM).
    )

The expert's cross-attn layer pattern (``cross_indices``) is recovered at
build time from the checkpoint's k_proj input dim: layers whose k_proj
``in_features != expert_hidden`` are cross-attending to the VLM (see
``smolvla_exporter.build_expert_stack`` for the recovery logic).

NAME_MAPPING: empty per decision S-1 — SmolVLA checkpoint keys load directly
into the spine components (no rename needed).
"""
from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from tether.models.base_vla import BaseVLA
from tether.registry.components import VLAS


@VLAS.register
class SmolVLA(BaseVLA):
    """SmolVLA spine composition — SmolVLM2 + state_proj + cross-attn expert."""

    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = (
        "vision_backbone",
        "llm_backbone",
        "projector",
        "vla_head",
    )
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    # ── Construction helpers ────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        dtype: torch.dtype | None = None,
    ) -> "SmolVLA":
        """Build SmolVLA from a HuggingFace SmolVLM2 + raw SmolVLA state_dict.

        The vision tower + text_model + connector all come from a single
        ``AutoModelForImageTextToText.from_pretrained`` call so the SigLIP
        and SmolLM2 weights stay aligned with their parent VLM. The action
        expert is built from the SmolVLA checkpoint state_dict via
        ``build_expert_stack`` from ``smolvla_exporter`` (the canonical
        recovery path for cross-attn layer indices + GQA shapes).

        Args:
            hf_id: HuggingFace repo for SmolVLM2 (the VLM backbone).
            state_dict: SmolVLA action-expert weights. If None, SmolVLA
                composition is built without the expert head (useful for
                vision/llm-only tests).
            max_state_dim: input dim of the state vector (32 per SmolVLA
                default config).
            max_action_dim: padded action dim for the expert output.
            dtype: optional cast for the loaded VLM.

        Returns:
            SmolVLA instance ready for forward()/predict_action() (Phase B).
        """
        from transformers import AutoModelForImageTextToText

        from tether.models.heads.flow_matching_head import FlowMatchingHead
        from tether.models.llm.smolvla_llm_backbone import SmolVLALLMBackbone
        from tether.models.projectors.linear_projector import LinearProjector
        from tether.models.vision.smolvla_vision_backbone import SmolVLAVisionBackbone

        # 1. Load SmolVLM2 once; split between vision and llm slots.
        vlm = AutoModelForImageTextToText.from_pretrained(hf_id)
        if dtype is not None:
            vlm = vlm.to(dtype=dtype)

        vision = SmolVLAVisionBackbone(model=vlm.model.vision_model)
        llm = SmolVLALLMBackbone(model=vlm)

        # 2. State projector: 32-dim state → text-hidden (the same hidden
        # the expert reads, per lerobot VLAFlowMatching.__init__:574-576).
        state_proj = LinearProjector(
            in_dim=max_state_dim,
            out_dim=llm.text_hidden_size,
        )

        # 3. Action expert (cross-attn) from the SmolVLA state_dict.
        if state_dict is not None:
            from tether.exporters.smolvla import build_expert_stack
            stack, _meta = build_expert_stack(
                state_dict=state_dict,
                head_dim=llm.text_head_dim,
            )
            head = FlowMatchingHead(expert_stack=stack)
            # Load state_proj weights from the checkpoint if present.
            sp_w = state_dict.get("model.state_proj.weight")
            sp_b = state_dict.get("model.state_proj.bias")
            if sp_w is not None:
                state_proj.linear.weight = nn.Parameter(sp_w)
            if sp_b is not None:
                state_proj.linear.bias = nn.Parameter(sp_b)
        else:
            # No state_dict → leave head as a stub. Caller is responsible
            # for binding a real expert before predict_action.
            head = None

        if head is None:
            # All REQUIRED_SLOTS must be populated. Raise here with a
            # clearer message than the spine's generic "missing slot".
            raise ValueError(
                "SmolVLA.from_pretrained requires state_dict to build the "
                "action expert. For vision/llm-only tests, instantiate "
                "SmolVLA(...) directly with stub components."
            )

        return cls(
            vision_backbone=vision,
            llm_backbone=llm,
            projector=state_proj,
            vla_head=head,
        )

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Any]) -> Any:
        """Minimum-viable forward — language path only on already-merged
        ``inputs_embeds``. Predict_action is a separate entry point.

        Args:
            batch: dict with keys:
                - "inputs_embeds": pre-merged image+text embeddings
                - "attention_mask": optional [batch, seq]
                - "past_key_values": optional
        """
        return self.llm_backbone(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch.get("attention_mask"),
            past_key_values=batch.get("past_key_values"),
        )

    def predict_action(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Full inference — TODO Phase B. The runtime path is currently
        served by ``tether.runtime.smolvla_native`` / ``smolvla_onnx_server``;
        Phase B will move it onto the spine here for parity with Pi0VLA/Pi05VLA.
        """
        raise NotImplementedError(
            "SmolVLA.predict_action is a Phase B follow-up; today the runtime "
            "path is at tether.runtime.smolvla_native / smolvla_onnx_server."
        )


__all__ = ["SmolVLA"]
