"""Pi05VLA — pi0.5 composition class on the BaseVLA spine.

Lift #1 Day 5 per `features/03_export/basevla-spine_plan.md`. Mirrors Day 4's
Pi0VLA but for pi0.5. The flow-matching head + most components are shared;
pi0.5's divergence from pi0:

- **AdaRMSNorm time conditioning** (vs pi0's plain RMSNorm). Time embedding
  becomes per-layer norm conditioning, NOT a suffix token.
- **State-in-language** — no state_proj projector slot used. State info is
  encoded by lerobot's tokenizer into the language prompt itself.
- **Suffix is action-only** — no state token prepended; suffix = action_emb
  for `chunk_size` tokens (vs pi0's `state + action_emb` for chunk_size+1).

Wires:

    Pi05VLA = BaseVLA(
        vision_backbone = SigLIPBackbone (extracted from paligemma.vision_tower)
        llm_backbone    = PaliGemmaBackbone (PaliGemma minus vision_tower)
        vla_head        = FlowMatchingHead (wraps Pi05ExpertStackWithPrefix)
    )

Registered under VLAS per decision S-3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from reflex.models.base_vla import BaseVLA
from reflex.registry.components import VLAS

if TYPE_CHECKING:
    pass


@VLAS.register
class Pi05VLA(BaseVLA):
    """pi0.5 spine composition — PaliGemma (vision split out) + AdaRMSNorm expert.

    Slots:

    - vision_backbone: SigLIPBackbone (REQUIRED) — extracted from PaliGemma
    - llm_backbone:    PaliGemmaBackbone (REQUIRED) — PaliGemma minus vision_tower
    - vla_head:        FlowMatchingHead (REQUIRED) — wraps Pi05ExpertStack
                       (the AdaRMSNorm-conditioned variant)
    - projector:       not used (None) — state encoded in language tokens
    - vlm_backbone:    not used (None)
    - text_encoder:    not used (None)

    NAME_MAPPING: empty per decision S-1 — pi0.5 checkpoint keys map directly
    to component slots via the spine's default routing.
    """

    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = (
        "vision_backbone",
        "llm_backbone",
        "vla_head",
    )
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    # ── Construction helpers ────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str = "lerobot/pi05_libero_finetuned_v044",
        *,
        dtype: torch.dtype | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "Pi05VLA":
        """Build Pi05VLA from a HuggingFace pi0.5 checkpoint.

        IMPORTANT: like Pi0VLA.from_pretrained, this naive path is broken for
        lerobot pi0.5 checkpoints because they nest PaliGemma weights under
        `paligemma_with_expert.paligemma.*` — stock PaliGemma's loader sets
        all weights to random init. Use the parity-script pattern (build from
        a loaded lerobot policy) until a proper key-remap loader is shipped.

        Args:
            hf_id: HuggingFace repo (default lerobot/pi05_libero_finetuned_v044)
            dtype: cast loaded model to this dtype (e.g. torch.bfloat16)
            state_dict: pre-loaded raw state_dict for the expert build.

        Returns:
            Pi05VLA instance ready for forward() + (Phase B) predict_action().
        """
        from transformers import PaliGemmaForConditionalGeneration

        from reflex.models.heads.flow_matching_head import FlowMatchingHead
        from reflex.models.llm.paligemma_backbone import PaliGemmaBackbone
        from reflex.models.vision.siglip_backbone import SigLIPBackbone

        # 1. Load PaliGemma — full model, cast if requested.
        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(hf_id)
        if dtype is not None:
            paligemma = paligemma.to(dtype=dtype)

        # 2. Vision: extract vision_tower
        vision = SigLIPBackbone(model=paligemma.model.vision_tower)

        # 3. Language: wrap the rest of PaliGemma
        llm = PaliGemmaBackbone(model=paligemma)

        # 4. State_dict for expert build
        if state_dict is None:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            try:
                safetensors_path = hf_hub_download(
                    repo_id=hf_id, filename="model.safetensors",
                )
                state_dict = load_file(safetensors_path)
            except Exception:
                state_dict = {}

        # 5. Head: build the prefix-aware pi0.5 expert (with AdaRMSNorm).
        from reflex.exporters.pi0_prefix_exporter import build_pi05_expert_with_prefix
        expert_with_prefix, _meta = build_pi05_expert_with_prefix(state_dict)
        head = FlowMatchingHead(expert_stack=expert_with_prefix)

        return cls(
            vision_backbone=vision,
            llm_backbone=llm,
            vla_head=head,
        )

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Any]) -> Any:
        """Minimum-viable forward — runs the language path on
        already-merged inputs_embeds. Phase B will add the full pipeline.

        Args:
            batch: dict with keys:
                - "inputs_embeds": pre-merged image+text embeddings
                - "attention_mask": [batch, seq]
                - "past_key_values": optional
        """
        return self.llm_backbone(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch.get("attention_mask"),
            past_key_values=batch.get("past_key_values"),
        )

    def predict_action(
        self,
        *,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor] | None = None,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        chunk_size: int = 50,
    ) -> torch.Tensor:
        """Full pi0.5 inference. NOT YET IMPLEMENTED — Day 5 Phase B.

        Mirrors Pi0VLA.predict_action but with pi0.5 specifics:

        1. SigLIPBackbone encodes each camera image
        2. PaliGemmaBackbone.multi_modal_projector projects vision → text_hidden
        3. PaliGemmaBackbone.embed_tokens embeds language tokens (state IS
           encoded in lang_tokens via lerobot's processor — see knowledge
           insulation paper / arxiv 2505.23705)
        4. Concat [img_embs..., lang] → prefix_embs
        5. PaliGemmaBackbone.language_model prefill with use_cache=True
           → per-layer past_key_values
        6. Flow-matching denoise loop with AdaRMSNorm time conditioning:
           a. Build action_emb = action_in_proj(noisy_actions)
           b. Build adarms_cond = time_mlp_in→silu→time_mlp_out→silu(time_emb)
           c. vla_head (Pi05ExpertStackWithPrefix) ingests action_emb +
              per-layer prefix_k/prefix_v + adarms_cond
           d. Euler update: x_t = x_t + dt * v_t where dt = -1/num_steps
        7. Return [B, chunk_size, action_dim] denoised actions
        """
        raise NotImplementedError(
            "Pi05VLA.predict_action lands in Day 5 Phase B. See "
            "features/03_export/basevla-spine_plan.md."
        )


__all__ = ["Pi05VLA"]
