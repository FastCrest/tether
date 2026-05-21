"""Pi0VLA — pi0 composition class on the BaseVLA spine.

Lift #1 Day 4f per `features/03_export/basevla-spine_plan.md`. Wires:

    Pi0VLA = BaseVLA(
        vision_backbone = SigLIPBackbone (extracted from paligemma.vision_tower)
        llm_backbone    = PaliGemmaBackbone (PaliGemma minus vision_tower)
        projector       = LinearProjector (state_proj)
        vla_head        = FlowMatchingHead (wraps ExpertStack from build_pi0_expert_stack)
    )

The 4 component classes were added in Days 4a-e. This file wires them
together + provides `from_pretrained()` to load the canonical
`lerobot/pi0_base` checkpoint.

What's NOT in this PR (deferred to Day 4g):

- `predict_action()` full inference pipeline (vision → project → merge →
  language → flow-matching denoise loop). Day 4f ships the composition
  shape; Day 4g ships the inference path + the `--use-new-spine` CLI flag
  + the parity gate vs the legacy `src/reflex/exporters/pi0_exporter.py`
  + a Modal smoke validating bit-identical actions vs the OLD path.

This PR's scope: prove the composition class builds + composes via the
spine. The forward() returns the language hidden states (incomplete —
the head's flow-matching step is wired but the multimodal merging logic
in `predict_action` is the missing piece).

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
class Pi0VLA(BaseVLA):
    """pi0 spine composition — PaliGemma (vision split out) + flow-matching expert.

    Slots:

    - vision_backbone: SigLIPBackbone (REQUIRED) — extracted from PaliGemma
    - llm_backbone:    PaliGemmaBackbone (REQUIRED) — PaliGemma minus vision_tower
    - projector:       LinearProjector (REQUIRED) — robot state → VLM hidden
    - vla_head:        FlowMatchingHead (REQUIRED) — wraps ExpertStack
    - vlm_backbone:    not used (None)
    - text_encoder:    not used (None)

    NAME_MAPPING: empty per decision S-1 — the lerobot/pi0_base checkpoint's
    keys map directly to component slots via the spine's default routing
    (load_state_dict splits keys by leading `slot.` prefix). If a future
    pi0 release ships with different naming, add the renames here.
    """

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
        hf_id: str = "lerobot/pi0_base",
        *,
        dtype: torch.dtype | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "Pi0VLA":
        """Build Pi0VLA from a HuggingFace pi0 checkpoint.

        Loads PaliGemma once, then:

        1. Extracts `paligemma.model.vision_tower` → wraps as SigLIPBackbone
        2. Wraps the rest of PaliGemma (still has vision_tower attribute but
           it's no longer called at runtime) → PaliGemmaBackbone
        3. Builds projector from PaliGemma's `state_proj` weights if present,
           else random-init (parity tests will validate against checkpoint)
        4. Builds the ExpertStack via `build_pi0_expert_stack(state_dict)`
           → wraps as FlowMatchingHead
        5. Returns Pi0VLA composed of the 4 components

        Args:
            hf_id: HuggingFace repo (default lerobot/pi0_base)
            dtype: cast loaded model to this dtype (e.g. torch.bfloat16)
            state_dict: pre-loaded raw state_dict for the expert-stack build
                + the projector weights. If None, loads from HF.

        Returns:
            Pi0VLA instance ready for forward() + (Day 4g) predict_action().
        """
        from transformers import PaliGemmaForConditionalGeneration

        from reflex.models.heads.flow_matching_head import FlowMatchingHead
        from reflex.models.llm.paligemma_backbone import PaliGemmaBackbone
        from reflex.models.projectors.linear_projector import LinearProjector
        from reflex.models.vision.siglip_backbone import SigLIPBackbone

        # 1. Load PaliGemma — full model, cast if requested.
        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(hf_id)
        if dtype is not None:
            paligemma = paligemma.to(dtype=dtype)

        # 2. Vision: extract vision_tower from paligemma.model.vision_tower
        vision = SigLIPBackbone(model=paligemma.model.vision_tower)

        # 3. Language: wrap the rest of PaliGemma (vision_tower attribute
        #    stays on the model but prepare_triton + forward path skip it).
        llm = PaliGemmaBackbone(model=paligemma)

        # 4. Projector: state_proj from the pi0 state_dict if present, else
        #    randomly initialized at the expected shape. The pi0 checkpoint
        #    ships state_proj.weight at `model.state_proj.weight` (action_dim=32
        #    → text_hidden=2048 for PaliGemma-3B).
        if state_dict is None:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            try:
                safetensors_path = hf_hub_download(
                    repo_id=hf_id, filename="model.safetensors",
                )
                state_dict = load_file(safetensors_path)
            except Exception:
                # If the checkpoint doesn't have a state_dict load-able
                # this way, downstream Day 4g handles the fallback.
                state_dict = {}

        text_hidden = llm.text_hidden_size
        action_dim = 32  # pi0 padded action dim — matches pi0_exporter's PI0_MAX_ACTION_DIM
        projector = LinearProjector(in_dim=action_dim, out_dim=text_hidden)
        # Try to load state_proj weights from the checkpoint if available.
        state_proj_w = state_dict.get("model.state_proj.weight")
        state_proj_b = state_dict.get("model.state_proj.bias")
        if state_proj_w is not None:
            with torch.no_grad():
                projector.linear.weight.copy_(state_proj_w)
                if state_proj_b is not None:
                    projector.linear.bias.copy_(state_proj_b)

        # 5. Head: build ExpertStack via the existing pi0 builder, wrap.
        head = FlowMatchingHead(state_dict=state_dict, vla_family="pi0")

        return cls(
            vision_backbone=vision,
            llm_backbone=llm,
            projector=projector,
            vla_head=head,
        )

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Any]) -> Any:
        """Minimum-viable forward — runs the language path on
        already-merged inputs_embeds.

        This is incomplete relative to the legacy pi0_exporter's inference
        path; Day 4g adds the full vision→project→merge→head pipeline.

        Args:
            batch: dict with keys:
                - "inputs_embeds": pre-merged image+text embeddings
                - "attention_mask": [batch, seq]
                - "past_key_values": optional

        Returns:
            The language_model output (BaseModelOutput).
        """
        return self.llm_backbone(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch.get("attention_mask"),
            past_key_values=batch.get("past_key_values"),
        )

    def predict_action(
        self,
        *,
        images: Any,
        state: Any,
        instruction: str,
    ) -> Any:
        """NOT IMPLEMENTED in Day 4f — deferred to Day 4g.

        Day 4g implements the full inference pipeline:

        1. vision_backbone(images) → image embeds
        2. multi_modal_projector(image_embeds) → projected to text_hidden
        3. embed_tokens(input_ids) → text embeds
        4. merge image_embeds into text_embeds at image-token positions
        5. llm_backbone(inputs_embeds=merged) → hidden states + per-layer KV
        6. projector(state) → state in VLM hidden space
        7. flow-matching denoise loop (10 Euler steps):
           - noise → vla_head(noisy_actions, timestep, position_ids, vlm_k, vlm_v)
           - update: x_t = x_t + dt * v_t
        8. return final actions

        Parity gate vs the legacy `src/reflex/exporters/pi0_exporter.py` path
        rides alongside Day 4g + a Modal smoke validates cos=+1.0.
        """
        raise NotImplementedError(
            "Pi0VLA.predict_action ships in Day 4g (full inference pipeline + "
            "parity validation vs legacy pi0_exporter). Day 4f scope = "
            "composition shape only."
        )


__all__ = ["Pi0VLA"]
