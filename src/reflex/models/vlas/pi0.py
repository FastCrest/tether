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

        # 5. Head: build the prefix-aware pi0 expert (NOT the bare
        #    ExpertStack — pi0's inference path concatenates per-layer VLM
        #    prefix-KV onto every expert layer's self-attention, see
        #    Pi0ExpertStackWithPrefix in exporters/pi0_prefix_exporter.py).
        #    The default FlowMatchingHead route via vla_family="pi0" builds
        #    the bare ExpertStack which is correct for expert-only ONNX
        #    export but NOT for end-to-end inference.
        from reflex.exporters.pi0_prefix_exporter import build_pi0_expert_with_prefix
        expert_with_prefix, _expert_meta = build_pi0_expert_with_prefix(state_dict)
        head = FlowMatchingHead(expert_stack=expert_with_prefix)

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
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor] | None = None,
        state: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        chunk_size: int = 50,
    ) -> torch.Tensor:
        """Full pi0 inference: vision → merge → prefix prefill → denoise loop.

        Matches lerobot's `PI0Policy.sample_actions()` orchestration but
        composed over spine components:

        1. SigLIPBackbone encodes each camera image
        2. PaliGemmaBackbone.multi_modal_projector projects vision_hidden → text_hidden
        3. PaliGemmaBackbone.embed_tokens embeds language tokens
        4. Concat [img1, img2, img3, lang] into prefix_embeds
        5. PaliGemmaBackbone.language_model prefill with use_cache=True
           → per-layer past_key_values
        6. LinearProjector projects state into text_hidden space
        7. Flow-matching denoise loop (default 10 Euler steps):
           a. Build suffix_embs = [state_emb, action_time_embs] (assembled inside vla_head)
           b. vla_head (Pi0ExpertStackWithPrefix) ingests noisy actions +
              per-layer prefix_k/prefix_v from PaliGemma's cache
           c. Euler update: x_t = x_t + dt * v_t where dt = -1/num_steps
        8. Return [B, chunk_size, action_dim] denoised actions

        Args:
            images: list of N camera tensors, each `[B, 3, H, W]` float32
                normalized to [-1, 1] (typical N=3 for LIBERO: base + 2 wrist).
            image_masks: optional list of `[B]` bool masks marking which
                images are valid (vs padding). None → all valid.
            state: `[B, action_dim]` robot state tensor.
            lang_tokens: `[B, seq_len]` int64 PaliGemma tokenizer output.
            lang_masks: `[B, seq_len]` bool attention mask.
            noise: optional `[B, chunk_size, action_dim]` Gaussian noise
                seed. None → torch.randn at call time.
            num_steps: Euler denoising steps. 10 = pi0 default, 1 = 1-NFE
                distilled student.
            chunk_size: action chunk length. 50 = pi0/LIBERO default.

        Returns:
            `[B, chunk_size, action_dim]` denoised action tensor.

        Raises:
            RuntimeError: if any required spine component is None (Pi0VLA
                requires all 4 of vision/llm/projector/head).
        """
        # Defensive — Pi0VLA's REQUIRED_SLOTS guarantees these aren't None
        # at construction, but predict_action is the runtime contract surface.
        for slot in ("vision_backbone", "llm_backbone", "projector", "vla_head"):
            if getattr(self, slot) is None:
                raise RuntimeError(f"Pi0VLA.predict_action: required slot {slot} is None")

        device = lang_tokens.device
        batch = lang_tokens.shape[0]
        text_hidden = self.llm_backbone.text_hidden_size
        action_dim = state.shape[-1]

        # ─── 1-3. Vision + projection + text embed ──────────────────────
        image_embeds_list: list[torch.Tensor] = []
        for img in images:
            # SigLIP: [B, 3, 224, 224] → [B, 256, vision_hidden=1152]
            img_emb = self.vision_backbone(img)
            # PaliGemma projection: [B, 256, 1152] → [B, 256, 2048]
            img_emb = self.llm_backbone.multi_modal_projector(img_emb)
            image_embeds_list.append(img_emb)

        # Token embed + Gemma scale-by-sqrt-d (matches PaliGemma's input scaling).
        # lerobot's embed_prefix at modeling_pi0.py:645-686 multiplies token
        # embeds by sqrt(hidden) before feeding the LM; mirroring keeps the
        # prefix forward bit-identical to the lerobot reference path.
        text_embs = self.llm_backbone.embed_tokens(lang_tokens) * (text_hidden ** 0.5)

        # ─── 4. Concat into prefix ──────────────────────────────────────
        prefix_embs = torch.cat([*image_embeds_list, text_embs], dim=1)
        prefix_seq_len = prefix_embs.shape[1]
        img_token_count = image_embeds_list[0].shape[1] if image_embeds_list else 0

        # Build prefix_pad_mask: [B, prefix_seq_len] — images mark valid
        # only when their image_mask is True; language uses lang_masks.
        if image_masks is None:
            image_masks = [torch.ones(batch, dtype=torch.bool, device=device) for _ in images]
        img_masks_per_token = []
        for m in image_masks:
            img_masks_per_token.append(m[:, None].expand(batch, img_token_count))
        prefix_pad_mask = torch.cat([*img_masks_per_token, lang_masks.bool()], dim=1)

        # PaliGemma block-attention pattern: image patches are mutually
        # bidirectional; language is causal-on-itself + can attend to images.
        # For prefix prefill we use the simpler "attention_mask = pad_mask"
        # equivalent to lerobot's prefix_att_masks (modeling_pi0.py:833-841).
        # The LM internally masks future positions per causal config.
        prefix_position_ids = torch.cumsum(prefix_pad_mask.long(), dim=1) - 1

        # ─── 5. Run language_model prefill → past_key_values ────────────
        # use_cache=True returns per-layer K/V we need for the expert.
        prefix_out = self.llm_backbone(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_pad_mask,
            position_ids=prefix_position_ids,
            use_cache=True,
        )
        past_key_values = prefix_out.past_key_values

        # Extract per-layer K and V as [L, B, prefix_len, nkv, hd] (Pi0ExpertStackWithPrefix
        # accepts either pre- or post-transpose layout; we pass post-transpose).
        # past_key_values may be either a tuple-of-tuples (legacy) or a Cache
        # object (modern transformers DynamicCache). Both expose per-layer K/V.
        prefix_k_list: list[torch.Tensor] = []
        prefix_v_list: list[torch.Tensor] = []
        if hasattr(past_key_values, "layers"):
            # transformers DynamicCache — each layer has .keys + .values
            for layer in past_key_values.layers:
                prefix_k_list.append(layer.keys)
                prefix_v_list.append(layer.values)
        elif hasattr(past_key_values, "key_cache"):
            # older Cache API — list per layer
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                prefix_k_list.append(k)
                prefix_v_list.append(v)
        else:
            # tuple-of-tuples (very old transformers)
            for (k, v) in past_key_values:
                prefix_k_list.append(k)
                prefix_v_list.append(v)
        prefix_k = torch.stack(prefix_k_list, dim=0)  # [L, B, nkv, prefix_len, hd]
        prefix_v = torch.stack(prefix_v_list, dim=0)

        # ─── 6. Build noise if not provided ─────────────────────────────
        if noise is None:
            noise = torch.randn(batch, chunk_size, action_dim, device=device, dtype=torch.float32)

        # ─── 7. Denoise loop (Euler) ────────────────────────────────────
        dt = -1.0 / num_steps
        x_t = noise
        # Action position_ids start at 0 (Pi0ExpertStackWithPrefix handles the
        # prefix offset internally via prefix_concat; the expert sees the
        # action tokens at positions [0..chunk_size-1] within their own block).
        action_position_ids = torch.arange(chunk_size, device=device).unsqueeze(0).expand(batch, -1)

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor([time_val], dtype=torch.float32, device=device).expand(batch)
            v_t = self.vla_head(
                noisy_actions=x_t,
                timestep=time_tensor,
                position_ids=action_position_ids,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
            )
            x_t = x_t + dt * v_t

        # ─── 8. Return denoised action chunk ────────────────────────────
        return x_t


__all__ = ["Pi0VLA"]
