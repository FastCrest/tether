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
        from reflex.exporters.pi0_prefix import build_pi05_expert_with_prefix
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
        action_dim: int = 32,
    ) -> torch.Tensor:
        """Full pi0.5 inference: vision → merge → prefix prefill → denoise loop.

        Matches lerobot PI05Policy's denoise_step orchestration with the
        pi0.5 specifics (vs pi0):

        1. SigLIPBackbone encodes each camera image
        2. PaliGemmaBackbone.multi_modal_projector projects vision → text_hidden
        3. PaliGemmaBackbone.embed_tokens embeds language tokens — state IS
           encoded in lang_tokens via lerobot's processor (knowledge
           insulation per arxiv 2505.23705), NO separate state_proj.
        4. Concat [img1, img2, img3, lang] into prefix_embs
        5. PaliGemmaBackbone.language_model prefill with use_cache=True
           → per-layer past_key_values
        6. Flow-matching denoise loop (default 10 Euler steps):
           a. action_emb = action_in_proj(noisy_actions) (NO state, NO time concat)
           b. time conditioning fed to AdaRMSNorm per layer via the expert
           c. Pi05ExpertStackWithPrefix ingests action_emb + prefix_k/v + attn_mask
           d. Euler update: x_t = x_t + dt * v_t where dt = -1/num_steps
        7. Return [B, chunk_size, action_dim] denoised actions

        Args:
            images: list of N camera tensors, each `[B, 3, H, W]` float32
                normalized to [-1, 1].
            image_masks: optional list of `[B]` bool masks. None → all valid.
            lang_tokens: `[B, seq_len]` int64 PaliGemma tokenizer output
                (state info IS embedded here per pi0.5's knowledge insulation).
            lang_masks: `[B, seq_len]` bool attention mask.
            noise: optional `[B, chunk_size, action_dim]` Gaussian noise seed.
            num_steps: Euler denoising steps. 10 = pi0.5 default.
            chunk_size: action chunk length. 50 = pi0.5/LIBERO default.
            action_dim: padded action dim. 32 = pi0.5 default.

        Returns:
            `[B, chunk_size, action_dim]` denoised action tensor.

        Raises:
            RuntimeError: if any required spine component is None.
        """
        for slot in ("vision_backbone", "llm_backbone", "vla_head"):
            if getattr(self, slot) is None:
                raise RuntimeError(f"Pi05VLA.predict_action: required slot {slot} is None")

        device = lang_tokens.device
        batch = lang_tokens.shape[0]

        # ─── 1-3. Vision + projection + text embed ──────────────────────
        # Same scaling protocol as Pi0VLA: PiGemma omits internal sqrt(h)
        # scaling, so text needs explicit pre-scale; image stays raw
        # (lerobot patched_embed_image net = ×1).
        text_hidden = self.llm_backbone.text_hidden_size
        sqrt_h = text_hidden ** 0.5
        image_embeds_list: list[torch.Tensor] = []
        for img in images:
            img_emb = self.vision_backbone(img)
            img_emb = self.llm_backbone.multi_modal_projector(img_emb)
            image_embeds_list.append(img_emb)
        text_embs = self.llm_backbone.embed_tokens(lang_tokens) * sqrt_h

        # ─── 4. Concat into prefix (NO state token — state is in lang_tokens) ──
        prefix_embs = torch.cat([*image_embeds_list, text_embs], dim=1)
        prefix_seq_len = prefix_embs.shape[1]
        img_token_count = image_embeds_list[0].shape[1] if image_embeds_list else 0

        # prefix_pad_mask: True for valid prefix tokens
        if image_masks is None:
            image_masks = [torch.ones(batch, dtype=torch.bool, device=device) for _ in images]
        img_masks_per_token = []
        for m in image_masks:
            img_masks_per_token.append(m[:, None].expand(batch, img_token_count))
        prefix_pad_mask = torch.cat([*img_masks_per_token, lang_masks.bool()], dim=1)

        # 4D bidirectional-within-prefix mask + cumsum position_ids (same as pi0)
        valid_pair = prefix_pad_mask[:, :, None] & prefix_pad_mask[:, None, :]
        prefix_dtype = prefix_embs.dtype
        neg_inf = torch.finfo(prefix_dtype).min
        prefix_4d_mask = torch.where(
            valid_pair, torch.zeros((), dtype=prefix_dtype, device=device),
            torch.full((), neg_inf, dtype=prefix_dtype, device=device),
        ).unsqueeze(1)
        prefix_position_ids = torch.cumsum(prefix_pad_mask.long(), dim=1) - 1

        # Force eager attention for prefill — same reasoning as Pi0VLA.
        prev_attn_impl = self.llm_backbone.language_model.config._attn_implementation
        self.llm_backbone.language_model.config._attn_implementation = "eager"
        try:
            # ─── 5. Run language_model prefill → past_key_values ────────
            prefix_out = self.llm_backbone(
                inputs_embeds=prefix_embs,
                attention_mask=prefix_4d_mask,
                position_ids=prefix_position_ids,
                use_cache=True,
            )
        finally:
            self.llm_backbone.language_model.config._attn_implementation = prev_attn_impl
        past_key_values = prefix_out.past_key_values

        # Extract per-layer K and V — same as Pi0VLA.
        prefix_k_list: list[torch.Tensor] = []
        prefix_v_list: list[torch.Tensor] = []
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                prefix_k_list.append(layer.keys)
                prefix_v_list.append(layer.values)
        elif hasattr(past_key_values, "key_cache"):
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                prefix_k_list.append(k)
                prefix_v_list.append(v)
        else:
            for (k, v) in past_key_values:
                prefix_k_list.append(k)
                prefix_v_list.append(v)
        prefix_k = torch.stack(prefix_k_list, dim=0)
        prefix_v = torch.stack(prefix_v_list, dim=0)

        # ─── 6. Build noise if not provided ─────────────────────────────
        if noise is None:
            noise = torch.randn(batch, chunk_size, action_dim, device=device, dtype=torch.float32)

        # ─── 7. Denoise loop (Euler) ────────────────────────────────────
        # Suffix is ACTION-ONLY (chunk_size tokens). Position_ids absolute,
        # continuing from prefix_len: [prefix_len, prefix_len+1, ..., prefix_len+chunk_size-1].
        prefix_len_per_batch = prefix_pad_mask.long().sum(dim=-1, keepdim=True)
        suffix_pad_mask = torch.ones(batch, chunk_size, dtype=torch.long, device=device)
        suffix_position_ids = prefix_len_per_batch + torch.cumsum(suffix_pad_mask, dim=1) - 1

        # Suffix attention mask: all actions attend prefix bidirectionally;
        # all actions attend ALL other actions (mutual within action block).
        # Per lerobot embed_suffix att_masks = [1] + [0]*(chunk_size-1):
        # cumsum_suffix = [1, 1, 1, ..., 1] → all in same block → mutual.
        prefix_len = prefix_pad_mask.shape[1]
        total_len = prefix_len + chunk_size
        full_att = torch.zeros(batch, total_len, dtype=torch.long, device=device)
        full_att[:, prefix_len] = 1  # first action opens new block
        cumsum = torch.cumsum(full_att, dim=1)
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
        full_pad = torch.cat([prefix_pad_mask, suffix_pad_mask.bool()], dim=1)
        pad_2d = full_pad[:, None, :] & full_pad[:, :, None]
        full_2d_mask = att_2d & pad_2d
        suffix_attn_mask = full_2d_mask[:, prefix_len:, :].unsqueeze(1)

        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor([time_val], dtype=torch.float32, device=device).expand(batch)
            v_t = self.vla_head(
                noisy_actions=x_t,
                timestep=time_tensor,
                position_ids=suffix_position_ids,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                attn_mask=suffix_attn_mask,
            )
            x_t = x_t + dt * v_t

        return x_t


__all__ = ["Pi05VLA"]
