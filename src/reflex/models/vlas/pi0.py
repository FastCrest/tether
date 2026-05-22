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

        # 5. Head: build the prefix-aware pi0 expert FIRST (so we know expert_hidden
        #    for the projector dim). pi0's inference path concatenates per-layer
        #    VLM prefix-KV onto every expert layer's self-attention via
        #    Pi0ExpertStackWithPrefix; the bare ExpertStack would be correct only
        #    for expert-only ONNX export, not end-to-end inference.
        from reflex.exporters.pi0_prefix import build_pi0_expert_with_prefix
        expert_with_prefix, expert_meta = build_pi0_expert_with_prefix(state_dict)
        head = FlowMatchingHead(expert_stack=expert_with_prefix)

        # 4. Projector: pi0's state_proj is action_dim → expert_hidden (NOT
        #    text_hidden — the state token lives in the expert's residual
        #    stream, not paligemma's). lerobot modeling_pi0.py:582:
        #    `nn.Linear(config.max_state_dim, action_expert_config.width)`.
        action_dim = 32  # pi0 padded action dim — matches PI0_MAX_ACTION_DIM
        expert_hidden = expert_meta["expert_hidden"]
        projector = LinearProjector(in_dim=action_dim, out_dim=expert_hidden)
        # Try to load state_proj weights from the checkpoint if available.
        state_proj_w = state_dict.get("model.state_proj.weight")
        state_proj_b = state_dict.get("model.state_proj.bias")
        if state_proj_w is not None:
            with torch.no_grad():
                projector.linear.weight.copy_(state_proj_w)
                if state_proj_b is not None:
                    projector.linear.bias.copy_(state_proj_b)

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
        _use_suffix_attn_mask: bool = True,
        _scale_images: bool = True,
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
        action_dim = state.shape[-1]

        # ─── 1-3. Vision + projection + text embed ──────────────────────
        # CRITICAL: lerobot's PiGemmaModel (the language tower used by
        # paligemma_with_expert.paligemma) OMITS the `inputs_embeds *= sqrt(hidden)`
        # internal scaling that stock GemmaModel applies (modeling_gemma.py:400-401
        # vs pi_gemma.py:194-300 — line removed). lerobot compensates by
        # PRE-SCALING externally in embed_prefix (modeling_pi0.py:669 for text)
        # and via patched_embed_image (cancels stock get_image_features' /sqrt(h)
        # to net ×1 for image). Since our llm_backbone wraps lerobot's PiGemma
        # variant (extracted from the loaded policy), we follow the same
        # protocol: pre-scale text externally; image is raw multi_modal_projector
        # output (no additional scaling needed because the get_image_features
        # path isn't used — we call multi_modal_projector directly).
        text_hidden = self.llm_backbone.text_hidden_size
        sqrt_h = text_hidden ** 0.5
        image_embeds_list: list[torch.Tensor] = []
        for img in images:
            # SigLIP: [B, 3, 224, 224] → [B, 256, vision_hidden=1152]
            img_emb = self.vision_backbone(img)
            # PaliGemma projection: [B, 256, 1152] → [B, 256, 2048] (raw output;
            # lerobot's patched_embed_image net = ×1, matching this directly).
            img_emb = self.llm_backbone.multi_modal_projector(img_emb)
            image_embeds_list.append(img_emb)

        # Pre-scale text per lerobot embed_prefix at modeling_pi0.py:669.
        text_embs = self.llm_backbone.embed_tokens(lang_tokens) * sqrt_h

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

        # PaliGemma prefix attention pattern is FULLY BIDIRECTIONAL within the
        # prefix block (lerobot modeling_pi0.py:837-841 builds att_masks=[0]*N
        # and converts via make_att_2d_masks). Stock HF GemmaModel applies a
        # CAUSAL mask by default if we pass a 1D pad mask; we override with a
        # pre-built 4D bidirectional mask (`create_causal_mask` returns 4D as-is).
        # See transformers/masking_utils.py:768-770.
        # 4D mask shape: [B, 1, prefix_len, prefix_len].
        # value 0.0 = attendable, large negative = masked. We use 0/-inf via
        # the float dtype's neg-infinity since stock attention adds these to logits.
        valid_pair = prefix_pad_mask[:, :, None] & prefix_pad_mask[:, None, :]  # [B, N, N]
        # Convert bool → additive bias: 0 where attendable, large_neg where not.
        # Use a representative small dtype min — bf16 if model is bf16, else fp32.
        prefix_dtype = prefix_embs.dtype
        neg_inf = torch.finfo(prefix_dtype).min
        prefix_4d_mask = torch.where(
            valid_pair, torch.zeros((), dtype=prefix_dtype, device=device),
            torch.full((), neg_inf, dtype=prefix_dtype, device=device),
        ).unsqueeze(1)  # [B, 1, prefix_len, prefix_len]
        prefix_position_ids = torch.cumsum(prefix_pad_mask.long(), dim=1) - 1

        # Force eager attention for the prefill — SDPA/flash mask handling
        # for custom 4D masks differs across transformers versions. lerobot
        # does the same at modeling_pi0.py:844 before its own prefill.
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

        # Extract per-layer K and V. past_key_values may be DynamicCache
        # (`.layers[i].keys/.values`), older Cache (`.key_cache/.value_cache`),
        # or legacy tuple-of-tuples. Both layouts arrive as [B, nkv, prefix_len, hd].
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
        prefix_k = torch.stack(prefix_k_list, dim=0)  # [L, B, nkv, prefix_len, hd]
        prefix_v = torch.stack(prefix_v_list, dim=0)

        # ─── 6. State embedding (suffix's first token) ─────────────────
        # self.projector is state_proj: [B, action_dim=32] → [B, expert_hidden=1024].
        state_emb = self.projector(state).unsqueeze(1)  # [B, 1, expert_hidden]

        # ─── 7. Build noise if not provided ─────────────────────────────
        if noise is None:
            noise = torch.randn(batch, chunk_size, action_dim, device=device, dtype=torch.float32)

        # ─── 8. Denoise loop (Euler) ────────────────────────────────────
        # Suffix is [state_token, action_token_1..chunk_size] = chunk_size + 1 tokens.
        # position_ids for suffix tokens are ABSOLUTE, continuing from prefix_len.
        # lerobot modeling_pi0.py:912-913:
        #     prefix_offsets = sum(prefix_pad_masks); position_ids = prefix_offsets + cumsum(suffix_pad_masks) - 1.
        # With suffix_pad_masks all 1, suffix positions become
        # [prefix_len, prefix_len+1, ..., prefix_len+chunk_size].
        prefix_len_per_batch = prefix_pad_mask.long().sum(dim=-1, keepdim=True)  # [B, 1]
        suffix_pad_mask = torch.ones(batch, chunk_size + 1, dtype=torch.long, device=device)
        suffix_position_ids = prefix_len_per_batch + torch.cumsum(suffix_pad_mask, dim=1) - 1  # [B, chunk_size+1]

        # Build the suffix-attends-to-(prefix+suffix) 4D bool mask. lerobot's
        # `make_att_2d_masks` (modeling_pi0.py:111-140) uses an `att_masks`
        # per-token vector where `1` opens a new attention block. For the full
        # sequence [prefix..., state, action_0, action_1..chunk_size-1]:
        #   prefix att_masks = [0]*prefix_len      → all in block 0 (mutual)
        #   state att_mask   = 1                   → state opens block 1
        #   action_0 att_mask = 1                  → first action opens block 2
        #   action_i (i>=1) att_mask = 0           → remaining actions stay in block 2
        # Then attn[i,j] = cumsum[j] <= cumsum[i]. The query side is the suffix
        # only (the expert receives suffix queries; prefix queries already had
        # their hiddens computed in the prefill).
        prefix_len = prefix_pad_mask.shape[1]
        suffix_len = chunk_size + 1
        total_len = prefix_len + suffix_len
        full_att = torch.zeros(batch, total_len, dtype=torch.long, device=device)
        full_att[:, prefix_len] = 1         # state opens new block
        full_att[:, prefix_len + 1] = 1     # first action opens new block
        # Remaining actions stay in same block (att=0 default).
        cumsum = torch.cumsum(full_att, dim=1)  # [B, total_len]
        # att_2d[b, i, j] = cumsum[b, j] <= cumsum[b, i]
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, total_len, total_len]
        # Pad mask combined: full_pad = prefix_pad ⊕ suffix_pad
        full_pad = torch.cat([prefix_pad_mask, suffix_pad_mask.bool()], dim=1)  # [B, total_len]
        pad_2d = full_pad[:, None, :] & full_pad[:, :, None]
        full_2d_mask = att_2d & pad_2d
        # Slice to suffix queries only: [B, suffix_len, total_len]
        suffix_2d_mask = full_2d_mask[:, prefix_len:, :]
        # Expert layer expects [B, 1, suffix_len, total_kv_len] for broadcasting
        # across n_heads.
        suffix_attn_mask = suffix_2d_mask.unsqueeze(1)

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
                state_emb=state_emb,
                attn_mask=suffix_attn_mask if _use_suffix_attn_mask else None,
            )
            x_t = x_t + dt * v_t

        # ─── 9. Return denoised action chunk ────────────────────────────
        return x_t


__all__ = ["Pi0VLA"]
