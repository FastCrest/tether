"""GR00TVLA — NVIDIA GR00T N1.6 on the BaseVLA spine.

GR00T is the **critical proof point** for the BaseVLA spine's 6-slot
design (decision S-2). Pi0/Pi05/SmolVLA all split vision + language into
TWO slots (``vision_backbone`` + ``llm_backbone``). GR00T's Eagle
(SigLIP + Qwen2-0.5B + mlp1, internally cross-attending) is **fused** —
it lives in the ``vlm_backbone`` slot as a single unit::

    GR00TVLA = BaseVLA(
        vlm_backbone = Eagle (SigLIP + Qwen2-0.5B + mlp1, fused),
        vla_head     = DITHead (32-block diffusion transformer),
        # ALL OTHER SLOTS UNUSED (None):
        vision_backbone = None,
        llm_backbone    = None,
        projector       = None,
        text_encoder    = None,
    )

This composition validates that:

1. The spine accepts a 2-slot VLA (vs pi0/pi05's 3 + smolvla's 4) ✓
2. The fused-VLM slot composes correctly with a non-flow-matching head ✓
3. Per-embodiment encoders/decoders flow through the spine ✓

Per the lift #1 Day 7 plan, this is the day that proves the 6-slot
taxonomy was the right call.

NAME_MAPPING: empty per decision S-1 (GR00T N1.6 keys load directly).
"""
from __future__ import annotations

from typing import Any, ClassVar

import torch

from reflex.models.base_vla import BaseVLA
from reflex.registry.components import VLAS


@VLAS.register
class GR00TVLA(BaseVLA):
    """GR00T N1.6 spine composition — Eagle (fused VLM) + DiT head."""

    REQUIRED_SLOTS: ClassVar[tuple[str, ...]] = (
        "vlm_backbone",
        "vla_head",
    )
    OPTIONAL_SLOTS: ClassVar[tuple[str, ...]] = ()
    NAME_MAPPING: ClassVar[dict[str, str]] = {}

    # ── Construction helpers ────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str = "nvidia/GR00T-N1.6-3B",
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
        embodiment_id: int = 0,
        select_layer: int = -1,
    ) -> "GR00TVLA":
        """Build GR00TVLA from a GR00T N1.6 checkpoint state_dict.

        Both Eagle and the DiT head are built from the SAME state_dict —
        Eagle reads ``backbone.model.*`` keys; the head reads
        ``action_head.*`` keys. They don't overlap, so a single load is
        sufficient.

        Args:
            hf_id: HuggingFace repo (default ``nvidia/GR00T-N1.6-3B``).
                Used only if ``state_dict`` is not provided.
            state_dict: Pre-loaded GR00T checkpoint. If None, attempts
                to download from ``hf_id``.
            embodiment_id: Which embodiment's per-embodiment encoder /
                decoder weights to bind (0 = N1.6 default).
            select_layer: Which Qwen2 hidden layer Eagle surfaces to the
                DiT cross-attn (default -1 = last layer).

        Returns:
            GR00TVLA instance ready for predict_action()/forward().
        """
        from reflex.models.heads.dit_head import DITHead
        from reflex.models.vlm.eagle_backbone import EagleBackbone

        if state_dict is None:
            from reflex.checkpoint import load_checkpoint
            state_dict, _ = load_checkpoint(hf_id)

        vlm = EagleBackbone(state_dict=state_dict, select_layer=select_layer)
        head = DITHead(state_dict=state_dict, embodiment_id=embodiment_id)

        return cls(vlm_backbone=vlm, vla_head=head)

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
        """Load a GR00T safetensors checkpoint directly into a flat dict.

        Supports both single-file and sharded (index.json) checkpoints.
        For sharded: pass the directory containing the .safetensors files.
        """
        from pathlib import Path

        from reflex.models.vlas._groot_safetensors_mapping import (
            expand_tied_groot,
            groot_safetensors_to_flat,
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
            target_key = groot_safetensors_to_flat(src_key)
            if target_key is None:
                continue
            flat[target_key] = tensor

        return expand_tied_groot(flat)

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Any]) -> Any:
        """Minimum-viable forward — runs Eagle on (images, tokens) to produce
        the VLM-KV context, then DITHead on it.

        Args:
            batch: dict with keys:
                - "images": ``[B, 3, H, W]``
                - "input_ids": ``[B, seq]``
                - "attention_mask": optional ``[B, seq]``
                - "image_flags": optional ``[B]``
                - "noisy_actions": ``[B, chunk, raw_action_dim]``
                - "timestep": ``[B]``
                - "position_ids": ``[B, chunk]``
                - "state": optional ``[B, state_dim]``

        Returns:
            ``[B, chunk, raw_action_dim]`` velocity from DITHead.
        """
        vlm_kv = self.vlm_backbone(
            batch["images"],
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            image_flags=batch.get("image_flags"),
        )
        return self.vla_head(
            batch["noisy_actions"],
            batch.get("timestep"),
            batch.get("position_ids"),
            state=batch.get("state"),
            vlm_kv=vlm_kv,
        )

    def predict_action(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
        chunk_size: int = 16,
        raw_action_dim: int = 128,
        num_steps: int = 4,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full GR00T inference: Eagle prefill → DiT denoise loop.

        GR00T uses fixed 4-step diffusion denoising (vs flow-matching's
        variable Euler steps).

        Args:
            images: ``[B, 3, H, W]`` SigLIP pixels.
            input_ids: ``[B, seq]`` Qwen2 tokens.
            state: ``[B, state_dim=128]`` raw robot state.
            attention_mask: ``[B, seq]`` — None → all-ones.
            image_flags: ``[B]`` — None → all-ones.
            chunk_size: number of action tokens to denoise (16 default).
            raw_action_dim: padded raw action dim (128 default).
            num_steps: DiT denoising steps (4 default).
            noise: optional ``[B, chunk_size, raw_action_dim]`` seed.

        Returns:
            ``[B, chunk_size, raw_action_dim]`` denoised actions.
        """
        for slot in ("vlm_backbone", "vla_head"):
            if getattr(self, slot) is None:
                raise RuntimeError(f"GR00TVLA.predict_action: required slot {slot} is None")

        device = images.device
        batch = images.shape[0]

        # 1. Eagle prefill — joint VLM-KV
        vlm_kv = self.vlm_backbone(
            images, input_ids,
            attention_mask=attention_mask, image_flags=image_flags,
        )

        # 2. Denoise loop — Euler steps from t=1 (pure noise) to t=0 (action)
        if noise is None:
            noise = torch.randn(batch, chunk_size, raw_action_dim, device=device)
        x = noise
        position_ids = torch.arange(chunk_size, device=device).unsqueeze(0).expand(batch, -1)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch,), 1.0 - step * dt, device=device)
            v = self.vla_head(
                x, t, position_ids,
                state=state, vlm_kv=vlm_kv,
            )
            x = x - dt * v

        return x


__all__ = ["GR00TVLA"]
