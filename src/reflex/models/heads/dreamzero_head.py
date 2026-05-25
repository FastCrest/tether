"""DreamZeroHead — DiT action + video diffusion head on the BaseVLA spine.

Fills the ``vla_head`` slot for DreamZeroVLA. Contains the CausalWanModel
(DiT backbone) + FlowMatchScheduler. Ported from FluxVLA
``heads/dreamzero_head.py`` (Apache-2.0, LimX Dynamics).

Rewritten imports: ``fluxvla.models.third_party_models.dreamzero.modules``
→ ``reflex.models.third_party.dreamzero.modules``.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

logger = logging.getLogger(__name__)


def _import_dreamzero_modules():
    from reflex.models.third_party.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
    from reflex.models.third_party.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
    return CausalWanModel, FlowMatchScheduler


def _ensure_file(path, hf_filename):
    if path is not None and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="Wan-AI/Wan2.1-I2V-14B-480P", filename=hf_filename)


class DreamZeroHead(nn.Module):
    """DreamZero action head — joint video + action flow matching on the
    Wan 2.1 DiT backbone.

    Args:
        action_dim: Actual robot action dimension.
        max_action_dim: Padded action dim inside the DiT.
        action_horizon: Action steps per generation block.
        max_state_dim: Padded state dimension.
        num_frame_per_block: Latent frames per DiT block.
        num_action_per_block: Action steps per block.
        num_state_per_block: State tokens per block.
        dit_dim: DiT hidden dimension.
        dit_num_heads: DiT attention heads.
        dit_num_layers: DiT transformer blocks.
        num_inference_steps: Denoising steps at inference.
        pretrained_name_or_path: Wan 2.1 checkpoint directory.
    """

    def __init__(
        self,
        action_dim: int = 7,
        max_action_dim: int = 32,
        action_horizon: int = 10,
        max_state_dim: int = 64,
        num_frames: int = 9,
        num_frame_per_block: int = 2,
        num_action_per_block: int = 10,
        num_state_per_block: int = 1,
        hidden_size: int = 64,
        input_embedding_dim: int = 1536,
        dit_dim: int = 5120,
        dit_ffn_dim: int = 13824,
        dit_num_heads: int = 40,
        dit_num_layers: int = 40,
        dit_freq_dim: int = 256,
        dit_in_dim: int = 36,
        dit_out_dim: int = 16,
        max_num_embodiments: int = 32,
        frame_seqlen: int = 880,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_inference_steps: int = 4,
        pretrained_name_or_path: str | None = None,
        use_gradient_checkpointing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        CausalWanModel, FlowMatchScheduler = _import_dreamzero_modules()

        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.max_state_dim = max_state_dim
        self.num_frames = num_frames
        self.num_frame_per_block = num_frame_per_block
        self.noise_s = noise_s
        self.num_inference_steps = num_inference_steps
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.use_cache = False

        self.model = CausalWanModel(
            diffusion_model_pretrained_path=pretrained_name_or_path,
            model_type="i2v",
            frame_seqlen=frame_seqlen,
            dim=dit_dim,
            in_dim=dit_in_dim,
            ffn_dim=dit_ffn_dim,
            out_dim=dit_out_dim,
            freq_dim=dit_freq_dim,
            num_heads=dit_num_heads,
            num_layers=dit_num_layers,
            max_chunk_size=-1,
            num_frame_per_block=num_frame_per_block,
            action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            max_num_embodiments=max_num_embodiments,
            hidden_size=hidden_size,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

        if pretrained_name_or_path is not None:
            self._load_pretrained_weights(pretrained_name_or_path)

        if use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.reset_inference_state()
        self.scheduler.set_timesteps(1000, training=True)

    def _load_pretrained_weights(self, dit_path: str) -> None:
        if dit_path is not None and os.path.isdir(dit_path):
            import json
            from safetensors.torch import load_file

            index_path = os.path.join(dit_path, "diffusion_pytorch_model.safetensors.index.json")
            single_path = os.path.join(dit_path, "diffusion_pytorch_model.safetensors")
            state_dict: dict = {}
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                for shard in set(index["weight_map"].values()):
                    state_dict.update(load_file(os.path.join(dit_path, shard)))
            elif os.path.exists(single_path):
                state_dict = load_file(single_path)
            else:
                logger.warning("No DiT weights at %s", dit_path)
                return
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info("DiT missing keys: %s", missing[:5])
            if unexpected:
                logger.info("DiT unexpected keys: %s", unexpected[:5])

    def reset_inference_state(self) -> None:
        self.inference_kv_cache = None
        self.inference_clip_feas = None
        self.inference_ys = None
        self.inference_prompt_embs = None
        self.current_start_frame = 0

    def _create_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        return [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(self.model.num_layers)
        ]

    def _predict_action_stateless(
        self,
        prompt_embs: torch.Tensor,
        latents: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
        num_inference_steps: int,
        observed_latent_frames: int,
    ) -> torch.Tensor:
        from reflex.models.third_party.dreamzero.modules.flow_unipc_multistep_scheduler import (
            FlowUniPCMultistepScheduler,
        )

        device = states.device
        b = states.shape[0]
        observed_latents = latents[:, :, :observed_latent_frames]

        local_kv_cache = self._create_kv_cache(b, latents.dtype, device)

        # Append reference frame
        self.inference_kv_cache = local_kv_cache
        timestep_ref = torch.zeros(b, 1, dtype=torch.int64, device=device)
        frame_seqlen = int(latents.shape[3] * latents.shape[4] / 4)
        _, _, updated_kv = self.model(
            observed_latents[:, :, :1],
            timestep=timestep_ref,
            clip_feature=clip_feas,
            y=ys[:, :, :1],
            context=prompt_embs,
            seq_len=frame_seqlen,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            kv_cache=local_kv_cache,
            crossattn_cache=None,
            current_start_frame=0,
        )
        local_kv_cache = updated_kv
        self.inference_kv_cache = None

        denoise_frames = self.num_frame_per_block
        if observed_latent_frames <= 1:
            denoise_frames = 1

        num_channels, lat_h, lat_w = latents.shape[1], latents.shape[3], latents.shape[4]

        noisy_latents = torch.randn(b, num_channels, denoise_frames, lat_h, lat_w, device=device, dtype=latents.dtype)
        noisy_actions = torch.randn(b, self.action_horizon, self.max_action_dim, device=device, dtype=latents.dtype)

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps, shift=1, use_dynamic_shifting=False,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps, shift=1, use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(num_inference_steps, device=device, shift=5.0)
        sample_scheduler_action.set_timesteps(num_inference_steps, device=device, shift=5.0)

        denoise_seq_len = denoise_frames * frame_seqlen
        y_future = ys[:, :, :denoise_frames] if ys.shape[2] >= denoise_frames else ys[:, :, -denoise_frames:]

        for step_idx in range(len(sample_scheduler.timesteps)):
            vt = sample_scheduler.timesteps[step_idx]
            at = sample_scheduler_action.timesteps[step_idx]

            video_pred, action_pred, _ = self.model(
                noisy_latents,
                timestep=vt.expand(b, denoise_frames),
                clip_feature=clip_feas,
                y=y_future,
                context=prompt_embs,
                seq_len=denoise_seq_len,
                state=states.to(torch.bfloat16),
                embodiment_id=embodiment_ids,
                action=noisy_actions,
                timestep_action=at.expand(b, self.action_horizon),
                kv_cache=local_kv_cache,
                crossattn_cache=None,
                current_start_frame=1,
            )

            noisy_latents = sample_scheduler.step(
                model_output=video_pred.float(), timestep=vt, sample=noisy_latents.float(),
                step_index=step_idx, return_dict=False,
            )[0].to(latents.dtype)
            noisy_actions = sample_scheduler_action.step(
                model_output=action_pred.float(), timestep=at, sample=noisy_actions.float(),
                step_index=step_idx, return_dict=False,
            )[0].to(latents.dtype)

        return noisy_actions

    def predict_action(
        self,
        prompt_embs: torch.Tensor,
        latents: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
        num_inference_steps: int | None = None,
        observed_latent_frames: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        latents = latents.transpose(1, 2)
        _, _, num_lat_frames, lat_h, lat_w = latents.shape

        if observed_latent_frames is None:
            observed_latent_frames = num_lat_frames
        observed_latent_frames = max(1, min(observed_latent_frames, num_lat_frames))

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        self.reset_inference_state()
        return self._predict_action_stateless(
            prompt_embs=prompt_embs,
            latents=latents,
            clip_feas=clip_feas,
            ys=ys,
            states=states,
            embodiment_ids=embodiment_ids,
            num_inference_steps=num_inference_steps,
            observed_latent_frames=observed_latent_frames,
        )

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}{name}": param.detach().clone()
            for name, param in self.named_parameters()
        }


__all__ = ["DreamZeroHead"]
