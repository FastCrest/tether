"""SnapFlowPI0Pytorch — target_time-aware PI0Pytorch for SnapFlow distillation.

Replaces three v0.3 monkey-patches with proper method overrides:

  1. ``embed_suffix(state, x_t, t, target_time=None)`` — when ``target_time``
     is given, a zero-init learnable MLP maps ``sinusoidal(target_time)``
     to a correction vector added to the time channel before the
     action_time MLP. When ``target_time`` is None, behaves exactly like
     the parent (teacher inference path is unaffected).
  2. ``denoise_step`` does NOT ``copy.deepcopy(past_key_values)`` — the
     downstream ``paligemma_with_expert.forward`` is called with
     ``use_cache=False`` so past_kv is read-only; deepcopy is strictly
     defensive overhead (and breaks on graph-attached tensors during
     training — pytorch#103001).
  3. ``denoise_step`` does NOT force-cast ``suffix_out`` to fp32 — uses
     ``self.action_out_proj.weight.dtype`` instead, so bf16 training
     stays bf16 end-to-end.

## How to activate

    from tether.distill.snapflow_pi0_model import enable_snapflow
    enable_snapflow(student_policy.model)   # mutates in place

After the call:
  - ``student.model.__class__`` is the dynamically-built SnapFlowPI0Pytorch
    subclass of the installed lerobot ``PI0Pytorch``.
  - ``student.model.target_time_embed_mlp`` is registered as a submodule
    (zero-init), so ``student.parameters()`` includes its weights for
    the optimizer to update.
  - ``student.model.embed_suffix`` and ``.denoise_step`` accept the new
    ``target_time`` kwarg with default None (= backward compat).

The student's weights are NOT moved or copied; this is a pure behavioral
change on the same instance.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cached subclasses — built once per process at first enable_snapflow() call,
# which is when we can safely import lerobot.
_SNAPFLOW_PI0_CLASS: Any = None
_SNAPFLOW_PI05_CLASS: Any = None


def _detach_kv_cache_deep(past_kv: Any) -> Any:
    """Walk a KV cache and detach every tensor found, in place.

    Used by SnapFlowPI05Pytorch.denoise_step before a ``copy.deepcopy``
    that would otherwise raise on graph-attached tensors (pytorch#103001).
    Returns the same object with tensors detached.

    Handles:
      - bare tuples / lists / dicts
      - HF DynamicCache (key_cache + value_cache attrs)
      - any object with a __dict__ containing nested tensors
    """
    import torch

    def _walk(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        for attr in ("key_cache", "value_cache", "key_states", "value_states"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, _walk(getattr(obj, attr)))
                except AttributeError:
                    pass  # read-only property
        if hasattr(obj, "__dict__"):
            for k, v in list(obj.__dict__.items()):
                if isinstance(v, (torch.Tensor, list, tuple, dict)):
                    obj.__dict__[k] = _walk(v)
        return obj

    return _walk(past_kv)


def enable_snapflow(model: Any) -> None:
    """Mutate a pi-family Pytorch instance in place so it supports target_time.

    Dispatches by class:
      - PI0Pytorch  → SnapFlowPI0Pytorch  (state-aware embed_suffix)
      - PI05Pytorch → SnapFlowPI05Pytorch (no state, AdaRMS time conditioning)

    See module docstring for the three behavioral changes this enables.

    Args:
      model: an instance of either ``PI0Pytorch`` or ``PI05Pytorch``
        (typically ``policy.model`` after from_pretrained).

    Raises:
      TypeError: if ``model`` isn't one of the supported pi-family classes.
    """
    import torch
    import torch.nn as nn
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    if isinstance(model, PI0Pytorch):
        snapflow_cls = _resolve_snapflow_pi0_class()
        family = "pi0"
    elif isinstance(model, PI05Pytorch):
        snapflow_cls = _resolve_snapflow_pi05_class()
        family = "pi05"
    else:
        raise TypeError(
            f"enable_snapflow expects PI0Pytorch or PI05Pytorch; got {type(model).__name__}"
        )

    dim = model.action_in_proj.out_features
    hidden_dim = max(dim, 256)

    target_time_mlp = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, dim),
    )
    with torch.no_grad():
        target_time_mlp[-1].weight.zero_()
        target_time_mlp[-1].bias.zero_()

    ref_param = next(model.parameters())
    target_time_mlp = target_time_mlp.to(
        dtype=ref_param.dtype, device=ref_param.device,
    )

    model.add_module("target_time_embed_mlp", target_time_mlp)
    model.__class__ = snapflow_cls
    logger.info(
        "[snapflow] enabled target_time on %s (dim=%d, hidden=%d, init=zero)",
        family, dim, hidden_dim,
    )


def _resolve_snapflow_pi0_class() -> type:
    """Build (or return cached) SnapFlowPI0Pytorch subclass of lerobot's
    PI0Pytorch. Lazy so importing this module doesn't force lerobot import.
    """
    global _SNAPFLOW_PI0_CLASS
    if _SNAPFLOW_PI0_CLASS is not None:
        return _SNAPFLOW_PI0_CLASS

    import torch
    from lerobot.policies.pi0.modeling_pi0 import (
        PI0Pytorch,
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
    )

    class SnapFlowPI0Pytorch(PI0Pytorch):
        """PI0Pytorch + target_time embed_suffix + deepcopy-free denoise_step.

        Not constructed directly — produced by enable_snapflow() via
        __class__-swap on a loaded PI0Pytorch instance.
        """

        def embed_suffix(
            self,
            state,
            noisy_actions,
            timestep,
            target_time=None,
        ):
            """Parent embed_suffix + optional zero-init target_time contribution.

            When target_time is None, bit-exact with parent. When given a
            (B,) tensor, adds ``target_time_embed_mlp(sinusoidal(target_time))``
            to the time embedding before it's fused with the action embedding.
            """
            import torch.nn.functional as F

            embs = []
            pad_masks = []
            att_masks = []

            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]

            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=timestep.device,
            )
            time_emb = time_emb.type(dtype=timestep.dtype)

            if target_time is not None:
                tt = target_time
                if tt.ndim == 0:
                    tt = tt.expand(bsize)
                tt_emb = create_sinusoidal_pos_embedding(
                    tt,
                    self.action_in_proj.out_features,
                    min_period=self.config.min_period,
                    max_period=self.config.max_period,
                    device=tt.device,
                )
                tt_emb = tt_emb.type(dtype=time_emb.dtype)
                time_emb = time_emb + self.target_time_embed_mlp(tt_emb)

            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None

            embs.append(action_time_emb)
            bsize2, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(
                bsize2, action_time_dim, dtype=torch.bool, device=timestep.device,
            )
            pad_masks.append(action_time_mask)

            att_masks += [1] + ([0] * (self.config.chunk_size - 1))

            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
            att_masks = att_masks[None, :].expand(bsize2, len(att_masks))

            return embs, pad_masks, att_masks, adarms_cond

        def denoise_step(
            self,
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
            target_time=None,
        ):
            """Parent denoise_step minus the deepcopy and minus the fp32 cast.

            - Skips ``copy.deepcopy(past_key_values)``: safe because the
              downstream forward uses ``use_cache=False`` (read-only pass).
            - Skips the hardcoded ``suffix_out.to(fp32)``: casts to
              ``self.action_out_proj.weight.dtype`` instead, so bf16 training
              stays bf16.
            - Passes ``target_time`` through to ``embed_suffix``.
            """
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, timestep, target_time=target_time,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2,
            )

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        @torch.no_grad()
        def sample_actions_1step(
            self,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=None,
        ):
            """SnapFlow 1-NFE inference: single denoise_step at target_time=1.

            After SnapFlow distillation, the ``target_time_embed_mlp`` has
            learned to produce the 2-step-Euler shortcut velocity at
            target_time=1. This method runs exactly one denoise_step with
            time=1 (pure noise) and target_time=1 (request one-shot
            generation), then does one Euler step (dt=-1) to produce the
            action chunk.

            Analog of PI0Pytorch.sample_actions with num_steps=1 + the
            target_time kwarg threaded through. Parent prefix-cache setup
            is replicated verbatim; only the integration loop shrinks.
            """
            bsize = state.shape[0]
            device = state.device

            if noise is None:
                actions_shape = (
                    bsize, self.config.chunk_size, self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            time = torch.ones(bsize, dtype=torch.float32, device=device)
            # Single denoise call at time=1, target_time=1. Inputs match the
            # dtype of action_in_proj so embed_suffix doesn't hit a cast.
            action_dtype = self.action_in_proj.weight.dtype
            x_t = noise.to(action_dtype)
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                target_time=time,
            )
            # Euler step: x_{k+1} = x_k + dt * v_t, with dt=-1 and x_k=noise.
            # v_t from a well-distilled student approximates noise - action.
            return (x_t - v_t).to(noise.dtype)

    _SNAPFLOW_PI0_CLASS = SnapFlowPI0Pytorch
    return SnapFlowPI0Pytorch


def _resolve_snapflow_pi05_class() -> type:
    """Build (or return cached) SnapFlowPI05Pytorch subclass of lerobot's
    PI05Pytorch. Lazy so importing this module doesn't force lerobot import.

    Architectural differences vs pi0:
      - pi05 has NO state_proj. State is not in the suffix embedding.
      - pi05 uses AdaRMSNorm for time conditioning: time_emb flows through
        time_mlp_in → silu → time_mlp_out → silu, and the result becomes
        adarms_cond passed to the gemma_expert via the AdaRMS layer.
      - target_time injects ADDITIVELY to time_emb BEFORE the time_mlp
        (so the zero-init MLP starts identity-equivalent to teacher).
    """
    global _SNAPFLOW_PI05_CLASS
    if _SNAPFLOW_PI05_CLASS is not None:
        return _SNAPFLOW_PI05_CLASS

    import torch
    from lerobot.policies.pi05.modeling_pi05 import (
        PI05Pytorch,
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
    )

    class SnapFlowPI05Pytorch(PI05Pytorch):
        """PI05Pytorch + target_time embed_suffix + deepcopy-free denoise_step.

        Not constructed directly — produced by enable_snapflow() via
        __class__-swap on a loaded PI05Pytorch instance.
        """

        def embed_suffix(
            self,
            noisy_actions,
            timestep,
            target_time=None,
        ):
            """Parent embed_suffix + optional zero-init target_time contribution.

            target_time (when given) is sinusoidally encoded and routed
            through ``target_time_embed_mlp`` (zero-init), then ADDED to
            time_emb BEFORE the existing time_mlp_{in,out} chain. This way
            the AdaRMS conditioning vector (adarms_cond) carries the
            shortcut-velocity signal at target_time=1.
            """
            import torch.nn.functional as F

            embs = []
            pad_masks = []
            att_masks = []

            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=timestep.device,
            )
            time_emb = time_emb.type(dtype=timestep.dtype)

            if target_time is not None:
                tt = target_time
                if tt.ndim == 0:
                    tt = tt.expand(time_emb.shape[0])
                tt_emb = create_sinusoidal_pos_embedding(
                    tt,
                    self.action_in_proj.out_features,
                    min_period=self.config.min_period,
                    max_period=self.config.max_period,
                    device=tt.device,
                )
                # Cast to MLP's dtype (bf16 during training/inference),
                # not time_emb's dtype (which follows timestep, often fp32).
                # Otherwise bf16 model + fp32 timestep → F.linear dtype
                # mismatch at MLP's first Linear layer.
                mlp_dtype = self.target_time_embed_mlp[0].weight.dtype
                tt_emb = tt_emb.to(mlp_dtype)
                mlp_out = self.target_time_embed_mlp(tt_emb)
                time_emb = time_emb + mlp_out.to(time_emb.dtype)

            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

            embs.append(action_time_emb)
            bsize, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(
                bsize, action_time_dim, dtype=torch.bool, device=timestep.device,
            )
            pad_masks.append(action_time_mask)

            att_masks += [1] + ([0] * (self.config.chunk_size - 1))

            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
            att_masks = att_masks[None, :].expand(bsize, len(att_masks))

            return embs, pad_masks, att_masks, adarms_cond

        def denoise_step(
            self,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
            target_time=None,
            state=None,  # accepted for API compat with pi0 path; ignored
        ):
            """Parent denoise_step minus the fp32 cast; with DETACHED past_kv.

            Unlike the pi0 override (which keeps past_kv graph-attached so
            VLM gradients flow to the student), pi0.5's
            ``paligemma_with_expert.forward`` needs ``past_key_values`` in a
            state the DynamicCache can inspect — in practice this means it
            must be re-parseable as a cache object, which requires going
            through a ``copy.deepcopy`` that fails on graph-attached
            tensors (pytorch#103001).

            Workaround: detach-in-place + deepcopy before the forward.
            Consequence: pi0.5 student trains action expert + projections +
            target_time_embed_mlp, but VLM weights stay frozen (same as
            pi0 in v0.3 pre-v0.3.1). Full-VLM training is v0.3.2 work.

            ``state`` kwarg is accepted for cross-family API compatibility
            but unused — pi0.5 doesn't put state in the suffix embedding.
            """
            import copy

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep, target_time=target_time,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2,
            )

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            # Detach past_kv tensors so deepcopy succeeds on graph-attached
            # cache objects. Walks the cache recursively.
            past_key_values = _detach_kv_cache_deep(past_key_values)
            past_key_values = copy.deepcopy(past_key_values)

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        @torch.no_grad()
        def sample_actions_1step(
            self,
            images,
            img_masks,
            tokens,
            masks,
            noise=None,
        ):
            """SnapFlow 1-NFE inference for pi0.5 (no state argument).

            Mirror of PI05Pytorch.sample_actions but with num_steps=1 and
            target_time=1 threaded into the single denoise_step call.
            """
            bsize = tokens.shape[0]
            device = tokens.device

            if noise is None:
                actions_shape = (
                    bsize, self.config.chunk_size, self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            time = torch.ones(bsize, dtype=torch.float32, device=device)
            action_dtype = self.action_in_proj.weight.dtype
            x_t = noise.to(action_dtype)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                target_time=time,
            )
            return (x_t - v_t).to(noise.dtype)

    _SNAPFLOW_PI05_CLASS = SnapFlowPI05Pytorch
    return SnapFlowPI05Pytorch


# ─────────────────────────────────────────────────────────────────────
# v0.5 — StateOut variant: student takes proprio state as an explicit
# input (via state_proj), instead of pi0.5's default state-in-language
# tokenization. Unlocks the prefix KV cache in production because
# lang_tokens becomes stable per episode (only task description, no
# drifting state tokens). Design: ``reflex_vla/01_architecture/
# distill_state_out_pi05_design.md``. Status: scaffolding as of
# 2026-04-22 — distill training pipeline + preprocessor override to
# strip state from lang are the remaining work.
# ─────────────────────────────────────────────────────────────────────

_SNAPFLOW_PI05_STATE_OUT_CLASS: Any = None


def _resolve_snapflow_pi05_state_out_class() -> type:
    """Build the SnapFlowPI05StateOutPytorch subclass.

    Extends SnapFlowPI05Pytorch with:
      - embed_suffix that accepts explicit ``state`` tensor and prepends
        ``state_proj(state)`` to the suffix embedding (pi0 pattern),
      - denoise_step that threads state through,
      - sample_actions_1step with state as a first-class arg.

    Requires ``enable_snapflow_state_out(model)`` to have been called
    first so that ``model.state_proj`` is a registered submodule.
    """
    global _SNAPFLOW_PI05_STATE_OUT_CLASS
    if _SNAPFLOW_PI05_STATE_OUT_CLASS is not None:
        return _SNAPFLOW_PI05_STATE_OUT_CLASS

    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi05.modeling_pi05 import (
        create_sinusoidal_pos_embedding,
        make_att_2d_masks,
    )

    parent = _resolve_snapflow_pi05_class()

    class SnapFlowPI05StateOutPytorch(parent):
        """v0.5 StateOut student. Inherits target_time machinery from
        SnapFlowPI05Pytorch; adds explicit state input via state_proj.

        Design note (2026-04-22 v2): the additive-bias injection (v1)
        gave the action expert a too-weak conditioning signal — the
        student couldn't extract per-position state info from a broadcast
        bias. SnapFlow self-distillation (Eq. 11) had no teacher term to
        push state_proj to learn either, so the student converged to
        "ignore state" and LIBERO eval was 0/4.

        v2 reverts to pi0's PREPEND-AS-TOKEN pattern. Suffix is now
        (B, chunk_size + 1, dim): state token at position 0, then 50
        action tokens. Attention can selectively read state. Earlier
        attempt at prepend hit attention-mask shape errors that turned
        out to be from gradient_checkpointing forcing use_cache=False
        (separate bug, fixed in commit 88cfe7b). With gc disabled the
        prepend works.
        """

        def embed_suffix(  # type: ignore[override]
            self,
            noisy_actions,
            timestep,
            target_time=None,
            state=None,
        ):
            """Pi0 pattern: prepend state_proj(state) as a sequence
            token at position 0 of the suffix. Suffix shape becomes
            (B, chunk_size + 1, dim). State MUST be provided."""
            if state is None:
                raise ValueError(
                    "SnapFlowPI05StateOutPytorch.embed_suffix requires state"
                )
            if state.dtype != self.state_proj.weight.dtype:
                state = state.to(self.state_proj.weight.dtype)

            state_emb = self.state_proj(state)  # (B, dim)

            # Run parent to get the chunk_size-length suffix (action_time_emb).
            embs_no_state, pad_masks_no_state, att_masks_no_state, adarms_cond = (
                super().embed_suffix(noisy_actions, timestep, target_time=target_time)
            )

            bsize = state_emb.shape[0]
            device = state_emb.device

            # Prepend state as token 0 of the suffix.
            state_token = state_emb[:, None, :].to(embs_no_state.dtype)  # (B, 1, dim)
            state_pad_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            state_att_flag = torch.ones(
                bsize, 1, dtype=att_masks_no_state.dtype, device=device,
            )

            embs = torch.cat([state_token, embs_no_state], dim=1)
            pad_masks = torch.cat([state_pad_mask, pad_masks_no_state], dim=1)
            att_masks = torch.cat([state_att_flag, att_masks_no_state], dim=1)

            return embs, pad_masks, att_masks, adarms_cond

        def denoise_step(  # type: ignore[override]
            self,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
            target_time=None,
            state=None,
        ):
            """Parent denoise_step but with state threaded through
            embed_suffix instead of the ignored-state pi0.5 default."""
            import copy

            if state is None:
                raise ValueError(
                    "SnapFlowPI05StateOutPytorch.denoise_step requires state"
                )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep, target_time=target_time, state=state,
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len,
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2,
            )

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            past_key_values = _detach_kv_cache_deep(past_key_values)
            past_key_values = copy.deepcopy(past_key_values)

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        @torch.no_grad()
        def sample_actions_1step(  # type: ignore[override]
            self,
            images,
            img_masks,
            tokens,
            masks,
            state,
            noise=None,
        ):
            """SnapFlow 1-NFE inference for the state-out student. Note:
            signature now matches the pi0 path (state is a required arg).
            `tokens` and `masks` should be the task-description-only
            tokenization — strip state tokens at the harness / preprocessor
            layer; this class doesn't re-tokenize."""
            bsize = tokens.shape[0]
            device = tokens.device

            if noise is None:
                actions_shape = (
                    bsize, self.config.chunk_size, self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks,
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            time = torch.ones(bsize, dtype=torch.float32, device=device)
            action_dtype = self.action_in_proj.weight.dtype
            x_t = noise.to(action_dtype)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
                target_time=time,
                state=state,
            )
            return (x_t - v_t).to(noise.dtype)

    _SNAPFLOW_PI05_STATE_OUT_CLASS = SnapFlowPI05StateOutPytorch
    return SnapFlowPI05StateOutPytorch


def enable_snapflow_state_out(
    model: Any,
    state_dim: int | None = None,
    warm_init_from_pi0: str | None = None,
) -> None:
    """Add target_time_embed_mlp + state_proj and swap to the StateOut class.

    Mirrors ``enable_snapflow`` but for pi0.5 only, with two extra steps:
    1. Registers ``state_proj`` = Linear(state_dim → expert_hidden_dim).
    2. Optionally warm-initializes state_proj weights from a pretrained
       pi0 checkpoint (matches the pi0 architecture's known-working
       state encoding instead of small-random init).

    Args:
        model: PI05Pytorch instance.
        state_dim: proprio state dim. Defaults to model.config.max_state_dim.
        warm_init_from_pi0: HF repo id of a pi0 checkpoint to copy
            state_proj.weight + bias from. Pi0 and pi0.5 share PaliGemma
            backbone width (1024), so the Linear(32, 1024) weights are
            shape-compatible. Setting this avoids the gradient-bootstrap
            problem (state_proj starts at known-good values trained on
            real robot state, not 0.02-std noise).

            Recommended: 'lerobot/pi0_libero_finetuned_v044' for LIBERO,
            'lerobot/pi0_base' for general.
    """
    import torch
    import torch.nn as nn
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    if not isinstance(model, PI05Pytorch):
        raise TypeError(
            f"enable_snapflow_state_out expects PI05Pytorch; got {type(model).__name__}"
        )

    # Step 1: install target_time_embed_mlp + swap to SnapFlowPI05Pytorch.
    enable_snapflow(model)

    # Step 2: add state_proj. Default init = small-normal; if warm_init_from_pi0
    # is set, replace with weights from that pi0 checkpoint.
    state_dim = state_dim or getattr(model.config, "max_state_dim", None)
    if state_dim is None:
        raise ValueError(
            "state_dim not provided and model.config.max_state_dim missing"
        )
    expert_hidden = model.action_in_proj.out_features
    state_proj = nn.Linear(state_dim, expert_hidden)
    with torch.no_grad():
        state_proj.weight.normal_(mean=0.0, std=0.02)
        state_proj.bias.zero_()

    if warm_init_from_pi0:
        try:
            from huggingface_hub import snapshot_download
            import safetensors.torch as st
            from pathlib import Path
            pi0_dir = snapshot_download(warm_init_from_pi0)
            sf_path = Path(pi0_dir) / "model.safetensors"
            tensors = st.load_file(str(sf_path))
            # lerobot pi0 saves state_proj as model.state_proj.{weight,bias}
            w_key = "model.state_proj.weight"
            b_key = "model.state_proj.bias"
            if w_key in tensors and b_key in tensors:
                w = tensors[w_key]
                b = tensors[b_key]
                if w.shape == state_proj.weight.shape and b.shape == state_proj.bias.shape:
                    with torch.no_grad():
                        state_proj.weight.copy_(w)
                        state_proj.bias.copy_(b)
                    logger.info(
                        "[snapflow-state-out] warm-initialized state_proj from %s "
                        "(weight norm: %.4f, vs small-init norm: %.4f)",
                        warm_init_from_pi0,
                        float(state_proj.weight.norm()),
                        float(state_dim) ** 0.5 * 0.02,
                    )
                else:
                    logger.warning(
                        "[snapflow-state-out] warm_init shape mismatch (pi0 %s vs pi0.5 %s); "
                        "using small-random init instead",
                        tuple(w.shape), tuple(state_proj.weight.shape),
                    )
            else:
                logger.warning(
                    "[snapflow-state-out] state_proj.weight not found in %s "
                    "(keys: %d); using small-random init",
                    warm_init_from_pi0, len(tensors),
                )
        except Exception as e:
            logger.warning(
                "[snapflow-state-out] warm_init failed (%s); using small-random init",
                e,
            )

    ref_param = next(model.parameters())
    state_proj = state_proj.to(dtype=ref_param.dtype, device=ref_param.device)
    model.add_module("state_proj", state_proj)

    # Step 3: swap class to the StateOut variant (subclass of SnapFlowPI05).
    state_out_cls = _resolve_snapflow_pi05_state_out_class()
    model.__class__ = state_out_cls
    logger.info(
        "[snapflow-state-out] enabled on pi05 (state_dim=%d, expert_hidden=%d, warm_init=%s)",
        state_dim, expert_hidden, warm_init_from_pi0 or "small-random",
    )


def load_snapflow_student(checkpoint_path: Any) -> Any:
    """Load a SnapFlow-distilled student checkpoint from a tether-saved dir.

    The backend's ``_save_student_checkpoint`` dumps the student via
    ``PI05Policy.save_pretrained(...)``, which includes the
    ``target_time_embed_mlp.*`` keys. Plain ``PI05Policy.from_pretrained``
    may reject those keys (strict=True default).

    This loader:

      1. Reads ``model.safetensors``, splits off the
         ``model.target_time_embed_mlp.*`` keys into a separate dict.
      2. Writes the base-only safetensors (+ copies sibling config files)
         into a temp dir, loads via ``PI05Policy.from_pretrained``
         (or PI0 if that's the teacher family).
      3. Calls ``enable_snapflow(policy.model)`` to attach the target_time
         machinery + an empty target_time_embed_mlp submodule.
      4. Loads the extracted MLP weights into that submodule.

    Callers then typically do ``.eval().to(device).to(dtype)`` as usual,
    and invoke ``policy.model.sample_actions_1step(...)`` for 1-NFE
    inference or ``policy.model.sample_actions(...)`` for multi-step.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    path = Path(checkpoint_path)
    if not path.exists():
        # Fallback: treat as HF repo id and snapshot_download. Lets callers
        # point at, e.g., `lerobot/pi05_libero_finetuned_v044` for teacher-path
        # ONNX eval where the base-model policy (no SnapFlow MLP) is
        # acceptable — mlp_state will be empty, enable_snapflow attaches a
        # zero-init MLP that contributes zero to the forward, and inference
        # runs via the ORT session on the caller's ONNX anyway.
        s = str(checkpoint_path)
        looks_hf = "/" in s and not s.startswith(("./", "/", "~", ".."))
        if looks_hf:
            try:
                from huggingface_hub import snapshot_download
                logger.info("[load_snapflow_student] snapshot_download(%s)", s)
                path = Path(snapshot_download(s))
            except Exception as e:
                raise FileNotFoundError(
                    f"SnapFlow student checkpoint not found at {path} and "
                    f"HF snapshot_download for {s!r} failed: {e}"
                )
        else:
            raise FileNotFoundError(
                f"SnapFlow student checkpoint not found at {path}. Expected a "
                f"tether-saved pretrained_model/ dir with model.safetensors + "
                f"config.json, or an HF repo id."
            )

    sf_path = path / "model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(f"missing model.safetensors in {path}")

    full_state = load_file(str(sf_path))
    mlp_prefix = "model.target_time_embed_mlp."
    state_proj_prefix = "model.state_proj."
    base_state = {
        k: v for k, v in full_state.items()
        if not k.startswith(mlp_prefix) and not k.startswith(state_proj_prefix)
    }
    mlp_state = {
        k[len(mlp_prefix):]: v
        for k, v in full_state.items() if k.startswith(mlp_prefix)
    }
    state_proj_state = {
        k[len(state_proj_prefix):]: v
        for k, v in full_state.items() if k.startswith(state_proj_prefix)
    }

    # State-out variant (v0.5+) saves model.state_proj.{weight,bias} keys.
    # The base PI05Policy doesn't have a state_proj submodule, so we strip
    # these from base_state and route through enable_snapflow_state_out.
    is_state_out_student = bool(state_proj_state)

    policy_cls = _dispatch_policy_class(path)

    with tempfile.TemporaryDirectory(prefix="snapflow_load_") as td:
        td_path = Path(td)
        for f in path.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy(f, td_path / f.name)
        save_file(base_state, str(td_path / "model.safetensors"))
        # Strip tether-distill provenance keys from config.json — PI05Config
        # (via draccus) is strict about unknown fields and rejects them at
        # load time. The fields are written by _save_student_checkpoint as
        # provenance metadata; not needed for inference.
        import json
        cfg_path = td_path / "config.json"
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg_dict = json.load(f)
            stripped = {k: v for k, v in cfg_dict.items() if not k.startswith("_reflex_")}
            if stripped != cfg_dict:
                with cfg_path.open("w") as f:
                    json.dump(stripped, f, indent=2)
        policy = policy_cls.from_pretrained(str(td_path))

    if is_state_out_student:
        enable_snapflow_state_out(policy.model)
        logger.info(
            "[load_snapflow_student] detected state-out variant — enabled "
            "state_proj submodule (will load %d keys)",
            len(state_proj_state),
        )
    else:
        enable_snapflow(policy.model)

    if not mlp_state:
        logger.warning(
            "[load_snapflow_student] no target_time_embed_mlp weights in %s; "
            "loaded as zero-init student (behaves like teacher at target_time=1). "
            "If this is a trained distill output, the checkpoint is corrupt.",
            path,
        )
        if is_state_out_student and state_proj_state:
            policy.model.state_proj.load_state_dict(state_proj_state, strict=True)
            logger.info("[load_snapflow_student] loaded state_proj weights")
        return policy

    missing, unexpected = policy.model.target_time_embed_mlp.load_state_dict(
        mlp_state, strict=True,
    )
    if missing:
        logger.warning("[load_snapflow_student] missing mlp keys: %s", missing)
    if unexpected:
        logger.warning("[load_snapflow_student] unexpected mlp keys: %s", unexpected)

    if is_state_out_student and state_proj_state:
        sp_missing, sp_unexpected = policy.model.state_proj.load_state_dict(
            state_proj_state, strict=True,
        )
        if sp_missing:
            logger.warning("[load_snapflow_student] missing state_proj keys: %s", sp_missing)
        if sp_unexpected:
            logger.warning("[load_snapflow_student] unexpected state_proj keys: %s", sp_unexpected)

    logger.info("[load_snapflow_student] loaded student from %s", path)
    return policy


def _dispatch_policy_class(checkpoint_path: Any):
    """Pick PI0Policy vs PI05Policy from checkpoint config."""
    import json
    from pathlib import Path

    cfg_path = Path(checkpoint_path) / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        ptype = cfg.get("type") or cfg.get("_tether_distill_teacher_type", "")
        if ptype == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            return PI05Policy
        if ptype == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            return PI0Policy
    # Fallback: try pi0.5 first (current distill target), then pi0.
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy
    except ImportError:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy


__all__ = ["enable_snapflow", "load_snapflow_student"]
