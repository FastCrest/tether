"""Tests for Pi0VLA — pi0 composition class on the BaseVLA spine.

Lift #1 Day 4f per `features/03_export/basevla-spine_plan.md`. Validates
the composition shape:

- registration on the VLAS registry
- slot declarations (REQUIRED_SLOTS, OPTIONAL_SLOTS, NAME_MAPPING)
- construction via from_config (the spine's primary path)
- predict_action raises NotImplementedError per Day 4g deferral
- forward routes to llm_backbone

Day 4g adds the full inference pipeline + parity test vs the legacy
pi0_exporter. This file tests the composition shape only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from reflex.models.base_vla import BaseVLA
from reflex.models.heads import VLAHead
from reflex.models.llm import LLMBackbone
from reflex.models.projectors import Projector
from reflex.models.vision import VisionBackbone
from reflex.models.vlas.pi0 import Pi0VLA
from reflex.registry.components import VLAS


# ─── Registration + slot declarations ───────────────────────────────────


def test_pi0_vla_registered():
    assert "Pi0VLA" in VLAS
    assert VLAS.get("Pi0VLA") is Pi0VLA


def test_pi0_vla_is_basevla_subclass():
    assert issubclass(Pi0VLA, BaseVLA)


def test_pi0_vla_required_slots():
    """Pi0VLA declares 4 required slots: vision/llm/projector/head.
    vlm_backbone + text_encoder unused."""
    assert Pi0VLA.REQUIRED_SLOTS == (
        "vision_backbone", "llm_backbone", "projector", "vla_head",
    )
    assert Pi0VLA.OPTIONAL_SLOTS == ()


def test_pi0_vla_name_mapping_default_empty():
    """Decision S-1 — empty NAME_MAPPING is the v1 default (the lerobot/pi0_base
    checkpoint's keys map directly to component slots via load_state_dict's
    slot-prefix routing)."""
    assert Pi0VLA.NAME_MAPPING == {}


# ─── Construction via direct kwargs (the test path) ─────────────────────


def test_pi0_vla_constructs_with_4_stub_components():
    vla = Pi0VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    assert isinstance(vla.vision_backbone, _StubVision)
    assert isinstance(vla.llm_backbone, _StubLLM)
    assert isinstance(vla.projector, _StubProjector)
    assert isinstance(vla.vla_head, _StubHead)
    # Optional slots stay None
    assert vla.vlm_backbone is None
    assert vla.text_encoder is None


def test_pi0_vla_missing_required_slot_raises():
    """Per BaseVLA contract — missing required slot is a ValueError at
    construction."""
    with pytest.raises(ValueError, match="missing required slot"):
        Pi0VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            projector=_StubProjector(),
            # vla_head missing
        )


def test_pi0_vla_undeclared_slot_raises():
    """Per BaseVLA — passing vlm_backbone (not in REQUIRED + OPTIONAL) raises."""
    with pytest.raises(ValueError, match="undeclared"):
        Pi0VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            projector=_StubProjector(),
            vla_head=_StubHead(),
            vlm_backbone=_StubVision(),  # not in Pi0VLA's slots
        )


# ─── Construction via from_config (Registry path) ───────────────────────


def test_pi0_vla_from_config_with_prebuilt_instances():
    """from_config accepts pre-built component instances directly (the test
    path; from_pretrained handles the full HF load)."""
    vla = Pi0VLA.from_config({
        "vision_backbone": _StubVision(),
        "llm_backbone": _StubLLM(),
        "projector": _StubProjector(),
        "vla_head": _StubHead(),
    })
    assert isinstance(vla, Pi0VLA)
    assert vla.vision_backbone is not None


# ─── Forward routing ────────────────────────────────────────────────────


def test_forward_routes_to_llm_backbone():
    """forward(batch) calls llm_backbone with inputs_embeds + attention_mask
    + past_key_values."""
    stub_llm = _StubLLM()
    vla = Pi0VLA(
        vision_backbone=_StubVision(),
        llm_backbone=stub_llm,
        projector=_StubProjector(),
        vla_head=_StubHead(),
    )
    embeds = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = vla.forward({
        "inputs_embeds": embeds,
        "attention_mask": mask,
        "past_key_values": None,
    })
    assert stub_llm.last_call["inputs_embeds"] is embeds
    assert stub_llm.last_call["attention_mask"] is mask
    assert out.last_hidden_state.shape == (1, 5, 8)


# ─── predict_action — Day 4g full inference pipeline ───────────────────


def test_predict_action_runs_end_to_end_with_stubs():
    """Day 4g: predict_action wires SigLIP → multi_modal_projector → text
    embed → concat prefix → language_model prefill → denoise loop.

    Validated here with stub components that simulate the right tensor
    shapes. The shape-checked happy path. Numerical parity vs the lerobot
    reference path is the Day 4h Modal-fired test against the real
    `lerobot/pi0_base` checkpoint."""
    batch, chunk_size, action_dim, text_hidden = 1, 50, 32, 2048
    img_tokens, seq_len = 256, 16
    num_layers, nkv, head_dim = 2, 1, 256
    expert_hidden = 1024

    vla = Pi0VLA(
        vision_backbone=_StubVisionForPi0(img_tokens=img_tokens, hidden=1152),
        llm_backbone=_StubPaliGemmaForPi0(
            text_hidden=text_hidden, num_layers=num_layers, nkv=nkv, head_dim=head_dim,
        ),
        projector=_StubStateProjector(in_dim=action_dim, out_dim=expert_hidden),
        vla_head=_StubPrefixHead(num_layers=num_layers, chunk_size=chunk_size,
                                 action_dim=action_dim, expert_hidden=expert_hidden),
    )

    images = [torch.randn(batch, 3, 224, 224) for _ in range(3)]
    state = torch.randn(batch, action_dim)
    lang_tokens = torch.randint(0, 100, (batch, seq_len), dtype=torch.long)
    lang_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    noise = torch.randn(batch, chunk_size, action_dim)

    actions = vla.predict_action(
        images=images, state=state, lang_tokens=lang_tokens, lang_masks=lang_masks,
        noise=noise, num_steps=2, chunk_size=chunk_size,
    )
    assert actions.shape == (batch, chunk_size, action_dim)
    # Each denoise step changes x_t by `dt * v_t` (Euler) — assert it's not the
    # pristine input (proves the loop ran).
    assert not torch.allclose(actions, noise)


def test_predict_action_raises_if_required_slot_missing():
    """RuntimeError if a required slot was somehow cleared between
    construction and predict_action call (defensive — REQUIRED_SLOTS
    prevents the bad construction path)."""
    vla = Pi0VLA(
        vision_backbone=_StubVisionForPi0(),
        llm_backbone=_StubPaliGemmaForPi0(),
        projector=_StubProjector(),
        vla_head=_StubPrefixHead(),
    )
    vla.vision_backbone = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="vision_backbone is None"):
        vla.predict_action(
            images=[torch.randn(1, 3, 224, 224)],
            state=torch.zeros(1, 32),
            lang_tokens=torch.zeros(1, 4, dtype=torch.long),
            lang_masks=torch.ones(1, 4, dtype=torch.bool),
        )


# ─── Helpers ────────────────────────────────────────────────────────────


class _StubVision(VisionBackbone):
    def forward(self, images): return images


class _StubLLM(LLMBackbone, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.last_call: dict = {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *args,
        inputs_embeds=None,
        past_key_values=None,
        **kwargs,
    ):
        self.last_call = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _StubProjector(Projector):
    def forward(self, x, *args, **kwargs): return x


class _StubHead(VLAHead):
    def forward(self, context, *args, **kwargs): return context


# ─── Day 4g predict_action stubs ───────────────────────────────────────


class _StubVisionForPi0(VisionBackbone, nn.Module):
    """SigLIP-shaped stub: [B, 3, H, W] → [B, img_tokens, hidden].

    Uses an avg-pool + small linear to keep parameter count tiny — a naive
    `Linear(3*224*224, img_tokens*hidden)` is 44B params (176 GB) which OOMs
    the test process. The output shape contract matches real SigLIP."""

    def __init__(self, img_tokens: int = 256, hidden: int = 1152):
        nn.Module.__init__(self)
        self.img_tokens = img_tokens
        self.hidden = hidden
        self.proj = nn.Linear(3, hidden)  # tiny per-pixel projection

    def forward(self, images):
        b = images.shape[0]
        # Downsample to img_tokens tokens via adaptive pool, then per-token linear.
        # AdaptiveAvgPool2d → [B, 3, sqrt(img_tokens), sqrt(img_tokens)]
        side = int(self.img_tokens ** 0.5)
        pooled = nn.functional.adaptive_avg_pool2d(images, (side, side))
        tokens = pooled.permute(0, 2, 3, 1).reshape(b, side * side, 3)
        return self.proj(tokens)


class _StubPaliGemmaForPi0(LLMBackbone, nn.Module):
    """Stub PaliGemma exposing the property accessors Pi0VLA.predict_action
    relies on. Returns past_key_values that look like a transformers Cache."""

    def __init__(
        self,
        text_hidden: int = 2048,
        vision_hidden: int = 1152,
        vocab: int = 1024,  # small vocab — test only embeds tokens in [0, 100)
        num_layers: int = 2,
        nkv: int = 1,
        head_dim: int = 256,
    ):
        nn.Module.__init__(self)
        self.multi_modal_projector = nn.Linear(vision_hidden, text_hidden)
        self.embed_tokens = nn.Embedding(vocab, text_hidden)
        self.text_hidden_size = text_hidden
        self.num_layers = num_layers
        self.nkv = nkv
        self.head_dim = head_dim
        # Pi0VLA.predict_action toggles `language_model.config._attn_implementation`
        # to "eager" around the prefill — stub the read/write target.
        self.language_model = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="eager"),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *args,
        inputs_embeds=None,
        past_key_values=None,
        position_ids=None,
        use_cache=False,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        b, seq_len, _ = inputs_embeds.shape
        kv_shape = (b, self.nkv, seq_len, self.head_dim)
        pkv = SimpleNamespace(
            layers=[
                SimpleNamespace(keys=torch.randn(*kv_shape), values=torch.randn(*kv_shape))
                for _ in range(self.num_layers)
            ],
        )
        return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=pkv)


class _StubStateProjector(Projector, nn.Module):
    """Stub state projector: [B, action_dim] → [B, expert_hidden]. Matches
    lerobot's state_proj which maps max_state_dim → action_expert_config.width."""

    def __init__(self, in_dim: int = 32, out_dim: int = 1024):
        nn.Module.__init__(self)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, *args, **kwargs):
        return self.linear(x)


class _StubPrefixHead(VLAHead, nn.Module):
    """Stub FlowMatchingHead that simulates a prefix-aware expert. Returns
    a velocity tensor shaped like the noisy actions input."""

    def __init__(self, num_layers: int = 2, chunk_size: int = 50, action_dim: int = 32,
                 expert_hidden: int = 1024):
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.expert_hidden = expert_hidden
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, noisy_actions, timestep=None, position_ids=None, *,
                prefix_k=None, prefix_v=None, state_emb=None, attn_mask=None,
                **kwargs):
        if prefix_k is None or prefix_v is None:
            raise ValueError("prefix-aware stub head requires prefix_k + prefix_v")
        if state_emb is None:
            raise ValueError("prefix-aware stub head requires state_emb (suffix's first token)")
        if attn_mask is None:
            raise ValueError("prefix-aware stub head requires attn_mask (lerobot block pattern)")
        assert prefix_k.shape[0] == self.num_layers
        assert state_emb.shape[-1] == self.expert_hidden, (
            f"state_emb hidden ({state_emb.shape[-1]}) != expert_hidden ({self.expert_hidden})"
        )
        # position_ids should be suffix-shaped: [B, chunk_size+1]
        assert position_ids.shape[-1] == noisy_actions.shape[1] + 1
        # attn_mask should be [B, 1, chunk_size+1, prefix_len + chunk_size + 1]
        assert attn_mask.ndim == 4
        assert attn_mask.shape[2] == noisy_actions.shape[1] + 1
        return noisy_actions * self.scale
