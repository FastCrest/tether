"""Tests for Pi05VLA — pi0.5 composition class on the BaseVLA spine.

Lift #1 Day 5 Phase A per `features/03_export/basevla-spine_plan.md`. Validates
the composition shape (mirroring Pi0VLA tests):

- registration on the VLAS registry
- slot declarations (REQUIRED_SLOTS, OPTIONAL_SLOTS, NAME_MAPPING)
- construction via from_config (the spine's primary path)
- predict_action raises NotImplementedError (Phase B will land it)
- forward routes to llm_backbone

Phase B adds the full inference pipeline + parity test vs lerobot PI05Policy.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tether.models.base_vla import BaseVLA
from tether.models.heads import VLAHead
from tether.models.llm import LLMBackbone
from tether.models.vision import VisionBackbone
from tether.models.vlas.pi05 import Pi05VLA
from tether.registry.components import VLAS


# ─── Registration + slot declarations ───────────────────────────────────


def test_pi05_vla_registered():
    assert "Pi05VLA" in VLAS
    assert VLAS.get("Pi05VLA") is Pi05VLA


def test_pi05_vla_is_basevla_subclass():
    assert issubclass(Pi05VLA, BaseVLA)


def test_pi05_vla_required_slots():
    """Pi05VLA declares 3 required slots: vision/llm/head.
    projector + vlm_backbone + text_encoder unused (state-in-language)."""
    assert Pi05VLA.REQUIRED_SLOTS == ("vision_backbone", "llm_backbone", "vla_head")
    assert Pi05VLA.OPTIONAL_SLOTS == ()


def test_pi05_vla_name_mapping_default_empty():
    """Decision S-1 — empty NAME_MAPPING is the v1 default."""
    assert Pi05VLA.NAME_MAPPING == {}


# ─── Construction via direct kwargs (the test path) ─────────────────────


def test_pi05_vla_constructs_with_3_stub_components():
    vla = Pi05VLA(
        vision_backbone=_StubVision(),
        llm_backbone=_StubLLM(),
        vla_head=_StubHead(),
    )
    assert isinstance(vla.vision_backbone, _StubVision)
    assert isinstance(vla.llm_backbone, _StubLLM)
    assert isinstance(vla.vla_head, _StubHead)
    # Unused slots stay None
    assert vla.projector is None
    assert vla.vlm_backbone is None
    assert vla.text_encoder is None


def test_pi05_vla_missing_required_slot_raises():
    with pytest.raises(ValueError, match="missing required slot"):
        Pi05VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            # vla_head missing
        )


def test_pi05_vla_undeclared_slot_raises():
    """Per BaseVLA — passing vlm_backbone (not in REQUIRED + OPTIONAL) raises."""
    with pytest.raises(ValueError, match="undeclared"):
        Pi05VLA(
            vision_backbone=_StubVision(),
            llm_backbone=_StubLLM(),
            vla_head=_StubHead(),
            vlm_backbone=_StubVision(),
        )


# ─── Construction via from_config ───────────────────────────────────────


def test_pi05_vla_from_config_with_prebuilt_instances():
    vla = Pi05VLA.from_config({
        "vision_backbone": _StubVision(),
        "llm_backbone": _StubLLM(),
        "vla_head": _StubHead(),
    })
    assert isinstance(vla, Pi05VLA)
    assert vla.vision_backbone is not None


# ─── Forward routing ────────────────────────────────────────────────────


def test_forward_routes_to_llm_backbone():
    stub_llm = _StubLLM()
    vla = Pi05VLA(
        vision_backbone=_StubVision(),
        llm_backbone=stub_llm,
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
    assert out.last_hidden_state.shape == (1, 5, 8)


# ─── predict_action — Day 5 Phase B full inference ─────────────────────


def test_predict_action_runs_end_to_end_with_stubs():
    """Day 5 Phase B: predict_action wires SigLIP → multi_modal_projector →
    text embed (with state-in-language) → concat prefix → PaliGemma prefill →
    pi0.5 AdaRMSNorm denoise loop.

    Numerical parity vs lerobot is Day 5 Phase B's Modal-fired test against
    the real `lerobot/pi05_libero_finetuned_v044` checkpoint."""
    batch, chunk_size, action_dim, text_hidden = 1, 50, 32, 2048
    img_tokens, seq_len = 256, 16
    num_layers, nkv, head_dim = 2, 1, 256

    vla = Pi05VLA(
        vision_backbone=_StubVisionForPi05(img_tokens=img_tokens, hidden=1152),
        llm_backbone=_StubPaliGemmaForPi05(
            text_hidden=text_hidden, num_layers=num_layers, nkv=nkv, head_dim=head_dim,
        ),
        vla_head=_StubPi05Head(num_layers=num_layers, chunk_size=chunk_size, action_dim=action_dim),
    )

    images = [torch.randn(batch, 3, 224, 224) for _ in range(3)]
    lang_tokens = torch.randint(0, 100, (batch, seq_len), dtype=torch.long)
    lang_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    noise = torch.randn(batch, chunk_size, action_dim)

    actions = vla.predict_action(
        images=images, lang_tokens=lang_tokens, lang_masks=lang_masks,
        noise=noise, num_steps=2, chunk_size=chunk_size, action_dim=action_dim,
    )
    assert actions.shape == (batch, chunk_size, action_dim)
    assert not torch.allclose(actions, noise)


def test_predict_action_raises_if_required_slot_missing():
    vla = Pi05VLA(
        vision_backbone=_StubVisionForPi05(),
        llm_backbone=_StubPaliGemmaForPi05(),
        vla_head=_StubPi05Head(),
    )
    vla.vision_backbone = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="vision_backbone is None"):
        vla.predict_action(
            images=[torch.randn(1, 3, 224, 224)],
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


class _StubHead(VLAHead):
    def forward(self, context, *args, **kwargs): return context


# ─── Day 5 Phase B predict_action stubs ──────────────────────────────


class _StubVisionForPi05(VisionBackbone, nn.Module):
    """SigLIP-shaped: [B, 3, H, W] → [B, img_tokens, hidden]. Uses pool +
    small linear to keep param count tiny."""

    def __init__(self, img_tokens: int = 256, hidden: int = 1152):
        nn.Module.__init__(self)
        self.img_tokens = img_tokens
        self.hidden = hidden
        self.proj = nn.Linear(3, hidden)

    def forward(self, images):
        b = images.shape[0]
        side = int(self.img_tokens ** 0.5)
        pooled = nn.functional.adaptive_avg_pool2d(images, (side, side))
        tokens = pooled.permute(0, 2, 3, 1).reshape(b, side * side, 3)
        return self.proj(tokens)


class _StubPaliGemmaForPi05(LLMBackbone, nn.Module):
    """Stub PaliGemma — exposes the attrs Pi05VLA.predict_action uses."""

    def __init__(
        self,
        text_hidden: int = 2048,
        vision_hidden: int = 1152,
        vocab: int = 1024,
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


class _StubPi05Head(VLAHead, nn.Module):
    """Stub pi0.5 expert head. Validates that NO state_emb is passed (pi0.5
    has no state token) and attn_mask is the right shape (chunk_size only,
    not chunk_size+1 like pi0)."""

    def __init__(self, num_layers: int = 2, chunk_size: int = 50, action_dim: int = 32):
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, noisy_actions, timestep=None, position_ids=None, *,
                prefix_k=None, prefix_v=None, attn_mask=None, **kwargs):
        if prefix_k is None or prefix_v is None:
            raise ValueError("pi0.5 stub head requires prefix_k + prefix_v")
        if attn_mask is None:
            raise ValueError("pi0.5 stub head requires attn_mask")
        # pi0.5 should NOT receive state_emb (state is in language tokens)
        assert kwargs.get("state_emb") is None, \
            f"pi0.5 must not receive state_emb (got {kwargs.get('state_emb')})"
        assert prefix_k.shape[0] == self.num_layers
        # position_ids should be chunk_size-shaped (NOT chunk_size+1 like pi0)
        assert position_ids.shape[-1] == noisy_actions.shape[1]
        assert attn_mask.ndim == 4
        assert attn_mask.shape[2] == noisy_actions.shape[1]
        return noisy_actions * self.scale
