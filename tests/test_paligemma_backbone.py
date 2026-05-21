"""Tests for PaliGemmaBackbone — language-tower component on the BaseVLA spine.

Lift #1 Day 4c per `features/03_export/basevla-spine_plan.md`. Validates:

- registration on the LLM_BACKBONES registry
- construction via pre-built model (the Day 4f extraction path) + input
  validation (must provide exactly one of model_id / model)
- forward shape contract with stub model (text-only + inputs_embeds paths)
- prepare_triton excludes vision_tower weights (those belong to SigLIPBackbone)
- accessor properties (language_model / multi_modal_projector / embed_tokens
  / text_hidden_size)

The pi0 parity tests with the real PaliGemma-3B model live in Day 4f's
per-VLA tests + the Modal smoke. This file tests the wrapper contract.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from reflex.models.llm import LLMBackbone
from reflex.models.llm.paligemma_backbone import PaliGemmaBackbone
from reflex.registry.components import LLM_BACKBONES


# ─── Registration + ABC ─────────────────────────────────────────────────


def test_paligemma_backbone_registered():
    assert "PaliGemmaBackbone" in LLM_BACKBONES
    assert LLM_BACKBONES.get("PaliGemmaBackbone") is PaliGemmaBackbone


def test_paligemma_backbone_is_llm_backbone_subclass():
    assert issubclass(PaliGemmaBackbone, LLMBackbone)


# ─── Construction validation ────────────────────────────────────────────


def test_rejects_both_model_and_model_id():
    stub = _make_stub_paligemma()
    with pytest.raises(ValueError, match="exactly one"):
        PaliGemmaBackbone(model_id="anything", model=stub)


def test_rejects_neither_model_nor_model_id():
    with pytest.raises(ValueError, match="exactly one"):
        PaliGemmaBackbone()


def test_constructs_with_pre_built_model():
    stub = _make_stub_paligemma()
    backbone = PaliGemmaBackbone(model=stub)
    assert backbone.model is stub


def test_accessors_route_to_paligemma_internals():
    """language_model / multi_modal_projector / embed_tokens / text_hidden_size
    accessor properties navigate the paligemma.model.model.* structure."""
    stub = _make_stub_paligemma(text_hidden=64)
    backbone = PaliGemmaBackbone(model=stub)
    assert backbone.language_model is stub.model.language_model
    assert backbone.multi_modal_projector is stub.model.multi_modal_projector
    assert backbone.embed_tokens is stub.model.language_model.embed_tokens
    assert backbone.text_hidden_size == 64


# ─── Forward shape contract ─────────────────────────────────────────────


def test_forward_with_input_ids_embeds_internally():
    """Path 1: input_ids → embed → language_model."""
    stub = _make_stub_paligemma(text_hidden=8, vocab_size=16)
    backbone = PaliGemmaBackbone(model=stub)

    input_ids = torch.zeros(2, 5, dtype=torch.long)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    # Stub returns a SimpleNamespace with last_hidden_state of shape [2, 5, 8]
    assert out.last_hidden_state.shape == (2, 5, 8)


def test_forward_with_inputs_embeds_bypasses_embedding():
    """Path 2: caller-supplied inputs_embeds skip the embedding step."""
    stub = _make_stub_paligemma(text_hidden=8, vocab_size=16)
    backbone = PaliGemmaBackbone(model=stub)

    inputs_embeds = torch.randn(2, 5, 8)
    out = backbone(inputs_embeds=inputs_embeds, attention_mask=torch.ones(2, 5))
    assert out.last_hidden_state.shape == (2, 5, 8)


def test_forward_requires_one_of_input_ids_or_inputs_embeds():
    backbone = PaliGemmaBackbone(model=_make_stub_paligemma())
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        backbone()


# ─── prepare_triton excludes vision_tower ───────────────────────────────


def test_prepare_triton_excludes_vision_tower_weights():
    """vision_tower weights belong to SigLIPBackbone — must NOT appear here."""
    stub = _make_stub_paligemma_with_vision_tower(text_hidden=8)
    backbone = PaliGemmaBackbone(model=stub)
    weights = backbone.prepare_triton(prefix="pi0.llm.")

    # multi_modal_projector should be included
    assert any("multi_modal_projector" in k for k in weights), (
        f"Expected multi_modal_projector params; got keys: {sorted(weights)}"
    )

    # vision_tower must NOT be included
    assert not any("vision_tower" in k for k in weights), (
        f"vision_tower weights leaked into PaliGemmaBackbone.prepare_triton: "
        f"{sorted(k for k in weights if 'vision_tower' in k)}"
    )


def test_prepare_triton_with_default_prefix():
    stub = _make_stub_paligemma_with_vision_tower()
    backbone = PaliGemmaBackbone(model=stub)
    weights = backbone.prepare_triton()
    # Some key exists
    assert len(weights) > 0
    # No vision tower
    assert not any("vision_tower" in k for k in weights)


# ─── Helpers ────────────────────────────────────────────────────────────


def _make_stub_paligemma(text_hidden: int = 1152, vocab_size: int = 32000):
    """Minimal PaliGemma-shaped stub mirroring `paligemma.model.model.*` access."""

    # The actual structure is paligemma.model (PaliGemmaModel).{vision_tower,
    # multi_modal_projector, language_model}. PaliGemmaBackbone.language_model
    # returns self.model.model.language_model.
    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, text_hidden)

        def forward(self, *, inputs_embeds, attention_mask=None, past_key_values=None, **kw):
            # Echo the embeds as hidden states (preserves shape for test assertions)
            return SimpleNamespace(
                last_hidden_state=inputs_embeds,
                past_key_values=None,
            )

    class _Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, text_hidden)

    class _InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LM()
            self.multi_modal_projector = _Projector()

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            # paligemma.model = PaliGemmaModel(...) which holds language_model etc.
            self.model = _InnerModel()
            # paligemma.config.text_config.hidden_size — accessor uses this
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=text_hidden),
            )

    return _Stub()


def _make_stub_paligemma_with_vision_tower(text_hidden: int = 8):
    """Variant that includes vision_tower with a parameter, for prepare_triton
    exclusion testing."""

    class _VisionTower(nn.Module):
        def __init__(self):
            super().__init__()
            # Single parameter that should NOT appear in PaliGemmaBackbone.prepare_triton
            self.patch_embed = nn.Linear(3, 8)

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, text_hidden)

        def forward(self, *, inputs_embeds, **kw):
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class _Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, text_hidden)

    class _InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = _VisionTower()
            self.language_model = _LM()
            self.multi_modal_projector = _Projector()

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerModel()
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=text_hidden),
            )

    return _Stub()
