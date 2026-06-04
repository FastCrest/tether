"""Tests for the Phase B safetensors→flat-dict-direct loader.

The contract pinned here:

1. ``apply_prefix_mapping`` is first-match-wins; declaration order matters.
2. ``load_flat_dict_from_safetensors`` reads every tensor in the file, applies
   the mapping, and casts to the requested dtype. No extra tensors leak.
3. Multi-file dirs union correctly via ``load_flat_dict_from_safetensors_dir``;
   duplicate keys across files raise ``ValueError``.
4. Empty src_prefix in mapping raises ``ValueError`` (would match every key).
"""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tether.runtime.inference_weights.safetensors_direct import (
    apply_prefix_mapping,
    load_flat_dict_from_safetensors,
    load_flat_dict_from_safetensors_dir,
)


# ── apply_prefix_mapping ────────────────────────────────────────────────


def test_prefix_mapping_identity_when_empty():
    assert apply_prefix_mapping("foo.bar", {}) == "foo.bar"


def test_prefix_mapping_first_match_wins():
    mapping = {
        "model.paligemma_with_expert.paligemma.model.vision_tower.": "vision_backbone.model.",
        "model.paligemma_with_expert.paligemma.": "llm_backbone.",
    }
    # vision_tower is a more specific prefix; declared first, wins.
    src = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    expected = "vision_backbone.model.vision_model.embeddings.patch_embedding.weight"
    assert apply_prefix_mapping(src, mapping) == expected


def test_prefix_mapping_fallback_to_less_specific():
    mapping = {
        "model.paligemma_with_expert.paligemma.model.vision_tower.": "vision_backbone.model.",
        "model.paligemma_with_expert.paligemma.": "llm_backbone.",
    }
    # Not under vision_tower; falls through to the second rule.
    src = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    expected = "llm_backbone.model.language_model.embed_tokens.weight"
    assert apply_prefix_mapping(src, mapping) == expected


def test_prefix_mapping_no_match_passes_through():
    mapping = {"model.foo.": "bar."}
    assert apply_prefix_mapping("something.else.weight", mapping) == "something.else.weight"


def test_prefix_mapping_empty_src_raises():
    with pytest.raises(ValueError, match="empty src_prefix"):
        apply_prefix_mapping("anything", {"": "wat"})


# ── load_flat_dict_from_safetensors ─────────────────────────────────────


@pytest.fixture
def tiny_safetensors_file(tmp_path):
    """Build a tiny 3-tensor safetensors file representing a pi05-like nesting."""
    path = tmp_path / "tiny.safetensors"
    tensors = {
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight": torch.zeros(4, 3, dtype=torch.float32),
        "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": torch.ones(8, 6, dtype=torch.float32),
        "model.paligemma_with_expert.gemma_expert.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 2.0, dtype=torch.float32),
    }
    save_file(tensors, str(path))
    return path, tensors


def test_load_reads_every_tensor(tiny_safetensors_file):
    path, source_tensors = tiny_safetensors_file
    flat = load_flat_dict_from_safetensors(path, device="cpu")
    assert len(flat) == len(source_tensors)
    assert set(flat) == set(source_tensors)


def test_load_applies_mapping(tiny_safetensors_file):
    path, _ = tiny_safetensors_file
    mapping = {
        "model.paligemma_with_expert.paligemma.model.vision_tower.": "vision_backbone.model.",
        "model.paligemma_with_expert.paligemma.": "llm_backbone.",
        "model.paligemma_with_expert.gemma_expert.": "vla_head.expert_stack.",
    }
    flat = load_flat_dict_from_safetensors(path, name_mapping=mapping, device="cpu")
    assert "vision_backbone.model.vision_model.embeddings.patch_embedding.weight" in flat
    assert "llm_backbone.model.language_model.embed_tokens.weight" in flat
    assert "vla_head.expert_stack.layers.0.self_attn.q_proj.weight" in flat


def test_load_casts_to_bf16(tiny_safetensors_file):
    path, _ = tiny_safetensors_file
    flat = load_flat_dict_from_safetensors(path, dtype=torch.bfloat16, device="cpu")
    for tensor in flat.values():
        assert tensor.dtype == torch.bfloat16


def test_load_preserves_dtype_when_no_cast(tiny_safetensors_file):
    path, _ = tiny_safetensors_file
    flat = load_flat_dict_from_safetensors(path, dtype=None, device="cpu")
    for tensor in flat.values():
        assert tensor.dtype == torch.float32  # matches source


def test_load_preserves_values(tiny_safetensors_file):
    path, source_tensors = tiny_safetensors_file
    flat = load_flat_dict_from_safetensors(path, device="cpu")
    for src_key, src_tensor in source_tensors.items():
        assert torch.equal(flat[src_key], src_tensor)


def test_load_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_flat_dict_from_safetensors(tmp_path / "nope.safetensors")


# ── load_flat_dict_from_safetensors_dir ─────────────────────────────────


def test_load_dir_merges_files(tmp_path):
    save_file(
        {"a.weight": torch.zeros(2, 2), "b.weight": torch.ones(3, 3)},
        str(tmp_path / "shard_1.safetensors"),
    )
    save_file(
        {"c.weight": torch.full((2, 2), 2.0)},
        str(tmp_path / "shard_2.safetensors"),
    )
    flat = load_flat_dict_from_safetensors_dir(tmp_path, device="cpu")
    assert set(flat) == {"a.weight", "b.weight", "c.weight"}


def test_load_dir_raises_on_duplicate_keys(tmp_path):
    save_file(
        {"shared.weight": torch.zeros(2, 2)},
        str(tmp_path / "shard_1.safetensors"),
    )
    save_file(
        {"shared.weight": torch.ones(2, 2)},  # collision
        str(tmp_path / "shard_2.safetensors"),
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        load_flat_dict_from_safetensors_dir(tmp_path, device="cpu")


def test_load_dir_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        load_flat_dict_from_safetensors_dir(tmp_path)


def test_load_dir_raises_when_not_a_dir(tmp_path):
    not_a_dir = tmp_path / "actually_a_file"
    not_a_dir.write_text("x")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        load_flat_dict_from_safetensors_dir(not_a_dir)
