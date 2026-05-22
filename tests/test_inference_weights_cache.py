"""Lift #3 Day 2 — disk cache for `prepare_inference_weights()` output.

Per ``features/01_serve/inference-only-weights_plan.md`` Day 2.

4 test cases per the plan spec.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from reflex.models._inference_weights_cache import cache_path, load_or_build


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Point the cache root at a fresh tmp dir for each test."""
    monkeypatch.setattr(
        "reflex.models._inference_weights_cache.Path.home",
        classmethod(lambda cls: tmp_path),  # type: ignore[arg-type]
    )
    return tmp_path / ".cache" / "reflex" / "inference_weights"


def _builder() -> dict[str, torch.Tensor]:
    return {"foo.weight": torch.randn(4, 4), "bar.bias": torch.randn(8)}


# ─── Cold build → warm read produces byte-identical dict ─────────────


def test_cold_build_then_warm_read_is_byte_identical(tmp_cache_dir):
    """First call builds the dict + writes to cache. Second call reads
    from cache and returns byte-identical tensors."""
    torch.manual_seed(42)
    cold = load_or_build(
        model_id="test/model",
        checkpoint_sha="abc123",
        vla_type="TestVLA",
        builder=_builder,
        reflex_version="0.10.0",
        torch_version="2.5.0",
    )

    # 2nd call: same key tuple → reads from disk
    def _bad_builder():
        raise AssertionError("builder should NOT be called on cache hit")

    warm = load_or_build(
        model_id="test/model",
        checkpoint_sha="abc123",
        vla_type="TestVLA",
        builder=_bad_builder,
        reflex_version="0.10.0",
        torch_version="2.5.0",
    )

    assert set(cold.keys()) == set(warm.keys())
    for k in cold:
        assert torch.equal(cold[k], warm[k]), f"key {k!r} differs between cold + warm"


# ─── Cache invalidates on checkpoint_sha change ──────────────────────


def test_cache_invalidates_on_checkpoint_sha_change(tmp_cache_dir):
    """Different `checkpoint_sha` → different cache file → builder
    runs again."""
    builder_calls = {"n": 0}

    def _counting_builder():
        builder_calls["n"] += 1
        return {"key": torch.randn(4)}

    load_or_build(
        model_id="test/model",
        checkpoint_sha="sha_v1",
        vla_type="TestVLA",
        builder=_counting_builder,
        reflex_version="0.10.0",
        torch_version="2.5.0",
    )
    load_or_build(
        model_id="test/model",
        checkpoint_sha="sha_v2",  # CHANGED
        vla_type="TestVLA",
        builder=_counting_builder,
        reflex_version="0.10.0",
        torch_version="2.5.0",
    )

    assert builder_calls["n"] == 2, "builder should have been called twice (cache miss on different sha)"


# ─── Cache invalidates on reflex_version change ──────────────────────


def test_cache_invalidates_on_reflex_version_change(tmp_cache_dir):
    """Reflex refactor that changes spine slot layout would silently
    shape-mismatch a cached dict. Version bump must invalidate."""
    builder_calls = {"n": 0}

    def _counting_builder():
        builder_calls["n"] += 1
        return {"key": torch.randn(4)}

    load_or_build(
        model_id="test/model",
        checkpoint_sha="abc123",
        vla_type="TestVLA",
        builder=_counting_builder,
        reflex_version="0.10.0",
        torch_version="2.5.0",
    )
    load_or_build(
        model_id="test/model",
        checkpoint_sha="abc123",
        vla_type="TestVLA",
        builder=_counting_builder,
        reflex_version="0.11.0",  # CHANGED
        torch_version="2.5.0",
    )

    assert builder_calls["n"] == 2, "version bump should invalidate cache"


# ─── Cache file path matches the hash convention ─────────────────────


def test_cache_path_is_hash_pt_file_under_inference_weights_dir(tmp_cache_dir):
    """`cache_path(...)` returns ``~/.cache/reflex/inference_weights/<hash>.pt``.
    The directory is created on demand."""
    p = cache_path(
        model_id="test/model",
        checkpoint_sha="abc123",
        vla_type="TestVLA",
        reflex_version="0.10.0",
        torch_version="2.5.0",
    )
    assert p.suffix == ".pt"
    assert p.parent.name == "inference_weights"
    assert p.parent.parent.name == "reflex"
    assert p.parent.parent.parent.name == ".cache"
    # Hash should be a 16-hex-char filename (per the implementation).
    stem = p.stem
    assert len(stem) == 16, f"expected 16-char hex stem, got {stem!r}"
    assert all(c in "0123456789abcdef" for c in stem), f"non-hex char in stem: {stem!r}"
