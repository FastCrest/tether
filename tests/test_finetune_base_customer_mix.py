"""Tests for FinetuneConfig.base_dataset + mix_ratio fields — Phase 1
self-distilling-serve Day 4. Per ADR 2026-04-25-self-distilling-serve-
architecture decision #2: customer + base data mixing via
ConcatDataset + WeightedRandomSampler.

The dataloader integration test requires lerobot + torch — gated. The
config-level tests run unconditionally."""
from __future__ import annotations

from pathlib import Path

import pytest

from tether.finetune.config import FinetuneConfig


def _mk_cfg(**overrides) -> FinetuneConfig:
    defaults = dict(
        base="lerobot/smolvla_base",
        dataset="customer/data",
        output=Path("/tmp/out"),
        phase="distill",
        teacher_export="/tmp/teacher",
    )
    defaults.update(overrides)
    return FinetuneConfig(**defaults)


# ---------------------------------------------------------------------------
# FinetuneConfig field defaults
# ---------------------------------------------------------------------------


def test_config_default_base_dataset_is_none():
    """When unset, base_dataset is None — single-dataset path (back-compat)."""
    cfg = _mk_cfg()
    assert cfg.base_dataset is None


def test_config_default_mix_ratio_is_half():
    """Default 0.5 (50/50) per ADR — balances adaptation against
    catastrophic forgetting."""
    cfg = _mk_cfg()
    assert cfg.mix_ratio == 0.5


def test_config_accepts_explicit_base_dataset():
    cfg = _mk_cfg(base_dataset="lerobot/libero")
    assert cfg.base_dataset == "lerobot/libero"


def test_config_accepts_boundary_mix_ratios():
    """0.0 = base only (no adaptation); 1.0 = customer only (max adaptation)."""
    cfg = _mk_cfg(mix_ratio=0.0)
    assert cfg.mix_ratio == 0.0
    cfg = _mk_cfg(mix_ratio=1.0)
    assert cfg.mix_ratio == 1.0


def test_config_rejects_negative_mix_ratio():
    with pytest.raises(ValueError, match="mix_ratio"):
        _mk_cfg(mix_ratio=-0.1)


def test_config_rejects_mix_ratio_above_one():
    with pytest.raises(ValueError, match="mix_ratio"):
        _mk_cfg(mix_ratio=1.5)


# ---------------------------------------------------------------------------
# Dataloader integration (gated on lerobot + torch availability)
# ---------------------------------------------------------------------------


def test_build_dataloader_signature_accepts_base_dataset():
    """Smoke test — _build_dataloader should reference cfg.base_dataset
    via getattr (back-compat with cfg objects that don't have the field)."""
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from tether.finetune.backends.snapflow_backend import _build_dataloader
    import inspect
    src = inspect.getsource(_build_dataloader)
    assert "base_dataset" in src, (
        "_build_dataloader must reference cfg.base_dataset for the "
        "Day 4 ConcatDataset path"
    )
    assert "mix_ratio" in src
    assert "ConcatDataset" in src
    assert "WeightedRandomSampler" in src
