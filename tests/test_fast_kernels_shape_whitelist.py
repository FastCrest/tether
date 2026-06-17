"""Unit tests for the Lift #5 Day 3 shape whitelist.

The whitelist guards ``tether serve --fast-kernels`` from running on
non-PaliGemma-SigLIP-base models — silent-wrong outputs would result from
Triton block-size mismatches with the vendored kernel hard-codes.
"""
from __future__ import annotations

from tether.kernels._shape_whitelist import (
    PI05_SHAPE_SIGNATURES,
    REQUIRED_CONFIG_KEYS,
    supported_signatures,
    validate_shape_signature,
)


# ── Happy path ─────────────────────────────────────────────────────────


def _valid_paligemma_siglip_base_config() -> dict:
    """Mirror PI05_SHAPE_SIGNATURES['paligemma_siglip_base'] exactly."""
    return dict(PI05_SHAPE_SIGNATURES["paligemma_siglip_base"])


def test_paligemma_siglip_base_passes():
    config = _valid_paligemma_siglip_base_config()
    ok, msg = validate_shape_signature(config)
    assert ok
    assert msg == ""


def test_extra_keys_in_config_are_ignored():
    config = _valid_paligemma_siglip_base_config()
    config["random_unused_key"] = 999
    config["something_else"] = "doesnt matter"
    ok, msg = validate_shape_signature(config)
    assert ok, msg


def test_derived_num_patches_matches_passes():
    config = _valid_paligemma_siglip_base_config()
    config["num_patches"] = 256  # (224/14)^2
    ok, msg = validate_shape_signature(config)
    assert ok, msg


# ── Reject path: wrong shape values ────────────────────────────────────


def test_dinosiglip_shape_rejected():
    """DinoSigLIP has different hidden + intermediate."""
    config = _valid_paligemma_siglip_base_config()
    config["vit_hidden"] = 768  # DinoSigLIP-base value
    ok, msg = validate_shape_signature(config)
    assert not ok
    assert "vit_hidden" in msg
    assert "768" in msg
    assert "1152" in msg


def test_eva_clip_shape_rejected():
    """EVA-CLIP-large has different patch_size + image_size."""
    config = _valid_paligemma_siglip_base_config()
    config["patch_size"] = 16
    config["image_size"] = 336
    ok, msg = validate_shape_signature(config)
    assert not ok
    assert "patch_size" in msg or "image_size" in msg


def test_partial_shape_mismatch_lists_all_offenders():
    """When multiple shapes are off, the error message lists every offender."""
    config = _valid_paligemma_siglip_base_config()
    config["vit_hidden"] = 768
    config["vit_num_heads"] = 12
    ok, msg = validate_shape_signature(config)
    assert not ok
    assert "vit_hidden" in msg
    assert "vit_num_heads" in msg


# ── Reject path: missing required keys ─────────────────────────────────


def test_missing_required_key_rejected():
    config = _valid_paligemma_siglip_base_config()
    del config["vit_hidden"]
    ok, msg = validate_shape_signature(config)
    assert not ok
    assert "missing" in msg.lower()
    assert "vit_hidden" in msg


def test_empty_config_rejected():
    ok, msg = validate_shape_signature({})
    assert not ok
    # All REQUIRED_CONFIG_KEYS should appear in the message
    for key in REQUIRED_CONFIG_KEYS:
        assert key in msg, f"{key} should be flagged as missing"


# ── Reject path: num_patches inconsistent with derivation ──────────────


def test_inconsistent_num_patches_rejected():
    """Even if image_size + patch_size are right, an explicit wrong num_patches is flagged."""
    config = _valid_paligemma_siglip_base_config()
    config["num_patches"] = 999  # would be 256 derived from 224/14
    ok, msg = validate_shape_signature(config)
    assert not ok
    assert "num_patches" in msg


# ── API surface ────────────────────────────────────────────────────────


def test_supported_signatures_returns_only_pi05_for_v1():
    sigs = supported_signatures()
    assert sigs == ("paligemma_siglip_base",), f"V1 must only support pi0.5; got {sigs}"
