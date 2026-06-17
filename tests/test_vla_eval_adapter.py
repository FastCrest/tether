"""Tests for tether.runtime.adapters.vla_eval pure helpers.

These exercise image-picking and action-truncation logic without requiring
``vla-eval`` to be installed. The adapter class itself (built via
``build_adapter_class``) needs vla-eval + an export dir, so we don't cover
it here — the Modal LIBERO script exercises it end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from tether.runtime.adapters.vla_eval import (
    build_adapter_class,
    pick_image,
    truncate_actions,
)


# ---------------------------------------------------------------------------
# pick_image
# ---------------------------------------------------------------------------

def test_pick_image_by_key():
    obs = {
        "images": {
            "top": np.ones((10, 10, 3), dtype=np.float32),
            "wrist": np.zeros((10, 10, 3), dtype=np.float32),
        }
    }
    img = pick_image(obs, camera_key="wrist")
    assert img.shape == (10, 10, 3)
    assert img.sum() == 0


def test_pick_image_default_first():
    obs = {
        "images": {
            "top": np.ones((10, 10, 3), dtype=np.float32),
            "wrist": np.zeros((10, 10, 3), dtype=np.float32),
        }
    }
    img = pick_image(obs)
    assert img.sum() > 0


def test_pick_image_missing_key_falls_back_to_first():
    obs = {"images": {"top": np.ones((5, 5, 3), dtype=np.float32)}}
    img = pick_image(obs, camera_key="nonexistent")
    assert img.shape == (5, 5, 3)
    assert img.sum() > 0


def test_pick_image_no_images_returns_none():
    assert pick_image({}) is None
    assert pick_image({"images": {}}) is None
    assert pick_image({"images": None}) is None


# ---------------------------------------------------------------------------
# truncate_actions
# ---------------------------------------------------------------------------

def test_truncate_actions_down():
    actions = np.random.randn(5, 32).astype(np.float32)
    result = truncate_actions(actions, 7)
    assert result.shape == (5, 7)
    np.testing.assert_array_equal(result, actions[:, :7])


def test_truncate_actions_pad_up():
    actions = np.random.randn(5, 3).astype(np.float32)
    result = truncate_actions(actions, 7)
    assert result.shape == (5, 7)
    np.testing.assert_array_equal(result[:, :3], actions)
    np.testing.assert_array_equal(
        result[:, 3:], np.zeros((5, 4), dtype=np.float32)
    )


def test_truncate_actions_same_dim():
    actions = np.random.randn(5, 7).astype(np.float32)
    result = truncate_actions(actions, 7)
    np.testing.assert_array_equal(result, actions)


def test_truncate_actions_1d_promotes():
    actions = np.random.randn(32).astype(np.float32)
    result = truncate_actions(actions, 7)
    assert result.shape == (1, 7)
    np.testing.assert_array_equal(result[0], actions[:7])


def test_truncate_preserves_dtype():
    actions = np.random.randn(3, 32).astype(np.float64)
    result = truncate_actions(actions, 7)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# build_adapter_class — should raise a clear error if vla-eval is absent
# ---------------------------------------------------------------------------

def test_build_adapter_class_raises_without_vla_eval():
    try:
        import vla_eval  # noqa: F401
    except ImportError:
        # vla-eval absent (most dev machines) — confirm we get a helpful error.
        with pytest.raises(ImportError, match="vla-eval"):
            build_adapter_class()
    else:
        # vla-eval present — confirm we get a class that subclasses something.
        cls = build_adapter_class()
        assert isinstance(cls, type)
        assert hasattr(cls, "predict")
