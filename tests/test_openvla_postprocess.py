"""Tests for OpenVLA action postprocessing helpers."""

import numpy as np
import pytest

from tether.postprocess.openvla import (
    bins_to_normalized,
    decode_actions,
    logits_to_tokens,
    tokens_to_action_bins,
    unnormalize_actions,
)


def _center(
    index: int,
    n_bins: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> np.float32:
    edges = np.linspace(action_low, action_high, n_bins, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers[index]


class TestLogitsToTokens:
    def test_picks_argmax_last_action_dim(self):
        # Batch of 1, seq 10, vocab 100; force token 42 at positions -3:
        logits = np.zeros((1, 10, 100), dtype=np.float32)
        logits[0, -3:, 42] = 5.0
        tokens = logits_to_tokens(logits, action_dim=3)
        assert tokens.shape == (1, 3)
        assert (tokens == 42).all()

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError):
            logits_to_tokens(np.zeros((10, 100)), action_dim=7)


class TestTokensToActionBins:
    def test_top_token_is_bin_zero(self):
        # OpenVLA decodes against effective vocab_size=32000, not the padded LM size.
        tokens = np.array([[31999, 31998, 31997]])
        bins = tokens_to_action_bins(tokens, vocab_size=32000, n_bins=256)
        assert (bins == np.array([[0, 1, 2]])).all()

    def test_clips_out_of_range(self):
        # Any token below the action token band clips to the last valid center.
        tokens = np.array([[0, 100, 1000]])
        bins = tokens_to_action_bins(tokens, vocab_size=32000, n_bins=256)
        assert (bins == 254).all()


class TestBinsToNormalized:
    def test_bin_0_maps_to_first_center(self):
        bins = np.array([[0]])
        out = bins_to_normalized(bins, n_bins=256, action_low=-1.0, action_high=1.0)
        assert out[0, 0] == pytest.approx(_center(0))

    def test_bin_last_maps_to_last_center(self):
        bins = np.array([[255]])
        out = bins_to_normalized(bins)
        assert out[0, 0] == pytest.approx(_center(254))


class TestUnnormalizeActions:
    def test_applies_q01_q99(self):
        norm_stats = {
            "bridge": {"action": {"q01": [0.0, 0.0], "q99": [2.0, 4.0], "mask": [True, True]}}
        }
        # normalized=-1 → q01, normalized=1 → q99
        normalized = np.array([[-1.0, 1.0]], dtype=np.float32)
        out = unnormalize_actions(normalized, norm_stats, "bridge")
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(4.0)

    def test_mask_passes_through(self):
        norm_stats = {
            "robot": {
                "action": {
                    "q01": [0.0, 0.0],
                    "q99": [2.0, 4.0],
                    "mask": [True, False],  # dim 1 passes through
                }
            }
        }
        normalized = np.array([[-1.0, 0.5]], dtype=np.float32)
        out = unnormalize_actions(normalized, norm_stats, "robot")
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(0.5)  # unchanged

    def test_unknown_dataset_raises(self):
        norm_stats = {"bridge": {"action": {"q01": [0], "q99": [1], "mask": [True]}}}
        with pytest.raises(KeyError):
            unnormalize_actions(np.array([0.5]), norm_stats, "unknown")


class TestDecodeActions:
    def test_full_pipeline_normalized(self):
        # 1 batch, seq 8, padded vocab 32064; OpenVLA decode uses token 31999.
        logits = np.zeros((1, 8, 32064), dtype=np.float32)
        logits[0, -7:, 31999] = 10.0  # effective top token = bin 0 = first center
        out = decode_actions(logits, action_dim=7)
        assert out.shape == (1, 7)
        assert np.allclose(out, _center(0))

    def test_with_norm_stats(self):
        logits = np.zeros((1, 8, 32064), dtype=np.float32)
        logits[0, -7:, 31999] = 10.0
        norm_stats = {
            "bridge": {
                "action": {
                    "q01": [0.0] * 7,
                    "q99": [2.0] * 7,
                    "mask": [True] * 7,
                }
            }
        }
        out = decode_actions(logits, action_dim=7, norm_stats=norm_stats, dataset_name="bridge")
        assert out.shape == (1, 7)
        # q01=0, q99=2 maps normalized x to x + 1.
        assert np.allclose(out, _center(0) + 1.0)
