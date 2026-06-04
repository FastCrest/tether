"""Tests for OpenVLA action postprocessing helpers."""

import numpy as np
import pytest

from tether.postprocess.openvla import (
    logits_to_tokens,
    tokens_to_action_bins,
    bins_to_normalized,
    unnormalize_actions,
    decode_actions,
)


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
        # top-of-vocab token (vocab_size-1) → bin 0
        tokens = np.array([[32063, 32062, 32061]])
        bins = tokens_to_action_bins(tokens, vocab_size=32064, n_bins=256)
        assert (bins == np.array([[0, 1, 2]])).all()

    def test_clips_out_of_range(self):
        # Any token below the top 256 should clip to bin 255 (lowest bin)
        tokens = np.array([[0, 100, 1000]])
        bins = tokens_to_action_bins(tokens, vocab_size=32064, n_bins=256)
        assert (bins == 255).all()


class TestBinsToNormalized:
    def test_bin_0_maps_to_low(self):
        bins = np.array([[0]])
        out = bins_to_normalized(bins, n_bins=256, action_low=-1.0, action_high=1.0)
        assert out[0, 0] == pytest.approx(-1.0)

    def test_bin_last_maps_to_high(self):
        bins = np.array([[255]])
        out = bins_to_normalized(bins)
        assert out[0, 0] == pytest.approx(1.0)


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
        # 1 batch, seq 8, vocab 32064, pick top token at last 7 positions
        logits = np.zeros((1, 8, 32064), dtype=np.float32)
        logits[0, -7:, 32063] = 10.0  # topmost token = bin 0 = -1.0 normalized
        out = decode_actions(logits, action_dim=7)
        assert out.shape == (1, 7)
        assert np.all(out == -1.0)

    def test_with_norm_stats(self):
        logits = np.zeros((1, 8, 32064), dtype=np.float32)
        logits[0, -7:, 32063] = 10.0
        norm_stats = {
            "bridge": {"action": {"q01": [0.0]*7, "q99": [2.0]*7, "mask": [True]*7}}
        }
        out = decode_actions(logits, action_dim=7, norm_stats=norm_stats, dataset_name="bridge")
        assert out.shape == (1, 7)
        # normalized=-1, q01=0, q99=2 → 0.5*(-1+1)*2 + 0 = 0.0
        assert np.allclose(out, 0.0)
