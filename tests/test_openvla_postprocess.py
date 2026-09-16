"""Tests for OpenVLA action-token postprocessing helpers."""

import numpy as np
import pytest

from tether.postprocess.openvla import (
    bins_to_normalized,
    decode_actions,
    decode_token_ids,
    logits_to_tokens,
    tokens_to_action_bins,
    unnormalize_actions,
)


def _upstream_decode_reference(
    token_ids: np.ndarray,
    *,
    vocab_size: int = 32000,
    n_bins: int = 256,
) -> np.ndarray:
    """Independent transcription of upstream OpenVLA ActionTokenizer decode."""

    edges = np.linspace(-1.0, 1.0, n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    discrete = vocab_size - np.asarray(token_ids)
    indexes = np.clip(discrete - 1, 0, centers.shape[0] - 1)
    return centers[indexes]


class TestLogitsToTokens:
    def test_picks_argmax_for_each_generation_step(self):
        logits = np.zeros((1, 3, 32064), dtype=np.float32)
        logits[0, 0, 31999] = 5.0
        logits[0, 1, 31998] = 6.0
        logits[0, 2, 31997] = 7.0
        tokens = logits_to_tokens(logits, action_dim=3)
        assert tokens.shape == (1, 3)
        assert np.array_equal(tokens, np.array([[31999, 31998, 31997]]))

    def test_rejects_prompt_sequence_logits(self):
        with pytest.raises(ValueError, match="one next-token logit row"):
            logits_to_tokens(np.zeros((1, 8, 32064), dtype=np.float32), action_dim=7)

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="generation logits"):
            logits_to_tokens(np.zeros((7, 32064), dtype=np.float32), action_dim=7)


class TestTokensToActionBins:
    def test_top_effective_vocab_tokens_map_to_first_centers(self):
        tokens = np.array([[31999, 31998, 31997]], dtype=np.int64)
        bins = tokens_to_action_bins(tokens, vocab_size=32000, n_bins=256)
        assert np.array_equal(bins, np.array([[0, 1, 2]]))

    def test_last_upstream_discrete_index_clips_to_last_center(self):
        # token 31744 corresponds to upstream discretized index 256.
        bins = tokens_to_action_bins(np.array([[31744]], dtype=np.int64), 32000, 256)
        assert bins[0, 0] == 254

    def test_non_action_tokens_clip_to_last_center(self):
        tokens = np.array([[0, 100, 1000]], dtype=np.int64)
        bins = tokens_to_action_bins(tokens, vocab_size=32000, n_bins=256)
        assert np.array_equal(bins, np.array([[254, 254, 254]]))

    def test_requires_integer_tokens(self):
        with pytest.raises(ValueError, match="integers"):
            tokens_to_action_bins(np.array([[31999.0]], dtype=np.float32), 32000)


class TestBinsToNormalized:
    def test_matches_upstream_bin_centers(self):
        indexes = np.array([[0, 1, 253, 254]], dtype=np.int64)
        actual = bins_to_normalized(indexes)
        expected = _upstream_decode_reference(
            np.array([[31999, 31998, 31746, 31745]], dtype=np.int64)
        )
        assert np.allclose(actual, expected.astype(np.float32), atol=1e-7)

    def test_endpoints_are_not_returned(self):
        actual = bins_to_normalized(np.array([[0, 254]], dtype=np.int64))
        assert actual[0, 0] > -1.0
        assert actual[0, 1] < 1.0
        assert actual[0, 0] == pytest.approx(-1.0 + 1.0 / 255.0)
        assert actual[0, 1] == pytest.approx(1.0 - 1.0 / 255.0)

    def test_rejects_invalid_edge_count(self):
        with pytest.raises(ValueError, match="n_bins"):
            bins_to_normalized(np.array([[0]]), n_bins=1)


class TestUnnormalizeActions:
    def test_applies_q01_q99(self):
        norm_stats = {
            "bridge": {
                "action": {
                    "q01": [0.0, 0.0],
                    "q99": [2.0, 4.0],
                    "mask": [True, True],
                }
            }
        }
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
                    "mask": [True, False],
                }
            }
        }
        normalized = np.array([[-1.0, 0.5]], dtype=np.float32)
        out = unnormalize_actions(normalized, norm_stats, "robot")
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(0.5)

    def test_unknown_dataset_raises(self):
        stats = {"bridge": {"action": {"q01": [0], "q99": [1], "mask": [True]}}}
        with pytest.raises(KeyError):
            unnormalize_actions(np.array([[0.5]], dtype=np.float32), stats, "unknown")

    def test_dimension_mismatch_fails_closed(self):
        stats = {"bridge": {"action": {"q01": [0, 0], "q99": [1, 1]}}}
        with pytest.raises(ValueError, match="Action dimension"):
            unnormalize_actions(np.array([[0.5]], dtype=np.float32), stats, "bridge")


class TestDecodeTokenIds:
    def test_matches_independent_upstream_formula(self):
        tokens = np.array([[31999, 31998, 31900, 31745, 31744, 0, 31800]], dtype=np.int64)
        actual = decode_token_ids(tokens, vocab_size=32000)
        expected = _upstream_decode_reference(tokens, vocab_size=32000)
        assert actual.shape == (1, 7)
        assert np.allclose(actual, expected, atol=1e-7)

    def test_with_norm_stats(self):
        tokens = np.full((1, 7), 31999, dtype=np.int64)
        stats = {
            "bridge": {
                "action": {
                    "q01": [0.0] * 7,
                    "q99": [2.0] * 7,
                    "mask": [True] * 7,
                }
            }
        }
        normalized = _upstream_decode_reference(tokens).astype(np.float32)
        actual = decode_token_ids(
            tokens,
            vocab_size=32000,
            norm_stats=stats,
            dataset_name="bridge",
        )
        assert np.allclose(actual, normalized + 1.0, atol=1e-7)

    def test_requires_stats_and_dataset_together(self):
        with pytest.raises(ValueError, match="provided together"):
            decode_token_ids(np.array([[31999]], dtype=np.int64), norm_stats={})


class TestDecodeActions:
    def test_generation_step_logits_decode(self):
        logits = np.zeros((1, 7, 32064), dtype=np.float32)
        expected_tokens = np.array([[31999, 31998, 31997, 31996, 31995, 31994, 31993]])
        for step, token in enumerate(expected_tokens[0]):
            logits[0, step, token] = 10.0
        actual = decode_actions(logits, action_dim=7, vocab_size=32000)
        expected = _upstream_decode_reference(expected_tokens, vocab_size=32000)
        assert np.allclose(actual, expected, atol=1e-7)
