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


class TestBinCentersAgainstAnIndependentOracle:
    """Check the decode against arithmetic derived from OpenVLA's published
    constants, not from this module's own constants.

    ``_center`` above re-derives centers with the same two numpy calls the
    implementation uses, so it would follow the implementation into an error.
    These assertions instead use the closed form for the centres of ``n_bins``
    equally spaced edges over ``[low, high]``::

        center(j) = low + (2j + 1) * (high - low) / (2 * (n_bins - 1))

    and the vocabulary arithmetic ``modeling_prismatic.py`` publishes::

        vocab_size = text_config.vocab_size - pad_to_multiple_of  # 32064 - 64
        bin_index  = vocab_size - token_id - 1, clipped to [0, n_bins - 2]

    Edges and centres differ by exactly one half-bin, ``1/255`` for the shipped
    256-bin configuration, so a tolerance of 1e-6 separates a correct decode
    from the pre-fix edge decode by five orders of magnitude while staying well
    clear of the float32 rounding of ``np.linspace`` (~3e-8).
    """

    # Published in openvla/openvla-7b config.json.
    PADDED_VOCAB = 32064
    PAD_TO_MULTIPLE_OF = 64
    N_ACTION_BINS = 256

    @property
    def effective_vocab(self) -> int:
        return self.PADDED_VOCAB - self.PAD_TO_MULTIPLE_OF

    @staticmethod
    def _closed_form_center(
        index: np.ndarray | int,
        n_bins: int = 256,
        low: float = -1.0,
        high: float = 1.0,
    ) -> np.ndarray:
        span = high - low
        return low + (2 * np.asarray(index, dtype=np.float64) + 1.0) * span / (2 * (n_bins - 1))

    def test_effective_vocab_is_the_unpadded_text_vocab(self):
        assert self.effective_vocab == 32000

    def test_there_are_one_fewer_centers_than_edges(self):
        # 256 edges bound 255 intervals, so the largest valid index is 254.
        # A decode that indexed 256 centers would accept 255 here.
        highest = bins_to_normalized(np.array([[10**6]]), n_bins=self.N_ACTION_BINS)[0, 0]
        assert highest == pytest.approx(self._closed_form_center(self.N_ACTION_BINS - 2), abs=1e-6)

    def test_top_action_token_decodes_to_the_first_center_not_the_first_edge(self):
        token = self.effective_vocab - 1  # 31999
        bins = tokens_to_action_bins(
            np.array([[token]]), vocab_size=self.effective_vocab, n_bins=self.N_ACTION_BINS
        )
        value = bins_to_normalized(bins, n_bins=self.N_ACTION_BINS)[0, 0]
        assert bins[0, 0] == 0
        assert value == pytest.approx(self._closed_form_center(0), abs=1e-6)
        # The pre-fix decode returned the edge, -1.0, a half-bin away.
        assert abs(float(value) - (-1.0)) == pytest.approx(1.0 / 255.0, abs=1e-6)

    def test_lowest_action_token_decodes_to_the_last_center(self):
        # 32000 - 31745 - 1 == 254, the last center.
        token = self.effective_vocab - self.N_ACTION_BINS + 1  # 31745
        bins = tokens_to_action_bins(
            np.array([[token]]), vocab_size=self.effective_vocab, n_bins=self.N_ACTION_BINS
        )
        value = bins_to_normalized(bins, n_bins=self.N_ACTION_BINS)[0, 0]
        assert bins[0, 0] == self.N_ACTION_BINS - 2
        assert value == pytest.approx(self._closed_form_center(self.N_ACTION_BINS - 2), abs=1e-6)

    def test_padding_tokens_above_the_effective_vocab_clip_to_bin_zero(self):
        # Tokens 32000..32063 exist in the padded LM head but are not action
        # tokens; vocab_size - token - 1 is negative and must clip to 0, as
        # modeling_prismatic.py's a_min=0 does.
        tokens = np.array([[self.effective_vocab, self.PADDED_VOCAB - 1]])
        bins = tokens_to_action_bins(
            tokens, vocab_size=self.effective_vocab, n_bins=self.N_ACTION_BINS
        )
        assert (bins == 0).all()

    def test_every_token_id_matches_the_reference_decode(self):
        # Differential oracle over the whole padded vocabulary. The reference is
        # modeling_prismatic.py's published arithmetic, recomputed here in
        # float64 from the closed form rather than read from this module.
        tokens = np.arange(self.PADDED_VOCAB, dtype=np.int64).reshape(1, -1)
        expected_index = np.clip(self.effective_vocab - tokens - 1, 0, self.N_ACTION_BINS - 2)
        expected = self._closed_form_center(expected_index, n_bins=self.N_ACTION_BINS)

        bins = tokens_to_action_bins(
            tokens, vocab_size=self.effective_vocab, n_bins=self.N_ACTION_BINS
        )
        actual = bins_to_normalized(bins, n_bins=self.N_ACTION_BINS)

        assert np.array_equal(bins, expected_index)
        assert float(np.max(np.abs(actual.astype(np.float64) - expected))) < 1e-6

    def test_decode_actions_defaults_to_the_effective_vocab(self):
        # decode_actions' vocab_size default must be the unpadded 32000. If it
        # defaulted to the padded 32064, token 31999 would land 64 bins away.
        logits = np.zeros((1, 8, self.PADDED_VOCAB), dtype=np.float32)
        logits[0, -7:, self.effective_vocab - 1] = 10.0
        out = decode_actions(logits, action_dim=7)
        assert np.allclose(out, self._closed_form_center(0), atol=1e-6)

    def test_unnormalization_matches_the_published_affine(self):
        # actions = 0.5 * (normalized + 1) * (q99 - q01) + q01, per
        # modeling_prismatic.py. Computed here from q01/q99 directly.
        q01, q99 = -0.3, 1.7
        norm_stats = {"d": {"action": {"q01": [q01], "q99": [q99], "mask": [True]}}}
        center = float(self._closed_form_center(0))
        out = unnormalize_actions(np.array([[center]], dtype=np.float32), norm_stats, "d")
        assert out[0, 0] == pytest.approx(0.5 * (center + 1.0) * (q99 - q01) + q01, abs=1e-6)
