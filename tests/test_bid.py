"""Unit tests for BID (Bidirectional Decoding) chunk selection.

Per src/tether/correction/bid.py + the 2026-04-29 a2c2-correction
research-revisit Lens 4 contender survey.
"""
from __future__ import annotations

import numpy as np
import pytest

from tether.correction.bid import (
    BIDConfig,
    predict_chunk_bid,
    score_backward_coherence,
    select_chunk_bid,
)


# ---------------------------------------------------------------------------
# BIDConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults_match_paper_baseline():
    cfg = BIDConfig()
    assert cfg.n_candidates == 8
    assert cfg.coherence_window == 5
    assert cfg.coherence_metric == "l2"


def test_config_rejects_n_candidates_below_2():
    with pytest.raises(ValueError, match="n_candidates"):
        BIDConfig(n_candidates=1)


def test_config_rejects_zero_window():
    with pytest.raises(ValueError, match="coherence_window"):
        BIDConfig(coherence_window=0)


def test_config_rejects_unknown_metric():
    with pytest.raises(ValueError, match="coherence_metric"):
        BIDConfig(coherence_metric="manhattan")


# ---------------------------------------------------------------------------
# score_backward_coherence
# ---------------------------------------------------------------------------


def test_score_l2_perfect_coherence_is_zero():
    """A candidate whose first K actions exactly match the previous chunk's
    last K actions should get score 0 (negated L2 distance = -0)."""
    chunk_size, action_dim = 50, 7
    rng = np.random.default_rng(0)
    prev = rng.standard_normal((chunk_size, action_dim)).astype(np.float32)
    candidate = np.zeros_like(prev)
    candidate[:5] = prev[-5:]  # perfect alignment in the first 5
    score = score_backward_coherence(candidate, prev, window=5, metric="l2")
    assert score == pytest.approx(0.0, abs=1e-6)


def test_score_l2_higher_is_more_coherent():
    """A candidate with smaller L2 distance to previous chunk's tail should
    score HIGHER (because we negate L2)."""
    chunk_size, action_dim = 50, 7
    rng = np.random.default_rng(1)
    prev = rng.standard_normal((chunk_size, action_dim)).astype(np.float32)

    close = np.zeros((chunk_size, action_dim), dtype=np.float32)
    close[:5] = prev[-5:] + 0.01 * rng.standard_normal((5, action_dim)).astype(np.float32)

    far = np.zeros((chunk_size, action_dim), dtype=np.float32)
    far[:5] = prev[-5:] + 5.0 * rng.standard_normal((5, action_dim)).astype(np.float32)

    score_close = score_backward_coherence(close, prev, window=5, metric="l2")
    score_far = score_backward_coherence(far, prev, window=5, metric="l2")
    assert score_close > score_far


def test_score_cos_perfect_alignment_is_one():
    """Cosine of identical vectors = 1.0 averaged across the window."""
    chunk_size, action_dim = 50, 7
    rng = np.random.default_rng(2)
    prev = rng.standard_normal((chunk_size, action_dim)).astype(np.float32)
    candidate = np.zeros_like(prev)
    candidate[:5] = prev[-5:]  # exact match
    score = score_backward_coherence(candidate, prev, window=5, metric="cos")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_score_cos_anti_aligned_is_minus_one():
    """Cosine of opposite-direction vectors = -1.0."""
    chunk_size, action_dim = 50, 7
    rng = np.random.default_rng(3)
    prev = rng.standard_normal((chunk_size, action_dim)).astype(np.float32)
    candidate = np.zeros_like(prev)
    candidate[:5] = -prev[-5:]  # anti-aligned
    score = score_backward_coherence(candidate, prev, window=5, metric="cos")
    assert score == pytest.approx(-1.0, abs=1e-5)


def test_score_rejects_shape_mismatch():
    a = np.zeros((50, 7), dtype=np.float32)
    b = np.zeros((50, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="!="):
        score_backward_coherence(a, b)


def test_score_rejects_window_larger_than_chunk():
    a = np.zeros((50, 7), dtype=np.float32)
    b = np.zeros((50, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="window"):
        score_backward_coherence(a, b, window=100)


# ---------------------------------------------------------------------------
# select_chunk_bid
# ---------------------------------------------------------------------------


def test_select_cold_start_returns_first_candidate():
    """No previous_chunk → return candidate 0 (no scoring)."""
    cfg = BIDConfig(n_candidates=4)
    candidates = [np.random.standard_normal((50, 7)).astype(np.float32) for _ in range(4)]
    best_idx, scores = select_chunk_bid(candidates, previous_chunk=None, config=cfg)
    assert best_idx == 0
    assert scores == [0.0] * 4


def test_select_picks_most_coherent():
    """Among 3 candidates, the one with smallest L2 to previous tail wins."""
    cfg = BIDConfig(n_candidates=3, coherence_window=5)
    rng = np.random.default_rng(5)
    prev = rng.standard_normal((50, 7)).astype(np.float32)

    # Candidate 0: random (likely large L2)
    cand_0 = rng.standard_normal((50, 7)).astype(np.float32)
    # Candidate 1: near-perfect alignment
    cand_1 = np.zeros((50, 7), dtype=np.float32)
    cand_1[:5] = prev[-5:] + 0.001 * rng.standard_normal((5, 7)).astype(np.float32)
    # Candidate 2: medium alignment
    cand_2 = np.zeros((50, 7), dtype=np.float32)
    cand_2[:5] = prev[-5:] + 0.5 * rng.standard_normal((5, 7)).astype(np.float32)

    candidates = [cand_0, cand_1, cand_2]
    best_idx, scores = select_chunk_bid(candidates, previous_chunk=prev, config=cfg)
    assert best_idx == 1, f"expected candidate 1 (closest match); got {best_idx} with scores={scores}"


def test_select_rejects_single_candidate():
    cfg = BIDConfig(n_candidates=2)  # cfg won't reject; selection logic does
    candidates = [np.zeros((50, 7), dtype=np.float32)]
    with pytest.raises(ValueError, match=">= 2"):
        select_chunk_bid(candidates, previous_chunk=None, config=cfg)


# ---------------------------------------------------------------------------
# predict_chunk_bid (end-to-end with mocked sample function)
# ---------------------------------------------------------------------------


def test_predict_calls_sample_fn_n_times():
    cfg = BIDConfig(n_candidates=8)
    call_log = []

    def sample_fn(i: int) -> np.ndarray:
        call_log.append(i)
        return np.random.standard_normal((50, 7)).astype(np.float32)

    chosen, telemetry = predict_chunk_bid(
        sample_fn, previous_chunk=None, config=cfg,
    )
    assert len(call_log) == 8
    assert call_log == list(range(8))
    assert telemetry["n_candidates"] == 8
    assert telemetry["selected_idx"] == 0  # cold-start


def test_predict_returns_correct_shape():
    cfg = BIDConfig(n_candidates=4)

    def sample_fn(i: int) -> np.ndarray:
        return np.random.standard_normal((50, 7)).astype(np.float32) * (i + 1)

    chosen, _ = predict_chunk_bid(
        sample_fn, previous_chunk=None, config=cfg,
    )
    assert chosen.shape == (50, 7)


def test_predict_with_previous_chunk_picks_via_coherence():
    """Mock the sample fn to return chunks where candidate 3 is most coherent
    with the previous chunk. predict_chunk_bid should pick candidate 3."""
    cfg = BIDConfig(n_candidates=4, coherence_window=5)
    rng = np.random.default_rng(7)
    prev = rng.standard_normal((50, 7)).astype(np.float32)

    candidates_in_order = []
    for i in range(4):
        c = rng.standard_normal((50, 7)).astype(np.float32)
        if i == 3:
            c[:5] = prev[-5:]  # near-perfect alignment
        candidates_in_order.append(c)

    call_count = [0]

    def sample_fn(i: int) -> np.ndarray:
        c = candidates_in_order[call_count[0]]
        call_count[0] += 1
        return c

    chosen, telemetry = predict_chunk_bid(
        sample_fn, previous_chunk=prev, config=cfg,
    )
    assert telemetry["selected_idx"] == 3
    np.testing.assert_array_equal(chosen, candidates_in_order[3])


def test_predict_telemetry_includes_all_fields():
    cfg = BIDConfig(n_candidates=4, coherence_window=3, coherence_metric="cos")
    prev = np.random.standard_normal((50, 7)).astype(np.float32)

    def sample_fn(i: int) -> np.ndarray:
        return np.random.standard_normal((50, 7)).astype(np.float32)

    _, telemetry = predict_chunk_bid(sample_fn, previous_chunk=prev, config=cfg)
    assert "selected_idx" in telemetry
    assert "scores" in telemetry
    assert len(telemetry["scores"]) == 4
    assert telemetry["n_candidates"] == 4
    assert telemetry["coherence_window"] == 3
    assert telemetry["coherence_metric"] == "cos"
