"""Bidirectional Decoding (BID) test-time chunk selection — alternative to
A2C2 head correction per arxiv 2408.17355 (Liu et al., ICLR 2025).

Background:
A2C2 head correction (kernels/a2c2_correction.py + correction/a2c2_training.py)
hit a fundamental architectural ceiling on 2026-04-29 across Phases 1-3:
the bounded-correction approach has a magnitude/derailment tension at the
deployment latency regime that L2 + scale tuning cannot escape (per
features/01_serve/subfeatures/_rtc_a2c2/a2c2-correction/a2c2-correction_research_revisit.md
Lens 4 contender survey).

BID sidesteps the magnitude problem entirely. Instead of correcting an
emitted action, it samples N candidate chunks from the (frozen) policy by
varying the flow-matching noise + picks the one most coherent with the
previously-executed chunk. No correction head, no training, no magnitude
to bound — just selection.

Algorithm (per arxiv 2408.17355):

    candidates = [policy.predict(noise=rng_i) for i in range(N)]
    score(c) = backward_coherence(c, previous_chunk) - lambda * forward_contrast(c, weak_ref)
    return candidates[argmax(scores)]

  - backward_coherence: aligns the FIRST K actions of the new chunk with
    the LAST K actions of the previously-emitted chunk (no chunk-boundary jumps).
  - forward_contrast: penalizes candidates that match a "weak" reference
    (e.g., a policy with shorter context). Higher = more distinct from weak ref.

Phase 1 (this module): backward_coherence only — minimal viable BID. The
forward_contrast term needs a "weak reference" policy which we don't have
natively; deferred to Phase 1.5 if measured signal warrants.

Composition with A2C2 hook:
    `--a2c2-checkpoint=<path>` : current head-based correction (default OFF; opt-in)
    `--bid-num-candidates=<N>` : NEW — enables BID selection at inference
    `--bid-coherence-window=<K>`: how many trailing actions of previous chunk to score against
Mutually exclusive in Phase 1: BID and head correction are both "fix
staleness" approaches; running both at once compounds risk + cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BIDConfig:
    """BID inference-time selection config.

    Defaults match arxiv 2408.17355 §4 baseline:
    - n_candidates=8: balance between selection quality (+monotonic with N up to ~16)
      and inference cost (Nx flow-matching denoise loops per chunk).
    - coherence_window=5: last 5 actions of previous chunk vs first 5 of new chunk.
      Robust to per-step noise; captures motion direction at chunk boundary.
    - coherence_metric='l2': L2 norm of (new[:K] - prev[-K:]) — lower is better.
      Alternatives ('cos' for cosine similarity) deferred to Phase 1.5 if needed.
    """

    n_candidates: int = 8
    coherence_window: int = 5
    coherence_metric: str = "l2"  # 'l2' | 'cos'

    def __post_init__(self):
        if self.n_candidates < 2:
            raise ValueError(f"n_candidates must be >= 2 to make a selection, got {self.n_candidates}")
        if self.coherence_window < 1:
            raise ValueError(f"coherence_window must be >= 1, got {self.coherence_window}")
        if self.coherence_metric not in ("l2", "cos"):
            raise ValueError(f"coherence_metric must be 'l2' or 'cos', got {self.coherence_metric!r}")


def score_backward_coherence(
    candidate: np.ndarray,
    previous_chunk: np.ndarray,
    *,
    window: int = 5,
    metric: str = "l2",
) -> float:
    """Score a candidate chunk by alignment with the trailing window of the
    previously-executed chunk.

    Args:
        candidate: shape (chunk_size, action_dim) — the candidate's actions.
        previous_chunk: same shape — what was actually executed last call.
        window: how many timesteps to compare (last K of prev vs first K of new).
        metric: 'l2' (lower is better — returns -L2 so higher = more coherent) or
                'cos' (higher is better — cosine similarity averaged across the window).

    Returns:
        Coherence score: higher = more coherent. Caller picks argmax.
    """
    if candidate.shape != previous_chunk.shape:
        raise ValueError(
            f"candidate {candidate.shape} != previous_chunk {previous_chunk.shape}"
        )
    if window > candidate.shape[0]:
        raise ValueError(
            f"window={window} exceeds chunk_size={candidate.shape[0]}"
        )

    head = candidate[:window].astype(np.float64)            # first K of new
    tail = previous_chunk[-window:].astype(np.float64)      # last K of prev

    if metric == "l2":
        # Negate L2 distance so higher score = more coherent.
        return -float(np.linalg.norm(head - tail))

    # Cosine: averaged per-timestep cosine similarity over the K-window.
    head_norms = np.linalg.norm(head, axis=-1) + 1e-9
    tail_norms = np.linalg.norm(tail, axis=-1) + 1e-9
    cos_per_step = np.sum(head * tail, axis=-1) / (head_norms * tail_norms)
    return float(np.mean(cos_per_step))


def select_chunk_bid(
    candidates: list[np.ndarray],
    previous_chunk: np.ndarray | None,
    config: BIDConfig,
) -> tuple[int, list[float]]:
    """Pick the best candidate via backward coherence with previous_chunk.

    Cold-start (previous_chunk is None) returns candidate 0 — the policy's
    "natural" first sample. Equivalent to current behavior.

    Args:
        candidates: list of N chunks, each shape (chunk_size, action_dim).
        previous_chunk: same shape, or None for cold-start.
        config: BIDConfig.

    Returns:
        (best_idx, scores): index into `candidates` of the chosen chunk, and
        the per-candidate scores for telemetry / regression debugging.
    """
    if len(candidates) < 2:
        raise ValueError(f"need >= 2 candidates to select; got {len(candidates)}")
    if previous_chunk is None:
        # Cold-start: no history to score against. Return the first candidate.
        return 0, [0.0] * len(candidates)

    scores = [
        score_backward_coherence(
            c, previous_chunk,
            window=config.coherence_window,
            metric=config.coherence_metric,
        )
        for c in candidates
    ]
    best_idx = int(np.argmax(scores))
    return best_idx, scores


def predict_chunk_bid(
    sample_chunk_fn: Callable[[int], np.ndarray],
    previous_chunk: np.ndarray | None,
    config: BIDConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """End-to-end BID prediction: sample N candidates + select.

    `sample_chunk_fn(i)` is called with the candidate index 0..N-1 and must
    return a chunk of shape (chunk_size, action_dim). The caller is responsible
    for varying the flow-matching noise per call (or whatever sampling
    randomness the policy supports).

    Returns:
        (chosen_chunk, telemetry) where telemetry includes:
          - 'selected_idx': which candidate was picked
          - 'scores': per-candidate coherence scores
          - 'n_candidates': how many were sampled
    """
    candidates: list[np.ndarray] = []
    for i in range(config.n_candidates):
        c = sample_chunk_fn(i)
        candidates.append(c)

    best_idx, scores = select_chunk_bid(candidates, previous_chunk, config)
    chosen = candidates[best_idx]
    return chosen, {
        "selected_idx": best_idx,
        "scores": scores,
        "n_candidates": config.n_candidates,
        "coherence_window": config.coherence_window,
        "coherence_metric": config.coherence_metric,
    }
