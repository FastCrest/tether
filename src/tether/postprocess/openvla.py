"""OpenVLA action decoding.

OpenVLA emits actions as the last `action_dim` tokens of the LM output,
which map onto the top N bins of the vocabulary via:

    bin_idx = vocab_size - token_id - 1
    action_normalized = linspace(action_low, action_high, N)[bin_idx]
    action_unnorm = unnormalize(action_normalized, norm_stats[dataset])

This module provides the postprocessing step that wraps a standard
Llama ONNX/PyTorch forward pass and turns its logits into actions.
Works with either the full LM output (logits over vocab) or the
pre-argmaxed token IDs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def logits_to_tokens(logits: np.ndarray, action_dim: int) -> np.ndarray:
    """Take the argmax of the last `action_dim` positions.

    Args:
        logits: shape [batch, seq_len, vocab] — raw LM logits
        action_dim: number of action dimensions (commonly 7)

    Returns:
        token ids of shape [batch, action_dim]
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [b, seq, vocab], got {logits.shape}")
    last = logits[:, -action_dim:, :]
    return np.argmax(last, axis=-1)


def tokens_to_action_bins(
    token_ids: np.ndarray,
    vocab_size: int,
    n_bins: int = 256,
) -> np.ndarray:
    """Convert predicted tokens to bin indices in [0, n_bins).

    OpenVLA assigns the top n_bins tokens of the Llama vocab to actions:
        bin_idx = vocab_size - token_id - 1
    """
    bin_idx = vocab_size - token_ids - 1
    return np.clip(bin_idx, 0, n_bins - 1)


def bins_to_normalized(
    bin_idx: np.ndarray,
    n_bins: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> np.ndarray:
    """Map bin indices to normalized actions in [action_low, action_high]."""
    bins = np.linspace(action_low, action_high, n_bins, dtype=np.float32)
    return bins[bin_idx]


def unnormalize_actions(
    normalized: np.ndarray,
    norm_stats: dict[str, Any],
    dataset_name: str,
) -> np.ndarray:
    """Apply OpenVLA's per-dataset unnormalization (q01 / q99 based).

    OpenVLA's config.json embeds norm_stats per training dataset. The
    unnormalization formula is:

        action = 0.5 * (normalized + 1) * (q99 - q01) + q01

    for dims where `mask` is true; unchanged otherwise.

    Args:
        normalized: shape [..., action_dim], values in [-1, 1]
        norm_stats: dict loaded from OpenVLA's config.json["norm_stats"]
        dataset_name: key into norm_stats, e.g. "bridge_orig"
    """
    if dataset_name not in norm_stats:
        raise KeyError(
            f"Dataset '{dataset_name}' not in norm_stats "
            f"(available: {list(norm_stats.keys())[:5]}...)"
        )
    stats = norm_stats[dataset_name]["action"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(q01)))

    # Linear unnormalization for masked dims; passthrough for unmasked.
    unnorm = 0.5 * (normalized + 1.0) * (q99 - q01) + q01
    return np.where(mask, unnorm, normalized)


def decode_actions(
    logits: np.ndarray,
    action_dim: int,
    norm_stats: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    vocab_size: int = 32064,
    n_bins: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> np.ndarray:
    """End-to-end OpenVLA action decode: logits → unnormalized actions.

    If `norm_stats` and `dataset_name` are provided, returns actions in
    the dataset's native scale (m/s, rad/s, etc.). Otherwise returns
    normalized actions in [action_low, action_high].

    Args:
        logits: [batch, seq_len, vocab] from LM forward
        action_dim: number of action dimensions (e.g. 7 for Bridge)
        norm_stats: OpenVLA config.json["norm_stats"] dict (optional)
        dataset_name: key into norm_stats (optional)

    Returns:
        actions of shape [batch, action_dim]
    """
    token_ids = logits_to_tokens(logits, action_dim)
    bin_idx = tokens_to_action_bins(token_ids, vocab_size, n_bins)
    normalized = bins_to_normalized(bin_idx, n_bins, action_low, action_high)

    if norm_stats is not None and dataset_name is not None:
        return unnormalize_actions(normalized, norm_stats, dataset_name)
    return normalized
