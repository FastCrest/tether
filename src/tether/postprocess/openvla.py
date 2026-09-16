"""OpenVLA action decoding.

OpenVLA is autoregressive: it generates one action token per action dimension,
then maps those token IDs onto centers between 256 action-bin edges.

Upstream semantics are:

    discretized = tokenizer_vocab_size - action_token_ids
    bin_idx = clip(discretized - 1, 0, len(bin_centers) - 1)
    action_normalized = bin_centers[bin_idx]
    action_unnorm = unnormalize(action_normalized, norm_stats[dataset])

For openvla-7b the tokenizer vocabulary is 32000 even though the language-model
logit dimension may be padded to 32064. Decode against the tokenizer vocabulary,
not the padded output width.

A single prompt-forward logits tensor is not equivalent to OpenVLA generation.
`decode_actions` therefore accepts exactly one retained next-token logit row per
autoregressive action step: shape [batch, action_dim, padded_vocab]. Callers that
already have generated token IDs should use `decode_token_ids` directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def logits_to_tokens(logits: np.ndarray, action_dim: int) -> np.ndarray:
    """Argmax one retained next-token logit row per generated action step.

    Args:
        logits: [batch, action_dim, padded_vocab], where row i is the next-token
            distribution observed at autoregressive action-generation step i.
        action_dim: exact number of generated action tokens.

    Returns:
        Token IDs with shape [batch, action_dim].
    """

    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError("action_dim must be a positive integer")
    if logits.ndim != 3:
        raise ValueError(f"Expected generation logits [batch, action_dim, vocab], got {logits.shape}")
    if logits.shape[1] != action_dim:
        raise ValueError(
            "OpenVLA decode requires exactly one next-token logit row per generated action step; "
            f"expected sequence dimension {action_dim}, got {logits.shape[1]}."
        )
    return np.argmax(logits, axis=-1)


def tokens_to_action_bins(
    token_ids: np.ndarray,
    vocab_size: int,
    n_bins: int = 256,
) -> np.ndarray:
    """Convert generated token IDs to upstream OpenVLA bin-center indices.

    `n_bins` is the number of *edges*. There are therefore `n_bins - 1`
    centers and valid returned indices are [0, n_bins - 2].
    """

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    tokens = np.asarray(token_ids)
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("token_ids must contain integers")
    discretized_actions = vocab_size - tokens
    return np.clip(discretized_actions - 1, 0, n_bins - 2)


def bins_to_normalized(
    bin_idx: np.ndarray,
    n_bins: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> np.ndarray:
    """Map bin-center indices to normalized actions using upstream semantics."""

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    if not np.isfinite(action_low) or not np.isfinite(action_high) or action_low >= action_high:
        raise ValueError("action_low/action_high must be finite with action_low < action_high")
    edges = np.linspace(action_low, action_high, n_bins, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    safe_idx = np.clip(np.asarray(bin_idx), 0, centers.shape[0] - 1)
    return centers[safe_idx]


def unnormalize_actions(
    normalized: np.ndarray,
    norm_stats: dict[str, Any],
    dataset_name: str,
) -> np.ndarray:
    """Apply OpenVLA's per-dataset q01/q99 action unnormalization."""

    if dataset_name not in norm_stats:
        raise KeyError(
            f"Dataset '{dataset_name}' not in norm_stats "
            f"(available: {list(norm_stats.keys())[:5]}...)"
        )
    stats = norm_stats[dataset_name]["action"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(q01)), dtype=bool)
    normalized = np.asarray(normalized, dtype=np.float32)

    if q01.shape != q99.shape or q01.shape != mask.shape:
        raise ValueError("OpenVLA q01, q99 and mask must have identical shapes")
    if normalized.shape[-1] != q01.shape[-1]:
        raise ValueError(
            f"Action dimension {normalized.shape[-1]} does not match norm stats {q01.shape[-1]}"
        )
    if not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError("OpenVLA q01/q99 norm stats must be finite")

    unnormalized = 0.5 * (normalized + 1.0) * (q99 - q01) + q01
    return np.where(mask, unnormalized, normalized)


def decode_token_ids(
    token_ids: np.ndarray,
    *,
    norm_stats: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    vocab_size: int = 32000,
    n_bins: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> np.ndarray:
    """Decode exact generated OpenVLA action token IDs to continuous actions."""

    bins = tokens_to_action_bins(token_ids, vocab_size=vocab_size, n_bins=n_bins)
    normalized = bins_to_normalized(
        bins,
        n_bins=n_bins,
        action_low=action_low,
        action_high=action_high,
    )
    if (norm_stats is None) != (dataset_name is None):
        raise ValueError("norm_stats and dataset_name must be provided together")
    if norm_stats is not None and dataset_name is not None:
        return unnormalize_actions(normalized, norm_stats, dataset_name)
    return normalized


def decode_actions(
    logits: np.ndarray,
    action_dim: int,
    norm_stats: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    vocab_size: int = 32000,
    n_bins: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> np.ndarray:
    """Decode retained autoregressive generation-step logits to actions.

    `logits` must contain exactly one next-token distribution per action step,
    not the sequence positions from one prompt forward pass.
    """

    token_ids = logits_to_tokens(logits, action_dim)
    return decode_token_ids(
        token_ids,
        norm_stats=norm_stats,
        dataset_name=dataset_name,
        vocab_size=vocab_size,
        n_bins=n_bins,
        action_low=action_low,
        action_high=action_high,
    )


__all__ = [
    "logits_to_tokens",
    "tokens_to_action_bins",
    "bins_to_normalized",
    "unnormalize_actions",
    "decode_token_ids",
    "decode_actions",
]
