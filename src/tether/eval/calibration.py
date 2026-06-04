"""Calibration metrics for VLA action predictions.

Zollo 2025 showed that ECE (Expected Calibration Error), Brier Score, and
NLL (Negative Log-Likelihood) are the non-simulation metrics most
monotonically correlated with downstream real-task success. For VLA
deployment tooling, these are the most load-bearing statistics that don't
require a robot or a simulator:

- **ECE** (Guo 2017): average gap between predicted confidence and actual
  accuracy, binned across confidence buckets. Lower is better.
- **Brier Score** (Brier 1950): mean squared error between predicted
  probabilities and one-hot ground truth. Lower is better. Unlike NLL,
  Brier is bounded in [0, 2] for binary and doesn't blow up on confident
  wrong predictions — useful when NLL is dominated by one bad sample.
- **NLL** (standard): `-log p(y_true | x)` averaged across samples.
  Dominant-metric for well-calibrated model comparison.

All three work on **discrete** predicted distributions (classes or action
bins). For continuous flow-matching VLAs we discretize the action chunk
via a per-dimension quantile-bin scheme: each action-dim gets K bins, and
the "probability" at each position is an empirical density from multiple
flow-matching samples.

The `tether eval --calibration` CLI will:
  1. Load a lerobot dataset (e.g. `lerobot/libero-10`).
  2. For each (obs, ground_truth_action) pair, run the model K times with
     different noise seeds to get a sample distribution over action-bin
     probabilities.
  3. Compute ECE / Brier / NLL against the ground-truth bin.
  4. Write the numbers to `<export_dir>/CALIBRATION.md`.

This module provides the pure-NumPy metric functions. The dataset loop is
in `scripts/modal_calibration_eval.py` (to be written).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _check_probs(probs: np.ndarray) -> None:
    """Validate a probability tensor. Shape: [N, K]. Each row sums to 1."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D [N, K]; got shape {probs.shape}")
    if probs.shape[1] < 2:
        raise ValueError(f"probs must have >=2 classes; got {probs.shape[1]}")
    # Each row should be a probability distribution — tolerate mild drift
    # from float arithmetic (<1e-4).
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(
            f"probs rows don't sum to 1: min={row_sums.min():.4f} "
            f"max={row_sums.max():.4f}"
        )


def _check_labels(labels: np.ndarray, num_classes: int) -> None:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape {labels.shape}")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"labels out of range [0, {num_classes}); "
            f"got min={labels.min()} max={labels.max()}"
        )


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> dict[str, Any]:
    """Expected Calibration Error (Guo et al. 2017).

    Bin predictions by their top-class confidence, then compare each
    bin's average confidence to its actual accuracy. ECE is the
    weighted mean absolute gap.

    Args:
        probs: [N, K] predicted probabilities per sample.
        labels: [N] ground-truth class indices.
        num_bins: confidence-bucket count. 15 is the Guo 2017 default.

    Returns:
        {
          "ece": float,
          "num_bins": int,
          "per_bin_accuracy": list[float],
          "per_bin_confidence": list[float],
          "per_bin_count": list[int],
        }
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _check_probs(probs)
    _check_labels(labels, probs.shape[1])

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    n = len(labels)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    per_bin_accuracy: list[float] = []
    per_bin_confidence: list[float] = []
    per_bin_count: list[int] = []
    for i in range(num_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        # Upper bin edge is inclusive on the last bin so confidence=1.0 lands.
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        per_bin_count.append(count)
        if count == 0:
            per_bin_accuracy.append(0.0)
            per_bin_confidence.append(0.0)
            continue
        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        per_bin_accuracy.append(bin_acc)
        per_bin_confidence.append(bin_conf)
        ece += (count / n) * abs(bin_acc - bin_conf)

    return {
        "ece": float(ece),
        "num_bins": num_bins,
        "per_bin_accuracy": per_bin_accuracy,
        "per_bin_confidence": per_bin_confidence,
        "per_bin_count": per_bin_count,
    }


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Multi-class Brier Score = mean over samples of
    sum_k (p_k - onehot_k)^2. Bounded in [0, 2].

    Lower is better. A model that predicts the correct class with
    probability 1 has Brier=0; a uniform model over K classes has
    Brier = 1 - 1/K.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _check_probs(probs)
    _check_labels(labels, probs.shape[1])

    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    per_sample = np.sum((probs - onehot) ** 2, axis=1)
    return {
        "brier": float(per_sample.mean()),
        "per_sample_std": float(per_sample.std()),
        "n": n,
        "k": k,
    }


def compute_nll(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Negative log-likelihood = -mean(log p(y_true | x)).

    Clamps probabilities to [eps, 1] before log to avoid -inf on
    confident wrong predictions.

    Lower is better. A model that's always 100% confident in the
    correct class has NLL=0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _check_probs(probs)
    _check_labels(labels, probs.shape[1])

    n = len(labels)
    p_true = np.clip(probs[np.arange(n), labels], eps, 1.0)
    per_sample = -np.log(p_true)
    return {
        "nll": float(per_sample.mean()),
        "per_sample_std": float(per_sample.std()),
        "n": n,
    }


def discretize_action_samples(
    samples: np.ndarray,
    num_bins: int = 20,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Convert N flow-matching samples of a continuous action into a
    discrete probability distribution over action bins.

    Args:
        samples: [N_samples] array of continuous action values for one
            (obs, action-dim) pair — from running predict() N times with
            different noise seeds.
        num_bins: discretization resolution. 20 bins over [-1, 1] →
            each bin is 0.1 wide.
        value_range: action's numerical range (defaults to normalized
            [-1, 1] convention).

    Returns: [num_bins] probability array (sums to 1).

    Used by the /eval harness to build the K-way probability distribution
    the ECE/Brier/NLL functions expect.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1D; got shape {samples.shape}")

    lo, hi = value_range
    if hi <= lo:
        raise ValueError(f"value_range must have hi > lo; got {value_range}")

    edges = np.linspace(lo, hi, num_bins + 1)
    # Clip out-of-range samples to the nearest edge.
    clipped = np.clip(samples, lo, hi - 1e-9)
    bin_idx = np.digitize(clipped, edges) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    counts = np.bincount(bin_idx, minlength=num_bins)
    return counts.astype(np.float64) / counts.sum()


__all__ = [
    "compute_ece",
    "compute_brier",
    "compute_nll",
    "discretize_action_samples",
]
