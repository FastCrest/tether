"""A2C2 trainer — pure-numpy backprop + Adam optimizer.

Trains the inference-time A2C2Head (kernels/a2c2_correction.py) end-to-end
without torch. The head's runtime forward pass is single-sample numpy; this
module adds a vectorized batch forward + analytic backward + Adam step so
training scripts (train_a2c2.py, modal_b4_gate_fire.py, validate_a2c2_transfer.py)
can converge on the same architecture the runtime loads.

Replaces the prior PyTorch-based trainer in correction/a2c2_head.py. Path A
of the dual-implementation unification (see 2026-04-25 experiment note + ADR):
trainers and runtime now share the same A2C2Config + A2C2Head class. The
`.npz` checkpoint produced here loads directly into runtime/a2c2_hook.py.

Architecture (matches kernels/a2c2_correction.py exactly):
    Input  = concat([base_action, observation, positional_encoding(chunk_pos), [latency_ms]])
    Hidden = num_hidden_layers × hidden_dim, GELU activations
    Output = correction (action_dim,)

Loss: MSE between predicted correction and target_residual.
Optimizer: Adam with bias-correction (β1=0.9, β2=0.999, ε=1e-8 by default).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tether.kernels.a2c2_correction import (
    OUTPUT_SATURATION_SCALE,
    A2C2Config,
    A2C2Head,
    positional_encoding,
)


# Phase 1 loss fixes per 2026-04-29-a2c2-correction_research_revisit.
# Huber δ caps gradient on outliers; L2 magnitude penalty discourages
# the head from learning large-magnitude residuals to fit the tail of
# the target distribution.
HUBER_DELTA = 0.1
L2_MAGNITUDE_PENALTY = 0.01

logger = logging.getLogger(__name__)


def build_a2c2_input_batch(
    base_actions: np.ndarray,
    observations: np.ndarray,
    chunk_positions: np.ndarray,
    latency_ms_per_step: np.ndarray,
    cfg: A2C2Config,
) -> np.ndarray:
    """Vectorized version of A2C2Head's input assembly. Shape: (N, input_dim).

    Mirrors the per-sample concatenation in A2C2Head.forward() so train+eval+runtime
    see the same input layout. Positional encoding is computed via a small
    Python loop — sufficient for the typical dataset size (~25k rows × 5 epochs).
    Vectorize further if profiling shows it's hot.
    """
    if base_actions.ndim != 2 or base_actions.shape[1] != cfg.action_dim:
        raise ValueError(
            f"base_actions must be (N, {cfg.action_dim}), got {base_actions.shape}"
        )
    if observations.shape != (base_actions.shape[0], cfg.obs_dim):
        raise ValueError(
            f"observations must be (N, {cfg.obs_dim}), got {observations.shape}"
        )
    if chunk_positions.shape != (base_actions.shape[0],):
        raise ValueError(
            f"chunk_positions must be (N,), got {chunk_positions.shape}"
        )
    if latency_ms_per_step.shape != (base_actions.shape[0],):
        raise ValueError(
            f"latency_ms_per_step must be (N,), got {latency_ms_per_step.shape}"
        )

    n = base_actions.shape[0]
    pe_batch = np.empty((n, cfg.position_encoding_dim), dtype=np.float32)
    for i in range(n):
        pos = int(chunk_positions[i])
        pos = max(0, min(pos, cfg.chunk_size - 1))
        pe_batch[i] = positional_encoding(pos, cfg.position_encoding_dim)

    lat_col = latency_ms_per_step.astype(np.float32, copy=False).reshape(-1, 1)
    return np.concatenate(
        [
            base_actions.astype(np.float32, copy=False),
            observations.astype(np.float32, copy=False),
            pe_batch,
            lat_col,
        ],
        axis=-1,
    )


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (approximation form, matches PyTorch default + the
    runtime forward in kernels/a2c2_correction._gelu)."""
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(c0 * (x + c1 * x * x * x)))


def _gelu_prime(x: np.ndarray) -> np.ndarray:
    """Analytic derivative of the GELU approximation.

    g(x)  = 0.5 * x * (1 + tanh(u))   where u = c0 * (x + c1 * x^3)
    g'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * c0 * (1 + 3 * c1 * x^2)
    """
    c0 = 0.7978845608028654
    c1 = 0.044715
    u = c0 * (x + c1 * x * x * x)
    tanh_u = np.tanh(u)
    return 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u * tanh_u) * c0 * (1.0 + 3.0 * c1 * x * x)


def _forward_batch(
    head: A2C2Head, x_in: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Batched forward pass that retains intermediate activations for backward.

    x_in:  (N, input_dim)
    Returns:
        correction: (N, action_dim)
        activations: [h0=x_in, h1, ..., h_last_hidden]   (post-GELU)
        pre_activations: [z0, z1, ...]                   (pre-GELU)
    """
    cfg = head.config
    activations: list[np.ndarray] = [x_in]
    pre_activations: list[np.ndarray] = []
    h = x_in
    for i in range(cfg.num_hidden_layers):
        # Weights stored as (out, in); batched GEMM = h @ W.T + b
        z = h @ head._weights[i].T + head._biases[i]
        pre_activations.append(z)
        h = _gelu(z)
        activations.append(h)
    # Output layer + bounded saturation. Per-head scale (Phase 3) read from
    # config; default 3.0 matches Phase 1. The pre-saturation z_out is not
    # stored separately; the backward pass recovers d/dz via
    # 1 - (correction/scale)^2.
    z_out = h @ head._weights[-1].T + head._biases[-1]
    scale = cfg.output_saturation_scale
    correction = np.tanh(z_out / scale) * scale
    return correction, activations, pre_activations


def _backward_batch(
    head: A2C2Head,
    activations: list[np.ndarray],
    pre_activations: list[np.ndarray],
    correction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    """Compute gradients wrt all weights and biases.

    Loss per 2026-04-29 research-revisit:
        Huber(diff, δ=HUBER_DELTA) summed over action_dim, mean over batch
      + L2_MAGNITUDE_PENALTY * mean(||correction||^2)

    Backprop accounts for the tanh saturation on the output layer
    (correction = scale * tanh(z_out/scale)). Returns: (loss_value,
    grad_weights, grad_biases) with the same length as head._weights /
    head._biases.
    """
    cfg = head.config
    n = correction.shape[0]
    diff = correction - target  # (N, action_dim) — diff in post-tanh space

    # Huber loss (per-element, summed over action_dim, mean over batch).
    abs_diff = np.abs(diff)
    huber = np.where(
        abs_diff < HUBER_DELTA,
        0.5 * diff * diff,
        HUBER_DELTA * (abs_diff - 0.5 * HUBER_DELTA),
    )
    loss_huber = float(np.mean(np.sum(huber, axis=-1)))

    # L2 magnitude penalty on correction. Discourages the tail of the
    # target distribution from training the head toward large outputs.
    loss_mag = L2_MAGNITUDE_PENALTY * float(np.mean(np.sum(correction * correction, axis=-1)))
    loss = loss_huber + loss_mag

    # Gradient wrt correction (post-tanh):
    #   ∂Huber/∂correction = clip(diff, ±δ)
    #   ∂(λ‖correction‖²)/∂correction = 2λ * correction
    grad_huber = np.where(abs_diff < HUBER_DELTA, diff, HUBER_DELTA * np.sign(diff))
    grad_mag = 2.0 * L2_MAGNITUDE_PENALTY * correction
    grad_corr = (grad_huber + grad_mag) / n  # (N, action_dim)

    # Backprop through tanh saturation: d(scale * tanh(z/scale))/dz =
    # 1 - tanh²(z/scale) = 1 - (correction/scale)². Scale read from config
    # (Phase 3); default 3.0 matches Phase 1.
    scale = head.config.output_saturation_scale
    saturation_factor = 1.0 - (correction / scale) ** 2
    grad_out = grad_corr * saturation_factor  # (N, action_dim) — wrt z_out

    # Output layer (now uses post-tanh-backprop grad_out).
    h_last = activations[-1]  # (N, hidden_dim)
    grad_w_out = grad_out.T @ h_last  # (action_dim, hidden_dim)
    grad_b_out = grad_out.sum(axis=0)  # (action_dim,)
    grad_h = grad_out @ head._weights[-1]  # (N, hidden_dim)

    grad_weights: list[np.ndarray] = [None] * (cfg.num_hidden_layers + 1)  # type: ignore
    grad_biases: list[np.ndarray] = [None] * (cfg.num_hidden_layers + 1)  # type: ignore
    grad_weights[-1] = grad_w_out
    grad_biases[-1] = grad_b_out

    # Hidden layers (back-to-front)
    for i in range(cfg.num_hidden_layers - 1, -1, -1):
        grad_z = grad_h * _gelu_prime(pre_activations[i])  # (N, hidden_dim)
        grad_w = grad_z.T @ activations[i]  # (out_i, in_i)
        grad_b = grad_z.sum(axis=0)
        grad_weights[i] = grad_w
        grad_biases[i] = grad_b
        grad_h = grad_z @ head._weights[i]  # (N, in_i) for the next iter

    return loss, grad_weights, grad_biases


@dataclass
class _AdamState:
    """First and second moment buffers for one parameter tensor."""

    m: np.ndarray
    v: np.ndarray


class _AdamOptimizer:
    """Plain Adam with bias correction. Per-parameter state keyed by id."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._state: dict[int, _AdamState] = {}
        self._t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        for p, g in zip(params, grads):
            key = id(p)
            st = self._state.get(key)
            if st is None:
                st = _AdamState(m=np.zeros_like(p), v=np.zeros_like(p))
                self._state[key] = st
            st.m = self.beta1 * st.m + (1.0 - self.beta1) * g
            st.v = self.beta2 * st.v + (1.0 - self.beta2) * g * g
            m_hat = st.m / bc1
            v_hat = st.v / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _init_head_for_training(cfg: A2C2Config, seed: int) -> A2C2Head:
    """Xavier-init weights, zero biases, **zero output layer** (residual head
    starts as identity-correction).

    Mirrors the PyTorch original's zero-output trick so untrained heads return
    zero correction and don't perturb base actions during cold-start.
    """
    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    in_dim = cfg.input_dim
    h = cfg.hidden_dim
    # First layer
    w = rng.standard_normal((h, in_dim)).astype(np.float32) * (1.0 / math.sqrt(in_dim))
    weights.append(w)
    biases.append(np.zeros(h, dtype=np.float32))
    # Hidden layers
    for _ in range(cfg.num_hidden_layers - 1):
        w = rng.standard_normal((h, h)).astype(np.float32) * (1.0 / math.sqrt(h))
        weights.append(w)
        biases.append(np.zeros(h, dtype=np.float32))
    # Output: zero-initialized so untrained head emits zero correction
    weights.append(np.zeros((cfg.action_dim, h), dtype=np.float32))
    biases.append(np.zeros(cfg.action_dim, dtype=np.float32))
    return A2C2Head(config=cfg, weights=weights, biases=biases)


@dataclass
class A2C2TrainResult:
    head: A2C2Head
    metrics: dict[str, Any]


def train_a2c2_head(
    *,
    base_actions: np.ndarray,
    observations: np.ndarray,
    chunk_positions: np.ndarray,
    latency_ms_per_step: np.ndarray,
    target_residuals: np.ndarray,
    cfg: A2C2Config | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.1,
    seed: int = 42,
    log_every_epoch: bool = True,
) -> A2C2TrainResult:
    """Train an A2C2Head on (base, obs, chunk_pos, latency, target_residual) rows.

    All inputs are numpy arrays. The trainer:
    1. Initializes a fresh A2C2Head with the given config (Xavier + zero output)
    2. Splits rows into train / val by `val_split`
    3. Runs Adam for `epochs` epochs over mini-batches of size `batch_size`
    4. Returns the trained head + per-epoch metrics dict

    The head can be saved via `head.save(path)` to produce the .npz that
    runtime/a2c2_hook.py loads directly.
    """
    cfg = cfg or A2C2Config()
    n = base_actions.shape[0]
    if n == 0:
        raise ValueError("no training rows")
    if target_residuals.shape != base_actions.shape:
        raise ValueError(
            f"target_residuals shape {target_residuals.shape} != base_actions {base_actions.shape}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    head = _init_head_for_training(cfg, seed=seed)
    opt = _AdamOptimizer(lr=lr)

    # Pre-build the full feature matrix once — the same rows are sampled
    # repeatedly across epochs. ~N * input_dim * 4 bytes; for N=25k, input=296,
    # that's ~30 MB — fine.
    x_all = build_a2c2_input_batch(
        base_actions, observations, chunk_positions, latency_ms_per_step, cfg
    )
    y_all = target_residuals.astype(np.float32, copy=False)

    metrics: dict[str, Any] = {
        "epochs": [],
        "config": {
            "action_dim": cfg.action_dim,
            "obs_dim": cfg.obs_dim,
            "chunk_size": cfg.chunk_size,
            "hidden_dim": cfg.hidden_dim,
            "num_hidden_layers": cfg.num_hidden_layers,
            "position_encoding_dim": cfg.position_encoding_dim,
        },
    }

    params = [*head._weights, *head._biases]

    for epoch in range(epochs):
        # Shuffle the train indices each epoch
        rng.shuffle(train_idx)
        train_losses: list[float] = []
        for s in range(0, train_idx.shape[0], batch_size):
            batch = train_idx[s : s + batch_size]
            x_b = x_all[batch]
            y_b = y_all[batch]
            correction, acts, pres = _forward_batch(head, x_b)
            loss, gw, gb = _backward_batch(head, acts, pres, correction, y_b)
            grads = [*gw, *gb]
            opt.step(params, grads)
            train_losses.append(loss)

        # Validation — same loss formula as training (Huber + L2 mag
        # penalty) so train/val are directly comparable.
        x_v = x_all[val_idx]
        y_v = y_all[val_idx]
        pred_v, _, _ = _forward_batch(head, x_v)
        diff_v = pred_v - y_v
        abs_v = np.abs(diff_v)
        huber_v = np.where(
            abs_v < HUBER_DELTA,
            0.5 * diff_v * diff_v,
            HUBER_DELTA * (abs_v - 0.5 * HUBER_DELTA),
        )
        loss_huber_v = float(np.mean(np.sum(huber_v, axis=-1)))
        loss_mag_v = L2_MAGNITUDE_PENALTY * float(np.mean(np.sum(pred_v * pred_v, axis=-1)))
        val_loss = loss_huber_v + loss_mag_v
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        metrics["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        if log_every_epoch:
            logger.info(
                "epoch %d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss
            )

    metrics["final_val_mse"] = (
        metrics["epochs"][-1]["val_loss"] if metrics["epochs"] else float("nan")
    )
    return A2C2TrainResult(head=head, metrics=metrics)


def evaluate_mse(
    head: A2C2Head,
    *,
    base_actions: np.ndarray,
    observations: np.ndarray,
    chunk_positions: np.ndarray,
    latency_ms_per_step: np.ndarray,
    target_residuals: np.ndarray,
    batch_size: int = 256,
) -> float:
    """Compute element-wise MSE(predicted_correction, target_residual).

    Uses the same MSE definition as the training loss path so train + eval
    + gate report all reference the same metric.
    """
    n = base_actions.shape[0]
    if n == 0:
        return float("nan")
    cfg = head.config
    x_all = build_a2c2_input_batch(
        base_actions, observations, chunk_positions, latency_ms_per_step, cfg
    )
    sq = 0.0
    cnt = 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        pred, _, _ = _forward_batch(head, x_all[s:e])
        err = pred - target_residuals[s:e]
        sq += float(np.sum(err * err))
        cnt += err.size
    return sq / max(cnt, 1)


__all__ = [
    "A2C2TrainResult",
    "build_a2c2_input_batch",
    "evaluate_mse",
    "train_a2c2_head",
]
