"""A2C2 correction head — paper arxiv 2509.23224.

Plug-in MLP residual that fixes RTC overshoot/undershoot on high-latency
deployments without retraining the base VLA. Lightweight (~100 KB weights),
sub-ms forward pass on Orin Nano.

Per a2c2-correction execution plan (Phase B.5 Day 1):
- Architecture: 3-4 hidden layers, 128-256 hidden dims, GELU activation
- Inputs: base_action (action_dim) + observation_encoding (256-512D) +
  chunk_position (positional encoding) + latency_estimate_ms
- Output: correction (action_dim, same shape as action) — added to base_action
- Auto-skip: serve runtime decides when to invoke based on latency_p95 +
  recent_success_rate (Day 3 wiring)

Loss + training live in scripts/train_a2c2_lerobot.py (Modal A100).
This module ships the inference-time substrate: architecture + forward
pass + checkpoint load + size/latency invariants.

Composition:
- RTC adapter (B.3) handles chunk-stitching on top of the policy's chunked
  output. A2C2 corrects per-step micro-deltas inside each chunk.
- Phase 2 SV-VLA macro-replan sits above A2C2 and triggers the heavy VLA
  re-plan only when divergence exceeds a threshold.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Output saturation scale (tanh(z/scale) * scale) per
# 2026-04-29-a2c2-correction_research_revisit. Bounds head output to
# [-3, 3] in normalized action space, matching typical pi0.5 action
# 3σ range. Prevents the magnitude-7 catastrophe observed in the
# 2026-04-26 N=50 LIBERO run while preserving zero-init cold-start.
OUTPUT_SATURATION_SCALE = 3.0


# Default architectural constants — tuned to the paper's ~100 KB target.
# 3 hidden layers × 128 dim ≈ 96 KB at FP32; 50% margin under the 150 KB
# ceiling enforced by test_a2c2_checkpoint_size().
_DEFAULT_HIDDEN_DIM = 128
_DEFAULT_NUM_HIDDEN_LAYERS = 3
_DEFAULT_OBS_DIM = 256
_DEFAULT_CHUNK_SIZE = 50
_DEFAULT_ACTION_DIM = 7

# Positional encoding for chunk position (which step within the 50-step
# chunk). 32-dim is enough resolution for 50 positions; cheap to include.
_POSITION_ENCODING_DIM = 32


@dataclass(frozen=True)
class A2C2Config:
    """Frozen architecture config for A2C2 head. Locked at construction so
    checkpoint load can verify shape compatibility."""

    action_dim: int = _DEFAULT_ACTION_DIM
    obs_dim: int = _DEFAULT_OBS_DIM
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    num_hidden_layers: int = _DEFAULT_NUM_HIDDEN_LAYERS
    position_encoding_dim: int = _POSITION_ENCODING_DIM
    # Phase 3 (2026-04-29): per-head saturation scale. Default 3.0 matches
    # the Phase 1 deploy (kernels/a2c2_correction.py shipped 2026-04-29 in
    # v0.7.2). Phase 2/2.1 found that L2 penalty alone hits a bimodal cliff
    # because tanh saturation at scale=3.0 is the binding constraint. Heads
    # trained with scale=1.5 (or other) bound output to ±scale and allow
    # the L2 penalty to actually steer corrections in [0, scale] range.
    output_saturation_scale: float = 3.0

    @property
    def input_dim(self) -> int:
        # base_action + obs + position_encoding + scalar latency_estimate
        return (
            self.action_dim
            + self.obs_dim
            + self.position_encoding_dim
            + 1
        )

    def estimated_param_count(self) -> int:
        """Estimated total parameters (weights + biases). Used by the size
        invariant test."""
        in_dim = self.input_dim
        h = self.hidden_dim
        # Layer 1: in_dim → h (weights + bias)
        params = in_dim * h + h
        # Hidden layers: h → h × (num_hidden_layers - 1)
        params += (h * h + h) * (self.num_hidden_layers - 1)
        # Output: h → action_dim
        params += h * self.action_dim + self.action_dim
        return params

    def estimated_size_bytes(self, dtype_bytes: int = 4) -> int:
        """Estimated checkpoint size assuming the given dtype (default FP32)."""
        return self.estimated_param_count() * dtype_bytes


def positional_encoding(position: int, dim: int) -> np.ndarray:
    """Sinusoidal positional encoding for chunk-position [0, chunk_size).

    Standard transformer-style: alternating sin/cos at exponentially-decaying
    frequencies. Cheap to compute (no learned params); deterministic; identical
    for the same position across calls.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    pe = np.zeros(dim, dtype=np.float32)
    for i in range(0, dim, 2):
        denom = math.pow(10000.0, i / dim)
        pe[i] = math.sin(position / denom)
        if i + 1 < dim:
            pe[i + 1] = math.cos(position / denom)
    return pe


class A2C2Head:
    """Inference-time A2C2 correction head — pure NumPy forward pass.

    PyTorch is the training backend (scripts/train_a2c2_lerobot.py); we ship
    a NumPy forward pass for inference because:
    - No torch import on the hot path (Tether serve doesn't require torch)
    - Sub-ms on CPU at this size (3 × 128 hidden, ~25k FLOPs per call)
    - Deterministic across machines (no cuDNN nondeterminism)

    Checkpoint format (Phase 1):
    - .npz file with keys: w0, b0, w1, b1, ..., wN, bN, action_dim, obs_dim,
      chunk_size, hidden_dim, num_hidden_layers
    - Loaded via `A2C2Head.from_checkpoint(path)`; shape validated against
      the embedded config to fail loud on schema drift

    Forward pass:
        head = A2C2Head.from_checkpoint("a2c2_lerobot_v1.npz")
        correction = head.forward(
            base_action=np.array([...], dtype=np.float32),  # (action_dim,)
            observation=np.array([...], dtype=np.float32),  # (obs_dim,)
            chunk_position=12,                              # int in [0, chunk_size)
            latency_estimate_ms=45.0,
        )
        actuated = base_action + correction  # apply
    """

    __slots__ = ("_config", "_weights", "_biases")

    def __init__(
        self,
        config: A2C2Config,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
    ):
        if len(weights) != config.num_hidden_layers + 1:
            raise ValueError(
                f"expected {config.num_hidden_layers + 1} weight matrices "
                f"(N hidden layers + 1 output), got {len(weights)}"
            )
        if len(biases) != len(weights):
            raise ValueError(
                f"weights and biases length mismatch: {len(weights)} vs {len(biases)}"
            )
        # Verify layer shapes against config.
        in_dim = config.input_dim
        h = config.hidden_dim
        if weights[0].shape != (h, in_dim):
            raise ValueError(
                f"first layer weight shape mismatch: expected ({h}, {in_dim}), "
                f"got {weights[0].shape}"
            )
        for i in range(1, config.num_hidden_layers):
            if weights[i].shape != (h, h):
                raise ValueError(
                    f"hidden layer {i} weight shape mismatch: expected ({h}, {h}), "
                    f"got {weights[i].shape}"
                )
        if weights[-1].shape != (config.action_dim, h):
            raise ValueError(
                f"output layer weight shape mismatch: expected "
                f"({config.action_dim}, {h}), got {weights[-1].shape}"
            )
        self._config = config
        self._weights = [w.astype(np.float32, copy=False) for w in weights]
        self._biases = [b.astype(np.float32, copy=False) for b in biases]

    @property
    def config(self) -> A2C2Config:
        return self._config

    @property
    def num_layers(self) -> int:
        return len(self._weights)

    def forward(
        self,
        *,
        base_action: np.ndarray,
        observation: np.ndarray,
        chunk_position: int,
        latency_estimate_ms: float,
    ) -> np.ndarray:
        """Compute the correction for one (base_action, observation) pair.

        Returns a same-shape array as `base_action`. Caller adds it to
        `base_action` to get the actuated value.

        Deterministic — same input always produces same output (no RNG).
        """
        cfg = self._config
        if base_action.shape != (cfg.action_dim,):
            raise ValueError(
                f"base_action shape mismatch: expected ({cfg.action_dim},), "
                f"got {base_action.shape}"
            )
        if observation.shape != (cfg.obs_dim,):
            raise ValueError(
                f"observation shape mismatch: expected ({cfg.obs_dim},), "
                f"got {observation.shape}"
            )
        if not (0 <= chunk_position < cfg.chunk_size):
            raise ValueError(
                f"chunk_position must be in [0, {cfg.chunk_size}), got "
                f"{chunk_position}"
            )

        # Build input vector.
        pe = positional_encoding(chunk_position, cfg.position_encoding_dim)
        x = np.concatenate([
            base_action.astype(np.float32, copy=False),
            observation.astype(np.float32, copy=False),
            pe,
            np.array([float(latency_estimate_ms)], dtype=np.float32),
        ])

        # Forward through hidden layers (GELU activation).
        for i in range(cfg.num_hidden_layers):
            x = self._weights[i] @ x + self._biases[i]
            x = _gelu(x)

        # Output layer with bounded saturation. Per-head scale (Phase 3,
        # 2026-04-29) read from config; default 3.0 preserves Phase 1 ship.
        z_out = self._weights[-1] @ x + self._biases[-1]
        scale = self._config.output_saturation_scale
        correction = np.tanh(z_out / scale) * scale
        return correction

    def to_checkpoint_dict(self) -> dict[str, np.ndarray]:
        """Serialize to a dict suitable for np.savez_compressed."""
        out: dict[str, Any] = {
            "action_dim": np.int32(self._config.action_dim),
            "obs_dim": np.int32(self._config.obs_dim),
            "chunk_size": np.int32(self._config.chunk_size),
            "hidden_dim": np.int32(self._config.hidden_dim),
            "num_hidden_layers": np.int32(self._config.num_hidden_layers),
            "position_encoding_dim": np.int32(self._config.position_encoding_dim),
            "output_saturation_scale": np.float32(self._config.output_saturation_scale),
        }
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            out[f"w{i}"] = w
            out[f"b{i}"] = b
        return out

    def save(self, path: str | Path) -> None:
        """Save to .npz. Use np.savez_compressed for ~3-5× smaller files."""
        path = Path(path)
        np.savez_compressed(path, **self.to_checkpoint_dict())

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "A2C2Head":
        """Load + verify schema. Backward-compat: pre-Phase-3 checkpoints
        (no output_saturation_scale field) load with the default 3.0,
        matching their training-time behavior."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"A2C2 checkpoint not found: {path}")
        data = np.load(path)
        config = A2C2Config(
            action_dim=int(data["action_dim"]),
            obs_dim=int(data["obs_dim"]),
            chunk_size=int(data["chunk_size"]),
            hidden_dim=int(data["hidden_dim"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            position_encoding_dim=int(data["position_encoding_dim"]),
            output_saturation_scale=(
                float(data["output_saturation_scale"])
                if "output_saturation_scale" in data.files
                else 3.0
            ),
        )
        n_layers = config.num_hidden_layers + 1
        weights = [data[f"w{i}"] for i in range(n_layers)]
        biases = [data[f"b{i}"] for i in range(n_layers)]
        return cls(config=config, weights=weights, biases=biases)

    @classmethod
    def random_init(
        cls,
        config: A2C2Config | None = None,
        *,
        seed: int = 0,
    ) -> "A2C2Head":
        """Create a randomly-initialized head — used for tests + skeleton
        checkpoints before training. Production use should always
        `from_checkpoint(...)` a trained file."""
        if config is None:
            config = A2C2Config()
        rng = np.random.default_rng(seed)
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        in_dim = config.input_dim
        h = config.hidden_dim
        # Layer 1
        w = rng.standard_normal((h, in_dim)).astype(np.float32) * (1.0 / math.sqrt(in_dim))
        b = np.zeros(h, dtype=np.float32)
        weights.append(w)
        biases.append(b)
        # Hidden layers
        for _ in range(config.num_hidden_layers - 1):
            w = rng.standard_normal((h, h)).astype(np.float32) * (1.0 / math.sqrt(h))
            b = np.zeros(h, dtype=np.float32)
            weights.append(w)
            biases.append(b)
        # Output
        w = rng.standard_normal((config.action_dim, h)).astype(np.float32) * (1.0 / math.sqrt(h))
        b = np.zeros(config.action_dim, dtype=np.float32)
        weights.append(w)
        biases.append(b)
        return cls(config=config, weights=weights, biases=biases)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (NumPy implementation; matches PyTorch default).

    Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(c0 * (x + c1 * x * x * x)))


__all__ = [
    "A2C2Config",
    "A2C2Head",
    "positional_encoding",
]
