"""Unit tests for the A2C2 trainer + numpy input encoder.

Path A unification (2026-04-25): the prior PyTorch A2C2Head was dropped.
Trainer + runtime now share the numpy A2C2Head from
`tether.kernels.a2c2_correction`. The numpy class itself is exercised by
tests/test_a2c2_correction_head.py + tests/test_a2c2_serve_integration.py;
this file covers the trainer-side encoder + the train/eval round-trip.

Pure-numpy; no torch, no Modal, no LIBERO. ~200 ms total.
"""
from __future__ import annotations

import numpy as np
import pytest

from tether.correction import (
    A2C2Config,
    A2C2Head,
    build_a2c2_input_batch,
    evaluate_mse,
    train_a2c2_head,
)
from tether.correction.a2c2_training import _gelu, _gelu_prime


class TestBuildBatchInput:
    def test_batch_shape(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, position_encoding_dim=8)
        n = 4
        x = build_a2c2_input_batch(
            base_actions=np.zeros((n, 7), dtype=np.float32),
            observations=np.zeros((n, 10), dtype=np.float32),
            chunk_positions=np.array([0, 1, 2, 3], dtype=np.int64),
            latency_ms_per_step=np.array([20.0, 40.0, 80.0, 100.0], dtype=np.float32),
            cfg=cfg,
        )
        assert x.shape == (n, cfg.input_dim)
        assert x.dtype == np.float32

    def test_input_dim_includes_scalar_latency(self):
        cfg = A2C2Config(action_dim=7, obs_dim=256, position_encoding_dim=32)
        # action(7) + obs(256) + pe(32) + scalar_latency(1) = 296
        assert cfg.input_dim == 296

    def test_action_dim_mismatch_raises(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, position_encoding_dim=8)
        with pytest.raises(ValueError, match="base_actions"):
            build_a2c2_input_batch(
                base_actions=np.zeros((4, 6), dtype=np.float32),
                observations=np.zeros((4, 10), dtype=np.float32),
                chunk_positions=np.zeros(4, dtype=np.int64),
                latency_ms_per_step=np.zeros(4, dtype=np.float32),
                cfg=cfg,
            )

    def test_obs_dim_mismatch_raises(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, position_encoding_dim=8)
        with pytest.raises(ValueError, match="observations"):
            build_a2c2_input_batch(
                base_actions=np.zeros((4, 7), dtype=np.float32),
                observations=np.zeros((4, 11), dtype=np.float32),
                chunk_positions=np.zeros(4, dtype=np.int64),
                latency_ms_per_step=np.zeros(4, dtype=np.float32),
                cfg=cfg,
            )

    def test_chunk_position_clamped_to_chunk_size(self):
        cfg = A2C2Config(
            action_dim=7, obs_dim=10, chunk_size=8, position_encoding_dim=8
        )
        # Position beyond chunk_size should clamp, not raise (matches the
        # runtime hook's "clamp positions" behavior for over-long chunks).
        x = build_a2c2_input_batch(
            base_actions=np.zeros((1, 7), dtype=np.float32),
            observations=np.zeros((1, 10), dtype=np.float32),
            chunk_positions=np.array([99], dtype=np.int64),
            latency_ms_per_step=np.array([10.0], dtype=np.float32),
            cfg=cfg,
        )
        assert x.shape == (1, cfg.input_dim)


class TestGeluDerivative:
    """Sanity-check the analytic gelu' against a finite-difference baseline."""

    def test_gelu_prime_matches_finite_difference(self):
        x = np.linspace(-3.0, 3.0, 50, dtype=np.float64)
        eps = 1e-5
        fd = (_gelu(x + eps) - _gelu(x - eps)) / (2 * eps)
        analytic = _gelu_prime(x)
        np.testing.assert_allclose(analytic, fd, atol=1e-4)

    def test_gelu_prime_zero_at_zero_is_half(self):
        # gelu'(0) = 0.5 * (1 + tanh(0)) + 0 = 0.5
        np.testing.assert_allclose(_gelu_prime(np.array([0.0])), [0.5], atol=1e-6)


class TestTrainerRoundtrip:
    def test_train_synthetic_loss_decreases(self):
        """Trainer should drive train_loss down over epochs on a learnable signal."""
        cfg = A2C2Config(
            action_dim=4, obs_dim=8, chunk_size=10,
            hidden_dim=16, num_hidden_layers=2, position_encoding_dim=8,
        )
        rng = np.random.default_rng(0)
        n = 200
        base = rng.standard_normal((n, cfg.action_dim)).astype(np.float32) * 0.3
        obs = rng.standard_normal((n, cfg.obs_dim)).astype(np.float32) * 0.5
        chunk_idx = rng.integers(0, cfg.chunk_size, size=n).astype(np.int64)
        latency = rng.uniform(20, 80, size=n).astype(np.float32)
        # Latency-conditioned target so the head has something to learn.
        scale = ((latency - 20) / 60.0)[:, None]
        target = (
            rng.standard_normal((n, cfg.action_dim)).astype(np.float32) * 0.05 * scale
        )

        result = train_a2c2_head(
            base_actions=base,
            observations=obs,
            chunk_positions=chunk_idx,
            latency_ms_per_step=latency,
            target_residuals=target,
            cfg=cfg,
            epochs=20, batch_size=32, lr=1e-2, val_split=0.1, seed=0,
            log_every_epoch=False,
        )
        first = result.metrics["epochs"][0]["train_loss"]
        last = result.metrics["epochs"][-1]["train_loss"]
        assert last < first * 0.5, f"loss did not decrease: first={first} last={last}"

    def test_train_zero_target_loss_decays_to_near_zero(self):
        """Output layer is zero-init -> initial correction is 0 -> loss already near 0
        when target is also zero. After training, loss should remain low."""
        cfg = A2C2Config(
            action_dim=4, obs_dim=8, chunk_size=10,
            hidden_dim=16, num_hidden_layers=2, position_encoding_dim=8,
        )
        rng = np.random.default_rng(0)
        n = 100
        base = rng.standard_normal((n, cfg.action_dim)).astype(np.float32)
        obs = rng.standard_normal((n, cfg.obs_dim)).astype(np.float32)
        chunk_idx = rng.integers(0, cfg.chunk_size, size=n).astype(np.int64)
        latency = rng.uniform(20, 80, size=n).astype(np.float32)
        target = np.zeros_like(base)

        result = train_a2c2_head(
            base_actions=base, observations=obs,
            chunk_positions=chunk_idx, latency_ms_per_step=latency,
            target_residuals=target,
            cfg=cfg, epochs=3, batch_size=16, lr=1e-3, val_split=0.1, seed=0,
            log_every_epoch=False,
        )
        # First-epoch loss starts at 0 (zero-init output layer * zero target)
        # and stays small. Use a relaxed bound — random init in early layers
        # can perturb things slightly through Adam.
        assert result.metrics["epochs"][-1]["val_loss"] < 0.5

    def test_save_load_roundtrip(self, tmp_path):
        cfg = A2C2Config(
            action_dim=4, obs_dim=8, chunk_size=10,
            hidden_dim=16, num_hidden_layers=2, position_encoding_dim=8,
        )
        rng = np.random.default_rng(0)
        n = 50
        base = rng.standard_normal((n, cfg.action_dim)).astype(np.float32)
        obs = rng.standard_normal((n, cfg.obs_dim)).astype(np.float32)
        chunk_idx = rng.integers(0, cfg.chunk_size, size=n).astype(np.int64)
        latency = rng.uniform(20, 80, size=n).astype(np.float32)
        scale = ((latency - 20) / 60.0)[:, None]
        target = rng.standard_normal((n, cfg.action_dim)).astype(np.float32) * 0.05 * scale

        result = train_a2c2_head(
            base_actions=base, observations=obs,
            chunk_positions=chunk_idx, latency_ms_per_step=latency,
            target_residuals=target,
            cfg=cfg, epochs=2, batch_size=16, lr=1e-2, val_split=0.1, seed=0,
            log_every_epoch=False,
        )
        path = tmp_path / "head.npz"
        result.head.save(path)
        assert path.exists()

        loaded = A2C2Head.from_checkpoint(path)
        assert loaded.config == cfg
        # Forward on the same input should match exactly (deterministic).
        i = 7
        a = result.head.forward(
            base_action=base[i], observation=obs[i],
            chunk_position=int(chunk_idx[i]), latency_estimate_ms=float(latency[i]),
        )
        b = loaded.forward(
            base_action=base[i], observation=obs[i],
            chunk_position=int(chunk_idx[i]), latency_estimate_ms=float(latency[i]),
        )
        np.testing.assert_array_equal(a, b)


class TestEvaluateMse:
    def test_zero_data_returns_nan(self):
        cfg = A2C2Config(action_dim=4, obs_dim=8, position_encoding_dim=8)
        head = A2C2Head.random_init(cfg, seed=0)
        out = evaluate_mse(
            head,
            base_actions=np.zeros((0, 4), dtype=np.float32),
            observations=np.zeros((0, 8), dtype=np.float32),
            chunk_positions=np.zeros(0, dtype=np.int64),
            latency_ms_per_step=np.zeros(0, dtype=np.float32),
            target_residuals=np.zeros((0, 4), dtype=np.float32),
        )
        assert np.isnan(out)

    def test_zero_init_head_against_zero_target_is_near_zero(self):
        """Output layer of train_a2c2_head's _init starts at zero, so an
        untrained head emits correction=0 -> MSE vs zero target is exactly 0."""
        from tether.correction.a2c2_training import _init_head_for_training
        cfg = A2C2Config(action_dim=4, obs_dim=8, position_encoding_dim=8)
        head = _init_head_for_training(cfg, seed=0)
        rng = np.random.default_rng(1)
        n = 20
        out = evaluate_mse(
            head,
            base_actions=rng.standard_normal((n, 4)).astype(np.float32),
            observations=rng.standard_normal((n, 8)).astype(np.float32),
            chunk_positions=rng.integers(0, 50, size=n).astype(np.int64),
            latency_ms_per_step=rng.uniform(20, 80, size=n).astype(np.float32),
            target_residuals=np.zeros((n, 4), dtype=np.float32),
        )
        assert out == 0.0
