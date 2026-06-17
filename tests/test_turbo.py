"""Tests for turbo action head optimization."""

import numpy as np
import torch
import torch.nn as nn
import pytest

from tether.kernels.turbo import TurboOptimizer, TurboConfig, TurboResult


class SimpleDenoisingModel(nn.Module):
    """Minimal model that mimics a VLA denoising step."""

    def __init__(self, action_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + 1 + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # Simple position embedding
        self.pos_embed = nn.Embedding(100, hidden)

    def forward(self, noisy_actions, timestep, position_ids):
        b, c, d = noisy_actions.shape
        pos = self.pos_embed(position_ids)  # [b, c, hidden]
        t = timestep.unsqueeze(1).unsqueeze(2).expand(b, c, 1)
        x = torch.cat([noisy_actions, t, pos], dim=-1)
        return self.net(x)


@pytest.fixture
def model():
    return SimpleDenoisingModel(action_dim=6, hidden=16)


@pytest.fixture
def dummy_inputs():
    noisy = torch.randn(1, 10, 6)
    pos = torch.arange(10).unsqueeze(0)
    return noisy, pos


class TestTurboFixed:
    def test_returns_correct_shape(self, model, dummy_inputs):
        optimizer = TurboOptimizer(TurboConfig(strategy="fixed"))
        noisy, pos = dummy_inputs
        result = optimizer.denoise_fixed(model, noisy, pos, num_steps=5)
        assert result.actions.shape == (1, 10, 6)
        assert result.steps_used == 5

    def test_records_velocity_norms(self, model, dummy_inputs):
        optimizer = TurboOptimizer(TurboConfig(strategy="fixed"))
        noisy, pos = dummy_inputs
        result = optimizer.denoise_fixed(model, noisy, pos, num_steps=5)
        assert len(result.per_step_velocity_norm) == 5
        assert all(v >= 0 for v in result.per_step_velocity_norm)

    def test_latency_recorded(self, model, dummy_inputs):
        optimizer = TurboOptimizer(TurboConfig(strategy="fixed"))
        noisy, pos = dummy_inputs
        result = optimizer.denoise_fixed(model, noisy, pos, num_steps=5)
        assert result.latency_ms > 0


class TestTurboAdaptive:
    def test_may_use_fewer_steps(self, model, dummy_inputs):
        config = TurboConfig(
            strategy="adaptive",
            min_steps=2,
            max_steps=10,
            warmup_steps=1,
            convergence_threshold=100.0,  # Very high threshold = converge fast
        )
        optimizer = TurboOptimizer(config)
        noisy, pos = dummy_inputs
        result = optimizer.denoise_adaptive(model, noisy, pos)
        assert result.steps_used <= 10
        # With high threshold, should converge early
        assert result.converged_early

    def test_respects_min_steps(self, model, dummy_inputs):
        config = TurboConfig(
            strategy="adaptive",
            min_steps=5,
            max_steps=10,
            warmup_steps=1,
            convergence_threshold=100.0,
        )
        optimizer = TurboOptimizer(config)
        noisy, pos = dummy_inputs
        result = optimizer.denoise_adaptive(model, noisy, pos)
        assert result.steps_used >= 5

    def test_speedup_calculated(self, model, dummy_inputs):
        optimizer = TurboOptimizer(TurboConfig(strategy="adaptive"))
        noisy, pos = dummy_inputs
        # Run fixed first to set baseline
        optimizer.denoise_fixed(model, noisy.clone(), pos, num_steps=10)
        result = optimizer.denoise_adaptive(model, noisy, pos)
        assert result.speedup_vs_fixed > 0


class TestTurboDenoise:
    def test_dispatch_fixed(self, model, dummy_inputs):
        optimizer = TurboOptimizer(TurboConfig(strategy="fixed"))
        noisy, pos = dummy_inputs
        result = optimizer.denoise(model, noisy, pos, num_steps=5)
        assert result.steps_used == 5

    def test_dispatch_adaptive(self, model, dummy_inputs):
        optimizer = TurboOptimizer(TurboConfig(strategy="adaptive"))
        noisy, pos = dummy_inputs
        result = optimizer.denoise(model, noisy, pos)
        assert result.steps_used <= 10


class TestBenchmarkStrategies:
    def test_compares_strategies(self, model):
        optimizer = TurboOptimizer()
        results = optimizer.benchmark_strategies(
            model, action_dim=6, chunk_size=10, device="cpu", n_trials=3
        )
        assert len(results["fixed"]) == 3
        assert len(results["adaptive"]) == 3
