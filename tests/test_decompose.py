"""Tests for RMSNorm and RoPE decomposition."""

import torch
import torch.nn as nn
import pytest

from tether.decompose import (
    DecomposedRMSNorm,
    DecomposedRotaryEmbedding,
    decompose_rmsnorm,
    decompose_rope,
    prepare_for_export,
)


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).to(x.dtype)


class TestDecomposedRMSNorm:
    def test_matches_original(self):
        dim = 64
        original = SimpleRMSNorm(dim)
        decomposed = DecomposedRMSNorm(original.weight.data, eps=original.eps)
        x = torch.randn(2, 10, dim)
        with torch.no_grad():
            ref = original(x)
            dec = decomposed(x)
        assert torch.allclose(ref, dec, atol=1e-5)

    def test_different_dtypes(self):
        dim = 32
        weight = torch.ones(dim)
        decomposed = DecomposedRMSNorm(weight)
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(1, 5, dim, dtype=dtype)
            out = decomposed(x)
            assert out.dtype == dtype


class TestDecomposedRotaryEmbedding:
    def test_output_shape(self):
        dim = 64
        rope = DecomposedRotaryEmbedding(dim, max_seq_len=128)
        q = torch.randn(1, 4, 10, dim)
        k = torch.randn(1, 4, 10, dim)
        pos = torch.arange(10).unsqueeze(0)
        q_out, k_out = rope(q, k, pos)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_different_positions(self):
        dim = 32
        rope = DecomposedRotaryEmbedding(dim)
        q = torch.randn(1, 2, 5, dim)
        k = torch.randn(1, 2, 5, dim)
        pos1 = torch.arange(5).unsqueeze(0)
        pos2 = torch.arange(5, 10).unsqueeze(0)
        q1, _ = rope(q, k, pos1)
        q2, _ = rope(q, k, pos2)
        assert not torch.allclose(q1, q2)


class TestModelDecomposition:
    def test_decompose_rmsnorm_in_model(self):
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = SimpleRMSNorm(64)
                self.norm2 = SimpleRMSNorm(64)
                self.linear = nn.Linear(64, 64)

        model = FakeModel()
        count = decompose_rmsnorm(model)
        assert count == 2
        assert isinstance(model.norm1, DecomposedRMSNorm)
        assert isinstance(model.norm2, DecomposedRMSNorm)
        assert isinstance(model.linear, nn.Linear)

    def test_prepare_for_export(self):
        class FakeVLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = SimpleRMSNorm(32)
                self.linear = nn.Linear(32, 32)

        model = FakeVLA()
        result = prepare_for_export(model)
        assert result["rmsnorm_replaced"] == 1
        assert result["rope_replaced"] == 0
