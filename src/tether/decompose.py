"""Op decomposition for ONNX/TensorRT compatibility.

RMSNorm and RoPE are not supported by TensorRT's ONNX parser.
This module replaces them with equivalent elementwise operations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DecomposedRMSNorm(nn.Module):
    """RMSNorm decomposed into elementwise ops for ONNX export.

    Replaces: y = x * rsqrt(mean(x^2) + eps) * weight

    Used for Llama-family RMSNorm (weight centered at 1). For Gemma-family
    (weight centered at 0 with `(1 + weight)` scaling), use DecomposedGemmaRMSNorm
    or pre-transform the weight via `weight_for_gemma(orig_weight)`.
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).to(x.dtype)


class DecomposedGemmaRMSNorm(nn.Module):
    """Gemma-family RMSNorm with `(1 + weight)` parameterization.

    Gemma trains the norm weight centered at 0, then applies `(1 + weight)`
    at inference. Used by PaliGemma (pi0's backbone). If you swap GemmaRMSNorm
    with this variant and copy the weight directly, the numerics match. If you
    try to use DecomposedRMSNorm you must pre-transform: `weight -> (1 + weight)`.
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return (x_normed * (1.0 + self.weight)).to(x.dtype)


def swap_rmsnorm_variants(model: nn.Module) -> int:
    """Replace every RMSNorm-family layer with its decomposed equivalent.

    Handles both Llama-family (RMSNorm, LlamaRMSNorm, SmolLM2RMSNorm) and
    Gemma-family (GemmaRMSNorm) via the correct `(1 + weight)` parameterization.
    Returns the number of layers swapped.

    Pattern: walk all modules, match by class name, setattr on the parent.
    Safe to call multiple times — already-swapped layers are skipped.
    """
    llama_family = {"RMSNorm", "LlamaRMSNorm", "SmolLM2RMSNorm"}
    gemma_family = {"GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm"}

    count = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            tn = type(child).__name__
            w = getattr(child, "weight", None)
            if w is None:
                continue
            eps = getattr(child, "variance_epsilon", None)
            if eps is None:
                eps = getattr(child, "eps", 1e-6)

            if tn in llama_family:
                new_norm = DecomposedRMSNorm(w.detach().clone(), eps=eps)
            elif tn in gemma_family:
                new_norm = DecomposedGemmaRMSNorm(w.detach().clone(), eps=eps)
            else:
                continue

            new_norm.to(dtype=w.dtype, device=w.device)
            setattr(parent, child_name, new_norm)
            count += 1
    return count


class DecomposedAdaRMSNorm(nn.Module):
    """Adaptive (time-conditioned) RMSNorm for pi0.5 expert layers.

    Given a time embedding, projects it through a dense layer into
    (scale, shift, gate) chunks, then modulates the normalized activations:

        normalized = x * rsqrt(mean(x^2) + eps)
        scale, shift, gate = dense(time_emb).chunk(3, dim=-1)
        out = gate * (normalized * (1 + scale) + shift)

    Follows AdaLN-style modulation from DiT. The `gate` applies on the
    residual side (after attention/MLP block). In pi0.5 the layernorm
    produces pre-norm output + gate; the gate is applied when the module
    is called with `return_gate=True`.

    The `dense.weight` has shape [3*hidden, time_dim] and `dense.bias`
    has shape [3*hidden] — confirmed on lerobot/pi05_base.
    """

    def __init__(
        self,
        hidden_size: int,
        time_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dense = nn.Linear(time_dim or hidden_size, 3 * hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # RMSNorm
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = (x * torch.rsqrt(variance + self.eps)).to(x.dtype)

        # Time conditioning
        projection = self.dense(time_emb)  # [..., 3*hidden]
        scale, shift, gate = projection.chunk(3, dim=-1)

        # Broadcast scale/shift/gate to [b, seq, hidden] from [b, hidden] or [b, 1, hidden]
        if scale.dim() == normed.dim() - 1:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            gate = gate.unsqueeze(1)

        modulated = normed * (1 + scale) + shift
        if return_gate:
            return modulated, gate
        return modulated


class DecomposedRotaryEmbedding(nn.Module):
    """Rotary Position Embedding decomposed for ONNX export.

    Precomputes sin/cos tables and applies via elementwise mul + gather.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        cos_cached = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin_cached = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)
        return q_embed, k_embed

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


def find_rmsnorm_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find all RMSNorm modules in a model."""
    results = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "RMSNorm" in module_type or "rmsnorm" in module_type.lower():
            results.append((name, module))
    return results


def find_rope_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find all RotaryEmbedding modules in a model."""
    results = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Rotary" in module_type or "rotary" in module_type.lower():
            results.append((name, module))
    return results


def replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named submodule in a parent module."""
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def decompose_rmsnorm(model: nn.Module) -> int:
    """Replace all RMSNorm modules with ONNX-compatible decompositions.

    Returns the number of modules replaced.
    """
    count = 0
    for name, module in find_rmsnorm_modules(model):
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        eps = getattr(module, "eps", 1e-6) or getattr(module, "variance_epsilon", 1e-6)
        decomposed = DecomposedRMSNorm(weight, eps=eps)
        replace_module(model, name, decomposed)
        count += 1
    return count


def decompose_rope(model: nn.Module, max_seq_len: int = 4096) -> int:
    """Replace all RotaryEmbedding modules with ONNX-compatible decompositions.

    Returns the number of modules replaced.
    """
    count = 0
    for name, module in find_rope_modules(model):
        dim = getattr(module, "dim", None)
        if dim is None:
            # Try to infer from inv_freq
            inv_freq = getattr(module, "inv_freq", None)
            if inv_freq is not None:
                dim = inv_freq.shape[0] * 2
            else:
                continue
        base = getattr(module, "base", 10000.0)
        decomposed = DecomposedRotaryEmbedding(dim, max_seq_len=max_seq_len, base=base)
        replace_module(model, name, decomposed)
        count += 1
    return count


def prepare_for_export(model: nn.Module, max_seq_len: int = 4096) -> dict[str, int]:
    """Decompose all ONNX-incompatible ops in a VLA model.

    Returns counts of replaced modules.
    """
    model.eval()
    rmsnorm_count = decompose_rmsnorm(model)
    rope_count = decompose_rope(model, max_seq_len=max_seq_len)
    return {"rmsnorm_replaced": rmsnorm_count, "rope_replaced": rope_count}
