# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Per-channel FP8 weights and the raw-accumulator FP8 GEMM.

``quantize_weight`` stores each output channel with its own scale
(``amax / 448``). ``gemm`` multiplies FP8 activations by FP8 weights with unit
scales and writes the BF16 accumulator; the consumer applies
``row_scale[m] * col_scale[n]``. BF16 keeps the full accumulator range while
its rounding error (2^-9 relative) stays far below E4M3 input quantization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0


@dataclass
class Fp8Weight:
    w: torch.Tensor  # [N, K] E4M3, row-major (K contiguous)
    scale: torch.Tensor  # [N] FP32 per-output-channel dequantization scale
    bias: torch.Tensor | None = None  # [N] FP32

    @property
    def shape(self):
        return tuple(self.w.shape)


def quantize_weight(w: torch.Tensor, bias: torch.Tensor | None = None, device=None) -> Fp8Weight:
    """Quantize ``w`` [N, K] (any float dtype) per output channel on ``device``."""
    w = w.to(device=device or w.device, dtype=torch.float32)
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (w / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(E4M3)
    b = None if bias is None else bias.to(device=w.device, dtype=torch.float32).contiguous()
    return Fp8Weight(q.contiguous(), scale.contiguous(), b)


_ONE: dict = {}


def _one(device):
    t = _ONE.get(device)
    if t is None:
        t = _ONE[device] = torch.ones((), device=device, dtype=torch.float32)
    return t


def gemm(a: torch.Tensor, w: Fp8Weight, out: torch.Tensor) -> torch.Tensor:
    """out[M, N] (BF16) = a[M, K] (E4M3) @ w.w[N, K]^T, unscaled."""
    one = _one(a.device)
    return torch._scaled_mm(a, w.w.t(), scale_a=one, scale_b=one, out_dtype=out.dtype, out=out)
