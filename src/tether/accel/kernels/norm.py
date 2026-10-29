# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Row-wise residual update, normalization and FP8 quantization in one pass.

``residual_norm_quant`` is the single building block for every pre-norm site:

    if delta:  res[m] += (delta[m] * delta_row[m] * delta_col + bias) * gate
    y[m] = norm(res[m])                       # RMS or mean-centred LayerNorm
    y[m] = y[m] * weight + beta               # optional learned affine
    y[m] = y[m] * (1 + mod_scale) + mod_shift # optional adaptive (AdaRMS/AdaLN)
    q[m], s[m] = fp8(y[m] / s[m]), amax(|y[m]|) / 448

``res`` (FP32, or BF16 in the LingBot-VA engine) is updated in place: the sum is
formed in FP32 and the value stored in ``res``'s dtype is the one normalized.
``delta`` is a raw FP8-GEMM accumulator (BF16) whose per-row/per-column scales
are applied here. Each program owns one row, so the quantization scale is exact
for that token.
"""

from __future__ import annotations

import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _residual_norm_quant_kernel(
    RES,
    DELTA,
    DROW,
    DCOL,
    BIAS,
    GATE,
    W,
    B,
    MOD_SCALE,
    MOD_SHIFT,
    Q,
    QS,
    Y,
    RW_W,
    RW_B,
    RW_OUT,
    MOD_IDX,
    stride_res,
    stride_delta,
    stride_q,
    stride_y,
    D,
    eps,
    HAS_DELTA: tl.constexpr,
    HAS_DROW: tl.constexpr,
    HAS_DCOL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    WRITE_RES: tl.constexpr,
    LAYERNORM: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    QUANT: tl.constexpr,
    WRITE_Y: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTER_E: tl.constexpr,
    ROUTER_K: tl.constexpr,
    ROUTER_SCALE: tl.constexpr,
    HAS_MOD_IDX: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    if HAS_MOD_IDX:
        # Per-row modulation: gate / scale / shift tables are [n, D]; row r uses entry MOD_IDX[r].
        mofs = tl.load(MOD_IDX + row) * D
    else:
        mofs = 0
    x = tl.load(RES + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DELTA:
        d = tl.load(DELTA + row * stride_delta + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_DROW:
            d = d * tl.load(DROW + row)
        if HAS_DCOL:
            d = d * tl.load(DCOL + cols, mask=mask, other=0.0)
        if HAS_BIAS:
            d = d + tl.load(BIAS + cols, mask=mask, other=0.0)
        if HAS_GATE:
            d = d * tl.load(GATE + mofs + cols, mask=mask, other=0.0)
        x = x + d
        if WRITE_RES:
            tl.store(RES + row * stride_res + cols, x.to(RES.dtype.element_ty), mask=mask)
            x = x.to(RES.dtype.element_ty).to(tl.float32)
    if LAYERNORM:
        mean = tl.sum(x, 0) / D
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, 0) / D
        y = xc * tl.math.rsqrt(var + eps)
    else:
        var = tl.sum(x * x, 0) / D
        y = x * tl.math.rsqrt(var + eps)
    if HAS_W:
        y = y * tl.load(W + cols, mask=mask, other=0.0)
    if HAS_B:
        y = y + tl.load(B + cols, mask=mask, other=0.0)
    if ADAPTIVE:
        y = y * (1.0 + tl.load(MOD_SCALE + mofs + cols, mask=mask, other=0.0)) + tl.load(
            MOD_SHIFT + mofs + cols, mask=mask, other=0.0
        )
    if WRITE_Y:
        tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)
    if QUANT:
        amax = tl.max(tl.where(mask, tl.abs(y), 0.0), 0)
        amax = tl.maximum(amax, 1e-8)
        tl.store(QS + row, amax / 448.0)
        tl.store(Q + row * stride_q + cols, (y * (448.0 / amax)).to(tl.float8e4nv), mask=mask)
    if ROUTER_E > 0:
        # Sigmoid top-k MoE routing on the normalized row (selection uses the correction bias).
        e = tl.arange(0, ROUTER_E)
        wr = tl.load(RW_W + e[:, None] * D + cols[None, :], mask=mask[None, :], other=0.0)
        logits = tl.sum(wr * tl.where(mask, y, 0.0)[None, :], 1)
        sg = 1.0 / (1.0 + tl.exp(-logits))
        sel = sg + tl.load(RW_B + e)
        chosen = tl.zeros((ROUTER_E,), tl.float32)
        for _ in tl.static_range(ROUTER_K):
            i = tl.argmax(sel, 0)
            chosen = tl.where(e == i, 1.0, chosen)
            sel = tl.where(e == i, -float("inf"), sel)
        ws = sg * chosen
        tl.store(RW_OUT + row * ROUTER_E + e, ws / (tl.sum(ws, 0) + 1e-20) * ROUTER_SCALE)


def residual_norm_quant(
    res,
    *,
    delta=None,
    delta_row=None,
    delta_col=None,
    bias=None,
    gate=None,
    write_res=True,
    layernorm=False,
    weight=None,
    beta=None,
    mod_scale=None,
    mod_shift=None,
    q=None,
    q_scale=None,
    y=None,
    eps=1e-6,
    num_warps=None,
    router=None,
    mod_index=None,
):
    """``router=(gate_w [E, D], e_bias [E], out [M, E], top_k, scale)`` additionally writes MoE
    routing weights computed from the normalized row."""
    """See module docstring. ``res`` [M, D] FP32 (BF16 for LingBot-VA); optional tensors broadcast over rows."""
    M, D = res.shape
    dummy = res
    block = triton.next_power_of_2(D)
    warps = num_warps or (4 if block <= 1024 else 8)
    _residual_norm_quant_kernel[(M,)](
        res,
        delta if delta is not None else dummy,
        delta_row if delta_row is not None else dummy,
        delta_col if delta_col is not None else dummy,
        bias if bias is not None else dummy,
        gate if gate is not None else dummy,
        weight if weight is not None else dummy,
        beta if beta is not None else dummy,
        mod_scale if mod_scale is not None else dummy,
        mod_shift if mod_shift is not None else dummy,
        q if q is not None else dummy,
        q_scale if q_scale is not None else dummy,
        y if y is not None else dummy,
        router[0] if router else dummy,
        router[1] if router else dummy,
        router[2] if router else dummy,
        mod_index if mod_index is not None else dummy,
        res.stride(0),
        delta.stride(0) if delta is not None else 0,
        q.stride(0) if q is not None else 0,
        y.stride(0) if y is not None else 0,
        D,
        eps,
        HAS_DELTA=delta is not None,
        HAS_DROW=delta_row is not None,
        HAS_DCOL=delta_col is not None,
        HAS_BIAS=bias is not None,
        HAS_GATE=gate is not None,
        WRITE_RES=write_res and delta is not None,
        LAYERNORM=layernorm,
        HAS_W=weight is not None,
        HAS_B=beta is not None,
        ADAPTIVE=mod_scale is not None,
        QUANT=q is not None,
        WRITE_Y=y is not None,
        BLOCK=block,
        ROUTER_E=router[0].shape[0] if router else 0,
        ROUTER_K=router[3] if router else 0,
        ROUTER_SCALE=float(router[4]) if router else 1.0,
        HAS_MOD_IDX=mod_index is not None,
        num_warps=warps,
    )


@triton.jit
def _row_quant_kernel(
    X,
    ROW,
    COL,
    BIAS,
    Q,
    QS,
    stride_x,
    stride_q,
    D,
    HAS_ROW: tl.constexpr,
    HAS_COL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GELU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_ROW:
        x = x * tl.load(ROW + row)
    if HAS_COL:
        x = x * tl.load(COL + cols, mask=mask, other=0.0)
    if HAS_BIAS:
        x = x + tl.load(BIAS + cols, mask=mask, other=0.0)
    if GELU:
        # tanh approximation, as used by SigLIP / Gemma ("gelu_pytorch_tanh").
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        x = 0.5 * x * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(x), 0.0), 0), 1e-8)
    tl.store(QS + row, amax / 448.0)
    tl.store(Q + row * stride_q + cols, (x * (448.0 / amax)).to(tl.float8e4nv), mask=mask)


def row_quant(x, q, q_scale, *, row=None, col=None, bias=None, gelu=False, num_warps=None):
    """q[m] = fp8((x[m] * row[m] * col + bias) [-> gelu] / s[m]) with per-row s."""
    M, D = x.shape
    block = triton.next_power_of_2(D)
    _row_quant_kernel[(M,)](
        x,
        row if row is not None else x,
        col if col is not None else x,
        bias if bias is not None else x,
        q,
        q_scale,
        x.stride(0),
        q.stride(0),
        D,
        HAS_ROW=row is not None,
        HAS_COL=col is not None,
        HAS_BIAS=bias is not None,
        GELU=gelu,
        BLOCK=block,
        num_warps=num_warps or (4 if block <= 2048 else 8),
    )


@triton.jit
def _dequant_rows_kernel(
    X,
    ROW,
    COL,
    BIAS,
    Y,
    stride_x,
    stride_y,
    D,
    HAS_ROW: tl.constexpr,
    HAS_COL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_ROW:
        x = x * tl.load(ROW + row)
    if HAS_COL:
        x = x * tl.load(COL + cols, mask=mask, other=0.0)
    if HAS_BIAS:
        x = x + tl.load(BIAS + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride_y + cols, x.to(Y.dtype.element_ty), mask=mask)


def dequant_rows(x, y, *, row=None, col=None, bias=None, block=None, num_warps=4):
    """y[m] = x[m] * row[m] * col + bias (any float dtypes)."""
    M, D = x.shape
    block = block or min(1024, triton.next_power_of_2(D))
    _dequant_rows_kernel[(M, triton.cdiv(D, block))](
        x,
        row if row is not None else x,
        col if col is not None else x,
        bias if bias is not None else x,
        y,
        x.stride(0),
        y.stride(0),
        D,
        HAS_ROW=row is not None,
        HAS_COL=col is not None,
        HAS_BIAS=bias is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
