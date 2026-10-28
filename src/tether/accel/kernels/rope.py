# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Split a fused QKV accumulator: rescale, optional per-head norm, rotary, store.

The QKV projection is one FP8 GEMM whose raw BF16 accumulator holds
``[Q heads | K heads | V heads]`` per token. This kernel applies the row/column
dequantization and optional bias, optional per-head RMSNorm of Q and K
(Qwen3-style ``q_norm``/``k_norm``), rotary embedding with split-half pairing,
and writes Q, K, V to their destinations (K/V may be KV-cache views).

Rotary angles are computed in FP32 as ``pos[t, section[i]] * inv_freq[i]``.
``pos`` is a device table ``[T, S]`` (S = 1 for 1D RoPE, 3 for M-RoPE), so a
captured graph follows whatever positions the current prompt needs. K/V rows
are written at ``kv_base + t`` where ``kv_base`` may also live on the device.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_split_kernel(
    QKV,
    ROW,
    COL,
    BIAS,
    QNW,
    KNW,
    QO,
    KO,
    VO,
    POS,
    SEC,
    INV_FREQ,
    KV_BASE,
    stride_qkv,
    sqo_t,
    sqo_h,
    sko_t,
    sko_h,
    svo_t,
    svo_h,
    stride_pos,
    kv_base_static,
    eps,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    HAS_ROW: tl.constexpr,
    HAS_COL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    QK_NORM: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_SEC: tl.constexpr,
    DYNAMIC_BASE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)  # [0, HQ) query heads, [HQ, HQ+HKV) key heads, then value heads
    HALF: tl.constexpr = HD // 2
    i = tl.arange(0, BLOCK_HALF)
    m = i < HALF
    base = QKV + t * stride_qkv + h * HD
    lo = tl.load(base + i, mask=m, other=0.0).to(tl.float32)
    hi = tl.load(base + HALF + i, mask=m, other=0.0).to(tl.float32)
    if HAS_ROW:
        r = tl.load(ROW + t)
        lo = lo * r
        hi = hi * r
    if HAS_COL:
        lo = lo * tl.load(COL + h * HD + i, mask=m, other=0.0)
        hi = hi * tl.load(COL + h * HD + HALF + i, mask=m, other=0.0)
    if HAS_BIAS:
        lo = lo + tl.load(BIAS + h * HD + i, mask=m, other=0.0)
        hi = hi + tl.load(BIAS + h * HD + HALF + i, mask=m, other=0.0)
    if h < HQ + HKV:
        if QK_NORM:
            rs = tl.math.rsqrt((tl.sum(lo * lo, 0) + tl.sum(hi * hi, 0)) / HD + eps)
            if h < HQ:
                wlo = tl.load(QNW + i, mask=m, other=0.0)
                whi = tl.load(QNW + HALF + i, mask=m, other=0.0)
            else:
                wlo = tl.load(KNW + i, mask=m, other=0.0)
                whi = tl.load(KNW + HALF + i, mask=m, other=0.0)
            lo = lo * rs * wlo
            hi = hi * rs * whi
        if ROPE:
            if HAS_SEC:
                sec = tl.load(SEC + i, mask=m, other=0)
            else:
                sec = tl.zeros((BLOCK_HALF,), tl.int32)
            p = tl.load(POS + t * stride_pos + sec, mask=m, other=0.0)
            ang = p * tl.load(INV_FREQ + i, mask=m, other=0.0)
            c = tl.cos(ang)
            s = tl.sin(ang)
            rlo = lo * c - hi * s
            hi = hi * c + lo * s
            lo = rlo
    if DYNAMIC_BASE:
        row = tl.load(KV_BASE) + t
    else:
        row = kv_base_static + t
    if h < HQ:
        dst = QO + t * sqo_t + h * sqo_h
    elif h < HQ + HKV:
        dst = KO + row * sko_t + (h - HQ) * sko_h
    else:
        dst = VO + row * svo_t + (h - HQ - HKV) * svo_h
    tl.store(dst + i, lo.to(QO.dtype.element_ty), mask=m)
    tl.store(dst + HALF + i, hi.to(QO.dtype.element_ty), mask=m)


def rope_inv_freq(head_dim, theta, device):
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim
    return (1.0 / (theta**exponent)).float()


def qkv_split(
    qkv,
    q_out,
    k_out,
    v_out,
    *,
    heads_q,
    heads_kv,
    head_dim,
    row=None,
    col=None,
    bias=None,
    inv_freq=None,
    pos=None,
    sections=None,
    q_norm=None,
    k_norm=None,
    norm_eps=1e-6,
    kv_base=0,
):
    """q_out [T, HQ, HD]; k_out/v_out [S, HKV, HD] written at rows ``kv_base + t``.

    Rotary is applied when ``inv_freq`` is given; ``pos`` is then a float32 device table
    ``[T, S]`` of per-token positions (``sections`` maps each frequency to a column).
    """
    if q_out.dtype != k_out.dtype or k_out.dtype != v_out.dtype:
        raise ValueError("q/k/v outputs must share a dtype")
    if inv_freq is not None and pos is None:
        raise ValueError("rotary needs a position table")
    T = qkv.shape[0]
    dynamic = torch.is_tensor(kv_base)
    dummy = qkv
    _qkv_split_kernel[(T, heads_q + 2 * heads_kv)](
        qkv,
        row if row is not None else dummy,
        col if col is not None else dummy,
        bias if bias is not None else dummy,
        q_norm if q_norm is not None else dummy,
        k_norm if k_norm is not None else dummy,
        q_out,
        k_out,
        v_out,
        pos if pos is not None else dummy,
        sections if sections is not None else dummy,
        inv_freq if inv_freq is not None else dummy,
        kv_base if dynamic else dummy,
        qkv.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        pos.stride(0) if pos is not None else 0,
        0 if dynamic else int(kv_base),
        float(norm_eps),
        HQ=heads_q,
        HKV=heads_kv,
        HD=head_dim,
        BLOCK_HALF=triton.next_power_of_2(head_dim // 2),
        HAS_ROW=row is not None,
        HAS_COL=col is not None,
        HAS_BIAS=bias is not None,
        QK_NORM=q_norm is not None,
        ROPE=inv_freq is not None,
        HAS_SEC=sections is not None,
        DYNAMIC_BASE=dynamic,
        num_warps=1 if head_dim <= 128 else 2,
    )
