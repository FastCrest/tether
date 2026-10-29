# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Gated MLP activation fused with the raw-accumulator rescale and FP8 quantization.

    g = gu[m, :F] * row[m] * col[:F]          u = gu[m, F:] * row[m] * col[F:]
    h = act(g) * u                            act in {gelu_tanh, silu}
    q[m], s[m] = fp8(h / s[m]), amax(|h|) / 448

One program owns a whole row (F <= 16384), so the row maximum is exact and the
hidden activation is read once and never written back in 16-bit form.
"""

from __future__ import annotations

import triton
import triton.language as tl

GELU_TANH = 0
SILU = 1


@triton.jit
def _gated_act_quant_kernel(
    GU,
    ROW,
    COL,
    Q,
    QS,
    stride_gu,
    stride_q,
    F,
    ACT: tl.constexpr,
    HAS_ROW: tl.constexpr,
    HAS_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < F
    g = tl.load(GU + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + row * stride_gu + F + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_ROW:
        r = tl.load(ROW + row)
        g = g * r
        u = u * r
    if HAS_COL:
        g = g * tl.load(COL + cols, mask=mask, other=0.0)
        u = u * tl.load(COL + F + cols, mask=mask, other=0.0)
    if ACT == 0:
        inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        a = 0.5 * g * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))
    else:
        a = g / (1.0 + tl.exp(-g))
    h = a * u
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(h), 0.0), 0), 1e-8)
    tl.store(QS + row, amax / 448.0)
    tl.store(Q + row * stride_q + cols, (h * (448.0 / amax)).to(tl.float8e4nv), mask=mask)


def gated_act_quant(gu, q, q_scale, *, row=None, col=None, act=GELU_TANH, num_warps=None):
    M, F2 = gu.shape
    F = F2 // 2
    block = triton.next_power_of_2(F)
    if block > 16384:
        raise ValueError("gated_act_quant supports hidden widths up to 16384")
    warps = num_warps or (4 if block <= 2048 else 8 if block <= 4096 else 16)
    _gated_act_quant_kernel[(M,)](
        gu,
        row if row is not None else gu,
        col if col is not None else gu,
        q,
        q_scale,
        gu.stride(0),
        q.stride(0),
        F,
        ACT=act,
        HAS_ROW=row is not None,
        HAS_COL=col is not None,
        BLOCK=block,
        num_warps=warps,
    )


# ---------------------------------------------------------------------------
# Mixture-of-experts (decode-sized batches)
#
# With a few dozen tokens and top-k routing nearly every expert is selected, so
# every expert's weights are read regardless of the dispatch strategy. The
# routed experts (plus an always-on shared expert) are therefore executed as two
# dense FP8 GEMMs over concatenated expert weights; routing enters only as a
# per-(token, expert) weight on the hidden activation, zero for experts a token
# did not select, so their down-projection contribution vanishes exactly.
# ---------------------------------------------------------------------------


@triton.jit
def _moe_router_kernel(
    X,
    W,
    EB,
    RW,
    stride_x,
    D,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    e = tl.arange(0, E)
    dm = d < D
    x = tl.load(X + t * stride_x + d, mask=dm, other=0.0).to(tl.float32)
    w = tl.load(W + e[:, None] * D + d[None, :], mask=dm[None, :], other=0.0)
    logits = tl.sum(w * x[None, :], 1)
    s = 1.0 / (1.0 + tl.exp(-logits))
    sel = s + tl.load(EB + e)
    chosen = tl.zeros((E,), tl.float32)
    for _ in tl.static_range(TOPK):
        i = tl.argmax(sel, 0)
        chosen = tl.where(e == i, 1.0, chosen)
        sel = tl.where(e == i, -float("inf"), sel)
    wsel = s * chosen
    wsel = wsel / (tl.sum(wsel, 0) + 1e-20) * SCALE
    tl.store(RW + t * E + e, wsel)


def moe_router(x, gate_w, e_bias, route_w, *, top_k, scale):
    """route_w[t, e] = normalized sigmoid weight for the top-k experts of token t, else 0.

    Selection uses ``sigmoid(logits) + e_bias``; the weights use the unbiased sigmoid.
    """
    M, D = x.shape
    E = gate_w.shape[0]
    _moe_router_kernel[(M,)](
        x,
        gate_w,
        e_bias,
        route_w,
        x.stride(0),
        D,
        E=E,
        TOPK=top_k,
        SCALE=float(scale),
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )


@triton.jit
def _moe_act_quant_kernel(
    GU,
    ROW,
    COL,
    RW,
    Q,
    QS,
    stride_gu,
    stride_q,
    F,
    n_routed,
    E: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    r = tl.load(ROW + row)
    amax = tl.zeros((BLOCK,), tl.float32)
    for p in tl.static_range(2):
        if p == 1:
            a = tl.maximum(tl.max(amax, 0), 1e-8)
            inv = 448.0 / a
        for c0 in range(0, F, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            m = cols < F
            g = tl.load(GU + row * stride_gu + cols, mask=m, other=0.0).to(tl.float32) * r
            u = tl.load(GU + row * stride_gu + F + cols, mask=m, other=0.0).to(tl.float32) * r
            g = g * tl.load(COL + cols, mask=m, other=0.0)
            u = u * tl.load(COL + F + cols, mask=m, other=0.0)
            grp = tl.minimum(cols // GROUP, E - 1)
            wr = tl.where(cols < n_routed, tl.load(RW + row * E + grp, mask=m, other=0.0), 1.0)
            h = g / (1.0 + tl.exp(-g)) * u * wr
            if p == 0:
                amax = tl.maximum(amax, tl.where(m, tl.abs(h), 0.0))
            else:
                tl.store(Q + row * stride_q + cols, (h * inv).to(tl.float8e4nv), mask=m)
    tl.store(QS + row, a / 448.0)


def moe_act_quant(
    gu, q, q_scale, route_w, *, row, col, experts, group, n_routed, block=4096, num_warps=8
):
    """h = silu(g) * u * route_w[t, col // group] (shared columns >= n_routed get weight 1)."""
    M, F2 = gu.shape
    _moe_act_quant_kernel[(M,)](
        gu,
        row,
        col,
        route_w,
        q,
        q_scale,
        gu.stride(0),
        q.stride(0),
        F2 // 2,
        int(n_routed),
        E=experts,
        GROUP=group,
        BLOCK=block,
        num_warps=num_warps,
    )
