# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Fused attention (non-causal, or causal for the LingBot-VLA-V2 prefix) for short robotics sequences.

One Triton kernel covers every attention site of the accel engines: grouped-query
heads are folded into the query-row dimension (rows ``token * group + head``),
so a single key/value head is loaded once per tile and reused by the whole
group, and the softmax never leaves registers. The valid key count can be read
from device memory, which lets a captured graph serve every prompt length that
fits its physical buffers.

Layouts (strides in elements, last dimension contiguous):
    q   [B, Tq, Hq, D]     o  [B, Tq, Hq, D]
    k,v [B, Tk, Hkv, D]    Hq = Hkv * group
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tether.accel.kernels._triton import compile_for, tc5_arch


@triton.jit
def _attn_kernel(
    Q,
    K,
    V,
    O,  # noqa: E741 (upstream kernel naming)
    KV_LEN,
    KV_EXTRA,
    sqb,
    sqt,
    sqh,
    skb,
    skt,
    skh,
    svb,
    svt,
    svh,
    sob,
    sot,
    soh,
    n_rows,
    kv_len_static,
    max_extra,
    causal_offset,
    scale,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DYNAMIC_LEN: tl.constexpr,
    TC5: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    CAUSAL: tl.constexpr,
    SPLITS: tl.constexpr,
    PO=None,
    PM=None,
    PL=None,
    KV_IDX=None,
    HAS_IDX: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    kvh = tl.program_id(1) // SPLITS
    split = tl.program_id(1) % SPLITS
    b = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = rows // GROUP
    head = kvh * GROUP + rows % GROUP
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM
    rmask = rows < n_rows
    q_ptrs = Q + b * sqb + tok[:, None] * sqt + head[:, None] * sqh + d[None, :]
    q = tl.load(q_ptrs, mask=rmask[:, None] & dmask[None, :], other=0.0)
    if DYNAMIC_LEN:
        kv_len = tl.load(KV_LEN)
    else:
        kv_len = kv_len_static
    # Per-query-row visible key count.
    limit = tl.zeros((BLOCK_M,), tl.int32) + kv_len
    if HAS_EXTRA:
        limit = limit + tl.load(KV_EXTRA + tok, mask=rmask, other=0)
    if CAUSAL:
        limit = tl.minimum(limit, tok + causal_offset + 1)
    end = tl.max(tl.where(rmask, limit, 0), 0)
    start0 = 0
    if SPLITS > 1:
        per = tl.cdiv(tl.cdiv(end, SPLITS), BLOCK_N) * BLOCK_N
        start0 = split * per
        end = tl.minimum(end, start0 + per)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = scale * 1.4426950408889634
    k_base = K + b * skb + kvh * skh
    v_base = V + b * svb + kvh * svh
    for start in range(start0, end, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        nmask = n < end
        if HAS_IDX:
            kr = tl.load(KV_IDX + n, mask=nmask, other=0)
        else:
            kr = n
        k = tl.load(
            k_base + kr[:, None] * skt + d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where((n[None, :] < limit[:, None]) & nmask[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            v_base + kr[:, None] * svt + d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    if SPLITS > 1:
        # Partial softmax state for the combine kernel; rows are (batch, kv head, row).
        n_kv = tl.num_programs(1) // SPLITS
        base = ((split * tl.num_programs(2) + b) * n_kv + kvh) * n_rows + rows
        tl.store(PM + base, m_i, mask=rmask)
        tl.store(PL + base, l_i, mask=rmask)
        tl.store(PO + base[:, None] * BLOCK_D + d[None, :], acc, mask=rmask[:, None])
    else:
        out = acc / l_i[:, None]
        o_ptrs = O + b * sob + tok[:, None] * sot + head[:, None] * soh + d[None, :]
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=rmask[:, None] & dmask[None, :])


# Per-(device, head_dim, group) launch choices measured on the target hardware.
# Missing entries fall back to the generic default below.
_TUNED: dict = {}
_DEFAULT = (32, 64, 4, 2)


def config_for(head_dim: int, group: int, rows: int):
    key = (torch.cuda.get_device_capability(), head_dim, group)
    return _TUNED.get(key, _DEFAULT)


@triton.jit
def _attn_combine(
    PO,
    PM,
    PL,
    O,  # noqa: E741 (upstream kernel naming)
    Q,
    QS,
    sob,
    sot,
    soh,
    stride_q,
    n_rows,
    n_batch,
    GROUP: tl.constexpr,
    HKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLITS: tl.constexpr,
    WRITE_O: tl.constexpr,
    QUANT: tl.constexpr,
):
    """Merge split-KV partials for one (batch, token); optionally FP8-quantize its full row."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.arange(0, GROUP)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM
    rows = t * GROUP + h
    amax = tl.zeros((GROUP,), tl.float32)
    for pass_ in tl.static_range(2 if QUANT else 1):
        for kvh in tl.static_range(HKV):
            m = tl.full((GROUP,), -float("inf"), tl.float32)
            for sp in tl.static_range(SPLITS):
                m = tl.maximum(m, tl.load(PM + ((sp * n_batch + b) * HKV + kvh) * n_rows + rows))
            l = tl.zeros((GROUP,), tl.float32)  # noqa: E741 (upstream kernel naming)
            acc = tl.zeros((GROUP, BLOCK_D), tl.float32)
            for sp in tl.static_range(SPLITS):
                base = ((sp * n_batch + b) * HKV + kvh) * n_rows + rows
                w = tl.exp2(tl.load(PM + base) - m)
                l += w * tl.load(PL + base)  # noqa: E741
                acc += w[:, None] * tl.load(PO + base[:, None] * BLOCK_D + d[None, :])
            out = acc / l[:, None]
            head = kvh * GROUP + h
            if pass_ == 0:
                if WRITE_O:
                    tl.store(
                        O + b * sob + t * sot + head[:, None] * soh + d[None, :],
                        out.to(O.dtype.element_ty),
                        mask=dmask[None, :],
                    )
                if QUANT:
                    amax = tl.maximum(amax, tl.max(tl.where(dmask[None, :], tl.abs(out), 0.0), 1))
            else:
                a = tl.maximum(tl.max(amax, 0), 1e-8)
                tl.store(
                    Q + t * stride_q + head[:, None] * HEAD_DIM + d[None, :],
                    (out * (448.0 / a)).to(tl.float8e4nv),
                    mask=dmask[None, :],
                )
                if kvh == 0:
                    tl.store(QS + t, a / 448.0)


class SplitWorkspace:
    """Partial-softmax storage for split-KV attention of one fixed geometry."""

    def __init__(self, batch, tq, hq, hkv, head_dim, splits, device):
        rows = tq * (hq // hkv)
        width = triton.next_power_of_2(head_dim)
        self.splits = splits
        self.po = torch.empty(
            splits * batch * hkv * rows * width, device=device, dtype=torch.float32
        )
        self.pm = torch.empty(splits * batch * hkv * rows, device=device, dtype=torch.float32)
        self.pl = torch.empty_like(self.pm)


def attention(
    q,
    k,
    v,
    out=None,
    *,
    scale=None,
    kv_len=None,
    kv_extra=None,
    max_extra=0,
    causal=False,
    causal_offset=0,
    config=None,
    tc5=False,
    split=None,
    q8=None,
    q8_scale=None,
    kv_index=None,
):
    """Non-causal (or causal) attention; query row ``t`` sees keys ``[0, limit_t)``.

    ``limit_t = kv_len (+ kv_extra[t]) (min causal_offset + t + 1)``. ``kv_len`` may be an int
    (default: all keys) or a 1-element int32 CUDA tensor read at execution time; ``kv_extra``
    is an int32 tensor [Tq] whose maximum must not exceed ``max_extra``.

    ``kv_index`` (int32 [>= kv_len]) makes key ``n`` the cache row ``kv_index[n]`` (pooled caches).

    ``split`` (a :class:`SplitWorkspace`) spreads the keys over several programs per query
    block and merges them in a second kernel, which can also emit each token's full output
    row as per-token FP8 (``q8`` [B*Tq, Hq*D], ``q8_scale`` [B*Tq]); ``out`` is then optional.
    """
    B, Tq, Hq, D = q.shape
    Hkv = k.shape[2]
    group = Hq // Hkv
    if out is None and (split is None or q8 is None):
        out = torch.empty_like(q)
    for t in (q, k, v) + ((out,) if out is not None else ()):
        if t.stride(-1) != 1:
            raise ValueError("attention requires a contiguous last dimension")
    if scale is None:
        scale = D**-0.5
    dynamic = torch.is_tensor(kv_len)
    static_len = k.shape[1] if kv_len is None or dynamic else int(kv_len)
    block_m, block_n, warps, stages = config or config_for(D, group, Tq * group)
    n_rows = Tq * group
    splits = split.splits if split is not None else 1
    grid = (triton.cdiv(n_rows, block_m), Hkv * splits, B)
    arch = tc5_arch(q.device) if tc5 else None
    o_ = out if out is not None else q
    with compile_for(arch):
        _attn_kernel[grid](
            q,
            k,
            v,
            o_,
            kv_len if dynamic else q,
            kv_extra if kv_extra is not None else q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o_.stride(0),
            o_.stride(1),
            o_.stride(2),
            n_rows,
            static_len,
            int(max_extra),
            int(causal_offset),
            float(scale),
            GROUP=group,
            HEAD_DIM=D,
            BLOCK_D=triton.next_power_of_2(D),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DYNAMIC_LEN=dynamic,
            TC5=arch is not None,
            HAS_EXTRA=kv_extra is not None,
            CAUSAL=causal,
            SPLITS=splits,
            PO=split.po if split is not None else None,
            PM=split.pm if split is not None else None,
            PL=split.pl if split is not None else None,
            KV_IDX=kv_index,
            HAS_IDX=kv_index is not None,
            num_warps=warps,
            num_stages=stages,
        )
    if split is not None:
        if q8 is not None and B != 1:
            raise ValueError("fused FP8 output supports batch 1")
        _attn_combine[(B, Tq)](
            split.po,
            split.pm,
            split.pl,
            o_,
            q8 if q8 is not None else split.po,
            q8_scale if q8_scale is not None else split.pm,
            o_.stride(0),
            o_.stride(1),
            o_.stride(2),
            q8.stride(0) if q8 is not None else 0,
            n_rows,
            B,
            GROUP=group,
            HKV=Hkv,
            HEAD_DIM=D,
            BLOCK_D=triton.next_power_of_2(D),
            SPLITS=splits,
            WRITE_O=out is not None,
            QUANT=q8 is not None,
            num_warps=4,
        )
    return out


# ---------------------------------------------------------------------------
# Chunk attention: a short query chunk (e.g. 50 action tokens) attends a cached
# prefix plus its own keys. The chunk's Q/K/V come straight from the raw QKV
# GEMM accumulator: dequantization and rotary are applied while loading, and
# the chunk's K/V never touch the cache. The prefix is split across programs
# (split-KV) and the last split owns the chunk keys; a combine kernel merges
# the partial softmax states and quantizes each token's full output row.
# ---------------------------------------------------------------------------


@triton.jit
def _load_rope_half(
    RAW,
    ROW,
    COL,
    INV_FREQ,
    tok,
    col0,
    pos,
    d,
    tmask,
    stride_raw,
    HALF: tl.constexpr,
    ROPE: tl.constexpr,
):
    r = tl.load(ROW + tok, mask=tmask, other=0.0)
    lo = tl.load(
        RAW + tok[:, None] * stride_raw + col0[:, None] + d[None, :], mask=tmask[:, None], other=0.0
    )
    hi = tl.load(
        RAW + tok[:, None] * stride_raw + col0[:, None] + HALF + d[None, :],
        mask=tmask[:, None],
        other=0.0,
    )
    lo = lo.to(tl.float32) * r[:, None] * tl.load(COL + col0[:, None] + d[None, :])
    hi = hi.to(tl.float32) * r[:, None] * tl.load(COL + col0[:, None] + HALF + d[None, :])
    if ROPE:
        ang = pos.to(tl.float32)[:, None] * tl.load(INV_FREQ + d)[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)
        rlo = lo * c - hi * s
        hi = hi * c + lo * s
        lo = rlo
    return lo, hi


@triton.jit
def _chunk_attn_partial(
    RAW,
    ROW,
    COL,
    KC,
    VC,
    POS,
    INV_FREQ,
    PO,
    PM,
    PL,
    stride_raw,
    stride_kc,
    stride_vc,
    n_tok,
    qk_scale,
    GROUP: tl.constexpr,
    HD: tl.constexpr,
    KV_COL: tl.constexpr,
    V_COL: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TC5: tl.constexpr,
):
    HALF: tl.constexpr = HD // 2
    pid_m = tl.program_id(0)
    split = tl.program_id(1)
    n_rows = n_tok * GROUP
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < n_rows
    tok = rows // GROUP
    head = rows % GROUP
    d = tl.arange(0, HALF)
    p0 = tl.load(POS)
    q_lo, q_hi = _load_rope_half(
        RAW, ROW, COL, INV_FREQ, tok, head * HD, p0 + tok, d, rmask, stride_raw, HALF, True
    )
    q_lo = (q_lo * qk_scale).to(tl.float16)
    q_hi = (q_hi * qk_scale).to(tl.float16)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc_lo = tl.zeros((BLOCK_M, HALF), tl.float32)
    acc_hi = tl.zeros((BLOCK_M, HALF), tl.float32)
    if split < SPLITS - 1:
        per = tl.cdiv(tl.cdiv(p0, SPLITS - 1), BLOCK_N) * BLOCK_N
        start = split * per
        end = tl.minimum(start + per, p0)
        for n0 in range(start, end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            nmask = n < end
            k_lo = tl.load(KC + n[:, None] * stride_kc + d[None, :], mask=nmask[:, None], other=0.0)
            k_hi = tl.load(
                KC + n[:, None] * stride_kc + HALF + d[None, :], mask=nmask[:, None], other=0.0
            )
            s = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))
            s = tl.where(nmask[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v_lo = tl.load(VC + n[:, None] * stride_vc + d[None, :], mask=nmask[:, None], other=0.0)
            v_hi = tl.load(
                VC + n[:, None] * stride_vc + HALF + d[None, :], mask=nmask[:, None], other=0.0
            )
            pf = p.to(tl.float16)
            acc_lo = acc_lo * alpha[:, None] + tl.dot(pf, v_lo)
            acc_hi = acc_hi * alpha[:, None] + tl.dot(pf, v_hi)
            m_i = m_new
    else:
        for n0 in range(0, n_tok, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            nmask = n < n_tok
            kcol = tl.zeros((BLOCK_N,), tl.int32) + KV_COL
            vcol = tl.zeros((BLOCK_N,), tl.int32) + V_COL
            k_lo, k_hi = _load_rope_half(
                RAW, ROW, COL, INV_FREQ, n, kcol, p0 + n, d, nmask, stride_raw, HALF, True
            )
            v_lo, v_hi = _load_rope_half(
                RAW, ROW, COL, INV_FREQ, n, vcol, p0 + n, d, nmask, stride_raw, HALF, False
            )
            s = tl.dot(q_lo, tl.trans(k_lo.to(tl.float16))) + tl.dot(
                q_hi, tl.trans(k_hi.to(tl.float16))
            )
            s = tl.where(nmask[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            pf = p.to(tl.float16)
            acc_lo = acc_lo * alpha[:, None] + tl.dot(pf, v_lo.to(tl.float16))
            acc_hi = acc_hi * alpha[:, None] + tl.dot(pf, v_hi.to(tl.float16))
            m_i = m_new
    base = split * n_rows + rows
    tl.store(PM + base, m_i, mask=rmask)
    tl.store(PL + base, l_i, mask=rmask)
    tl.store(PO + base[:, None] * HD + d[None, :], acc_lo, mask=rmask[:, None])
    tl.store(PO + base[:, None] * HD + HALF + d[None, :], acc_hi, mask=rmask[:, None])


@triton.jit
def _chunk_attn_combine(
    PO,
    PM,
    PL,
    Q,
    QS,
    O16,
    n_tok,
    stride_q,
    stride_o,
    GROUP: tl.constexpr,
    HD: tl.constexpr,
    SPLITS: tl.constexpr,
    QUANT: tl.constexpr,
    WRITE_O: tl.constexpr,
):
    t = tl.program_id(0)
    n_rows = n_tok * GROUP
    h = tl.arange(0, GROUP)
    d = tl.arange(0, HD)
    rows = t * GROUP + h
    m = tl.full((GROUP,), -float("inf"), tl.float32)
    for s in range(SPLITS):
        m = tl.maximum(m, tl.load(PM + s * n_rows + rows))
    l = tl.zeros((GROUP,), tl.float32)  # noqa: E741 (upstream kernel naming)
    acc = tl.zeros((GROUP, HD), tl.float32)
    for s in range(SPLITS):
        w = tl.exp2(tl.load(PM + s * n_rows + rows) - m)
        l += w * tl.load(PL + s * n_rows + rows)  # noqa: E741
        acc += w[:, None] * tl.load(PO + (s * n_rows + rows)[:, None] * HD + d[None, :])
    out = acc / l[:, None]
    cols = h[:, None] * HD + d[None, :]
    if WRITE_O:
        tl.store(O16 + t * stride_o + cols, out.to(O16.dtype.element_ty))
    if QUANT:
        amax = tl.maximum(tl.max(tl.max(tl.abs(out), 1), 0), 1e-8)
        tl.store(QS + t, amax / 448.0)
        tl.store(Q + t * stride_q + cols, (out * (448.0 / amax)).to(tl.float8e4nv))


class ChunkAttention:
    """Prepared chunk attention for fixed geometry; owns its split workspace."""

    def __init__(
        self,
        n_tok,
        group,
        head_dim,
        device,
        *,
        splits=4,
        block_m=32,
        block_n=64,
        num_warps=4,
        num_stages=2,
        tc5=False,
    ):
        self.tc5 = tc5
        self.n_tok, self.group, self.hd = n_tok, group, head_dim
        self.splits, self.block_m, self.block_n = splits, block_m, block_n
        self.num_warps, self.num_stages = num_warps, num_stages
        rows = n_tok * group
        self.po = torch.empty(splits * rows * head_dim, device=device, dtype=torch.float32)
        self.pm = torch.empty(splits * rows, device=device, dtype=torch.float32)
        self.pl = torch.empty(splits * rows, device=device, dtype=torch.float32)

    def __call__(
        self,
        raw,
        row_scale,
        col_scale,
        kc,
        vc,
        prefix_len,
        inv_freq,
        *,
        q=None,
        q_scale=None,
        out=None,
        scale=None,
    ):
        """raw [n_tok, (group+2)*hd] accumulator; kc/vc [S, hd] cache rows [0, prefix_len)."""
        hd, g = self.hd, self.group
        rows = self.n_tok * g
        qk_scale = (scale if scale is not None else hd**-0.5) * 1.4426950408889634
        grid = (triton.cdiv(rows, self.block_m), self.splits)
        arch = tc5_arch(raw.device) if self.tc5 else None
        with compile_for(arch):
            _chunk_attn_partial[grid](
                raw,
                row_scale,
                col_scale,
                kc,
                vc,
                prefix_len,
                inv_freq,
                self.po,
                self.pm,
                self.pl,
                raw.stride(0),
                kc.stride(0),
                vc.stride(0),
                self.n_tok,
                qk_scale,
                GROUP=g,
                HD=hd,
                KV_COL=g * hd,
                V_COL=(g + 1) * hd,
                SPLITS=self.splits,
                BLOCK_M=self.block_m,
                BLOCK_N=self.block_n,
                TC5=arch is not None,
                num_warps=self.num_warps,
                num_stages=self.num_stages,
            )
        dummy = self.po
        _chunk_attn_combine[(self.n_tok,)](
            self.po,
            self.pm,
            self.pl,
            q if q is not None else dummy,
            q_scale if q_scale is not None else dummy,
            out if out is not None else dummy,
            self.n_tok,
            q.stride(0) if q is not None else 0,
            out.stride(0) if out is not None else 0,
            GROUP=g,
            HD=hd,
            SPLITS=self.splits,
            QUANT=q is not None,
            WRITE_O=out is not None,
            num_warps=4,
        )
