# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in src/tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28.
"""GPU checks of tether.accel.kernels against plain PyTorch references (skipped without CUDA)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() < (8, 9):
    pytest.skip("FP8 kernels need sm_89+", allow_module_level=True)

from tether.accel.kernels.activation import GELU_TANH, SILU, gated_act_quant  # noqa: E402
from tether.accel.kernels.attention import attention  # noqa: E402
from tether.accel.kernels.fp8 import gemm, quantize_weight  # noqa: E402
from tether.accel.kernels.norm import dequant_rows, residual_norm_quant, row_quant  # noqa: E402
from tether.accel.kernels.rope import qkv_split, rope_inv_freq  # noqa: E402

DEV = "cuda"


def _deq(q, s):
    return q.float() * s[:, None]


def _ref_attention(q, k, v, limits, scale):
    g = q.shape[2] // k.shape[2]
    kk = k.float().repeat_interleave(g, 2)
    vv = v.float().repeat_interleave(g, 2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * scale
    n = torch.arange(k.shape[1], device=DEV)
    s = s.masked_fill(~(n[None, :] < limits[:, None])[None, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), vv)


@pytest.mark.parametrize("hd,hq,hkv", [(256, 8, 1), (72, 16, 16), (128, 16, 2)])
def test_attention_dynamic_length(hd, hq, hkv):
    torch.manual_seed(0)
    q = torch.randn(2, 40, hq, hd, device=DEV, dtype=torch.float16)
    k = torch.randn(2, 64, hkv, hd, device=DEV, dtype=torch.float16)
    v = torch.randn_like(k)
    n = torch.tensor([53], device=DEV, dtype=torch.int32)
    out = attention(q, k, v, kv_len=n, config=(32, 32, 4, 2))
    ref = _ref_attention(q, k, v, torch.full((40,), 53, device=DEV), hd**-0.5)
    assert (out.float() - ref).abs().max() < 2e-3


def test_attention_causal_and_extra():
    torch.manual_seed(1)
    q = torch.randn(1, 20, 4, 128, device=DEV, dtype=torch.float16)
    k = torch.randn(1, 48, 2, 128, device=DEV, dtype=torch.float16)
    v = torch.randn_like(k)
    out = attention(q, k, v, causal=True, causal_offset=5, config=(16, 16, 4, 1))
    lim = torch.minimum(torch.full((20,), 48, device=DEV), torch.arange(20, device=DEV) + 6)
    assert (out.float() - _ref_attention(q, k, v, lim, 128**-0.5)).abs().max() < 2e-3
    extra = torch.tensor([1] + [11] * 19, device=DEV, dtype=torch.int32)
    base = torch.tensor([30], device=DEV, dtype=torch.int32)
    out = attention(q, k, v, kv_len=base, kv_extra=extra, max_extra=11, config=(16, 16, 4, 1))
    assert (out.float() - _ref_attention(q, k, v, 30 + extra, 128**-0.5)).abs().max() < 2e-3


def test_qkv_split_rope_and_qk_norm():
    torch.manual_seed(2)
    T, HQ, HKV, HD = 7, 4, 2, 128
    raw = torch.randn(T, (HQ + 2 * HKV) * HD, device=DEV).to(torch.bfloat16)
    row = torch.rand(T, device=DEV) + 0.5
    col = torch.rand(raw.shape[1], device=DEV) + 0.5
    bias = torch.randn(raw.shape[1], device=DEV)
    qn, kn = torch.rand(HD, device=DEV) + 0.5, torch.rand(HD, device=DEV) + 0.5
    inv = rope_inv_freq(HD, 1e6, DEV)
    pos = (
        torch.stack([torch.arange(T), torch.arange(T) * 2, torch.arange(T) * 3], 1).float().to(DEV)
    )
    sec = (torch.arange(HD // 2, device=DEV) % 3).int()
    q = torch.empty(T, HQ, HD, device=DEV, dtype=torch.float16)
    kc = torch.zeros(20, HKV, HD, device=DEV, dtype=torch.float16)
    vc = torch.zeros_like(kc)
    base = torch.tensor([9], device=DEV, dtype=torch.int32)
    qkv_split(
        raw,
        q,
        kc,
        vc,
        heads_q=HQ,
        heads_kv=HKV,
        head_dim=HD,
        row=row,
        col=col,
        bias=bias,
        inv_freq=inv,
        pos=pos,
        sections=sec,
        q_norm=qn,
        k_norm=kn,
        kv_base=base,
    )
    x = raw.float() * row[:, None] * col + bias
    xq, xk, xv = (
        x[:, : HQ * HD].view(T, HQ, HD),
        x[:, HQ * HD : (HQ + HKV) * HD].view(T, HKV, HD),
        x[:, (HQ + HKV) * HD :].view(T, HKV, HD),
    )

    def rms(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    ang = pos.gather(1, sec.long()[None].expand(T, -1)) * inv

    def rope(t):
        c, s = ang.cos()[:, None], ang.sin()[:, None]
        lo, hi = t[..., : HD // 2], t[..., HD // 2 :]
        return torch.cat([lo * c - hi * s, hi * c + lo * s], -1)

    assert (q.float() - rope(rms(xq, qn))).abs().max() < 1e-2
    assert (kc[9:16].float() - rope(rms(xk, kn))).abs().max() < 1e-2
    assert (vc[9:16].float() - xv).abs().max() < 1e-2
    assert kc[:9].abs().max() == 0 and kc[16:].abs().max() == 0


@pytest.mark.parametrize("layernorm", [False, True])
def test_residual_norm_quant(layernorm):
    torch.manual_seed(3)
    M, D = 33, 1152
    res = torch.randn(M, D, device=DEV)
    delta = torch.randn(M, D, device=DEV).to(torch.bfloat16)
    dr, dc, b, gate = (
        torch.rand(M, device=DEV),
        torch.rand(D, device=DEV),
        torch.randn(D, device=DEV),
        torch.rand(D, device=DEV),
    )
    w, beta = torch.rand(D, device=DEV), torch.randn(D, device=DEV)
    ms, mh = torch.randn(D, device=DEV) * 0.1, torch.randn(D, device=DEV) * 0.1
    q = torch.empty(M, D, device=DEV, dtype=torch.float8_e4m3fn)
    qs = torch.empty(M, device=DEV)
    expect = res + (delta.float() * dr[:, None] * dc + b) * gate
    residual_norm_quant(
        res,
        delta=delta,
        delta_row=dr,
        delta_col=dc,
        bias=b,
        gate=gate,
        layernorm=layernorm,
        weight=w,
        beta=beta,
        mod_scale=ms,
        mod_shift=mh,
        q=q,
        q_scale=qs,
    )
    assert (res - expect).abs().max() < 1e-4
    x = expect - expect.mean(-1, keepdim=True) if layernorm else expect
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    y = (y * w + beta) * (1 + ms) + mh
    assert ((_deq(q, qs) - y).abs() / y.abs().amax(-1, keepdim=True)).max() < 0.07


@pytest.mark.parametrize("act", [GELU_TANH, SILU])
def test_gated_act_quant(act):
    torch.manual_seed(4)
    M, F = 9, 4096
    gu = torch.randn(M, 2 * F, device=DEV).to(torch.bfloat16)
    row, col = torch.rand(M, device=DEV), torch.rand(2 * F, device=DEV)
    q = torch.empty(M, F, device=DEV, dtype=torch.float8_e4m3fn)
    qs = torch.empty(M, device=DEV)
    gated_act_quant(gu, q, qs, row=row, col=col, act=act)
    x = gu.float() * row[:, None] * col
    g, u = x[:, :F], x[:, F:]
    a = (
        torch.nn.functional.gelu(g, approximate="tanh")
        if act == GELU_TANH
        else torch.nn.functional.silu(g)
    )
    h = a * u
    assert ((_deq(q, qs) - h).abs() / h.abs().amax(-1, keepdim=True)).max() < 0.07


def test_fp8_gemm_and_consumer_scaling():
    torch.manual_seed(5)
    x = torch.randn(50, 1024, device=DEV)
    w = torch.randn(2560, 1024, device=DEV) * 0.02
    wq = quantize_weight(w, torch.randn(2560, device=DEV))
    q = torch.empty(50, 1024, device=DEV, dtype=torch.float8_e4m3fn)
    qs = torch.empty(50, device=DEV)
    row_quant(x, q, qs)
    raw = torch.empty(50, 2560, device=DEV, dtype=torch.bfloat16)
    gemm(q, wq, raw)
    y = torch.empty(50, 2560, device=DEV)
    dequant_rows(raw, y, row=qs, col=wq.scale, bias=wq.bias)
    ref = x @ w.t() + wq.bias
    assert ((y - ref).abs().max() / ref.abs().max()) < 0.05


@pytest.mark.parametrize("hd,hq,hkv,splits", [(256, 8, 1, 3), (128, 16, 2, 2), (72, 16, 16, 4)])
def test_split_attention_with_fused_quant(hd, hq, hkv, splits):
    from tether.accel.kernels.attention import SplitWorkspace

    torch.manual_seed(6)
    T = 50
    q = torch.randn(1, T, hq, hd, device=DEV, dtype=torch.float16)
    k = torch.randn(1, 700, hkv, hd, device=DEV, dtype=torch.float16)
    v = torch.randn_like(k)
    base = torch.tensor([600], device=DEV, dtype=torch.int32)
    extra = torch.tensor([1] + [T] * (T - 1), device=DEV, dtype=torch.int32)
    ws = SplitWorkspace(1, T, hq, hkv, hd, splits, DEV)
    out = torch.empty_like(q)
    q8 = torch.empty(T, hq * hd, device=DEV, dtype=torch.float8_e4m3fn)
    qs = torch.empty(T, device=DEV)
    attention(
        q,
        k,
        v,
        out=out,
        kv_len=base,
        kv_extra=extra,
        max_extra=T,
        split=ws,
        q8=q8,
        q8_scale=qs,
        config=(32, 64, 4, 2),
    )
    ref = _ref_attention(q, k, v, 600 + extra, hd**-0.5)
    assert (out.float() - ref).abs().max() < 2e-3
    r = ref.reshape(T, -1)
    assert ((_deq(q8, qs) - r).abs() / r.abs().amax(-1, keepdim=True)).max() < 0.07


def test_moe_router_and_routed_activation():
    from tether.accel.kernels.activation import moe_act_quant, moe_router

    torch.manual_seed(7)
    M, D, E, G, S = 51, 768, 32, 512, 704
    x = torch.randn(M, D, device=DEV)
    gw = torch.randn(E, D, device=DEV) * 0.05
    eb = torch.zeros(E, device=DEV)
    rw = torch.empty(M, E, device=DEV)
    moe_router(x, gw, eb, rw, top_k=4, scale=4.0)
    s = torch.sigmoid(x @ gw.t())
    idx = torch.topk(s + eb, 4, -1).indices
    ref = torch.zeros(M, E, device=DEV).scatter(1, idx, s.gather(1, idx))
    ref = ref / (ref.sum(-1, keepdim=True) + 1e-20) * 4.0
    assert (rw - ref).abs().max() < 1e-5
    F = E * G + S
    gu = torch.randn(M, 2 * F, device=DEV).to(torch.bfloat16)
    row, col = torch.rand(M, device=DEV) + 0.5, torch.rand(2 * F, device=DEV) + 0.5
    q = torch.empty(M, F, device=DEV, dtype=torch.float8_e4m3fn)
    qs = torch.empty(M, device=DEV)
    moe_act_quant(gu, q, qs, rw, row=row, col=col, experts=E, group=G, n_routed=E * G)
    xg = gu.float() * row[:, None] * col
    g, u = xg[:, :F], xg[:, F:]
    wcol = torch.cat([rw.repeat_interleave(G, 1), torch.ones(M, S, device=DEV)], 1)
    h = torch.nn.functional.silu(g) * u * wcol
    assert ((_deq(q, qs) - h).abs() / h.abs().amax(-1, keepdim=True)).max() < 0.07


def test_residual_norm_fused_router_matches_standalone():
    from tether.accel.kernels.activation import moe_router

    torch.manual_seed(8)
    M, D, E = 51, 768, 32
    res = torch.randn(M, D, device=DEV)
    ms, mh = torch.randn(D, device=DEV) * 0.1, torch.randn(D, device=DEV) * 0.1
    gw, eb = torch.randn(E, D, device=DEV) * 0.05, torch.rand(E, device=DEV) * 0.01
    y = torch.empty(M, D, device=DEV)
    rw_fused, rw_ref = torch.empty(M, E, device=DEV), torch.empty(M, E, device=DEV)
    residual_norm_quant(
        res.clone(), mod_scale=ms, mod_shift=mh, y=y, router=(gw, eb, rw_fused, 4, 4.0)
    )
    moe_router(y, gw, eb, rw_ref, top_k=4, scale=4.0)
    assert (rw_fused - rw_ref).abs().max() < 1e-5
