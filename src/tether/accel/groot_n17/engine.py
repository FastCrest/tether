# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Accel GR00T N1.7 action head: VLSA over backbone features + 4-step AlternateVLDiT.

One CUDA graph per sequence-length bucket covers: vlln + 4 VLSA blocks, the text/image
row partition (on device), the 16 cross-attention K/V projections, the state encoder,
and four Euler steps of the 32-block DiT with the action encoder and decoder.
"""

from __future__ import annotations

import torch

from tether.accel._common import Arena, GraphCache, bucket
from tether.accel.kernels.attention import attention
from tether.accel.kernels.fp8 import E4M3, gemm
from tether.accel.kernels.norm import dequant_rows, residual_norm_quant, row_quant

VL_D, VL_H, VL_HD, VL_FF = 2048, 32, 64, 8192
D, H, HD, FF = 1536, 32, 48, 6144
from tether.accel.groot_n17 import ACT_DIM, HORIZON  # noqa: E402

SA = HORIZON + 1
LN_EPS, OUT_EPS = 1e-5, 1e-6

ATTENTION_SITES = {
    "vlsa": (64, 64, 4, 2, True),
    "self": (32, 64, 4, 2, False),
    "cross": (32, 64, 4, 2, False),
}


class GrootN17Engine:
    def __init__(self, weights, device, *, max_seq=512, bucket_size=32):
        self.w = weights
        self.dev = torch.device(device)
        self.bucket_size = int(bucket_size)
        self.max_seq = bucket(max_seq, bucket_size)
        self.attn = {k: (v[:4], v[4]) for k, v in ATTENTION_SITES.items()}
        dev, f32, f16, bf16 = self.dev, torch.float32, torch.float16, torch.bfloat16
        S = self.max_seq
        self.feat = torch.zeros(S, VL_D, dtype=f32, device=dev)
        self.img_mask = torch.zeros(S, dtype=torch.bool, device=dev)
        self.valid = torch.zeros(S, dtype=torch.bool, device=dev)
        self.seq_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self.n_img = torch.zeros(1, dtype=torch.int32, device=dev)
        self.n_txt = torch.zeros(1, dtype=torch.int32, device=dev)
        self.state = torch.zeros(1, ACT_DIM, dtype=f32, device=dev)
        self.noise = torch.zeros(HORIZON, ACT_DIM, dtype=f32, device=dev)
        rows = max(S, SA)
        self.q8 = Arena(rows * VL_FF, E4M3, dev)
        self.raw = Arena(rows * VL_FF, bf16, dev)
        self.scale = [torch.zeros(rows, dtype=f32, device=dev) for _ in range(2)]
        self.vx = torch.zeros(S, VL_D, dtype=f32, device=dev)
        self.enc = {k: torch.zeros(S, VL_D, dtype=f32, device=dev) for k in ("text", "image")}
        self.qkv16 = torch.zeros(S, 3 * VL_D, dtype=f16, device=dev)
        self.o16 = torch.zeros(S, VL_D, dtype=f16, device=dev)
        n_cross = sum(b.cross for b in weights.dit)
        self.kc = torch.zeros(n_cross, S, H, HD, dtype=f16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.h = torch.zeros(SA, D, dtype=f32, device=dev)
        self.y = torch.zeros(SA, D, dtype=f32, device=dev)
        self.a_t = torch.zeros(HORIZON, ACT_DIM, dtype=f32, device=dev)
        self.state_emb = torch.zeros(1, D, dtype=f32, device=dev)
        self.enc_tmp = torch.zeros(1, 1024, dtype=f32, device=dev)
        self._graphs = GraphCache(dev)
        self._rows = None
        self.graph_captured = False
        self.replays = 0

    def infer(self, features, image_mask, attention_mask, state, noise):
        """features [S, 2048]; masks [S] bool; state [132]; noise [40, 132] -> actions [40, 132]."""
        n = int(features.shape[0])
        if not 1 <= n <= self.max_seq:
            raise ValueError(f"GR00T engine supports 1..{self.max_seq} backbone tokens, got {n}")
        rows = bucket(n, self.bucket_size)
        self.feat[:n].copy_(features)
        self.feat[n:rows].zero_()
        self.img_mask[:n].copy_(image_mask)
        self.img_mask[n:].zero_()
        self.valid[:n].copy_(attention_mask)
        self.valid[n:].zero_()
        self.seq_len.fill_(n)
        self.state.copy_(state.reshape(1, ACT_DIM))
        self.noise.copy_(noise.reshape(HORIZON, ACT_DIM))
        self._rows = rows
        self._graphs.run(rows, self._forward)
        self.graph_captured = True
        self.replays += 1
        return self.a_t

    # ----------------------------------------------------------------- forward
    def _forward(self, M):
        self._vlsa(M)
        self._cross_kv(M)
        self._dit()

    def _vlsa(self, M):
        w, q8, raw, s = self.w, self.q8, self.raw, self.scale
        residual_norm_quant(
            self.feat[:M], layernorm=True, weight=w.vlln_w, beta=w.vlln_b, y=self.vx[:M], eps=LN_EPS
        )
        L0 = w.vlsa[0]
        residual_norm_quant(
            self.vx[:M],
            layernorm=True,
            weight=L0.ln1_w,
            beta=L0.ln1_b,
            q=q8(M, VL_D),
            q_scale=s[0],
            eps=LN_EPS,
        )
        view = (1, M, VL_H, VL_HD)
        cfg, tc5 = self.attn["vlsa"]
        for i, L in enumerate(w.vlsa):
            gemm(q8(M, VL_D), L.qkv, raw(M, 3 * VL_D))
            dequant_rows(
                raw(M, 3 * VL_D),
                self.qkv16[:M],
                row=s[0],
                col=L.qkv.scale,
                bias=L.qkv.bias,
                block=512,
            )
            qkv = self.qkv16[:M].view(1, M, 3, VL_H, VL_HD)
            attention(
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                out=self.o16[:M].view(view),
                kv_len=self.seq_len,
                config=cfg,
                tc5=tc5,
            )
            row_quant(self.o16[:M], q8(M, VL_D), s[1], num_warps=1)
            gemm(q8(M, VL_D), L.o, raw(M, VL_D))
            residual_norm_quant(
                self.vx[:M],
                delta=raw(M, VL_D),
                delta_row=s[1],
                delta_col=L.o.scale,
                bias=L.o.bias,
                layernorm=True,
                weight=L.ln3_w,
                beta=L.ln3_b,
                q=q8(M, VL_D),
                q_scale=s[0],
                eps=LN_EPS,
            )
            gemm(q8(M, VL_D), L.ff1, raw(M, VL_FF))
            row_quant(
                raw(M, VL_FF),
                q8(M, VL_FF),
                s[1],
                row=s[0],
                col=L.ff1.scale,
                bias=L.ff1.bias,
                gelu=True,
                num_warps=8,
            )
            gemm(q8(M, VL_FF), L.ff2, raw(M, VL_D))
            nxt = w.vlsa[i + 1] if i + 1 < len(w.vlsa) else None
            if nxt is not None:
                residual_norm_quant(
                    self.vx[:M],
                    delta=raw(M, VL_D),
                    delta_row=s[1],
                    delta_col=L.ff2.scale,
                    bias=L.ff2.bias,
                    layernorm=True,
                    weight=nxt.ln1_w,
                    beta=nxt.ln1_b,
                    q=q8(M, VL_D),
                    q_scale=s[0],
                    eps=LN_EPS,
                )
            else:
                residual_norm_quant(
                    self.vx[:M],
                    delta=raw(M, VL_D),
                    delta_row=s[1],
                    delta_col=L.ff2.scale,
                    bias=L.ff2.bias,
                    eps=LN_EPS,
                )

    def _cross_kv(self, M):
        """Gather text / image rows to the front of two buffers and project every cross block's K/V."""
        img = self.img_mask[:M] & self.valid[:M]
        txt = ~self.img_mask[:M] & self.valid[:M]
        for key, sel, count in (("image", img, self.n_img), ("text", txt, self.n_txt)):
            order = torch.argsort((~sel).to(torch.int8), stable=True)
            torch.index_select(self.vx[:M], 0, order, out=self.enc[key][:M])
            count.copy_(sel.sum().to(torch.int32).view(1))
        q8, raw, s = self.q8, self.raw, self.scale
        for key in ("text", "image"):
            row_quant(self.enc[key][:M], q8(M, VL_D), s[0], num_warps=4)
            for i, blk in enumerate(self.w.dit):
                if not blk.cross or blk.text != (key == "text"):
                    continue
                gemm(q8(M, VL_D), blk.kv, raw(M, 2 * D))
                ci = i // 2
                dequant_rows(
                    raw(M, 2 * D)[:, :D],
                    self.kc[ci, :M].view(M, D),
                    row=s[0],
                    col=blk.kv.scale[:D],
                    bias=blk.kv.bias[:D],
                    block=512,
                )
                dequant_rows(
                    raw(M, 2 * D)[:, D:],
                    self.vc[ci, :M].view(M, D),
                    row=s[0],
                    col=blk.kv.scale[D:],
                    bias=blk.kv.bias[D:],
                    block=512,
                )

    def _dit(self):
        w, q8, raw, s = self.w, self.q8, self.raw, self.scale
        torch.addmm(w.state_b1, self.state, w.state_w1, out=self.enc_tmp)
        torch.relu_(self.enc_tmp)
        torch.addmm(w.state_b2, self.enc_tmp, w.state_w2, out=self.state_emb)
        self.a_t.copy_(self.noise)
        view = (1, SA, H, HD)
        qkv16 = self.qkv16[:SA]
        o16 = self.o16[:SA, :D]
        last = len(w.dit) - 1
        for step in range(len(w.t_disc)):
            mod = w.mod[step]
            self.h[:1].copy_(self.state_emb)
            act = torch.addmm(w.act_c[step], self.a_t, w.act_w)
            act = act * torch.sigmoid(act)
            torch.addmm(w.act_b3, act, w.act_w3, out=self.h[1:])
            self.h[1:].add_(w.pos)
            residual_norm_quant(
                self.h,
                layernorm=True,
                mod_scale=mod[0, 0],
                mod_shift=mod[0, 1],
                q=q8(SA, D),
                q_scale=s[0],
                eps=LN_EPS,
            )
            for i, blk in enumerate(w.dit):
                if blk.cross:
                    gemm(q8(SA, D), blk.qkv, raw(SA, D))
                    dequant_rows(
                        raw(SA, D),
                        qkv16[:, :D],
                        row=s[0],
                        col=blk.qkv.scale,
                        bias=blk.qkv.bias,
                        block=512,
                    )
                    ci = i // 2
                    cfg, tc5 = self.attn["cross"]
                    attention(
                        qkv16[:, :D].view(view),
                        self.kc[ci][None],
                        self.vc[ci][None],
                        out=o16.view(view),
                        kv_len=self.n_txt if blk.text else self.n_img,
                        config=cfg,
                        tc5=tc5,
                    )
                else:
                    gemm(q8(SA, D), blk.qkv, raw(SA, 3 * D))
                    dequant_rows(
                        raw(SA, 3 * D),
                        qkv16[:, : 3 * D],
                        row=s[0],
                        col=blk.qkv.scale,
                        bias=blk.qkv.bias,
                        block=512,
                    )
                    qkv = qkv16[:, : 3 * D].view(1, SA, 3, H, HD)
                    cfg, tc5 = self.attn["self"]
                    attention(
                        qkv[:, :, 0],
                        qkv[:, :, 1],
                        qkv[:, :, 2],
                        out=o16.view(view),
                        config=cfg,
                        tc5=tc5,
                    )
                row_quant(o16, q8(SA, D), s[1], num_warps=1)
                gemm(q8(SA, D), blk.o, raw(SA, D))
                residual_norm_quant(
                    self.h,
                    delta=raw(SA, D),
                    delta_row=s[1],
                    delta_col=blk.o.scale,
                    bias=blk.o.bias,
                    layernorm=True,
                    q=q8(SA, D),
                    q_scale=s[0],
                    eps=LN_EPS,
                )
                gemm(q8(SA, D), blk.ff1, raw(SA, FF))
                row_quant(
                    raw(SA, FF),
                    q8(SA, FF),
                    s[1],
                    row=s[0],
                    col=blk.ff1.scale,
                    bias=blk.ff1.bias,
                    gelu=True,
                    num_warps=8,
                )
                gemm(q8(SA, FF), blk.ff2, raw(SA, D))
                if i < last:
                    nm = mod[i + 1]
                    residual_norm_quant(
                        self.h,
                        delta=raw(SA, D),
                        delta_row=s[1],
                        delta_col=blk.ff2.scale,
                        bias=blk.ff2.bias,
                        layernorm=True,
                        mod_scale=nm[0],
                        mod_shift=nm[1],
                        q=q8(SA, D),
                        q_scale=s[0],
                        eps=LN_EPS,
                    )
                else:
                    om = w.out_mod[step]
                    residual_norm_quant(
                        self.h,
                        delta=raw(SA, D),
                        delta_row=s[1],
                        delta_col=blk.ff2.scale,
                        bias=blk.ff2.bias,
                        layernorm=True,
                        mod_scale=om[1],
                        mod_shift=om[0],
                        y=self.y,
                        eps=OUT_EPS,
                    )
            out = torch.addmm(w.proj_out_b, self.y[1:], w.proj_out_w)
            v = torch.addmm(w.dec_b2, torch.relu(torch.addmm(w.dec_b1, out, w.dec_w1)), w.dec_w2)
            self.a_t.add_(v, alpha=w.dt)
