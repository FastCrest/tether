# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Accel pi05 engine: SigLIP + Gemma-2B prefix + Gemma-300M flow-matching expert.

Request path (one CUDA graph per prompt-length bucket and camera count):

    images [V,3,224,224] -> SigLIP (27 layers) -> projector -> prefix rows
    token embeddings -----------------------------------> prefix rows
    prefix (18 layers, bidirectional) -> K/V cache
    noise -> 10 Euler steps of the action expert (attends prefix K/V + chunk)

All projections are per-channel FP8 with per-token activation scales
(:mod:`tether.accel.kernels`). The graph is keyed by the *physical* prefix
row count only; the valid prompt length lives in device memory, so changing
the prompt never recalibrates and rarely recaptures.
"""

from __future__ import annotations

import numpy as np
import torch

from tether.accel.kernels.activation import GELU_TANH, gated_act_quant
from tether.accel.kernels.attention import attention
from tether.accel.kernels.fp8 import E4M3, gemm
from tether.accel.kernels.norm import dequant_rows, residual_norm_quant, row_quant
from tether.accel.kernels.rope import qkv_split, rope_inv_freq

VISION_TOKENS = 256
VIS_D, VIS_HEADS, VIS_HD, VIS_FF = 1152, 16, 72, 4304
LM_D, LM_FF, EX_D, EX_FF = 2048, 16384, 1024, 4096
HEADS, HEAD_DIM = 8, 256
MAX_ACTION_DIM = 32  # the model's padded action width (action_in/out_proj)
EMBED_SCALE = 45.25  # bf16(sqrt(2048)), as the checkpoint's Gemma embedding applies it


class _Arena:
    """Flat static buffers viewed as contiguous [rows, cols] matrices of any width."""

    def __init__(self, elems, dtype, device):
        self.buf = torch.zeros(elems, dtype=dtype, device=device)

    def __call__(self, rows, cols):
        return self.buf[: rows * cols].view(rows, cols)


# Attention launch choices per site: (block_m, block_n, warps, stages, tcgen05).
# Measured on Jetson Thor (SM110) with warmed clocks; other devices use the
# same tiles without the tcgen05 compile target.
ATTENTION_SITES = {
    "vision": (128, 64, 4, 3, True),
    "prefix": (128, 64, 8, 2, True),
    "expert": (32, 128, 8, 2, False),
}


class Pi05Engine:
    def __init__(self, weights, num_views, device, *, max_prompt=256, chunk=50, bucket=16):
        self.w = weights
        self.attn = {site: (cfg[:4], cfg[4]) for site, cfg in ATTENTION_SITES.items()}
        self.V = int(num_views)
        self.dev = torch.device(device)
        self.chunk = int(chunk)
        self.bucket = int(bucket)
        self.max_prompt = int(max_prompt)
        self.steps = weights.steps
        T = self.V * VISION_TOKENS
        smax = T + -(-self.max_prompt // self.bucket) * self.bucket
        self.smax = smax
        dev, f32, f16, bf16 = self.dev, torch.float32, torch.float16, torch.bfloat16
        # inputs
        self.img = torch.zeros(self.V, 3, 224, 224, dtype=f32, device=dev)
        self.lang = torch.zeros(self.max_prompt + self.bucket, LM_D, dtype=f32, device=dev)
        self.noise = torch.zeros(self.chunk, 32, dtype=f32, device=dev)
        self.prefix_len = torch.zeros(1, dtype=torch.int32, device=dev)
        self.kv_len = torch.zeros(1, dtype=torch.int32, device=dev)
        # activations
        rows = max(smax, T)
        self.q8 = _Arena(rows * LM_FF, E4M3, dev)
        self.raw = _Arena(rows * 2 * LM_FF, bf16, dev)
        self.scale = [torch.zeros(rows, dtype=f32, device=dev) for _ in range(2)]
        self.vx = torch.zeros(T, VIS_D, dtype=f32, device=dev)
        self.vqkv = torch.zeros(T, 3 * VIS_D, dtype=f16, device=dev)
        self.patch_w = weights.patch_w.reshape(VIS_D, -1).t().contiguous().to(f16)  # [588, 1152]
        self.vo = torch.zeros(T, VIS_D, dtype=f16, device=dev)
        self.px = torch.zeros(smax, LM_D, dtype=f32, device=dev)
        self.q16 = torch.zeros(smax, HEADS, HEAD_DIM, dtype=f16, device=dev)
        self.o16 = torch.zeros(smax, HEADS * HEAD_DIM, dtype=f16, device=dev)
        self.kc = torch.zeros(18, smax + self.chunk, 1, HEAD_DIM, dtype=f16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.x_t = torch.zeros(self.chunk, 32, dtype=f32, device=dev)
        self.dx = torch.zeros(self.chunk, EX_D, dtype=f32, device=dev)
        self.dy = torch.zeros(self.chunk, EX_D, dtype=f32, device=dev)
        self.inv_freq = rope_inv_freq(HEAD_DIM, 10000.0, dev)
        # Rotary positions: prefix rows are 0..S-1; chunk rows continue after the valid prefix.
        self.pos_prefix = torch.arange(smax, dtype=f32, device=dev)[:, None].contiguous()
        self.pos_chunk = torch.zeros(self.chunk, 1, dtype=f32, device=dev)
        dt = -1.0 / self.steps
        self.out_w_dt = (weights.action_out_w * dt).t().contiguous()
        self.out_b_dt = (weights.action_out_b * dt).contiguous()
        self.in_w_t = weights.action_in_w.t().contiguous()
        self._graphs = {}
        self._pool = None
        self._rows = None
        self._host_lang = torch.zeros(self.max_prompt, LM_D, dtype=f32).pin_memory()

    # ------------------------------------------------------------------ inputs
    def set_prompt(self, token_ids):
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64))
        n = int(ids.numel())
        if not 1 <= n <= self.max_prompt:
            raise ValueError(f"pi05 engine supports 1..{self.max_prompt} prompt tokens, got {n}")
        emb = self.w.embed.index_select(0, ids).float() * EMBED_SCALE
        self._host_lang[:n].copy_(emb)
        self.lang[n:].zero_()
        self.lang[:n].copy_(self._host_lang[:n], non_blocking=True)
        prefix = self.V * VISION_TOKENS + n
        self.prefix_len.fill_(prefix)
        self.kv_len.fill_(prefix + self.chunk)
        torch.arange(prefix, prefix + self.chunk, out=self.pos_chunk[:, 0])
        self._rows = self.V * VISION_TOKENS + -(-n // self.bucket) * self.bucket

    def infer(self, images, noise=None):
        """images: [V,3,224,224] float in [-1,1] (CUDA or CPU); noise: [chunk,32] or None."""
        if self._rows is None:
            raise RuntimeError("set_prompt must be called before infer")
        self.img.copy_(images, non_blocking=True)
        if noise is None:
            self.noise.normal_()
        else:
            self.noise.copy_(torch.as_tensor(noise), non_blocking=True)
        graph = self._graphs.get(self._rows)
        if graph is None:
            graph = self._capture(self._rows)
        graph.replay()
        return self.x_t

    # ----------------------------------------------------------------- capture
    def _capture(self, rows):
        stream = torch.cuda.Stream(self.dev)
        stream.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(rows)
        torch.cuda.current_stream(self.dev).wait_stream(stream)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            self._forward(rows)
        self._graphs[rows] = graph
        return graph

    # ----------------------------------------------------------------- forward
    def _forward(self, M):
        self._vision()
        self._prefix(M)
        self._expert()

    def _vision(self):
        w, V, T = self.w, self.V, self.V * VISION_TOKENS
        q8, raw = self.q8, self.raw
        s = self.scale
        # 14x14/stride-14 patch embedding as one FP16 GEMM over reshaped pixels (HWC patch order
        # matching the conv weight's [C, kh, kw] flattening).
        pix = (
            self.img.view(V, 3, 16, 14, 16, 14)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(T, 588)
            .to(torch.float16)
        )
        self.vx.copy_(torch.mm(pix, self.patch_w)).add_(w.patch_b)
        self.vx.view(V, VISION_TOKENS, VIS_D).add_(w.pos_emb)
        L0 = w.vision[0]
        residual_norm_quant(
            self.vx, layernorm=True, weight=L0.ln1_w, beta=L0.ln1_b, q=q8(T, VIS_D), q_scale=s[0]
        )
        view = (V, VISION_TOKENS, VIS_HEADS, VIS_HD)
        for i, L in enumerate(w.vision):
            gemm(q8(T, VIS_D), L.qkv, raw(T, 3 * VIS_D))
            dequant_rows(
                raw(T, 3 * VIS_D), self.vqkv, row=s[0], col=L.qkv.scale, bias=L.qkv.bias, block=512
            )
            qkv = self.vqkv.view(V, VISION_TOKENS, 3, VIS_HEADS, VIS_HD)
            attention(
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                out=self.vo.view(view),
                scale=VIS_HD**-0.5,
                config=self.attn["vision"][0],
                tc5=self.attn["vision"][1],
            )
            row_quant(self.vo, q8(T, VIS_D), s[1])
            gemm(q8(T, VIS_D), L.o, raw(T, VIS_D))
            residual_norm_quant(
                self.vx,
                delta=raw(T, VIS_D),
                delta_row=s[1],
                delta_col=L.o.scale,
                bias=L.o.bias,
                layernorm=True,
                weight=L.ln2_w,
                beta=L.ln2_b,
                q=q8(T, VIS_D),
                q_scale=s[0],
            )
            gemm(q8(T, VIS_D), L.fc1, raw(T, VIS_FF))
            row_quant(
                raw(T, VIS_FF),
                q8(T, VIS_FF),
                s[1],
                row=s[0],
                col=L.fc1.scale,
                bias=L.fc1.bias,
                gelu=True,
                num_warps=4,
            )
            gemm(q8(T, VIS_FF), L.fc2, raw(T, VIS_D))
            nxt = w.vision[i + 1] if i + 1 < len(w.vision) else None
            residual_norm_quant(
                self.vx,
                delta=raw(T, VIS_D),
                delta_row=s[1],
                delta_col=L.fc2.scale,
                bias=L.fc2.bias,
                layernorm=True,
                weight=nxt.ln1_w if nxt else w.post_ln_w,
                beta=nxt.ln1_b if nxt else w.post_ln_b,
                q=q8(T, VIS_D),
                q_scale=s[0],
            )
        gemm(q8(T, VIS_D), w.projector, raw(T, LM_D))
        dequant_rows(
            raw(T, LM_D), self.px[:T], row=s[0], col=w.projector.scale, bias=w.projector.bias
        )

    def _prefix(self, M):
        w, T = self.w, self.V * VISION_TOKENS
        q8, raw, s = self.q8, self.raw, self.scale
        self.px[T:M].copy_(self.lang[: M - T])
        residual_norm_quant(self.px[:M], q=q8(M, LM_D), q_scale=s[0])
        qv = self.q16[:M].view(1, M, HEADS, HEAD_DIM)
        for i, L in enumerate(w.prefix):
            gemm(q8(M, LM_D), L.qkv, raw(M, 2560))
            qkv_split(
                raw(M, 2560),
                self.q16[:M],
                self.kc[i],
                self.vc[i],
                heads_q=HEADS,
                heads_kv=1,
                head_dim=HEAD_DIM,
                inv_freq=self.inv_freq,
                pos=self.pos_prefix,
                row=s[0],
                col=L.qkv.scale,
            )
            if L.o is None:
                break
            attention(
                qv,
                self.kc[i][None],
                self.vc[i][None],
                out=self.o16[:M].view(1, M, HEADS, HEAD_DIM),
                kv_len=self.prefix_len,
                scale=HEAD_DIM**-0.5,
                config=self.attn["prefix"][0],
                tc5=self.attn["prefix"][1],
            )
            row_quant(self.o16[:M], q8(M, LM_D), s[1], num_warps=1)
            gemm(q8(M, LM_D), L.o, raw(M, LM_D))
            residual_norm_quant(
                self.px[:M],
                delta=raw(M, LM_D),
                delta_row=s[1],
                delta_col=L.o.scale,
                q=q8(M, LM_D),
                q_scale=s[0],
            )
            gemm(q8(M, LM_D), L.gate_up, raw(M, 2 * LM_FF))
            gated_act_quant(
                raw(M, 2 * LM_FF),
                q8(M, LM_FF),
                s[1],
                row=s[0],
                col=L.gate_up.scale,
                act=GELU_TANH,
                num_warps=8,
            )
            gemm(q8(M, LM_FF), L.down, raw(M, LM_D))
            residual_norm_quant(
                self.px[:M],
                delta=raw(M, LM_D),
                delta_row=s[1],
                delta_col=L.down.scale,
                q=q8(M, LM_D),
                q_scale=s[0],
            )

    def _expert(self):
        w, A = self.w, self.chunk
        q8, raw, s = self.q8, self.raw, self.scale
        self.x_t.copy_(self.noise)
        qv = self.q16[:A].view(1, A, HEADS, HEAD_DIM)
        ov = self.o16[:A].view(1, A, HEADS, HEAD_DIM)
        last = len(w.expert) - 1
        for step in range(self.steps):
            torch.addmm(w.action_in_b, self.x_t, self.in_w_t, out=self.dx)
            ma, mf = w.mod_attn[step], w.mod_ffn[step]
            residual_norm_quant(
                self.dx, mod_scale=ma[0, 0], mod_shift=ma[0, 1], q=q8(A, EX_D), q_scale=s[0]
            )
            for i, L in enumerate(w.expert):
                gemm(q8(A, EX_D), L.qkv, raw(A, 2560))
                qkv_split(
                    raw(A, 2560),
                    self.q16[:A],
                    self.kc[i],
                    self.vc[i],
                    heads_q=HEADS,
                    heads_kv=1,
                    head_dim=HEAD_DIM,
                    inv_freq=self.inv_freq,
                    pos=self.pos_chunk,
                    row=s[0],
                    col=L.qkv.scale,
                    kv_base=self.prefix_len,
                )
                attention(
                    qv,
                    self.kc[i][None],
                    self.vc[i][None],
                    out=ov,
                    kv_len=self.kv_len,
                    scale=HEAD_DIM**-0.5,
                    config=self.attn["expert"][0],
                    tc5=self.attn["expert"][1],
                )
                row_quant(self.o16[:A], q8(A, 2048), s[1], num_warps=1)
                gemm(q8(A, 2048), L.o, raw(A, EX_D))
                residual_norm_quant(
                    self.dx,
                    delta=raw(A, EX_D),
                    delta_row=s[1],
                    delta_col=L.o.scale,
                    gate=ma[i, 2],
                    mod_scale=mf[i, 0],
                    mod_shift=mf[i, 1],
                    q=q8(A, EX_D),
                    q_scale=s[0],
                )
                gemm(q8(A, EX_D), L.gate_up, raw(A, 2 * EX_FF))
                gated_act_quant(
                    raw(A, 2 * EX_FF),
                    q8(A, EX_FF),
                    s[1],
                    row=s[0],
                    col=L.gate_up.scale,
                    act=GELU_TANH,
                    num_warps=16,
                )
                gemm(q8(A, EX_FF), L.down, raw(A, EX_D))
                if i < last:
                    residual_norm_quant(
                        self.dx,
                        delta=raw(A, EX_D),
                        delta_row=s[1],
                        delta_col=L.down.scale,
                        gate=mf[i, 2],
                        mod_scale=ma[i + 1, 0],
                        mod_shift=ma[i + 1, 1],
                        q=q8(A, EX_D),
                        q_scale=s[0],
                    )
                else:
                    mfin = w.mod_final[step]
                    residual_norm_quant(
                        self.dx,
                        delta=raw(A, EX_D),
                        delta_row=s[1],
                        delta_col=L.down.scale,
                        gate=mf[i, 2],
                        mod_scale=mfin[0],
                        mod_shift=mfin[1],
                        y=self.dy,
                    )
            self.x_t.addmm_(self.dy, self.out_w_dt).add_(self.out_b_dt)
