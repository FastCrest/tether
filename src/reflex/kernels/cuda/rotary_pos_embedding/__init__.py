"""Vendored generic RoPE CUDA extension (Apache-2.0, LimX Dynamics).

JIT-compiles via ``reflex.kernels.cuda._jit_loader``.
"""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def _load() -> object:
    from reflex.kernels.cuda._jit_loader import jit_load_cuda_extension
    return jit_load_cuda_extension(
        name="rotary_pos_embedding_ext",
        sources_dir=_HERE / "src",
    )


rotary_pos_embedding_ext = _load()

from .rotary_pos_embedding import *  # noqa: E402, F401, F403
