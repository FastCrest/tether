"""Vendored cublasLt matmul+bias extension (Apache-2.0, LimX Dynamics).

JIT-compiles the C++/CUDA sources at first import via
``tether.kernels.cuda._jit_loader``. See ``src/tether/kernels/ATTRIBUTION.txt``.
"""
from __future__ import annotations

from pathlib import Path

# Lazy JIT-compile the .cpp/.cu sources. `matmul_bias.py` imports
# `matmul_bias_ext` from this package; expose it under that name so the
# original FluxVLA import paths work unchanged.
_HERE = Path(__file__).parent


def _load() -> object:
    from tether.kernels.cuda._jit_loader import jit_load_cuda_extension
    return jit_load_cuda_extension(
        name="matmul_bias_ext",
        sources_dir=_HERE / "src",
    )


matmul_bias_ext = _load()

from .matmul_bias import *  # noqa: E402, F401, F403
