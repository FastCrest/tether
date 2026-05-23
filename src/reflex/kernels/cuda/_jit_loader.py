"""JIT loader for vendored CUDA C++ extensions (Lift #5).

The vendored extensions (``matmul_bias``, ``gemma_rotary_embedding``,
``rotary_pos_embedding``) ship as ``.cpp`` + ``.cu`` source files in their
``src/`` subdirs. FluxVLA's setup.py builds these as a wheel-time step;
reflex's pyproject.toml installs a pure-Python wheel, so we JIT-compile via
``torch.utils.cpp_extension.load(...)`` at first import on a CUDA host.

Compilation takes ~30-60s per extension on first use. The result is cached
under ``~/.cache/torch_extensions/`` so subsequent imports are instant.

Re-vendor cadence note (ATTRIBUTION.txt): when torch / CUDA toolkit drift
breaks these extensions, manually re-vendor from FluxVLA's main branch and
rebuild the cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def jit_load_cuda_extension(
    name: str,
    sources_dir: Path | str,
    *,
    extra_cuda_cflags: list[str] | None = None,
    extra_include_paths: list[str] | None = None,
) -> Any:
    """JIT-compile a CUDA extension from .cpp + .cu source files.

    Args:
        name: Module name (matches the import target — e.g.
            ``"matmul_bias_ext"`` → registered under that name).
        sources_dir: Directory containing the ``.cpp`` and ``.cu`` files.
        extra_cuda_cflags: Optional extra NVCC flags.
        extra_include_paths: Optional extra include directories.

    Returns:
        The compiled extension module. Cached under
        ``~/.cache/torch_extensions/``; subsequent imports are instant.
    """
    from torch.utils.cpp_extension import load

    sources_dir = Path(sources_dir)
    sources = sorted(sources_dir.glob("*.cpp")) + sorted(sources_dir.glob("*.cu"))
    if not sources:
        raise FileNotFoundError(f"No .cpp/.cu sources found in {sources_dir}")

    return load(
        name=name,
        sources=[str(s) for s in sources],
        extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math"],
        extra_include_paths=extra_include_paths or [],
        verbose=False,
    )


__all__ = ["jit_load_cuda_extension"]
