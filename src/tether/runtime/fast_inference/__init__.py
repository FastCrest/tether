"""Lift #5 second-runtime fast-inference path (Triton kernels + CUDA Graph).

Sibling of ``tether.runtime`` (the ORT-based ``PolicyRuntime``). Same external
interface (state-in, action-out, RTC-refused, record-replay-compatible) but
completely different inner execution: PyTorch CUDA tensors throughout, calls
vendored Triton kernels directly, captures the entire Pi0.5 pipeline in one
``torch.cuda.CUDAGraph()``.

V1 scope: Pi0.5 only (T-2). GR00T + SmolVLA defer to Phase 2.5+.

See ``reflex_context/features/01_serve/triton-fast-kernels_plan.md`` for the
day-by-day execution plan + ``triton-fast-kernels_research.md`` for the
load-bearing architecture decisions.
"""
from __future__ import annotations

# Lazy re-export — importing Pi05FastKernelsInference eagerly here would pull
# in Triton + the vendored CUDA kernels, which fails fast on hosts without
# Triton/CUDA installed. Callers explicitly import the submodule they need.

__all__: list[str] = []
