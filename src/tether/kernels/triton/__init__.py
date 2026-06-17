"""Vendored Triton kernels for `tether serve --fast-kernels` (Lift #5).

See ``src/tether/kernels/ATTRIBUTION.txt`` for the three-source license trail
(FluxVLA Apache-2.0 + LimX Apache-2.0 + Dexmal MIT). Re-vendoring requires
manual ``cp`` to preserve the in-file Dexmal SPDX headers.

Triton + PyTorch are pinned at install per T-5 (``triton>=3.0,<3.2`` +
``torch>=2.6,<2.8``). Import failure must trigger silent fallback to ORT/TRT
per the kill-trigger ADR
(``reflex_context/01_decisions/2026-05-20-fast-kernels-kill-triggers.md``).
"""
# Re-exports are intentionally lazy — importing a Triton kernel module on a
# host without a CUDA + Triton install would error out. Callers should
# import the specific kernel submodule they need.

__all__: list[str] = []
