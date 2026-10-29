# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Triton/FP8 kernels for the ``tether.accel`` engines (Thor, RTX 4090, RTX 5090).

Design rules shared by the engine kernels:

* Activations are quantized to FP8 E4M3 *per row* at the moment they are
  produced (norms, activations, attention outputs). A row's scale travels next
  to it, so no calibration pass is needed and one outlier token cannot degrade
  the resolution of every other token.
* Weights are quantized per output channel. FP8 GEMMs (``torch._scaled_mm``,
  :mod:`.fp8`) therefore run with unit scales and write their raw accumulator in
  BF16; the consuming kernel applies ``row_scale[m] * col_scale[n]`` (plus bias)
  while it reads the tile, so the rescale costs no extra launch or memory pass.
* Residual streams stay in FP32. (The kernels also accept a BF16 residual and a
  causal attention mode; no engine ported into Tether uses either yet.)
* Shapes that change between requests (prompt length) are read from device
  memory, so one captured graph serves every request that fits its buffers.
* On SM110 (Jetson Thor) matrix-product kernels can be compiled for sm100 so
  ``tl.dot`` uses tcgen05; kernels opt in per launch (:mod:`._triton`).

Importing this package does not import Triton or require a GPU; the Triton
kernel modules (activation, attention, norm, rope) import
Triton when they are themselves imported.
"""
