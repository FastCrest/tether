# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Triton compilation target selection.

Triton 3.6 compiles ``tl.dot`` for SM110 (Jetson Thor) with the legacy
``mma.sync`` path, although the device implements Blackwell's ``tcgen05``
tensor-core instructions. Compiling the IR for SM100 while still assembling
for the real device (Triton's ``override_arch`` only changes the IR target)
emits ``tcgen05`` code for Thor. Kernels opt in per launch through a ``TC5``
constexpr so both variants have distinct in-memory cache entries.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

_TC5_ARCH = {(11, 0): "sm100"}


def tc5_arch(device=None):
    """IR arch string that enables tcgen05 on this device, or None."""
    return _TC5_ARCH.get(torch.cuda.get_device_capability(device))


@contextmanager
def compile_for(arch):
    if arch is None:
        yield
        return
    import triton

    knobs = triton.knobs.runtime
    old = knobs.override_arch
    knobs.override_arch = arch
    try:
        yield
    finally:
        knobs.override_arch = old
