"""Hardware compatibility gate for the Triton fast-kernels path.

Refuses to enable ``tether serve --fast-kernels`` on hardware where the vendored
Triton kernels + ``torch.cuda.CUDAGraph()`` capture path is known not to work
or is untested. Caller is expected to surface the refusal reason and fall back
to ORT silently per the kill-trigger ADR
(``reflex_context/01_decisions/2026-05-20-fast-kernels-kill-triggers.md``).

V1 supported (T-2 = pi0.5 only):

- **sm 8.0 (A100)** — primary target, FluxVLA's published 9.64× benchmark.
- **sm 8.9 (RTX 4090 / L4)** — supported; FluxVLA published 6.99× on RTX 5090
  which is sm 12.0, but A40/L4 (sm 8.9) is the closer match for Phase 1.5.
- **sm 9.0 (H100 / H200)** — supported; same Triton dialect.
- **sm 10.0+ (B100 / B200 / RTX 5090)** — accepted as "Blackwell with TRT-LLM
  fallback" path; the fast-kernels path also works per FM-2 testing.

V1 refused:

- **sm 8.6 (A10G)** — Tier-degraded per the cuda-graphs ADR
  (``2026-04-24-cuda-graphs-architecture.md``); expert-only capture supported
  in Phase 2.5; V1 refuses.
- **sm 8.7 (Orin Nano / Orin AGX)** — untested; 8GB Orin Nano can't fit the
  Pi0.5 graph; AGX is plausible but not in V1 scope.
- **sm < 8.0 (Volta, Turing, older)** — Triton + flash-attn primitives
  incompatible.
- **No CUDA** (CPU-only host, MPS, etc.) — fundamentally incompatible.
"""
from __future__ import annotations

from typing import Any


# Supported compute capabilities (sm major.minor as tuples).
# V1 = pi0.5 on Ampere/Hopper/Blackwell. Add as we test.
_SUPPORTED_SM: frozenset[tuple[int, int]] = frozenset({
    (8, 0),   # A100
    (8, 9),   # RTX 4090, L4, L40, A40
    (9, 0),   # H100, H200
    (10, 0),  # B100 / RTX 5090
    (12, 0),  # RTX 5090 (post-naming; FluxVLA bench was here)
})


# Explicitly-refused compute capabilities (clearer error than "not supported").
_REFUSED_SM_WITH_REASON: dict[tuple[int, int], str] = {
    (8, 6): (
        "A10G (sm 8.6) is a tier-degraded device for fast-kernels — only the "
        "expert-only capture path will work, and that's deferred to Phase 2.5+. "
        "Drop --fast-kernels for V1."
    ),
    (8, 7): (
        "Orin (sm 8.7) is not in the V1 fast-kernels target hardware list. "
        "8GB Orin Nano can't fit the Pi0.5 graph; AGX is plausible but untested. "
        "Drop --fast-kernels for V1; ORT/TRT path supports Orin."
    ),
}


def is_fast_kernels_hardware_compatible(
    *,
    _cuda_available_override: bool | None = None,
    _device_capability_override: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    """Return ``(is_compatible, message)`` for the current host hardware.

    Args (testing only):
        _cuda_available_override: If not None, skip ``torch.cuda.is_available()``
            check and use this value. For mocked unit tests.
        _device_capability_override: If not None, skip the device-capability
            probe and use this ``(major, minor)`` tuple. For mocked unit tests.

    Returns:
        ``(True, "")`` if the host can run --fast-kernels. ``(False, reason)``
        otherwise. The reason should be surfaced verbatim in the refuse-to-load
        error so users know why they got ORT instead of Triton.
    """
    # Allow injection for unit tests that don't have CUDA.
    if _cuda_available_override is None:
        try:
            import torch
        except ImportError:
            return (False, "fast-kernels requires PyTorch (not installed)")
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = _cuda_available_override

    if not cuda_available:
        return (
            False,
            "fast-kernels requires CUDA. Detected no CUDA device on this host "
            "(MPS / CPU / Jetson-without-runtime won't work). Drop --fast-kernels.",
        )

    if _device_capability_override is None:
        import torch
        cap = torch.cuda.get_device_capability(0)
    else:
        cap = _device_capability_override

    if cap in _REFUSED_SM_WITH_REASON:
        return (False, _REFUSED_SM_WITH_REASON[cap])

    if cap not in _SUPPORTED_SM:
        return (
            False,
            f"fast-kernels has not been validated on sm {cap[0]}.{cap[1]}. "
            f"Supported V1 hardware: A100 (8.0), RTX 4090 / L4 / L40 / A40 (8.9), "
            f"H100 / H200 (9.0), B100 / RTX 5090 (10.0+). Drop --fast-kernels.",
        )

    return (True, "")


def supported_compute_capabilities() -> tuple[tuple[int, int], ...]:
    """List the supported (major, minor) compute capabilities for V1."""
    return tuple(sorted(_SUPPORTED_SM))


__all__ = [
    "is_fast_kernels_hardware_compatible",
    "supported_compute_capabilities",
]
