"""Unit tests for the Lift #5 Day 3 hardware gate.

Mock the CUDA device-capability probe so these run on any host (CI, Mac,
CPU-only). Real-hardware behavior is tested implicitly when Day 4-7 fires on
Modal A100 (sm 8.0) — the gate's PASS-on-A100 assumption is validated there.
"""
from __future__ import annotations

from tether.kernels._hardware_gate import (
    is_fast_kernels_hardware_compatible,
    supported_compute_capabilities,
)


# ── PASS path: supported hardware ──────────────────────────────────────


def test_a100_sm80_passes():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(8, 0),
    )
    assert ok
    assert msg == ""


def test_l4_sm89_passes():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(8, 9),
    )
    assert ok, msg


def test_h100_sm90_passes():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(9, 0),
    )
    assert ok, msg


def test_blackwell_sm100_passes():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(10, 0),
    )
    assert ok, msg


# ── REFUSE path: explicit refusals (clear error message) ───────────────


def test_a10g_sm86_refused_with_tier_message():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(8, 6),
    )
    assert not ok
    assert "A10G" in msg or "8.6" in msg
    assert "expert-only" in msg.lower() or "tier" in msg.lower()


def test_orin_sm87_refused_with_orin_message():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(8, 7),
    )
    assert not ok
    assert "Orin" in msg or "8.7" in msg


# ── REFUSE path: unsupported compute capabilities ──────────────────────


def test_turing_sm75_refused():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(7, 5),
    )
    assert not ok
    assert "7.5" in msg


def test_volta_sm70_refused():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=True,
        _device_capability_override=(7, 0),
    )
    assert not ok


# ── REFUSE path: no CUDA at all ────────────────────────────────────────


def test_no_cuda_refused():
    ok, msg = is_fast_kernels_hardware_compatible(
        _cuda_available_override=False,
    )
    assert not ok
    assert "CUDA" in msg


# ── API surface ────────────────────────────────────────────────────────


def test_supported_capabilities_returned_sorted():
    caps = supported_compute_capabilities()
    # Sorted check + must include the V1 primary target sm 8.0
    assert (8, 0) in caps
    assert (9, 0) in caps
    assert list(caps) == sorted(caps)
