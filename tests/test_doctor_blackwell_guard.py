"""Doctor's Blackwell guard — verify the version-comparison logic that
catches future customers from being stuck on pre-1.25.1 ORT on sm_120
hardware (the trap that cost rob 2 weeks 2026-04-28 → 2026-05-07).

The full doctor invocation is heavy (probes torch, ORT, CUDA, fastapi,
etc.); these tests target only the Version() comparison + the
_gpu_is_blackwell() integration, which is what the new check actually
adds.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from packaging.version import Version


# --- The guard's core logic, extracted for testing ---

MIN_BLACKWELL_SAFE_ORT = Version("1.25.1")


def needs_ort_upgrade_for_blackwell(ort_version: str) -> bool:
    """True iff customer running ORT < 1.25.1 on Blackwell hardware
    needs to upgrade. Pre-1.25.0 lacks sm_120 kernels entirely; 1.23
    and 1.24 actively regressed sm_120 with cudaErrorNoKernelImageForDevice.
    1.25.0 (2026-04-20) added sm_120; 1.25.1 (2026-04-27) is current
    stable + the floor we recommend."""
    return Version(ort_version) < MIN_BLACKWELL_SAFE_ORT


def test_pre_blackwell_versions_need_upgrade():
    """All pre-1.25.0 versions need upgrade for Blackwell support."""
    for v in ("1.20.0", "1.21.0", "1.22.0", "1.23.0", "1.23.5", "1.24.0", "1.24.4"):
        assert needs_ort_upgrade_for_blackwell(v), (
            f"{v} predates Blackwell support; should require upgrade"
        )


def test_1_25_0_still_needs_upgrade():
    """1.25.0 added sm_120 but we recommend 1.25.1 as the floor (current
    stable + has any 1.25.0-shipped fixes)."""
    assert needs_ort_upgrade_for_blackwell("1.25.0")


def test_1_25_1_does_not_need_upgrade():
    """1.25.1 is the current stable + recommended Blackwell floor."""
    assert not needs_ort_upgrade_for_blackwell("1.25.1")


def test_post_1_25_1_does_not_need_upgrade():
    """Future ORT releases (1.25.2, 1.26, etc.) inherit Blackwell
    support — the guard only fires below the minimum floor."""
    for v in ("1.25.2", "1.25.5", "1.26.0", "1.30.0", "2.0.0"):
        assert not needs_ort_upgrade_for_blackwell(v), (
            f"{v} is past the Blackwell floor; should NOT require upgrade"
        )


def test_dev_versions_compared_correctly():
    """Pre-release suffixes — 1.25.1.dev0 < 1.25.1 per packaging.version."""
    assert needs_ort_upgrade_for_blackwell("1.25.1.dev0")
    assert needs_ort_upgrade_for_blackwell("1.25.1rc1")  # release candidate
    assert not needs_ort_upgrade_for_blackwell("1.25.1.post1")  # post-release


def test_gpu_is_blackwell_pattern_matches_real_gpu_names():
    """Verify the _gpu_is_blackwell pattern set covers known Blackwell
    SKUs we've encountered or plan to support. Names lifted from past
    incidents (rob's RTX 5090 segfault 2026-04-28) + NVIDIA product
    pages."""
    from reflex.runtime.server import _BLACKWELL_GPU_PATTERNS

    real_blackwell_names = [
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5070",
        "NVIDIA RTX PRO 6000 Blackwell",
        "NVIDIA B200",
        "NVIDIA GB200 NVL72",
    ]
    for gpu_name in real_blackwell_names:
        lower = gpu_name.lower()
        matched = any(pat in lower for pat in _BLACKWELL_GPU_PATTERNS)
        assert matched, (
            f"GPU {gpu_name!r} should match one of the Blackwell patterns "
            f"but didn't. Patterns: {_BLACKWELL_GPU_PATTERNS}"
        )


def test_gpu_is_blackwell_does_not_match_non_blackwell():
    """Hopper / Ada / Ampere GPUs must NOT match Blackwell patterns
    (or the doctor would emit spurious upgrade warnings on supported
    hardware)."""
    from reflex.runtime.server import _BLACKWELL_GPU_PATTERNS

    non_blackwell_names = [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200 141GB HBM3e",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA A10G",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA L4",
        "Tesla V100-SXM2-32GB",
        "Jetson Orin Nano Developer Kit",
    ]
    for gpu_name in non_blackwell_names:
        lower = gpu_name.lower()
        matched = any(pat in lower for pat in _BLACKWELL_GPU_PATTERNS)
        assert not matched, (
            f"GPU {gpu_name!r} is NOT Blackwell but matched a pattern. "
            f"Patterns: {_BLACKWELL_GPU_PATTERNS}"
        )
