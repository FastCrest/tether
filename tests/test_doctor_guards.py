"""Doctor's silent-failure guards (v0.9.4).

Targets the version-comparison + arch-detection logic the new guards
add. Guards extracted from cli.py for testability:
  - multi-GPU mixed-arch detection
  - Jetson JetPack version comparison
  - CUDA driver vs cuDNN version skew

The empirical TRT EP session test is hard to unit-test (needs ORT +
GPU) — covered by the manual smoke + the 2026-04-29 v07-install-
validation experiment.
"""
from __future__ import annotations

import pytest


# --- Multi-GPU arch detection ---

def _arch_from_name(name: str) -> str:
    """Extracted from cli.py doctor() multi-GPU guard. Heuristic
    arch detection from `nvidia-smi --query-gpu=name` strings."""
    n = name.lower()
    if any(p in n for p in ("rtx 50", "rtx pro 60", "blackwell", "b200", "gb200")):
        return "blackwell"
    if any(p in n for p in ("h100", "h200", "hopper")):
        return "hopper"
    if any(p in n for p in ("rtx 40", "l4", "l40", "ada")):
        return "ada"
    if any(p in n for p in ("a100", "a10g", "a40", "ampere", "rtx 30")):
        return "ampere"
    if "orin" in n or "tegra" in n:
        return "orin"
    return "unknown"


@pytest.mark.parametrize("gpu_name,expected_arch", [
    # Blackwell
    ("NVIDIA GeForce RTX 5090", "blackwell"),
    ("NVIDIA GeForce RTX 5070", "blackwell"),
    ("NVIDIA RTX PRO 6000 Blackwell", "blackwell"),
    ("NVIDIA B200", "blackwell"),
    ("NVIDIA GB200 NVL72", "blackwell"),
    # Hopper
    ("NVIDIA H100 80GB HBM3", "hopper"),
    ("NVIDIA H200 141GB HBM3e", "hopper"),
    # Ada
    ("NVIDIA GeForce RTX 4090", "ada"),
    ("NVIDIA L4", "ada"),
    ("NVIDIA L40S", "ada"),
    # Ampere
    ("NVIDIA A100-SXM4-80GB", "ampere"),
    ("NVIDIA A10G", "ampere"),
    ("NVIDIA GeForce RTX 3090", "ampere"),
    # Orin (Jetson)
    ("Orin (nvgpu)", "orin"),
    ("NVIDIA Tegra Orin", "orin"),
    # Unknown
    ("Tesla V100-SXM2-32GB", "unknown"),  # Volta — not in our matrix
    ("NVIDIA Quadro RTX 8000", "unknown"),  # Turing
])
def test_arch_detection(gpu_name: str, expected_arch: str):
    assert _arch_from_name(gpu_name) == expected_arch


def test_arch_detection_handles_empty_string():
    assert _arch_from_name("") == "unknown"


def test_arch_detection_case_insensitive():
    """Arch detection should match regardless of case (nvidia-smi varies)."""
    assert _arch_from_name("nvidia geforce rtx 5090") == "blackwell"
    assert _arch_from_name("NVIDIA GEFORCE RTX 5090") == "blackwell"


def test_mixed_arch_detection_h100_plus_5090():
    """Mixed H100 + RTX 5090 box — common cloud-vs-dev scenario.
    Should detect 2 distinct archs."""
    gpu_names = ["NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 5090"]
    archs = {_arch_from_name(n) for n in gpu_names}
    archs.discard("unknown")
    assert archs == {"hopper", "blackwell"}


def test_uniform_arch_does_not_trigger_warning():
    """2x A100 setup — should be a single arch."""
    gpu_names = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"]
    archs = {_arch_from_name(n) for n in gpu_names}
    archs.discard("unknown")
    assert archs == {"ampere"}
    assert len(archs) == 1  # No mixed-arch warning fires


# --- Jetson JetPack version detection ---

def _parse_jetpack_version(release_content: str) -> int | None:
    """Extracted from cli.py doctor() Jetson guard. Parses a JetPack
    version int from the first line of /etc/nv_tegra_release.
    Format: '# R36 (release), REVISION: 4.0, GCID: ...'"""
    for line in release_content.splitlines():
        if line.startswith("# R"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1].lstrip("R"))
                except (TypeError, ValueError):
                    return None
    return None


def test_parse_jetpack_r36():
    """JetPack 6.x (Orin) ships R36 — CUDA 12, ORT 1.20+ compatible."""
    content = "# R36 (release), REVISION: 4.0, GCID: 12345, BOARD: t186ref"
    assert _parse_jetpack_version(content) == 36


def test_parse_jetpack_r35():
    """JetPack 5.x ships R35 — CUDA 11.4, NOT ORT 1.20+ compatible."""
    content = "# R35 (release), REVISION: 4.1, GCID: 67890"
    assert _parse_jetpack_version(content) == 35


def test_parse_jetpack_invalid_format():
    """Empty or malformed content returns None."""
    assert _parse_jetpack_version("") is None
    assert _parse_jetpack_version("not a release file") is None
    assert _parse_jetpack_version("# R(no number)") is None


def test_jetpack_threshold_36_ort_compat():
    """R36+ is the floor for ORT 1.20+ on Jetson (CUDA 12.x)."""
    assert _parse_jetpack_version("# R36 ...") >= 36
    assert _parse_jetpack_version("# R35 ...") < 36
    # R37 (hypothetical future) should also pass
    assert _parse_jetpack_version("# R37 (release)") >= 36


# --- CUDA driver vs cuDNN minor version requirements ---

def _min_driver_for_cudnn(cudnn_version: str) -> int:
    """Extracted from cli.py doctor() driver/cuDNN guard.
    cuDNN 9.5+ needs NVIDIA driver R555+; cuDNN 9.0-9.4 needs R550+."""
    try:
        cudnn_minor = int(cudnn_version.split(".")[1])
    except (ValueError, IndexError):
        return 550  # conservative default
    return 555 if cudnn_minor >= 5 else 550


@pytest.mark.parametrize("cudnn_version,expected_min_driver", [
    ("9.0.0", 550),
    ("9.4.99", 550),
    ("9.5.0", 555),  # threshold
    ("9.5.1", 555),
    ("9.10.0", 555),
    ("9.0", 550),  # short version string
])
def test_min_driver_for_cudnn(cudnn_version: str, expected_min_driver: int):
    assert _min_driver_for_cudnn(cudnn_version) == expected_min_driver


def test_min_driver_for_cudnn_handles_garbage():
    """Garbage version strings return conservative default (550)."""
    assert _min_driver_for_cudnn("") == 550
    assert _min_driver_for_cudnn("not.a.version") == 550


def test_driver_550_blocks_cudnn_95():
    """The combo that actually trips customers: driver R550 + cuDNN 9.5
    silently fails at first kernel call. Guard catches it."""
    cudnn = "9.5.0"
    min_required = _min_driver_for_cudnn(cudnn)
    actual_driver = 550
    assert actual_driver < min_required, (
        f"This combination should be flagged as blocking: "
        f"driver R{actual_driver}, cuDNN {cudnn}, needs R{min_required}+"
    )


def test_driver_555_passes_cudnn_95():
    """Driver R555+ is fine for cuDNN 9.5+."""
    cudnn = "9.5.0"
    min_required = _min_driver_for_cudnn(cudnn)
    actual_driver = 555
    assert actual_driver >= min_required


def test_driver_550_passes_cudnn_94():
    """Driver R550 is fine for cuDNN 9.0-9.4 (the pre-9.5 envelope)."""
    cudnn = "9.4.0"
    min_required = _min_driver_for_cudnn(cudnn)
    actual_driver = 550
    assert actual_driver >= min_required
