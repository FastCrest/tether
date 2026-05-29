"""Detect the local device class for `reflex go` to pick the right model variant.

Probe order: NVIDIA GPU (nvidia-smi) → Jetson (tegrastats) → CPU. Returns one
of the canonical device-class strings used in the model registry:
  "h200", "h100", "a100", "a10g" — datacenter GPUs
  "thor", "agx_orin", "orin_nano" — Jetson edge devices
  "cpu" — fallback, no GPU detected

The probe is deliberately simple — short subprocess timeouts (5s), no heavy
dependencies, graceful fallbacks. If detection is wrong, the customer can
override via `reflex go --device-class <name>` (TODO: wire the override).
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DeviceClass = str  # one of: h200, h100, a100, a10g, thor, agx_orin, orin_nano, cpu

CANONICAL_DEVICE_CLASSES: tuple[str, ...] = (
    "h200", "h100", "a100", "a10g", "thor", "agx_orin", "orin_nano", "cpu",
)

# Substring-keyed table for nvidia-smi GPU name → device class. Order matters:
# more specific patterns first (h200 before h100 before h, etc).
_NVIDIA_SMI_PATTERNS: tuple[tuple[str, str], ...] = (
    ("h200", "h200"),
    ("h100", "h100"),
    ("a100", "a100"),
    ("a10g", "a10g"),
    ("a10", "a10g"),  # plain "A10" → use a10g profile
    # Jetson SoCs (rarely show up via nvidia-smi but include for completeness)
    ("thor", "thor"),
    ("agx orin", "agx_orin"),
    ("orin nano", "orin_nano"),
    ("orin", "agx_orin"),  # generic "Orin" → AGX (more capable; safer default)
)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of `probe_device_class()`."""

    device_class: str
    raw_gpu_name: str = ""
    detection_method: str = ""  # nvidia-smi / tegrastats / fallback-cpu
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.notes is None:
            object.__setattr__(self, "notes", [])
        if self.device_class not in CANONICAL_DEVICE_CLASSES:
            raise ValueError(
                f"device_class {self.device_class!r} not in canonical set "
                f"{CANONICAL_DEVICE_CLASSES}"
            )


def _try_nvidia_smi(timeout_s: float = 5.0) -> tuple[str, str] | None:
    """Returns (gpu_name, raw_output) or None if nvidia-smi unavailable/fails."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    name = (proc.stdout or "").strip().split("\n")[0].strip()
    if not name:
        return None
    return name, proc.stdout.strip()


def _try_tegrastats(timeout_s: float = 3.0) -> tuple[str, str] | None:
    """Returns ('agx_orin' | 'orin_nano' | 'thor', raw_output) if tegrastats present."""
    out = ""
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        out = proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        # tegrastats is a streaming monitor and normally does not exit on its
        # own. subprocess.run preserves partial stdout on TimeoutExpired; use
        # that first sample instead of treating a healthy Jetson as CPU-only.
        partial = exc.stdout if exc.stdout is not None else exc.output
        if isinstance(partial, bytes):
            out = partial.decode(errors="ignore")
        else:
            out = partial or ""
    except (FileNotFoundError, OSError):
        return None
    if not out.strip():
        return None
    # Heuristic: tegrastats output mentions hardware. Default to agx_orin if
    # we can't disambiguate; the override flag exists for edge cases.
    raw = out[:400]
    lower = out.lower()
    if "thor" in lower:
        return "thor", raw
    if "nano" in lower:
        return "orin_nano", raw
    return "agx_orin", raw


def _gpu_name_to_device_class(name: str) -> str | None:
    """Match against NVIDIA_SMI_PATTERNS; first hit wins. Returns None if no match."""
    lname = name.lower()
    for needle, cls in _NVIDIA_SMI_PATTERNS:
        if needle in lname:
            return cls
    return None


def probe_device_class(
    timeout_s: float = 5.0,
    override: str | None = None,
) -> ProbeResult:
    """Detect the local device class.

    Args:
        timeout_s: max wall-clock for each probe step
        override: explicit device class override (must be in CANONICAL_DEVICE_CLASSES);
                  bypasses detection. Use when probe misclassifies (e.g., shared GPU
                  in a sandbox).

    Returns:
        ProbeResult with device_class set; never raises (falls back to "cpu").
    """
    if override is not None:
        if override not in CANONICAL_DEVICE_CLASSES:
            raise ValueError(
                f"override device_class {override!r} not in {CANONICAL_DEVICE_CLASSES}"
            )
        return ProbeResult(
            device_class=override,
            raw_gpu_name="",
            detection_method="override",
            notes=[f"explicit override via --device-class {override}"],
        )

    # 1. nvidia-smi
    smi = _try_nvidia_smi(timeout_s)
    if smi is not None:
        gpu_name, raw = smi
        cls = _gpu_name_to_device_class(gpu_name)
        if cls is not None:
            return ProbeResult(
                device_class=cls, raw_gpu_name=gpu_name, detection_method="nvidia-smi",
            )
        # Got an nvidia GPU we don't have a profile for — log + assume a10g (mid-tier)
        logger.warning(
            "nvidia-smi reports unrecognized GPU %r; defaulting to 'a10g' profile. "
            "Use `reflex go --device-class <h200|h100|a100|a10g>` to override.",
            gpu_name,
        )
        return ProbeResult(
            device_class="a10g",
            raw_gpu_name=gpu_name,
            detection_method="nvidia-smi-unknown-fallback",
            notes=[f"unrecognized GPU {gpu_name!r}; using a10g profile by default"],
        )

    # 2. tegrastats (Jetson)
    tegra = _try_tegrastats()
    if tegra is not None:
        cls, raw = tegra
        return ProbeResult(
            device_class=cls, raw_gpu_name="", detection_method="tegrastats",
        )

    # 3. CPU fallback
    return ProbeResult(
        device_class="cpu",
        raw_gpu_name="",
        detection_method="fallback-cpu",
        notes=["no GPU detected; reflex serve on CPU is unsupported for production"],
    )
