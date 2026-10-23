"""Detect the local device class for `tether go` to pick the right model variant.

Probe order: Jetson board model → NVIDIA GPU (nvidia-smi) → tegrastats → CPU. Returns one
of the canonical device-class strings used in the model registry:
  "h200", "h100", "a100", "a10g" — datacenter GPUs
  "thor", "agx_orin", "orin_nano" — Jetson edge devices
  "cpu" — fallback, no GPU detected

The probe is deliberately simple — short subprocess timeouts (5s), no heavy
dependencies, graceful fallbacks. If detection is wrong, the customer can
override via `tether go --device-class <name>`.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DeviceClass = str  # one of: h200, h100, a100, a10g, thor, agx_orin, orin_nano, cpu

CANONICAL_DEVICE_CLASSES: tuple[str, ...] = (
    "h200",
    "h100",
    "a100",
    "a10g",
    "thor",
    "agx_orin",
    "orin_nano",
    "cpu",
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
)

_DEVICE_TREE_MODEL_PATHS = (
    Path("/proc/device-tree/model"),
    Path("/sys/firmware/devicetree/base/model"),
)


def _try_jetson_model() -> tuple[str, str] | None:
    """Read bounded board identity; telemetry and RAM size do not identify a SKU."""
    for path in _DEVICE_TREE_MODEL_PATHS:
        try:
            with path.open("rb") as stream:
                raw = stream.read(4097)
        except OSError:
            continue
        if len(raw) > 4096:
            continue
        model = raw.decode("utf-8", errors="replace").replace("\x00", " ").strip()
        normalized = re.sub(r"[\s_-]+", " ", model.lower())
        if not any(word in normalized for word in ("jetson", "tegra", "orin")):
            continue
        # A generic Orin (or an unsupported NX/Xavier) is not an AGX Orin.
        # CPU means no supported GPU profile identified, with an explicit note.
        return _gpu_name_to_device_class(normalized) or "cpu", model
    return None


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of `probe_device_class()`."""

    device_class: str
    raw_gpu_name: str = ""
    detection_method: str = ""  # nvidia-smi / tegrastats / fallback-cpu
    notes: list[str] = None  # type: ignore[assignment]
    requires_device_override: bool = False

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
    for executable in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        try:
            proc = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            continue
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None
        name = (proc.stdout or "").strip().split("\n")[0].strip()
        if not name:
            return None
        return name, proc.stdout.strip()
    return None


def _try_tegrastats(timeout_s: float = 3.0) -> tuple[str, str] | None:
    """Return an explicit supported model, or CPU for unidentified telemetry."""
    out = ""
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
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
    # Ordinary tegrastats samples contain utilization, not board identity.
    # Never infer the higher-memory AGX profile from that sample alone.
    raw = out[:400]
    return _gpu_name_to_device_class(out) or "cpu", raw


def _gpu_name_to_device_class(name: str) -> str | None:
    """Match against NVIDIA_SMI_PATTERNS; first hit wins. Returns None if no match."""
    lname = re.sub(r"[\s_-]+", " ", name.lower())
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

    # Board identity takes precedence over generic SoC names and telemetry.
    board = _try_jetson_model()
    if board is not None:
        board_class, model = board
        return ProbeResult(
            device_class=board_class,
            raw_gpu_name=model,
            detection_method="device-tree",
            notes=_unidentified_jetson_notes() if board_class == "cpu" else [],
            requires_device_override=board_class == "cpu",
        )

    # 1. nvidia-smi
    smi = _try_nvidia_smi(timeout_s)
    if smi is not None:
        gpu_name, raw = smi
        cls = _gpu_name_to_device_class(gpu_name)
        if cls is not None:
            return ProbeResult(
                device_class=cls,
                raw_gpu_name=gpu_name,
                detection_method="nvidia-smi",
            )
        if "orin" in gpu_name.lower() or "jetson" in gpu_name.lower():
            return ProbeResult(
                device_class="cpu",
                raw_gpu_name=gpu_name,
                detection_method="nvidia-smi-unidentified-jetson",
                notes=_unidentified_jetson_notes(),
                requires_device_override=True,
            )
        # Got an nvidia GPU we don't have a profile for — log + assume a10g (mid-tier)
        logger.warning(
            "nvidia-smi reports unrecognized GPU %r; defaulting to 'a10g' profile. "
            "Use `tether go --device-class <h200|h100|a100|a10g>` to override.",
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
            device_class=cls,
            raw_gpu_name="",
            detection_method="tegrastats",
            notes=_unidentified_jetson_notes() if cls == "cpu" else [],
            requires_device_override=cls == "cpu",
        )

    # 3. CPU fallback
    return ProbeResult(
        device_class="cpu",
        raw_gpu_name="",
        detection_method="fallback-cpu",
        notes=["no GPU detected; tether serve on CPU is unsupported for production"],
    )


def _unidentified_jetson_notes() -> list[str]:
    return [
        "Jetson GPU profile could not be identified; no AGX profile assumed. "
        "Use --device-class orin_nano, agx_orin or thor only after checking the board model. "
        "CPU fallback is unsupported for production serving."
    ]
