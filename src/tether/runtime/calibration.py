"""Auto-calibration substrate — Phase 1 a2u-calibration feature.

Per ADR 2026-04-25-auto-calibration-architecture:
- SELECTION not tuning: pick among pre-shipped variants + bucketed values
- Greedy resolver order: variant → provider → NFE → chunk_size →
  latency_compensation_ms (strict partial order)
- Schema v1 with `schema_version` as the FIRST field for fast-path detection
- Cache key: (hardware_fingerprint, embodiment, model_hash)
- Hardware fingerprint: gpu_uuid + gpu_name + driver_major.minor + cuda +
  kernel + cpu_count + ram_gb + tether_version (major.minor only on driver
  to avoid invalidation on every patch)

This module is the substrate (Day 1 of the plan). The greedy resolver +
measurement harness + CLI integration are Days 2-9.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Schema version of the cache JSON. Bump on a breaking change; v1 readers
# refuse to load v2+. Phase 1 = v1; Phase 2 evolution is additive-only
# (new optional fields only, no rename / no remove).
SCHEMA_VERSION = 1

# Cold-start defaults for latency_compensation_ms by embodiment, used until
# the LatencyTracker has populated real samples (per ADR Lens 3 estimates).
COLD_START_LATENCY_COMP_MS_BY_EMBODIMENT: dict[str, float] = {
    "franka": 40.0,
    "so100": 60.0,
    "ur5": 40.0,
}
DEFAULT_COLD_START_LATENCY_COMP_MS = 40.0

# How long after a calibration entry was recorded before we treat it as
# stale (per ADR: 30 days OR fingerprint mismatch OR tether_version change).
DEFAULT_STALE_AFTER_DAYS = 30


@dataclass(frozen=True)
class HardwareFingerprint:
    """Per-host identity. Stable across reboots; insensitive to driver
    patch-version bumps (only major.minor matters for cache validity).

    `current()` probes the running host. Returns sentinel "unknown" values
    on probes that fail (e.g., nvidia-smi missing on a CPU-only host) so
    operators see one consistent fingerprint shape, not None / Optional.
    """

    gpu_uuid: str
    gpu_name: str
    driver_version_major: int
    driver_version_minor: int
    cuda_version_major: int
    cuda_version_minor: int
    kernel_release: str
    cpu_count: int
    ram_gb: int
    tether_version: str

    @classmethod
    def current(cls) -> "HardwareFingerprint":
        """Probe the running host. Always returns a valid fingerprint;
        unknown fields populated with sentinels."""
        gpu_uuid, gpu_name = _probe_gpu()
        driver_major, driver_minor = _probe_driver_version()
        cuda_major, cuda_minor = _probe_cuda_version()
        kernel = platform.release() or "unknown"
        cpu_count = os.cpu_count() or 0
        ram_gb = _probe_ram_gb()
        tether_version = _probe_tether_version()
        return cls(
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version_major=driver_major,
            driver_version_minor=driver_minor,
            cuda_version_major=cuda_major,
            cuda_version_minor=cuda_minor,
            kernel_release=kernel,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            tether_version=tether_version,
        )

    def matches(self, other: "HardwareFingerprint", *, strict: bool = False) -> bool:
        """Compare two fingerprints. Default mode ignores `kernel_release` +
        `ram_gb` minor differences (kernel patch + RAM rounding). Strict
        mode requires bitwise equality."""
        if strict:
            return self == other
        return (
            self.gpu_uuid == other.gpu_uuid
            and self.gpu_name == other.gpu_name
            and self.driver_version_major == other.driver_version_major
            and self.driver_version_minor == other.driver_version_minor
            and self.cuda_version_major == other.cuda_version_major
            and self.cuda_version_minor == other.cuda_version_minor
            and self.cpu_count == other.cpu_count
            and abs(self.ram_gb - other.ram_gb) <= 1  # tolerate ±1 GB rounding
            and self.tether_version == other.tether_version
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HardwareFingerprint":
        return cls(**d)


@dataclass(frozen=True)
class MeasurementQuality:
    """Quality of one measurement run — recorded into each calibration entry
    so operators can see how trustworthy the calibration is."""

    warmup_iters: int
    measurement_iters: int
    median_ms: float
    p99_ms: float
    n_outliers_dropped: int
    quality_score: float  # ∈ [0, 1] — 1.0 = clean (low variance), 0.0 = noisy

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(
                f"quality_score must be in [0, 1], got {self.quality_score}"
            )
        if self.warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {self.warmup_iters}")
        if self.measurement_iters < 1:
            raise ValueError(
                f"measurement_iters must be >= 1, got {self.measurement_iters}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementQuality":
        return cls(**d)


@dataclass(frozen=True)
class MeasurementContext:
    """Stack-version snapshot at calibration time. Drives staleness detection
    on the next boot: any field changing invalidates the cached entry."""

    ort_version: str
    torch_version: str | None  # None when torch isn't imported
    numpy_version: str
    onnx_version: str | None  # None when onnx isn't imported

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementContext":
        return cls(**d)

    @classmethod
    def current(cls) -> "MeasurementContext":
        """Probe versions of the runtime stack. Optional imports return None."""
        ort_v = _safe_pkg_version("onnxruntime")
        torch_v = _safe_pkg_version("torch")
        numpy_v = _safe_pkg_version("numpy") or "unknown"
        onnx_v = _safe_pkg_version("onnx")
        return cls(
            ort_version=ort_v or "unknown",
            torch_version=torch_v,
            numpy_version=numpy_v,
            onnx_version=onnx_v,
        )


@dataclass(frozen=True)
class CalibrationEntry:
    """One calibration result for a (hardware × embodiment × model_hash) tuple.

    Frozen + serializable — written once per calibration pass; never mutated.
    Re-calibration produces a new entry that overwrites the old via
    CalibrationCache.record().
    """

    chunk_size: int
    nfe: int
    latency_compensation_ms: float
    provider: str
    variant: str
    measurement_quality: MeasurementQuality
    measurement_context: MeasurementContext
    timestamp: str  # ISO 8601

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if not (1 <= self.nfe <= 50):
            raise ValueError(f"nfe must be in [1, 50], got {self.nfe}")
        if self.latency_compensation_ms < 0:
            raise ValueError(
                f"latency_compensation_ms must be >= 0, got "
                f"{self.latency_compensation_ms}"
            )
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if not self.variant:
            raise ValueError("variant must be non-empty")

    def age_seconds(self) -> float:
        """Seconds since the entry was recorded. Returns large positive value
        on parse failure (treats unparseable timestamp as stale)."""
        try:
            ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - ts).total_seconds()
        except Exception:  # noqa: BLE001
            return float("inf")

    def is_stale(self, max_age_days: float = DEFAULT_STALE_AFTER_DAYS) -> bool:
        return self.age_seconds() > max_age_days * 86_400.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "nfe": self.nfe,
            "latency_compensation_ms": self.latency_compensation_ms,
            "provider": self.provider,
            "variant": self.variant,
            "measurement_quality": self.measurement_quality.to_dict(),
            "measurement_context": self.measurement_context.to_dict(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationEntry":
        return cls(
            chunk_size=int(d["chunk_size"]),
            nfe=int(d["nfe"]),
            latency_compensation_ms=float(d["latency_compensation_ms"]),
            provider=str(d["provider"]),
            variant=str(d["variant"]),
            measurement_quality=MeasurementQuality.from_dict(d["measurement_quality"]),
            measurement_context=MeasurementContext.from_dict(d["measurement_context"]),
            timestamp=str(d["timestamp"]),
        )


@dataclass
class CalibrationCache:
    """Top-level cache container — one per host. Backed by a JSON file at
    `~/.tether/calibration.json` by default.

    Schema v1 layout (FIRST field is `schema_version` for fast-path version
    detection):

    {
      "schema_version": 1,
      "tether_version": "0.5.0",
      "calibration_date": "2026-04-25T10:00:00Z",
      "hardware_fingerprint": {...},
      "entries": {
        "franka::abc123": <CalibrationEntry dict>,
        "so100::def456": <CalibrationEntry dict>,
        ...
      }
    }

    Entry keys are `{embodiment}::{model_hash}`. Model hash comes from the
    existing compute_model_hash function (server.py:1246). Embodiment comes
    from the loaded EmbodimentConfig.embodiment field.

    Phase 2 evolution: ADD-ONLY. Never rename / remove a v1 field. Schema v2
    bump only on breaking changes; v1 readers refuse to load v2+ loud.
    """

    schema_version: int = SCHEMA_VERSION
    tether_version: str = ""
    calibration_date: str = ""
    hardware_fingerprint: HardwareFingerprint | None = None
    entries: dict[str, CalibrationEntry] = field(default_factory=dict)

    # Class constants surfaced for downstream consumers.
    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    @staticmethod
    def make_key(embodiment: str, model_hash: str) -> str:
        if not embodiment or not model_hash:
            raise ValueError(
                f"embodiment + model_hash must both be non-empty; got "
                f"{embodiment!r}, {model_hash!r}"
            )
        if "::" in embodiment or "::" in model_hash:
            raise ValueError(
                "embodiment + model_hash must not contain '::' separator"
            )
        return f"{embodiment}::{model_hash}"

    def lookup(
        self,
        *,
        embodiment: str,
        model_hash: str,
        require_fingerprint: HardwareFingerprint | None = None,
    ) -> CalibrationEntry | None:
        """Return the cached entry for this (embodiment, model_hash) tuple,
        or None if absent / hardware-mismatched.

        When `require_fingerprint` is provided, returns None unless the
        cached fingerprint matches (default tolerance — kernel patch +
        RAM rounding ignored)."""
        if (
            require_fingerprint is not None
            and self.hardware_fingerprint is not None
            and not self.hardware_fingerprint.matches(require_fingerprint)
        ):
            return None
        return self.entries.get(self.make_key(embodiment, model_hash))

    def record(
        self,
        *,
        embodiment: str,
        model_hash: str,
        entry: CalibrationEntry,
    ) -> None:
        """Insert or overwrite the entry. Updates calibration_date."""
        self.entries[self.make_key(embodiment, model_hash)] = entry
        self.calibration_date = _utcnow_iso()

    def is_stale(
        self,
        current: HardwareFingerprint,
        *,
        max_age_days: float = DEFAULT_STALE_AFTER_DAYS,
    ) -> bool:
        """Cache is stale when:
        - hardware fingerprint mismatches OR
        - calibration_date older than max_age_days
        """
        if self.hardware_fingerprint is None:
            return True
        if not self.hardware_fingerprint.matches(current):
            return True
        if not self.calibration_date:
            return True
        try:
            ts = datetime.fromisoformat(self.calibration_date.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
            return age_s > max_age_days * 86_400.0
        except Exception:  # noqa: BLE001
            return True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "tether_version": self.tether_version,
            "calibration_date": self.calibration_date,
            "hardware_fingerprint": (
                self.hardware_fingerprint.to_dict()
                if self.hardware_fingerprint is not None else None
            ),
            "entries": {
                k: v.to_dict() for k, v in self.entries.items()
            },
        }
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationCache":
        sv = int(d.get("schema_version", 0))
        if sv > SCHEMA_VERSION:
            raise ValueError(
                f"calibration cache schema_version={sv} exceeds supported "
                f"version {SCHEMA_VERSION}. Upgrade tether or delete "
                f"the cache file."
            )
        if sv < 1:
            raise ValueError(
                f"calibration cache schema_version={sv} is invalid (must be >= 1)"
            )
        fp = d.get("hardware_fingerprint")
        return cls(
            schema_version=sv,
            tether_version=str(d.get("tether_version", "")),
            calibration_date=str(d.get("calibration_date", "")),
            hardware_fingerprint=(
                HardwareFingerprint.from_dict(fp) if fp is not None else None
            ),
            entries={
                str(k): CalibrationEntry.from_dict(v)
                for k, v in d.get("entries", {}).items()
            },
        )

    def save(self, path: str | Path) -> None:
        """Atomic write via temp + rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationCache":
        """Load + validate. Raises FileNotFoundError when path doesn't exist."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def load_or_empty(cls, path: str | Path) -> "CalibrationCache":
        """Load if file exists; else return a fresh empty cache."""
        path = Path(path)
        if not path.exists():
            return cls(
                schema_version=SCHEMA_VERSION,
                tether_version=_probe_tether_version(),
                calibration_date="",
                hardware_fingerprint=None,
                entries={},
            )
        return cls.load(path)


# ---------------------------------------------------------------------------
# Hardware probes — defensive, all failures fall back to "unknown" sentinels
# ---------------------------------------------------------------------------


def _probe_gpu() -> tuple[str, str]:
    """Probe primary GPU UUID + name via nvidia-smi. Returns ("unknown",
    "unknown") on any failure (CPU-only host, NVIDIA driver missing, etc.)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return ("unknown", "unknown")
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            return ("unknown", "unknown")
        return (parts[0], parts[1])
    except Exception:  # noqa: BLE001
        return ("unknown", "unknown")


def _probe_driver_version() -> tuple[int, int]:
    """Probe NVIDIA driver version (major.minor only — patch ignored)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode != 0:
            return (0, 0)
        v = result.stdout.strip().split("\n")[0].strip()
        m = re.match(r"(\d+)\.(\d+)", v)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        return (0, 0)


def _probe_cuda_version() -> tuple[int, int]:
    """Probe CUDA version (major.minor).

    Tries 3 sources in order: torch.version.cuda (most reliable -- works
    in slim images without dev tools), nvcc --version (dev image),
    nvidia-smi (last resort, returns driver-cuda not toolkit-cuda).
    Returns (0, 0) when all sources fail (e.g., CPU-only host).

    Caught by 2026-04-25 calibration matrix Modal smoke: nvcc was
    absent in the slim image, falling through to sentinel (0, 0).
    """
    # 1. torch.version.cuda is the most reliable on inference machines
    #    (debian_slim images ship torch but not cuda-dev).
    try:
        import torch
        cuda_str = getattr(torch.version, "cuda", None)
        if cuda_str:
            m = re.match(r"(\d+)\.(\d+)", cuda_str)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        pass

    # 2. nvcc (dev images / full CUDA installs)
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode == 0:
            m = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        pass

    # 3. nvidia-smi reports driver's CUDA, which differs from toolkit
    #    but is better than nothing for image-cache-validity comparison.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode == 0:
            m = re.match(r"\s*(\d+)\.(\d+)", result.stdout)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        pass

    return (0, 0)


def _probe_ram_gb() -> int:
    """Probe total RAM in GB. Linux: parse /proc/meminfo. macOS: sysctl.
    Other: 0."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return max(1, round(kb / (1024 * 1024)))
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2.0,
            )
            if result.returncode == 0:
                bytes_total = int(result.stdout.strip())
                return max(1, round(bytes_total / (1024 ** 3)))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _probe_tether_version() -> str:
    """Probe installed tether version. Returns 'unknown' on any failure."""
    return _safe_pkg_version("tether") or "unknown"


def _safe_pkg_version(pkg: str) -> str | None:
    """Return version string for an installed pkg, or None when not installed."""
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:  # noqa: BLE001
        return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Measurement harness (Day 2)
# ---------------------------------------------------------------------------


# Default tail-trim for outlier rejection. 5% on each side drops single
# noisy outliers (GC pause, kernel scheduler hiccup) without distorting
# the median. Bench-revamp methodology defaults match (cf. bench/methodology.py).
_DEFAULT_OUTLIER_TRIM_FRAC = 0.05

# Quality-score thresholds mapping coefficient-of-variation (std / median)
# to a [0, 1] confidence. Below CLEAN, full credit; between CLEAN and NOISY,
# linear ramp; above NOISY, zero. Tuned so that healthy A10G measurements
# (typically CV ~ 0.05) score ~1.0 and obviously thrashing measurements
# (CV > 0.30) score 0 — refuses-to-write threshold downstream.
_QUALITY_CV_CLEAN = 0.10
_QUALITY_CV_NOISY = 0.30


def measure_latency_profile(
    predict_callable,
    *,
    n_iters: int = 100,
    warmup_iters: int = 10,
    outlier_trim_frac: float = _DEFAULT_OUTLIER_TRIM_FRAC,
) -> MeasurementQuality:
    """Bench-style timing harness.

    Args:
        predict_callable: zero-argument function that runs ONE forward pass.
            Caller is responsible for binding inputs (closure / partial).
        n_iters: number of measured forwards. Default 100 — enough for
            stable p99; small enough to keep calibration sub-second on
            cheap inferences.
        warmup_iters: number of pre-measurement forwards to discard.
            Default 10 — covers JIT warmup + ORT cuDNN algo selection.
        outlier_trim_frac: fraction of the SORTED measurements to drop
            from each tail (default 0.05 = 5% each side). Catches GC
            pauses + scheduler hiccups without distorting the median.

    Returns:
        A MeasurementQuality with median_ms + p99_ms + n_outliers_dropped
        + quality_score derived from coefficient of variation.

    Raises:
        ValueError: on invalid args. predict_callable exceptions are
            propagated unchanged — calibration must fail loud on a
            broken predict path, not silently average over crashes.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
    if not (0.0 <= outlier_trim_frac < 0.5):
        raise ValueError(
            f"outlier_trim_frac must be in [0, 0.5), got {outlier_trim_frac}"
        )
    if not callable(predict_callable):
        raise TypeError("predict_callable must be a callable")

    # Warmup — discarded.
    for _ in range(warmup_iters):
        predict_callable()

    # Measurement.
    samples_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        predict_callable()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        samples_ms.append(elapsed_ms)

    # Outlier rejection: trim equal fraction from each tail.
    samples_ms.sort()
    k = int(n_iters * outlier_trim_frac)
    n_outliers_dropped = 2 * k
    trimmed = samples_ms[k:n_iters - k] if k > 0 else samples_ms

    # Defensive: if trimming removed everything (shouldn't happen with valid
    # args), fall back to the un-trimmed samples to keep invariants sane.
    if not trimmed:
        trimmed = samples_ms
        n_outliers_dropped = 0

    median_ms = _median(trimmed)
    p99_ms = _percentile(trimmed, 0.99)
    quality_score = _quality_score(trimmed, median_ms)

    return MeasurementQuality(
        warmup_iters=warmup_iters,
        measurement_iters=n_iters,
        median_ms=float(median_ms),
        p99_ms=float(p99_ms),
        n_outliers_dropped=n_outliers_dropped,
        quality_score=float(quality_score),
    )


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_values[n // 2]
    return 0.5 * (sorted_values[n // 2 - 1] + sorted_values[n // 2])


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interp percentile q in [0, 1] over a SORTED list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _quality_score(samples_ms: list[float], median_ms: float) -> float:
    """Map coefficient-of-variation (std / median) to a [0, 1] quality.

    CV <= CLEAN (0.10) → 1.0 (high confidence)
    CV >= NOISY (0.30) → 0.0 (refuse to write downstream)
    Linear ramp between.

    Sub-microsecond medians are dominated by `perf_counter` jitter and
    don't reflect real workload variance — return 1.0 in that case. The
    floor is 0.001 ms (1 µs) which is well above clock resolution but
    below any meaningful inference cost on supported hardware.
    """
    n = len(samples_ms)
    if n < 2 or median_ms <= 0:
        return 0.0
    if median_ms < 0.001:
        # Below clock-jitter floor — nothing to measure; treat as clean.
        return 1.0
    mean = sum(samples_ms) / n
    var = sum((x - mean) ** 2 for x in samples_ms) / (n - 1)
    std = var ** 0.5
    cv = std / median_ms
    if cv <= _QUALITY_CV_CLEAN:
        return 1.0
    if cv >= _QUALITY_CV_NOISY:
        return 0.0
    # Linear ramp between CLEAN and NOISY
    return 1.0 - (cv - _QUALITY_CV_CLEAN) / (_QUALITY_CV_NOISY - _QUALITY_CV_CLEAN)


# ---------------------------------------------------------------------------
# Greedy resolver (Day 3) — strict partial order picker
# ---------------------------------------------------------------------------


# Bounded enum of variants in priority order (most preferred first). Each
# requires (a) the corresponding .onnx file to be present in the export
# AND (b) hardware capability (fp8 needs Hopper+; int8 needs a calibration
# cache from training; fp16 is universal on CUDA).
_VARIANT_PRIORITY: tuple[str, ...] = ("fp8", "int8", "fp16")

# Bounded enum of ORT providers, in priority order. TRT-EP only when
# variant is fp16 + max_batch == 1 (per ADR 2026-04-14-disable-trt-when-batch-gt-1).
_PROVIDER_PRIORITY: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

# CUDA SM compute capability gates per variant. Hopper = sm_90; sm_89 covers
# Ada Lovelace (4090, L40). fp8 needs sm_89+ for hardware FP8 support.
_FP8_MIN_SM = 89

# Candidate NFE values to consider, ordered LARGEST FIRST so the resolver
# returns the highest NFE that fits the chunk-period budget. 10 is the
# pi0.5 teacher default; 1 is the SnapFlow-distilled student.
_DEFAULT_CANDIDATE_NFES: tuple[int, ...] = (10, 8, 4, 2, 1)

# Safety margin: leave at least this fraction of the chunk_period for VLM
# prefix + RTC overhead. Cuts the budget for the expert denoise loop.
_CHUNK_PERIOD_SAFETY_MARGIN = 0.30


@dataclass(frozen=True)
class ResolverInputs:
    """Frozen inputs to GreedyResolver. Built once at calibration time;
    passes to the resolver verbatim."""

    available_variants: tuple[str, ...]
    available_providers: tuple[str, ...]
    candidate_nfes: tuple[int, ...]
    hardware: "HardwareFingerprint"
    embodiment: str
    chunk_size_default: int
    control_frequency_hz: float
    max_batch: int = 1  # per ADR, TRT requires == 1

    def __post_init__(self) -> None:
        if not self.available_variants:
            raise ValueError("available_variants must be non-empty")
        if not self.available_providers:
            raise ValueError("available_providers must be non-empty")
        if not self.candidate_nfes:
            raise ValueError("candidate_nfes must be non-empty")
        if self.chunk_size_default < 1:
            raise ValueError(
                f"chunk_size_default must be >= 1, got {self.chunk_size_default}"
            )
        if self.control_frequency_hz <= 0:
            raise ValueError(
                f"control_frequency_hz must be positive, got "
                f"{self.control_frequency_hz}"
            )

    @property
    def chunk_period_ms(self) -> float:
        """How long the robot has to consume one chunk (ms). Drives the
        NFE + chunk_size feasibility math."""
        return self.chunk_size_default / self.control_frequency_hz * 1000.0

    @property
    def expert_budget_ms(self) -> float:
        """Available budget for the expert_denoise loop after carving out
        VLM prefix + RTC overhead."""
        return self.chunk_period_ms * (1.0 - _CHUNK_PERIOD_SAFETY_MARGIN)


class GreedyResolver:
    """Resolves calibration parameters in strict partial order.

    Each resolve_* call narrows the search space for the next. Pure
    functions: same inputs always produce same outputs (no state mutation).

    Usage:

        inputs = ResolverInputs(
            available_variants=("fp16", "int8"),
            available_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            candidate_nfes=(10, 8, 4, 2, 1),
            hardware=HardwareFingerprint.current(),
            embodiment="franka",
            chunk_size_default=50,
            control_frequency_hz=20.0,
        )
        resolver = GreedyResolver(inputs)
        variant = resolver.resolve_variant()
        provider = resolver.resolve_provider(variant)
        nfe = resolver.resolve_nfe(variant, provider,
                                    expert_denoise_ms_per_step=measured_ms)
        chunk_size = resolver.resolve_chunk_size(nfe, expert_denoise_ms_per_step)
        latency_comp = resolver.resolve_latency_compensation(...)
    """

    __slots__ = ("_inputs",)

    def __init__(self, inputs: ResolverInputs):
        self._inputs = inputs

    @property
    def inputs(self) -> ResolverInputs:
        return self._inputs

    def resolve_variant(self) -> str:
        """Pick the best-supported variant. fp8 only on sm_89+; int8 + fp16
        broadly supported; fp16 is the universal fallback."""
        avail = set(self._inputs.available_variants)
        for v in _VARIANT_PRIORITY:
            if v not in avail:
                continue
            if v == "fp8" and not self._supports_fp8():
                continue
            return v
        # Should be unreachable — fp16 should always be in available_variants
        # if anything is. Fall back to the first available.
        return self._inputs.available_variants[0]

    def resolve_provider(self, variant: str) -> str:
        """TRT-EP for fp16 + batch=1 + TRT in available; CUDA-EP otherwise;
        CPU only as last resort."""
        avail = set(self._inputs.available_providers)
        for p in _PROVIDER_PRIORITY:
            if p not in avail:
                continue
            if p == "TensorrtExecutionProvider":
                # TRT EP rebuilds engine per batch shape — disabled when
                # max_batch > 1 per ADR 2026-04-14-disable-trt-when-batch-gt-1.
                if self._inputs.max_batch > 1:
                    continue
                # TRT EP works best with fp16; int8 + fp8 require pre-built
                # calibration profiles we don't ship in Phase 1.
                if variant != "fp16":
                    continue
            return p
        # Should be unreachable if CPU is in available_providers.
        return self._inputs.available_providers[0]

    def resolve_nfe(
        self,
        variant: str,
        provider: str,
        expert_denoise_ms_per_step: float,
    ) -> int:
        """Pick the largest NFE such that nfe * step_ms < expert_budget.

        Falls back to NFE=1 when no candidate fits — forces the SnapFlow-
        distilled path. This is the falsifiable claim from the ADR:
        A10G x franka x pi0.5-teacher x NFE=10 has no legal solution at
        20 Hz replan; resolver must drop NFE."""
        if expert_denoise_ms_per_step <= 0:
            # Defensive: invalid measurement -> pick lowest NFE
            logger.warning(
                "calibration.resolve_nfe: expert_denoise_ms_per_step=%s "
                "non-positive; falling back to NFE=1",
                expert_denoise_ms_per_step,
            )
            return min(self._inputs.candidate_nfes)
        budget_ms = self._inputs.expert_budget_ms
        candidates_desc = sorted(self._inputs.candidate_nfes, reverse=True)
        for nfe in candidates_desc:
            if nfe * expert_denoise_ms_per_step <= budget_ms:
                return nfe
        # No candidate fits — pick smallest. Operator should see the warning
        # in the caller and consider the SnapFlow distill path.
        smallest = min(candidates_desc)
        logger.warning(
            "calibration.resolve_nfe: no NFE fits budget — variant=%s "
            "provider=%s step_ms=%.2f budget_ms=%.2f. Falling back to NFE=%d. "
            "Consider re-exporting with SnapFlow distillation for 1-NFE inference.",
            variant, provider, expert_denoise_ms_per_step, budget_ms, smallest,
        )
        return smallest

    def resolve_chunk_size(
        self,
        nfe: int,
        expert_denoise_ms_per_step: float,
    ) -> int:
        """Use the embodiment default unless even NFE doesn't fit at that
        chunk size — then halve until it does (minimum 1)."""
        chunk_size = self._inputs.chunk_size_default
        budget_ms = self._inputs.expert_budget_ms
        if nfe * expert_denoise_ms_per_step <= budget_ms:
            return chunk_size
        # Shrink until feasible — but in practice this branch fires only
        # when resolve_nfe also fell back. Cap at 50% reduction floor 1.
        min_chunk = max(1, chunk_size // 2)
        while chunk_size > min_chunk:
            chunk_size //= 2
            new_budget = chunk_size / self._inputs.control_frequency_hz * 1000.0 * (1 - _CHUNK_PERIOD_SAFETY_MARGIN)
            if nfe * expert_denoise_ms_per_step <= new_budget:
                logger.warning(
                    "calibration.resolve_chunk_size: shrinking chunk_size "
                    "%d -> %d to fit NFE=%d at step_ms=%.2f",
                    self._inputs.chunk_size_default, chunk_size, nfe,
                    expert_denoise_ms_per_step,
                )
                return chunk_size
        # Even at min_chunk doesn't fit — return min anyway with a loud warning.
        logger.warning(
            "calibration.resolve_chunk_size: even chunk_size=%d insufficient "
            "for NFE=%d at step_ms=%.2f. Customer should re-export with "
            "SnapFlow distillation OR lower control_frequency_hz.",
            min_chunk, nfe, expert_denoise_ms_per_step,
        )
        return min_chunk

    def resolve_latency_compensation_ms(self) -> float:
        """Cold-start default by embodiment. Real value populates via the
        passive LatencyTracker warm-update (Day 5)."""
        return COLD_START_LATENCY_COMP_MS_BY_EMBODIMENT.get(
            self._inputs.embodiment, DEFAULT_COLD_START_LATENCY_COMP_MS,
        )

    # --- internals -------------------------------------------------------

    def _supports_fp8(self) -> bool:
        """fp8 requires sm_89+ (Ada Lovelace / Hopper / Blackwell). The
        hardware fingerprint doesn't carry SM compute capability today;
        approximate via gpu_name string match. Phase 2: probe via cuda
        runtime API for a precise check."""
        name = self._inputs.hardware.gpu_name.lower()
        # sm_89: Ada Lovelace (RTX 4090, L40, L4); sm_90+: Hopper (H100), Blackwell.
        if any(t in name for t in ("h100", "h200", "b100", "b200", "h800")):
            return True
        if any(t in name for t in ("4090", "4080", "4070", "l40", "l4 ")):
            return True
        # A100, A10, A40, T4, V100 etc. are all pre-Hopper; no fp8.
        return False


# ---------------------------------------------------------------------------
# Passive warm-up tracker (Day 5) — derives latency_compensation_ms from
# real /act traffic without an active boot-time probe.
# ---------------------------------------------------------------------------


class CalibrationWarmupTracker:
    """Rolling-window p95 stability detector that writes back to the cache.

    Per ADR 2026-04-25-auto-calibration-architecture:
    - No active probe at boot (avoid unintended first-move)
    - Cold-start: use embodiment default (franka 40ms, etc.)
    - Warm-up: sample real /act latencies; when p95 is stable for
      `stable_count` consecutive observations within `tolerance_ms`, write
      the new latency_compensation_ms to the cache entry + persist atomically

    Composition:
    - Created by `create_app()` lifespan when `--auto-calibrate` is set
    - `record_latency(latency_ms)` called from /act handler post-flush
    - `maybe_persist()` called periodically; returns True when a write
      happened. Caller (the /act handler) doesn't have to await — it's
      a synchronous in-memory op + atomic disk write
    """

    __slots__ = (
        "_cache", "_cache_path", "_embodiment", "_model_hash",
        "_window", "_stable_observations", "_window_size",
        "_tolerance_ms", "_stable_count_target", "_min_samples_to_persist",
        "_last_persisted_value_ms", "_lock",
    )

    def __init__(
        self,
        *,
        cache: CalibrationCache,
        cache_path: str | Path,
        embodiment: str,
        model_hash: str,
        window_size: int = 100,
        tolerance_ms: float = 5.0,
        stable_count_target: int = 3,
        min_samples_to_persist: int = 30,
    ):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if tolerance_ms < 0:
            raise ValueError(f"tolerance_ms must be >= 0, got {tolerance_ms}")
        if stable_count_target < 1:
            raise ValueError(
                f"stable_count_target must be >= 1, got {stable_count_target}"
            )
        if min_samples_to_persist < 1:
            raise ValueError(
                f"min_samples_to_persist must be >= 1, got {min_samples_to_persist}"
            )
        if not embodiment:
            raise ValueError("embodiment must be non-empty")
        if not model_hash:
            raise ValueError("model_hash must be non-empty")
        import collections
        import threading
        self._cache = cache
        self._cache_path = Path(cache_path)
        self._embodiment = embodiment
        self._model_hash = model_hash
        self._window: collections.deque[float] = collections.deque(maxlen=window_size)
        self._window_size = window_size
        self._tolerance_ms = float(tolerance_ms)
        self._stable_count_target = int(stable_count_target)
        self._min_samples_to_persist = int(min_samples_to_persist)
        self._stable_observations = 0
        self._last_persisted_value_ms: float | None = None
        self._lock = threading.Lock()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._window)

    @property
    def stable_observations(self) -> int:
        with self._lock:
            return self._stable_observations

    @property
    def last_persisted_value_ms(self) -> float | None:
        return self._last_persisted_value_ms

    def record_latency(self, latency_ms: float) -> None:
        """Record one /act-completed wall-clock latency. Drops non-positive +
        NaN values silently — hot-path reliability beats strict validation."""
        if latency_ms <= 0 or latency_ms != latency_ms:  # rejects NaN
            return
        with self._lock:
            self._window.append(float(latency_ms))

    def current_p95_ms(self) -> float | None:
        """Returns None when there aren't enough samples to form a stable
        estimate (< min_samples_to_persist)."""
        with self._lock:
            n = len(self._window)
            if n < self._min_samples_to_persist:
                return None
            samples_sorted = sorted(self._window)
        return _percentile(samples_sorted, 0.95)

    def maybe_persist(self) -> bool:
        """Check whether the rolling p95 is stable enough to write back to
        the cache + persist. Returns True when a write happened.

        Stability rule: the current p95 must be within `tolerance_ms` of
        the LAST persisted value for `stable_count_target` consecutive
        calls. On the first persist (no prior value), we only require
        `min_samples_to_persist` samples.
        """
        current = self.current_p95_ms()
        if current is None:
            return False

        with self._lock:
            last = self._last_persisted_value_ms
            if last is not None and abs(current - last) <= self._tolerance_ms:
                self._stable_observations += 1
            else:
                self._stable_observations = 1
                # Don't immediately persist on first sample — need more
                # observations to confirm stability.
                if last is not None:
                    self._last_persisted_value_ms = current
                    return False

            should_persist = (
                last is None  # first persist after warmup
                or self._stable_observations >= self._stable_count_target
            )
            if not should_persist:
                return False

            # Lookup the existing entry for (embodiment, model_hash); if
            # absent, we can't write back (resolver must run first to
            # create it). Day 5 only persists when an entry already exists.
            entry = self._cache.lookup(
                embodiment=self._embodiment, model_hash=self._model_hash,
            )
            if entry is None:
                return False

            # Construct an updated entry with the new latency_compensation_ms.
            updated = CalibrationEntry(
                chunk_size=entry.chunk_size,
                nfe=entry.nfe,
                latency_compensation_ms=current,
                provider=entry.provider,
                variant=entry.variant,
                measurement_quality=entry.measurement_quality,
                measurement_context=entry.measurement_context,
                timestamp=_utcnow_iso(),
            )
            self._cache.record(
                embodiment=self._embodiment,
                model_hash=self._model_hash,
                entry=updated,
            )
            self._last_persisted_value_ms = current
            self._stable_observations = 0  # reset; require fresh stability after a write

        # Persist outside the lock — atomic temp+rename, safe to release first.
        try:
            self._cache.save(self._cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "calibration_warmup.persist_failed path=%s: %s",
                self._cache_path, exc,
            )
            return False
        logger.info(
            "auto-calibrate: persisted latency_compensation_ms=%.1f for "
            "embodiment=%s model_hash=%s",
            current, self._embodiment, self._model_hash,
        )
        return True


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_STALE_AFTER_DAYS",
    "COLD_START_LATENCY_COMP_MS_BY_EMBODIMENT",
    "DEFAULT_COLD_START_LATENCY_COMP_MS",
    "HardwareFingerprint",
    "MeasurementQuality",
    "MeasurementContext",
    "CalibrationEntry",
    "CalibrationCache",
    "CalibrationWarmupTracker",
    "GreedyResolver",
    "ResolverInputs",
    "measure_latency_profile",
]
