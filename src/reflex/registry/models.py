"""Registry data model + filter helpers.

`ModelEntry` is the per-row schema. `REGISTRY` is the in-package list of curated
entries (loaded lazily from `data.py` to keep imports cheap). Filter helpers
implement the predicates `reflex models list --family X --device Y` consume.

Each entry carries:
- `model_id`: short-form, kebab-case (e.g. "pi05-base"). Stable across versions.
- `hf_repo`: full HuggingFace repo id (e.g. "lerobot/pi05_base").
- `hf_revision`: commit sha or tag for reproducibility. None = HEAD (discouraged).
- `family`: pi0 / pi05 / smolvla / openvla / groot — for `--family` filter.
- `action_dim`: action vector size (32 for raw pi0 / pi0.5; 7 for Franka-finetuned; etc.).
- `size_mb`: approximate on-disk weight size.
- `supported_embodiments`: list of strings matching `configs/embodiments/*.json`.
- `supported_devices`: list of strings ("orin_nano", "agx_orin", "thor", "a10g", "a100", "h100").
- `benchmarks`: dict of device -> ModelBenchmark. May be empty for "supported but unmeasured".
- `requires_export`: True if customer must run `reflex export` after pull (raw weights);
                     False if the repo already contains a Reflex-ready export (vlm_prefix.onnx etc.).
- `description`: one-sentence pitch.
- `license`: SPDX identifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelBenchmark:
    """Per-device latency + memory measurement for a single model."""

    device: str
    p50_ms: float
    p99_ms: float
    vram_mb: int
    measured_at: str = ""  # YYYY-MM-DD when this number landed


@dataclass(frozen=True)
class ModelEntry:
    """One curated registry row. Frozen so callers can stash them in sets/dicts."""

    model_id: str
    hf_repo: str
    family: str
    action_dim: int
    size_mb: int
    supported_embodiments: tuple[str, ...] = ()
    supported_devices: tuple[str, ...] = ()
    benchmarks: tuple[ModelBenchmark, ...] = ()
    requires_export: bool = True
    description: str = ""
    license: str = "unknown"
    hf_revision: str | None = None
    # Per lift #1 decision S-4 (Day 8): non-spine VLAs declare a special
    # vla_type marker. Spine VLAs (pi0/pi05/smolvla/groot) leave this None
    # (the BaseVLA registry name is the source of truth). OpenVLA sets
    # "_openvla_shim" to mark it as the optimum-cli shim path.
    vla_type: str | None = None

    def __post_init__(self):
        if not self.model_id:
            raise ValueError("model_id required")
        if "/" in self.model_id:
            raise ValueError(f"model_id must be kebab-case (no slashes): {self.model_id!r}")
        if "/" not in self.hf_repo:
            raise ValueError(f"hf_repo must be 'org/name' format: {self.hf_repo!r}")
        if self.family not in ("pi0", "pi05", "smolvla", "openvla", "groot"):
            raise ValueError(f"family must be one of pi0/pi05/smolvla/openvla/groot, got {self.family!r}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.vla_type is not None and not self.vla_type.startswith("_"):
            # Convention: marker types are prefixed with `_` so they don't
            # collide with real spine VLA class names (Pi0VLA, Pi05VLA, etc).
            raise ValueError(
                f"vla_type markers must start with `_` (got {self.vla_type!r}). "
                "Use e.g. `_openvla_shim` for non-spine VLAs."
            )

    def benchmark_for(self, device: str) -> ModelBenchmark | None:
        for b in self.benchmarks:
            if b.device == device:
                return b
        return None

    @property
    def resolved_vla_type(self) -> str:
        """Resolve this entry's vla_type for display + dispatch.

        - If ``vla_type`` is set (e.g., ``_openvla_shim``), return it verbatim.
        - Otherwise, derive from ``family``: pi0 → Pi0VLA, pi05 → Pi05VLA,
          smolvla → SmolVLA, groot → GR00TVLA. The returned name matches
          the spine ``VLAS`` registry key for that family.

        Added lift #1 Day 10 for ``reflex models list`` + ``reflex models info``
        + ``reflex export <model_id>`` dispatch.
        """
        if self.vla_type is not None:
            return self.vla_type
        return {
            "pi0": "Pi0VLA",
            "pi05": "Pi05VLA",
            "smolvla": "SmolVLA",
            "groot": "GR00TVLA",
        }.get(self.family, self.family)


# Lazy import — keeps `import reflex.registry` cheap if data.py grows large
def _load_registry() -> list[ModelEntry]:
    from reflex.registry.data import REGISTRY as _R
    return list(_R)


REGISTRY: list[ModelEntry] = _load_registry()


def by_id(model_id: str) -> ModelEntry | None:
    """O(N) lookup. N is small; not worth indexing."""
    for entry in REGISTRY:
        if entry.model_id == model_id:
            return entry
    return None


def filter_models(
    family: str | None = None,
    device: str | None = None,
    embodiment: str | None = None,
) -> list[ModelEntry]:
    """Apply --family / --device / --embodiment filters. AND-composed."""
    out: list[ModelEntry] = []
    for entry in REGISTRY:
        if family and entry.family != family:
            continue
        if device and device not in entry.supported_devices:
            continue
        if embodiment and embodiment not in entry.supported_embodiments:
            continue
        out.append(entry)
    return out


def list_families() -> list[str]:
    """Distinct families across the registry, in registration order."""
    seen: list[str] = []
    for entry in REGISTRY:
        if entry.family not in seen:
            seen.append(entry.family)
    return seen


def list_devices() -> list[str]:
    """Union of supported_devices across the registry, sorted."""
    seen: set[str] = set()
    for entry in REGISTRY:
        seen.update(entry.supported_devices)
    return sorted(seen)
