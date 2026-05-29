"""Pick the right model variant for a given (model_id, device_class) pair.

`reflex go --model pi05 --embodiment franka` calls into here. The resolver
consults the in-package registry (`reflex.registry.REGISTRY`) and applies
preference rules:

1. If the user passed a fully-qualified registry id (e.g. `pi05-libero`), use
   that exact entry unless the device is a strict memory-constrained edge target.
2. Otherwise interpret the model arg as a family name (`pi05` / `smolvla` /
   `pi0`) and pick the variant whose `supported_devices` includes the probed
   device_class AND whose `supported_embodiments` includes the requested
   embodiment. Prefer smaller models for edge devices, larger for datacenter.
3. If the chosen entry has `requires_export=True`, surface that in the
   ResolveResult so the caller can decide: orchestrate the export step or
   tell the user to run `reflex models export` first.

Resolver is pure-Python (no I/O); the calling layer in `reflex go` is what
actually pulls + serves.
"""
from __future__ import annotations

from dataclasses import dataclass
from reflex.registry import ModelEntry, REGISTRY, by_id


_STRICT_DEVICE_MATCH_CLASSES = {"orin_nano"}
_DEVICE_LABELS = {
    "orin_nano": "Jetson Orin Nano",
}


@dataclass(frozen=True)
class ResolveResult:
    """What the resolver picked + why."""

    entry: ModelEntry
    matched_strategy: str  # "exact-id" / "family-and-device" / "family-fallback"
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.notes is None:
            object.__setattr__(self, "notes", [])


class ModelResolverError(Exception):
    """Raised when no entry matches the request."""


def _device_label(device_class: str) -> str:
    return _DEVICE_LABELS.get(device_class, device_class)


def _alternatives_for_device(device_class: str, embodiment: str = "") -> list[str]:
    return [
        e.model_id for e in REGISTRY
        if device_class in e.supported_devices
        and (not embodiment or embodiment in e.supported_embodiments)
    ]


def _unsupported_device_error(
    model: str,
    device_class: str,
    supported_devices: tuple[str, ...],
    embodiment: str = "",
) -> ModelResolverError:
    alternatives = _alternatives_for_device(device_class, embodiment)
    suggestion = (
        f" Supported {_device_label(device_class)} choices: {alternatives}."
        if alternatives else
        f" Run `reflex models list --device {device_class}` to browse supported models."
    )
    return ModelResolverError(
        f"{model!r} is not supported on {_device_label(device_class)} "
        f"(supported devices: {list(supported_devices)})."
        f" Refusing to fall back because this edge target is memory-constrained."
        f"{suggestion}"
    )


def resolve_model(
    model: str,
    device_class: str,
    embodiment: str = "",
) -> ResolveResult:
    """Pick a registry entry for (model, device_class, embodiment).

    Args:
        model: either a fully-qualified registry id (e.g. "pi05-libero") OR a
               family name ("pi05", "smolvla", "pi0", "openvla", "groot").
        device_class: from `hardware_probe.probe_device_class()`.
        embodiment: optional ("franka" / "so100" / "ur5"). When set, only
                    entries that support the embodiment qualify.

    Returns:
        ResolveResult.

    Raises:
        ModelResolverError if no entry matches.
    """
    # Strategy 1: exact registry id
    entry = by_id(model)
    if entry is not None:
        notes: list[str] = []
        if device_class not in entry.supported_devices:
            if device_class in _STRICT_DEVICE_MATCH_CLASSES:
                raise _unsupported_device_error(
                    model,
                    device_class,
                    entry.supported_devices,
                    embodiment=embodiment,
                )
            notes.append(
                f"warning: {model!r} not listed as supported on {device_class!r} "
                f"(supported: {list(entry.supported_devices)}). Proceeding because "
                f"the user pinned this exact id."
            )
        if embodiment and embodiment not in entry.supported_embodiments:
            notes.append(
                f"warning: {model!r} not listed as supporting embodiment "
                f"{embodiment!r} (supported: {list(entry.supported_embodiments)})."
            )
        return ResolveResult(entry=entry, matched_strategy="exact-id", notes=notes)

    # Strategy 2: family + device_class + embodiment match
    family = model
    candidates = [
        e for e in REGISTRY
        if e.family == family
        and device_class in e.supported_devices
        and (not embodiment or embodiment in e.supported_embodiments)
    ]
    if candidates:
        # Edge devices: prefer smaller; datacenter: prefer larger (more capable)
        if device_class in ("orin_nano", "agx_orin", "thor", "cpu"):
            picked = min(candidates, key=lambda e: e.size_mb)
        else:
            picked = max(candidates, key=lambda e: e.size_mb)
        return ResolveResult(
            entry=picked,
            matched_strategy="family-and-device",
            notes=[
                f"family={family} on {device_class}: {len(candidates)} candidate(s); "
                f"picked {picked.model_id} ({'smallest' if device_class in ('orin_nano', 'agx_orin', 'thor', 'cpu') else 'largest'})"
            ],
        )

    # Strategy 3: family fallback (any device, with embodiment if specified)
    family_only = [
        e for e in REGISTRY
        if e.family == family
        and (not embodiment or embodiment in e.supported_embodiments)
    ]
    if family_only:
        if device_class in _STRICT_DEVICE_MATCH_CLASSES:
            supported_for_device = _alternatives_for_device(device_class, embodiment)
            suffix = (
                f" Supported {_device_label(device_class)} choices: {supported_for_device}."
                if supported_for_device else
                f" Run `reflex models list --device {device_class}` to browse supported models."
            )
            raise ModelResolverError(
                f"No {family!r} variant explicitly supports {_device_label(device_class)}. "
                f"Refusing family fallback because it can select an oversized model "
                f"for an 8GB edge target.{suffix}"
            )
        # Pick first by registry order (curated)
        picked = family_only[0]
        return ResolveResult(
            entry=picked,
            matched_strategy="family-fallback",
            notes=[
                f"warning: no {family} variant explicitly supports {device_class!r}; "
                f"falling back to {picked.model_id} (supports: "
                f"{list(picked.supported_devices)}). May not run optimally — "
                f"consider explicit --device-class or `reflex models list` to browse."
            ],
        )

    # No match
    available_families = sorted({e.family for e in REGISTRY})
    available_ids = sorted(e.model_id for e in REGISTRY)
    raise ModelResolverError(
        f"No registry entry matches model={model!r} device={device_class!r} "
        f"embodiment={embodiment!r}.\n"
        f"  Available families: {available_families}\n"
        f"  Available ids:      {available_ids}\n"
        f"  Try: reflex models list --device {device_class}"
    )
