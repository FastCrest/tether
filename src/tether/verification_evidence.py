"""Typed availability for parity-verification evidence.

Verification must distinguish a real measured zero from a measurement that was
never taken.  ``EvidenceValue`` is the only wire representation used for that
distinction by ``tether verify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Literal, Mapping, TypeVar, cast


class UnavailableReason(str, Enum):
    """Closed set from Verification Evidence v1."""

    MISSING_FIELD = "missing_field"
    CAPTURE_DISABLED = "capture_disabled"
    BACKEND_UNSUPPORTED = "backend_unsupported"
    MEASUREMENT_FAILED = "measurement_failed"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"


T = TypeVar("T")


@dataclass(frozen=True)
class EvidenceValue(Generic[T]):
    """A measured value or a bounded explanation for its absence."""

    status: Literal["available", "unavailable"]
    value: T | None = None
    reason: UnavailableReason | None = None

    def __post_init__(self) -> None:
        if self.status == "available":
            if self.reason is not None:
                raise ValueError("available evidence cannot have a reason")
            if self.value is None:
                raise ValueError("available evidence requires a value")
            if isinstance(self.value, (str, list, tuple, dict, set)) and not self.value:
                raise ValueError("available evidence cannot be an empty sentinel")
        elif self.status == "unavailable":
            if self.reason is None:
                raise ValueError("unavailable evidence requires a reason")
            if self.value is not None:
                raise ValueError("unavailable evidence cannot have a value")
        else:  # pragma: no cover - Literal protects typed callers
            raise ValueError(f"invalid evidence status: {self.status!r}")

    @classmethod
    def available(cls, value: T) -> "EvidenceValue[T]":
        return cls(status="available", value=value)

    @classmethod
    def unavailable(
        cls,
        reason: UnavailableReason | str,
    ) -> "EvidenceValue[T]":
        return cls(status="unavailable", reason=UnavailableReason(reason))

    @property
    def is_available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, Any]:
        if self.is_available:
            return {"status": "available", "value": self.value}
        assert self.reason is not None
        return {"status": "unavailable", "reason": self.reason.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceValue[Any]":
        status = payload.get("status")
        if status == "available":
            if "value" not in payload or payload["value"] is None:
                raise ValueError("available evidence payload requires a value")
            return cls.available(payload["value"])
        if status == "unavailable":
            return cls.unavailable(str(payload.get("reason")))
        raise ValueError(f"invalid evidence payload status: {status!r}")


def serialize_evidence(value: Any) -> dict[str, Any]:
    """Serialize either a typed value or a concrete legacy measurement."""

    if isinstance(value, EvidenceValue):
        return value.to_dict()
    return EvidenceValue.available(value).to_dict()


def normalize_verification_device(device: str) -> Literal["cpu", "cuda"]:
    """Return the one public verification device spelling or reject it."""

    normalized = device.strip().lower()
    if normalized not in ("cpu", "cuda"):
        raise ValueError(f"verification device must be exactly 'cpu' or 'cuda', got {device!r}")
    return cast(Literal["cpu", "cuda"], normalized)


def first_unavailable(
    evidence: Mapping[str, EvidenceValue[Any]],
    required: tuple[str, ...],
) -> tuple[str, EvidenceValue[Any]] | None:
    """Return the first unavailable required channel in deterministic order."""

    for name in required:
        item = evidence.get(name)
        if item is None:
            return name, EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
        if not item.is_available:
            return name, item
    return None


__all__ = [
    "EvidenceValue",
    "UnavailableReason",
    "first_unavailable",
    "normalize_verification_device",
    "serialize_evidence",
]
