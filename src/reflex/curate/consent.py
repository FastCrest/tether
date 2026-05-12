"""Curate consent — opt-in receipt for the data contribution program.

Per the architecture ADR (decision #3):
- Free tier: opt-in. Receipt at ~/.reflex/consent.json.
- Pro tier: opt-in by default with explicit disclosure during onboarding;
  same receipt format with tier="pro".
- Enterprise tier: negotiated; receipt with tier="enterprise".

The receipt is the single source of truth for "is this customer opted in?"
Both the runtime data-collection path and the CLI status command read it.
Revocation = delete the receipt + cascade purge through R2 + derived
datasets (cascade lives in `_compliance/consent-revoke.md`).

Phase 1 leaves consent_signature + server_acknowledgment as None — the
contribution-worker hasn't been deployed yet. The receipt format reserves
those fields so Phase 1.5 can add the dual-signature flow without changing
the receipt shape.

This module is intentionally separate from `reflex.pro.consent.ProConsent`
(which is for the legacy Pro data-collection-to-customer-disk flow). They
solve different problems:
  - ProConsent: customer's own disk, Pro license required, JWT-bound
  - CurateConsent: shared corpus on R2, Free + Pro tiers, anonymous contributor_id

A customer can have both receipts simultaneously — they're orthogonal.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

CONSENT_VERSION = 1
DEFAULT_CONSENT_PATH = "~/.reflex/consent.json"
TERMS_VERSION = "2026-05-05"

VALID_TIERS = ("free", "pro", "enterprise")
VALID_PRIVACY_MODES = ("hash_only", "raw_opt_in")

DEFAULT_SCOPE = {
    "data_types": ["recorded_episodes", "metadata"],
    "excludes": ["raw_images_unless_explicit_opt_in"],
}


@dataclass(frozen=True)
class ConsentScope:
    """Customer's choice of what is and isn't allowed in their contribution."""

    data_types: list[str]
    excludes: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"data_types": list(self.data_types), "excludes": list(self.excludes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConsentScope":
        return cls(
            data_types=list(d.get("data_types", [])),
            excludes=list(d.get("excludes", [])),
        )

    @classmethod
    def default(cls) -> "ConsentScope":
        return cls(
            data_types=list(DEFAULT_SCOPE["data_types"]),
            excludes=list(DEFAULT_SCOPE["excludes"]),
        )


@dataclass(frozen=True)
class CurateConsent:
    """Persisted consent receipt at `~/.reflex/consent.json`.

    Frozen — saved once on opt-in, never mutated. Opt-out / revoke = delete
    the file. Re-opt-in = create a new receipt with a fresh opted_in_at.
    """

    version: int
    contributor_id: str
    tier: str
    opted_in_at: str
    accepted_terms_version: str
    scope: ConsentScope
    privacy_mode: str
    geographic_residency: str
    revoke_url: str | None = None
    consent_signature: str | None = None
    server_acknowledgment: str | None = None

    SCHEMA_VERSION: ClassVar[int] = CONSENT_VERSION

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"tier must be one of {VALID_TIERS}, got {self.tier!r}"
            )
        if self.privacy_mode not in VALID_PRIVACY_MODES:
            raise ValueError(
                f"privacy_mode must be one of {VALID_PRIVACY_MODES}, "
                f"got {self.privacy_mode!r}"
            )
        if not self.contributor_id:
            raise ValueError("contributor_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "contributor_id": self.contributor_id,
            "tier": self.tier,
            "opted_in_at": self.opted_in_at,
            "accepted_terms_version": self.accepted_terms_version,
            "scope": self.scope.to_dict(),
            "privacy_mode": self.privacy_mode,
            "geographic_residency": self.geographic_residency,
            "revoke_url": self.revoke_url,
            "consent_signature": self.consent_signature,
            "server_acknowledgment": self.server_acknowledgment,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CurateConsent":
        return cls(
            version=int(d["version"]),
            contributor_id=str(d["contributor_id"]),
            tier=str(d["tier"]),
            opted_in_at=str(d["opted_in_at"]),
            accepted_terms_version=str(d.get("accepted_terms_version", TERMS_VERSION)),
            scope=ConsentScope.from_dict(d.get("scope", {})),
            privacy_mode=str(d.get("privacy_mode", "hash_only")),
            geographic_residency=str(d.get("geographic_residency", "auto-detect")),
            revoke_url=d.get("revoke_url"),
            consent_signature=d.get("consent_signature"),
            server_acknowledgment=d.get("server_acknowledgment"),
        )


class ConsentNotFound(Exception):
    """Raised when callers expect a consent receipt but none exists."""


class ConsentCorrupted(Exception):
    """Raised when the receipt file exists but is unparseable / version-drifted."""


def derive_contributor_id() -> str:
    """Anonymous, stable-across-reinstalls contributor identifier.

    Format: `free_<16-hex-of-hardware-hash>_<8-hex-of-uuid4>`

    The hardware piece is derived from `uuid.getnode()` (MAC-derived 48-bit
    node id) + a project salt. Stable across reinstalls on the same machine.
    The UUID4 suffix gives uniqueness when multiple contributors share a
    NAT'd MAC (common in container/VM environments).

    Privacy posture: the contributor_id is anonymous by design — we never
    record the original MAC, just its hash. The contributor_id reveals
    nothing about the underlying machine. Reverse-mapping requires both
    the MAC and the project salt and a fresh hash computation; even then,
    only the contributor's machine can prove ownership (by re-deriving).
    """
    node_bytes = uuid.getnode().to_bytes(8, "big")
    salt = b"reflex-curate:contributor:v1"
    h = hashlib.sha256(salt + node_bytes).hexdigest()[:16]
    suffix = uuid.uuid4().hex[:8]
    return f"free_{h}_{suffix}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: str | Path = DEFAULT_CONSENT_PATH) -> CurateConsent:
    """Load the receipt from disk. Raises ConsentNotFound when absent,
    ConsentCorrupted when present but unparseable / version-drifted."""
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise ConsentNotFound(
            f"no curate consent receipt at {path_obj}. "
            f"Run `reflex contribute --opt-in` to create one."
        )
    try:
        data = json.loads(path_obj.read_text())
        receipt = CurateConsent.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ConsentCorrupted(
            f"corrupted curate consent receipt at {path_obj}: {exc}. "
            f"Run `reflex contribute --opt-out` to clear it, then "
            f"`reflex contribute --opt-in` to re-create."
        ) from exc
    if receipt.version != CONSENT_VERSION:
        raise ConsentCorrupted(
            f"consent_version drift: receipt={receipt.version}, "
            f"current={CONSENT_VERSION}. Run `reflex contribute --opt-out` then "
            f"`reflex contribute --opt-in` to re-create."
        )
    return receipt


def is_opted_in(path: str | Path = DEFAULT_CONSENT_PATH) -> bool:
    """Cheap predicate: does a valid receipt exist on disk?"""
    try:
        load(path)
        return True
    except (ConsentNotFound, ConsentCorrupted):
        return False


def save(
    *,
    tier: str,
    contributor_id: str | None = None,
    scope: ConsentScope | None = None,
    privacy_mode: str = "hash_only",
    geographic_residency: str = "auto-detect",
    revoke_url: str | None = None,
    consent_signature: str | None = None,
    server_acknowledgment: str | None = None,
    path: str | Path = DEFAULT_CONSENT_PATH,
) -> CurateConsent:
    """Build + persist a fresh receipt. Atomic write, mode 0600.

    `contributor_id` is generated via `derive_contributor_id()` when None.
    Free tier callers should pass None; Pro tier callers should pass the
    license `customer_id` (Pro contributors are not anonymous because their
    revenue share requires they be identifiable to the billing pipeline).
    """
    receipt = CurateConsent(
        version=CONSENT_VERSION,
        contributor_id=contributor_id or derive_contributor_id(),
        tier=tier,
        opted_in_at=_utc_now_iso(),
        accepted_terms_version=TERMS_VERSION,
        scope=scope or ConsentScope.default(),
        privacy_mode=privacy_mode,
        geographic_residency=geographic_residency,
        revoke_url=revoke_url,
        consent_signature=consent_signature,
        server_acknowledgment=server_acknowledgment,
    )
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    tmp.replace(path_obj)
    os.chmod(path_obj, 0o600)
    logger.info(
        "curate consent saved — tier=%s contributor_id=%s path=%s",
        tier, receipt.contributor_id, path_obj,
    )
    return receipt


def revoke(path: str | Path = DEFAULT_CONSENT_PATH) -> bool:
    """Delete the local receipt. Idempotent — returns True iff a file was
    removed. Server-side cascade purge is a separate concern handled by
    `reflex contribute --revoke` (which calls this AND POSTs to the worker).
    """
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        return False
    path_obj.unlink()
    logger.warning("curate consent revoked locally — receipt removed at %s", path_obj)
    return True


__all__ = [
    "CONSENT_VERSION",
    "DEFAULT_CONSENT_PATH",
    "TERMS_VERSION",
    "VALID_TIERS",
    "VALID_PRIVACY_MODES",
    "ConsentCorrupted",
    "ConsentNotFound",
    "ConsentScope",
    "CurateConsent",
    "derive_contributor_id",
    "is_opted_in",
    "load",
    "revoke",
    "save",
]
