"""Pro-tier license loader — HW-bound JWT with 24h heartbeat semantics.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #5:
HW-bound signed JWT at `~/.reflex/pro.license`, bound to
machine_fingerprint, with 24h server heartbeat for revocation.

Phase 1 ships the substrate:
- License file format (JSON envelope; field shape locked)
- Validation logic (file exists + parses + not expired + HW matches +
  heartbeat fresh)
- HAS a `signature` field reserved for cryptographic verification but
  Phase 1 doesn't run actual crypto (license signing infra is a
  Phase 1.5 follow-up per ADR open-items)

Phase 1.5 wires actual signature verification + remote heartbeat
endpoint. The interface stays stable across the upgrade — customers'
license files don't get re-issued.

License is REQUIRED to enable `--pro` features. Absence = exit 1 with
a clear error pointing the customer at their account dashboard.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Bumped on a breaking license-format change.
#   v1 = unsigned dev license (Phase 1, accepted with warning when bundled
#        public key not yet deployed; will be rejected once Phase 1.5 lands)
#   v2 = Ed25519-signed via the license worker (Phase 1.5, required for paid)
# load_license() accepts both versions; v1 falls through with a deprecation
# warning, v2 must verify against the bundled public key.
LICENSE_VERSION = 2
LICENSE_VERSION_LEGACY_UNSIGNED = 1

# Default location. Customer can override via `--pro-license <path>`
# (Phase 1.5 wiring) OR `REFLEX_PRO_LICENSE` env var.
DEFAULT_LICENSE_PATH = "~/.reflex/pro.license"

# Heartbeat freshness window. After 24h without successful validation
# (which Phase 1.5 ties to a remote endpoint), the license is treated
# as stale + refused. Phase 1's heartbeat is purely local — increments
# on every successful validation. Operators who don't restart in 24h
# get caught.
HEARTBEAT_FRESHNESS_S = 24 * 3600


@dataclass(frozen=True)
class HardwareFingerprintLite:
    """Subset of HardwareFingerprint that the license binds against. We
    don't bind to driver_version (driver upgrades shouldn't break the
    license) or kernel_release (kernel patches don't change identity)."""

    gpu_uuid: str
    gpu_name: str
    cpu_count: int

    def matches(self, other: "HardwareFingerprintLite") -> bool:
        return (
            self.gpu_uuid == other.gpu_uuid
            and self.gpu_name == other.gpu_name
            and self.cpu_count == other.cpu_count
        )


@dataclass(frozen=True)
class ProLicense:
    """Frozen license. Loaded once at startup, validated, never mutated.

    Phase 1 fields (LOCKED — additive-only Phase 2 evolution):
    - license_version (int)
    - customer_id (str)
    - tier (str — "pro" today; "enterprise" reserved for Phase 2)
    - issued_at (ISO 8601 UTC)
    - expires_at (ISO 8601 UTC)
    - hardware_binding (HardwareFingerprintLite — matched at load time)
    - signature (str — base64 of HMAC-SHA256 in Phase 1.5; placeholder
      string in Phase 1)
    - last_heartbeat_at (ISO 8601 — local timestamp of most-recent
      successful validation; refreshed on save)
    """

    license_version: int
    customer_id: str
    tier: str
    issued_at: str
    expires_at: str
    hardware_binding: HardwareFingerprintLite
    signature: str = ""
    last_heartbeat_at: str = ""

    SCHEMA_VERSION: ClassVar[int] = LICENSE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "license_version": self.license_version,
            "customer_id": self.customer_id,
            "tier": self.tier,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "hardware_binding": asdict(self.hardware_binding),
            "signature": self.signature,
            "last_heartbeat_at": self.last_heartbeat_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProLicense":
        return cls(
            license_version=int(d["license_version"]),
            customer_id=str(d["customer_id"]),
            tier=str(d["tier"]),
            issued_at=str(d["issued_at"]),
            expires_at=str(d["expires_at"]),
            hardware_binding=HardwareFingerprintLite(**d["hardware_binding"]),
            signature=str(d.get("signature", "")),
            last_heartbeat_at=str(d.get("last_heartbeat_at", "")),
        )

    def is_expired(self) -> bool:
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > exp
        except Exception:
            return True  # unparseable = treat as expired

    def heartbeat_age_s(self) -> float:
        if not self.last_heartbeat_at:
            return float("inf")
        try:
            ts = datetime.fromisoformat(
                self.last_heartbeat_at.replace("Z", "+00:00")
            )
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            return float("inf")

    def is_heartbeat_stale(
        self, *, max_age_s: float = HEARTBEAT_FRESHNESS_S,
    ) -> bool:
        return self.heartbeat_age_s() > max_age_s


class LicenseError(Exception):
    """Base for license-failure exceptions. Caller maps to exit 1 with a
    clear message; never silently degrade."""


class LicenseMissing(LicenseError):
    """No license file found at the expected path."""


class LicenseExpired(LicenseError):
    """License `expires_at` is in the past."""


class LicenseHardwareMismatch(LicenseError):
    """`hardware_binding` doesn't match the running host. Customer has to
    re-issue (license server endpoint, Phase 1.5)."""


class LicenseHeartbeatStale(LicenseError):
    """Last successful validation is > HEARTBEAT_FRESHNESS_S old. Phase 1
    requires a server restart to refresh; Phase 1.5 wires remote heartbeat."""


class LicenseCorrupt(LicenseError):
    """Couldn't parse the license file."""


def load_license(
    *,
    path: str | Path = DEFAULT_LICENSE_PATH,
    current_hardware: HardwareFingerprintLite,
    skip_heartbeat_check: bool = False,
) -> ProLicense:
    """Load + validate the license. Refreshes the local heartbeat on
    success.

    Raises:
        LicenseMissing: file absent
        LicenseCorrupt: unparseable / wrong schema
        LicenseExpired: past expires_at
        LicenseHardwareMismatch: HW fingerprint different
        LicenseHeartbeatStale: > 24h since last validation
            (skip_heartbeat_check=True bypasses for tests / first-run)
    """
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise LicenseMissing(
            f"Pro license not found at {path_obj}. "
            f"Get yours at https://reflex.run/pro/license (Phase 1.5: real URL). "
            f"Set REFLEX_PRO_LICENSE env or pass --pro-license <path>."
        )

    try:
        data = json.loads(path_obj.read_text())
        license = ProLicense.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise LicenseCorrupt(
            f"Pro license at {path_obj} is corrupt or schema-mismatched: {exc}"
        ) from exc

    if license.license_version not in (LICENSE_VERSION, LICENSE_VERSION_LEGACY_UNSIGNED):
        raise LicenseCorrupt(
            f"Pro license version {license.license_version} not supported "
            f"(expected {LICENSE_VERSION} signed, or {LICENSE_VERSION_LEGACY_UNSIGNED} "
            f"legacy unsigned). Re-issue with the latest license worker."
        )

    # Phase 1.5: verify Ed25519 signature for v2 licenses. Legacy v1 licenses
    # fall through with a deprecation warning so existing dev installs don't
    # break overnight; future release will reject them entirely.
    if license.license_version == LICENSE_VERSION:
        try:
            from reflex.pro.signature import verify_license_signature
            verify_license_signature(data)
        except Exception as exc:  # noqa: BLE001 — wrap in LicenseCorrupt below
            raise LicenseCorrupt(
                f"Pro license signature verification failed: {exc}. "
                f"Either the license file was tampered with, or the bundled "
                f"public key in src/reflex/pro/_public_key.py is stale "
                f"(re-fetch via `wrangler tail` on the deployed license worker)."
            ) from exc
    else:
        logger.warning(
            "Pro license is v%d (legacy unsigned). Re-issue as v%d via the "
            "deployed license worker before this format is removed.",
            LICENSE_VERSION_LEGACY_UNSIGNED, LICENSE_VERSION,
        )

    if license.is_expired():
        raise LicenseExpired(
            f"Pro license expired at {license.expires_at}. "
            f"Renew at https://reflex.run/pro/renew."
        )

    if not license.hardware_binding.matches(current_hardware):
        raise LicenseHardwareMismatch(
            f"Pro license bound to different hardware: "
            f"license={license.hardware_binding}, current={current_hardware}. "
            f"Re-issue for this host at https://reflex.run/pro/rebind."
        )

    if not skip_heartbeat_check and license.is_heartbeat_stale():
        raise LicenseHeartbeatStale(
            f"Pro license heartbeat stale ({license.heartbeat_age_s():.0f}s "
            f"old, max {HEARTBEAT_FRESHNESS_S}s). Restart the server to "
            f"refresh; persistent staleness indicates a bypass attempt."
        )

    # Phase 1: signature is a placeholder; Phase 1.5 wires actual
    # cryptographic verification. We log when the signature is empty so
    # operators see the development-mode posture.
    if not license.signature:
        logger.warning(
            "Pro license has empty signature field — running in Phase 1 "
            "development mode (no cryptographic verification). Phase 1.5 "
            "will require a signed license."
        )

    # Refresh local heartbeat so the next validation passes for another 24h
    refreshed = ProLicense(
        license_version=license.license_version,
        customer_id=license.customer_id,
        tier=license.tier,
        issued_at=license.issued_at,
        expires_at=license.expires_at,
        hardware_binding=license.hardware_binding,
        signature=license.signature,
        last_heartbeat_at=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
    )
    try:
        tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
        tmp.write_text(json.dumps(refreshed.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path_obj)
        os.chmod(path_obj, 0o600)
    except Exception as exc:  # noqa: BLE001 — heartbeat write failure shouldn't kill startup
        logger.warning(
            "Pro license heartbeat write failed at %s: %s — license still "
            "valid for this session, but next validation will use the OLD "
            "heartbeat timestamp. Investigate filesystem permissions.",
            path_obj, exc,
        )

    logger.info(
        "Pro license valid — customer_id=%s tier=%s expires_at=%s",
        license.customer_id, license.tier, license.expires_at,
    )

    # Best-effort telemetry heartbeat (opt-out via REFLEX_NO_TELEMETRY=1).
    # Phase 1: minimal payload (customer_id + version); workload_type
    # ("vla_family", "hardware_tier") defaults to "unknown" because
    # license.py doesn't know what's being served. The runtime caller
    # can re-emit a richer heartbeat post-server-startup if desired.
    # Telemetry failure never blocks startup — see pro/telemetry.py.
    try:
        from reflex import __version__
        from reflex.pro.telemetry import emit as _emit_telemetry
        _emit_telemetry(
            customer_id=refreshed.customer_id,
            reflex_version=__version__,
        )
    except Exception:  # noqa: BLE001 — telemetry must never break licensing
        pass

    return refreshed


def issue_dev_license(
    *,
    customer_id: str,
    hardware: HardwareFingerprintLite,
    tier: str = "pro",
    valid_for_days: int = 30,
    path: str | Path = DEFAULT_LICENSE_PATH,
) -> ProLicense:
    """Issue a development-mode license for local testing.

    Phase 1 has no signing infra; this writes an unsigned license that
    `load_license` accepts (with a warning). Phase 1.5 will require all
    licenses to come from the signing endpoint; `issue_dev_license` will
    move to a `--dev-license` CLI flag that emits a warning at startup.

    Writes as license_version=LICENSE_VERSION_LEGACY_UNSIGNED so the
    loader takes the legacy unsigned path. Writing it as the current
    LICENSE_VERSION (=2) trips signature verification, which fails on
    the empty signature field — the dev path stops working.
    """
    now = datetime.now(timezone.utc)
    license = ProLicense(
        license_version=LICENSE_VERSION_LEGACY_UNSIGNED,
        customer_id=customer_id,
        tier=tier,
        issued_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        expires_at=(
            now.replace(microsecond=0).isoformat()
            if valid_for_days == 0
            else (now + _days(valid_for_days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ),
        hardware_binding=hardware,
        signature="",  # unsigned dev license
        last_heartbeat_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(license.to_dict(), indent=2, sort_keys=True))
    os.chmod(path_obj, 0o600)
    return license


def _days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


__all__ = [
    "DEFAULT_LICENSE_PATH",
    "HEARTBEAT_FRESHNESS_S",
    "LICENSE_VERSION",
    "HardwareFingerprintLite",
    "LicenseCorrupt",
    "LicenseError",
    "LicenseExpired",
    "LicenseHardwareMismatch",
    "LicenseHeartbeatStale",
    "LicenseMissing",
    "ProLicense",
    "issue_dev_license",
    "load_license",
]
