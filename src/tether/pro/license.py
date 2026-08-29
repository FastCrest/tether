"""Signed v2 Pro license loader with server-confirmed heartbeat validity."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Released runtimes accept exactly signed v2. The legacy constant remains only
# so migration tooling can identify old files; it is never an accepted version.
LICENSE_VERSION = 2
LICENSE_VERSION_LEGACY_UNSIGNED = 1

# Default location. Customer can override via `--pro-license <path>`
# (Phase 1.5 wiring) OR `TETHER_PRO_LICENSE` env var.
DEFAULT_LICENSE_PATH = "~/.tether/pro.license"

# Compatibility name for callers displaying the maximum server-confirmed
# validity. The signed attestation is authoritative; local timestamps are not.
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
    """Frozen signed license plus non-serialized runtime lease metadata."""

    license_version: int
    license_id: str
    customer_id: str
    tier: str
    issued_at: str
    expires_at: str
    max_seats: int
    hardware_binding: HardwareFingerprintLite
    signature: str
    key_id: str
    # Retained only for reading older v2 envelopes and CLI display. It has no
    # authority and is never refreshed locally.
    last_heartbeat_at: str = ""
    attestation_valid_until: int = field(default=0, compare=False, repr=False)
    heartbeat_cache_file: str = field(default="", compare=False, repr=False)

    SCHEMA_VERSION: ClassVar[int] = LICENSE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "license_version": self.license_version,
            "license_id": self.license_id,
            "customer_id": self.customer_id,
            "tier": self.tier,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "max_seats": self.max_seats,
            "hardware_binding": asdict(self.hardware_binding),
            "signature": self.signature,
            "key_id": self.key_id,
            "last_heartbeat_at": self.last_heartbeat_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProLicense":
        return cls(
            license_version=int(d["license_version"]),
            license_id=str(d["license_id"]),
            customer_id=str(d["customer_id"]),
            tier=str(d["tier"]),
            issued_at=str(d["issued_at"]),
            expires_at=str(d["expires_at"]),
            max_seats=int(d["max_seats"]),
            hardware_binding=HardwareFingerprintLite(**d["hardware_binding"]),
            signature=str(d["signature"]),
            key_id=str(d["key_id"]),
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
    """No usable server-signed heartbeat attestation remains."""


class LicenseCorrupt(LicenseError):
    """Couldn't parse the license file."""


def load_license(
    *,
    path: str | Path = DEFAULT_LICENSE_PATH,
    current_hardware: HardwareFingerprintLite,
    heartbeat_endpoint: str | None = None,
    heartbeat_timeout_s: float = 5.0,
) -> ProLicense:
    """Verify a signed v2 license and obtain server-confirmed validity.

    Every load attempts a fresh signed heartbeat. Only a temporary network
    failure may fall back to a previously persisted, still-valid signed
    attestation. The license file itself is never rewritten by this function.

    Raises:
        LicenseMissing: file absent
        LicenseCorrupt: unparseable / wrong schema
        LicenseExpired: past expires_at
        LicenseHardwareMismatch: HW fingerprint different
        LicenseHeartbeatStale: no fresh server response or usable signed cache
    """
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise LicenseMissing(
            f"Pro license not found at {path_obj}. "
            f"Get yours at https://tether.fastcrest.com/pro/license (Phase 1.5: real URL). "
            f"Set TETHER_PRO_LICENSE env or pass --pro-license <path>."
        )

    try:
        data = json.loads(path_obj.read_text())
        license = ProLicense.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise LicenseCorrupt(
            f"Pro license at {path_obj} is corrupt or schema-mismatched: {exc}"
        ) from exc

    if license.license_version != LICENSE_VERSION:
        raise LicenseCorrupt(
            f"Pro license version {license.license_version} is not accepted; "
            f"released runtimes require signed version {LICENSE_VERSION}. "
            "Re-issue with the current license worker."
        )

    try:
        from tether.pro.signature import verify_license_signature
        verify_license_signature(data)
    except Exception as exc:  # noqa: BLE001
        raise LicenseCorrupt(f"Pro license signature verification failed: {exc}") from exc

    if license.is_expired():
        raise LicenseExpired(
            f"Pro license expired at {license.expires_at}. "
            f"Renew at https://tether.fastcrest.com/pro/renew."
        )

    if not license.hardware_binding.matches(current_hardware):
        raise LicenseHardwareMismatch(
            f"Pro license bound to different hardware: "
            f"license={license.hardware_binding}, current={current_hardware}. "
            f"Re-issue for this host at https://tether.fastcrest.com/pro/rebind."
        )

    from tether import __version__
    from tether.pro.heartbeat import (
        HeartbeatNetworkError,
        HeartbeatProtocolError,
        enforce_attestation,
        heartbeat_cache_path,
        load_cached_attestation,
        send_heartbeat,
    )

    cache_path = heartbeat_cache_path(path_obj)
    fingerprint_text = (
        f"{current_hardware.gpu_uuid}|{current_hardware.gpu_name}|"
        f"{current_hardware.cpu_count}"
    )
    fingerprint = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:32]
    try:
        attestation = send_heartbeat(
            license_id=license.license_id,
            hardware_fingerprint=fingerprint,
            tether_version=__version__,
            license_expires_at=license.expires_at,
            cache_path=cache_path,
            endpoint=heartbeat_endpoint,
            timeout_s=heartbeat_timeout_s,
        )
    except HeartbeatNetworkError as exc:
        logger.warning(
            "License heartbeat unavailable; checking verified cache: %s", exc,
        )
        try:
            attestation = load_cached_attestation(
                cache_path=cache_path,
                license_id=license.license_id,
                license_expires_at=license.expires_at,
            )
            enforce_attestation(attestation, license_expires_at=license.expires_at)
        except Exception as cache_exc:  # noqa: BLE001
            raise LicenseHeartbeatStale(
                "No fresh server heartbeat and no usable signed heartbeat cache"
            ) from cache_exc
    except HeartbeatProtocolError as exc:
        raise LicenseHeartbeatStale(
            f"License heartbeat response was not trustworthy: {exc}"
        ) from exc

    validated = replace(
        license,
        attestation_valid_until=attestation.valid_until,
        heartbeat_cache_file=str(cache_path),
        last_heartbeat_at=datetime.fromtimestamp(
            attestation.issued_at, tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    logger.info(
        "Pro license valid — customer_id=%s tier=%s expires_at=%s",
        validated.customer_id, validated.tier, validated.expires_at,
    )

    # Best-effort telemetry heartbeat (opt-out via TETHER_NO_TELEMETRY=1).
    # Phase 1: minimal payload (customer_id + version); workload_type
    # ("vla_family", "hardware_tier") defaults to "unknown" because
    # license.py doesn't know what's being served. The runtime caller
    # can re-emit a richer heartbeat post-server-startup if desired.
    # Telemetry failure never blocks startup — see pro/telemetry.py.
    try:
        from tether.pro.telemetry import emit as _emit_telemetry
        _emit_telemetry(
            customer_id=validated.customer_id,
            tether_version=__version__,
        )
    except Exception:  # noqa: BLE001 — telemetry must never break licensing
        pass

    return validated


def issue_dev_license(
    *,
    customer_id: str,
    hardware: HardwareFingerprintLite,
    tier: str = "pro",
    valid_for_days: int = 30,
    path: str | Path = DEFAULT_LICENSE_PATH,
) -> ProLicense:
    """Write a legacy marker for tooling tests; released loaders reject it."""
    now = datetime.now(timezone.utc)
    license = ProLicense(
        license_version=LICENSE_VERSION_LEGACY_UNSIGNED,
        license_id=f"lic_dev_{int(now.timestamp())}",
        customer_id=customer_id,
        tier=tier,
        issued_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        expires_at=(
            now.replace(microsecond=0).isoformat()
            if valid_for_days == 0
            else (now + _days(valid_for_days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ),
        max_seats=1,
        hardware_binding=hardware,
        signature="",  # unsigned dev license
        key_id="",
        last_heartbeat_at="",
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
