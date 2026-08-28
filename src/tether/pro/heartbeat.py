"""Signed license-heartbeat attestations and verified local cache.

The server response, not the local clock or license file, is the revocation
authority. A verified active attestation is usable for at most 24 hours plus
five minutes of clock skew, and never beyond the signed license expiry.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tether.pro._public_key import TRUSTED_PUBLIC_KEYS_B64

logger = logging.getLogger(__name__)

HEARTBEAT_DOMAIN = "tether.license.heartbeat"
HEARTBEAT_VERSION = 1
HEARTBEAT_MAX_VALIDITY_S = 24 * 60 * 60
HEARTBEAT_CLOCK_SKEW_S = 5 * 60
HEARTBEAT_RETRY_S = 5 * 60
HEARTBEAT_STATUSES = frozenset({"active", "suspended", "expired", "revoked"})
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class HeartbeatError(Exception):
    """Base class for heartbeat failures."""


class HeartbeatNetworkError(HeartbeatError):
    """The authoritative service could not be reached temporarily."""


class HeartbeatProtocolError(HeartbeatError):
    """The service response was malformed, unsigned, or untrusted."""


class LicenseRevokedError(HeartbeatError):
    """The signed server attestation reports revocation."""


class LicenseExpiredAtServer(HeartbeatError):
    """The signed server attestation reports expiry."""


class LicenseSuspendedError(HeartbeatError):
    """The signed server attestation reports suspension."""


@dataclass(frozen=True)
class HeartbeatAttestation:
    domain: str
    issued_at: int
    key_id: str
    license_id: str
    request_nonce: str
    status: str
    v: int
    valid_until: int
    signature: str

    def signed_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("signature")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(obj: Any) -> bytes:
    """RFC 8785-compatible encoding for this integer/string-only schema."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _decode_base64url(value: str, *, expected_bytes: int, label: str) -> bytes:
    if not value or "=" in value or not _B64URL_RE.fullmatch(value):
        raise HeartbeatProtocolError(f"{label} is not unpadded base64url")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:  # noqa: BLE001
        raise HeartbeatProtocolError(f"{label} is not valid base64url") from exc
    if len(raw) != expected_bytes:
        raise HeartbeatProtocolError(
            f"{label} must decode to {expected_bytes} bytes, got {len(raw)}"
        )
    return raw


def _expires_epoch(expires_at: str) -> int:
    try:
        return int(datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp())
    except Exception as exc:  # noqa: BLE001
        raise HeartbeatProtocolError("license expires_at is invalid") from exc


def verify_heartbeat_attestation(
    value: dict[str, Any],
    *,
    expected_license_id: str,
    license_expires_at: str,
    expected_nonce: str | None = None,
    now: int | None = None,
) -> HeartbeatAttestation:
    """Verify the exact heartbeat envelope, signature, binding, and bounds."""
    required = {
        "domain", "issued_at", "key_id", "license_id", "request_nonce",
        "status", "v", "valid_until", "signature",
    }
    if set(value) != required:
        raise HeartbeatProtocolError(
            f"heartbeat fields mismatch: expected {sorted(required)}, got {sorted(value)}"
        )
    if isinstance(value.get("issued_at"), bool) or not isinstance(value.get("issued_at"), int):
        raise HeartbeatProtocolError("issued_at must be an integer Unix timestamp")
    if isinstance(value.get("valid_until"), bool) or not isinstance(value.get("valid_until"), int):
        raise HeartbeatProtocolError("valid_until must be an integer Unix timestamp")
    if value.get("domain") != HEARTBEAT_DOMAIN or value.get("v") != HEARTBEAT_VERSION:
        raise HeartbeatProtocolError("unsupported heartbeat domain or version")
    if value.get("license_id") != expected_license_id:
        raise HeartbeatProtocolError("heartbeat license_id does not match the loaded license")
    if value.get("status") not in HEARTBEAT_STATUSES:
        raise HeartbeatProtocolError(f"invalid heartbeat status: {value.get('status')!r}")

    nonce = str(value.get("request_nonce", ""))
    _decode_base64url(nonce, expected_bytes=16, label="request_nonce")
    if expected_nonce is not None and nonce != expected_nonce:
        raise HeartbeatProtocolError("heartbeat request_nonce does not match the request")

    key_id = str(value.get("key_id", ""))
    public_key_b64 = TRUSTED_PUBLIC_KEYS_B64.get(key_id)
    if not public_key_b64:
        raise HeartbeatProtocolError(f"unknown or retired heartbeat key_id={key_id!r}")
    signature = str(value.get("signature", ""))
    signature_raw = _decode_base64url(signature, expected_bytes=64, label="signature")

    attestation = HeartbeatAttestation(**value)
    try:
        public_raw = base64.b64decode(public_key_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(public_raw).verify(
            signature_raw, _canonical_json(attestation.signed_payload()),
        )
    except (ValueError, InvalidSignature) as exc:
        raise HeartbeatProtocolError("heartbeat signature verification failed") from exc

    issued_at = attestation.issued_at
    valid_until = attestation.valid_until
    expires_epoch = _expires_epoch(license_expires_at)
    if valid_until > issued_at + HEARTBEAT_MAX_VALIDITY_S:
        raise HeartbeatProtocolError("heartbeat validity exceeds 24 hours")
    if valid_until > expires_epoch:
        raise HeartbeatProtocolError("heartbeat validity exceeds license expiry")
    current = int(time.time()) if now is None else int(now)
    if issued_at > current + HEARTBEAT_CLOCK_SKEW_S:
        raise HeartbeatProtocolError("heartbeat issued_at exceeds five-minute future clock skew")
    if expected_nonce is not None and issued_at < current - HEARTBEAT_CLOCK_SKEW_S:
        raise HeartbeatProtocolError("fresh heartbeat issued_at exceeds five-minute clock skew")
    return attestation


def heartbeat_cache_path(license_path: str | Path) -> Path:
    path = Path(license_path).expanduser()
    return path.with_suffix(path.suffix + ".heartbeat")


def persist_verified_attestation(
    attestation: HeartbeatAttestation, *, cache_path: str | Path,
) -> None:
    path = Path(cache_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(attestation.to_dict(), indent=2, sort_keys=True))
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_cached_attestation(
    *,
    cache_path: str | Path,
    license_id: str,
    license_expires_at: str,
    now: int | None = None,
) -> HeartbeatAttestation:
    path = Path(cache_path).expanduser()
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HeartbeatProtocolError(f"no readable verified heartbeat cache at {path}") from exc
    if not isinstance(value, dict):
        raise HeartbeatProtocolError("heartbeat cache must contain a JSON object")
    return verify_heartbeat_attestation(
        value,
        expected_license_id=license_id,
        license_expires_at=license_expires_at,
        now=now,
    )


def attestation_deadline(attestation: HeartbeatAttestation, *, license_expires_at: str) -> int:
    return min(
        attestation.valid_until + HEARTBEAT_CLOCK_SKEW_S,
        _expires_epoch(license_expires_at),
    )


def enforce_attestation(
    attestation: HeartbeatAttestation,
    *,
    license_expires_at: str,
    now: int | None = None,
) -> None:
    if attestation.status == "revoked":
        raise LicenseRevokedError(f"License {attestation.license_id} is revoked")
    if attestation.status == "expired":
        raise LicenseExpiredAtServer(f"License {attestation.license_id} is expired")
    if attestation.status == "suspended":
        raise LicenseSuspendedError(f"License {attestation.license_id} is suspended")
    current = int(time.time()) if now is None else int(now)
    if current >= attestation_deadline(attestation, license_expires_at=license_expires_at):
        raise LicenseExpiredAtServer(
            f"License {attestation.license_id} heartbeat grace window elapsed"
        )


def send_heartbeat(
    *,
    license_id: str,
    hardware_fingerprint: str,
    tether_version: str,
    license_expires_at: str,
    cache_path: str | Path | None = None,
    endpoint: str | None = None,
    timeout_s: float = 5.0,
) -> HeartbeatAttestation:
    """Fetch, verify, optionally persist, and enforce a signed attestation."""
    from tether.pro.activate import DEFAULT_LICENSE_ENDPOINT

    url = (endpoint or os.environ.get("TETHER_LICENSE_ENDPOINT", DEFAULT_LICENSE_ENDPOINT)).rstrip("/")
    full_url = f"{url}/v1/heartbeat"
    nonce = base64.urlsafe_b64encode(secrets.token_bytes(16)).decode("ascii").rstrip("=")
    payload = {
        "license_id": license_id,
        "hardware_fingerprint": hardware_fingerprint,
        "tether_version": tether_version,
        "request_nonce": nonce,
    }
    try:
        import httpx
        response = httpx.post(full_url, json=payload, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        raise HeartbeatNetworkError(f"heartbeat request to {url} failed: {exc}") from exc
    if response.status_code >= 500:
        raise HeartbeatNetworkError(f"heartbeat service returned HTTP {response.status_code}")
    if response.status_code != 200:
        raise HeartbeatProtocolError(
            f"heartbeat service rejected the request with HTTP {response.status_code}"
        )
    try:
        value = response.json()
    except Exception as exc:  # noqa: BLE001
        raise HeartbeatProtocolError("heartbeat response was not JSON") from exc
    if not isinstance(value, dict):
        raise HeartbeatProtocolError("heartbeat response must be a JSON object")
    attestation = verify_heartbeat_attestation(
        value,
        expected_license_id=license_id,
        expected_nonce=nonce,
        license_expires_at=license_expires_at,
    )
    if cache_path is not None:
        persist_verified_attestation(attestation, cache_path=cache_path)
    enforce_attestation(attestation, license_expires_at=license_expires_at)
    return attestation


__all__ = [
    "HEARTBEAT_CLOCK_SKEW_S", "HEARTBEAT_MAX_VALIDITY_S", "HEARTBEAT_RETRY_S",
    "HeartbeatAttestation", "HeartbeatError", "HeartbeatNetworkError",
    "HeartbeatProtocolError", "LicenseExpiredAtServer", "LicenseRevokedError",
    "LicenseSuspendedError", "attestation_deadline", "enforce_attestation",
    "heartbeat_cache_path", "load_cached_attestation", "persist_verified_attestation",
    "send_heartbeat", "verify_heartbeat_attestation",
]
