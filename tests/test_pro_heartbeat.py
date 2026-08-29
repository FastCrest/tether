"""Contract tests for signed license-heartbeat attestations."""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tether.pro import heartbeat


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch):
    old = Ed25519PrivateKey.generate()
    current = Ed25519PrivateKey.generate()
    trusted = {
        "key_old": base64.b64encode(old.public_key().public_bytes_raw()).decode(),
        "key_current": base64.b64encode(current.public_key().public_bytes_raw()).decode(),
    }
    monkeypatch.setattr(heartbeat, "TRUSTED_PUBLIC_KEYS_B64", trusted)
    return old, current


def _expires(days: int = 2) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _signed(private, *, key_id="key_current", nonce=None, status="active", **overrides):
    now = int(time.time())
    payload = {
        "domain": "tether.license.heartbeat",
        "issued_at": now,
        "key_id": key_id,
        "license_id": "lic_test",
        "request_nonce": nonce or _b64url(b"0123456789abcdef"),
        "status": status,
        "v": 1,
        "valid_until": now + 3600 if status == "active" else now,
    }
    payload.update(overrides)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "signature": _b64url(private.sign(canonical))}


def test_current_and_overlap_keys_are_trusted(keys):
    old, current = keys
    for private, key_id in [(old, "key_old"), (current, "key_current")]:
        value = _signed(private, key_id=key_id)
        attestation = heartbeat.verify_heartbeat_attestation(
            value,
            expected_license_id="lic_test",
            expected_nonce=value["request_nonce"],
            license_expires_at=_expires(),
        )
        assert attestation.status == "active"


@pytest.mark.parametrize("mutation", ["nonce", "license", "status", "padding", "extra"])
def test_tampering_and_non_exact_envelopes_fail(keys, mutation):
    _, current = keys
    value = _signed(current)
    expected_nonce = value["request_nonce"]
    if mutation == "nonce":
        expected_nonce = _b64url(b"fedcba9876543210")
    elif mutation == "license":
        value["license_id"] = "lic_other"
    elif mutation == "status":
        value["status"] = "active-forever"
    elif mutation == "padding":
        value["signature"] += "="
    else:
        value["unexpected"] = True
    with pytest.raises(heartbeat.HeartbeatProtocolError):
        heartbeat.verify_heartbeat_attestation(
            value,
            expected_license_id="lic_test",
            expected_nonce=expected_nonce,
            license_expires_at=_expires(),
        )


def test_unknown_or_retired_key_fails(keys):
    _, current = keys
    value = _signed(current, key_id="key_retired")
    with pytest.raises(heartbeat.HeartbeatProtocolError, match="unknown or retired"):
        heartbeat.verify_heartbeat_attestation(
            value,
            expected_license_id="lic_test",
            license_expires_at=_expires(),
        )


def test_validity_is_bounded_by_24h_and_license_expiry(keys):
    _, current = keys
    now = int(time.time())
    too_long = _signed(current, issued_at=now, valid_until=now + 86401)
    with pytest.raises(heartbeat.HeartbeatProtocolError, match="exceeds 24 hours"):
        heartbeat.verify_heartbeat_attestation(
            too_long, expected_license_id="lic_test", license_expires_at=_expires(),
        )
    beyond_license = _signed(current, issued_at=now, valid_until=now + 3600)
    expires_soon = datetime.fromtimestamp(now + 60, timezone.utc).isoformat()
    with pytest.raises(heartbeat.HeartbeatProtocolError, match="license expiry"):
        heartbeat.verify_heartbeat_attestation(
            beyond_license, expected_license_id="lic_test", license_expires_at=expires_soon,
        )


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("revoked", heartbeat.LicenseRevokedError),
        ("expired", heartbeat.LicenseExpiredAtServer),
        ("suspended", heartbeat.LicenseSuspendedError),
    ],
)
def test_signed_non_active_statuses_fail_closed(keys, status, error):
    _, current = keys
    value = _signed(current, status=status)
    attestation = heartbeat.verify_heartbeat_attestation(
        value, expected_license_id="lic_test", license_expires_at=_expires(),
    )
    with pytest.raises(error):
        heartbeat.enforce_attestation(attestation, license_expires_at=_expires())


def test_send_echoes_nonce_verifies_and_persists_cache(tmp_path, monkeypatch, keys):
    _, current = keys
    observed = {}

    class Response:
        status_code = 200

        def __init__(self, value):
            self._value = value

        def json(self):
            return self._value

    def post(url, *, json, timeout):
        observed.update({"url": url, "json": json, "timeout": timeout})
        return Response(_signed(current, nonce=json["request_nonce"]))

    import httpx
    monkeypatch.setattr(httpx, "post", post)
    cache = tmp_path / "heartbeat.json"
    attestation = heartbeat.send_heartbeat(
        license_id="lic_test",
        hardware_fingerprint="fp",
        tether_version="1.2.3",
        license_expires_at=_expires(),
        cache_path=cache,
        endpoint="https://licenses.test",
    )
    assert attestation.request_nonce == observed["json"]["request_nonce"]
    assert len(base64.urlsafe_b64decode(attestation.request_nonce + "==")) == 16
    assert json.loads(cache.read_text())["signature"] == attestation.signature
    assert cache.stat().st_mode & 0o777 == 0o600


def test_grace_ends_at_earlier_of_validity_plus_skew_or_license_expiry(keys):
    _, current = keys
    now = int(time.time())
    value = _signed(current, issued_at=now - 4000, valid_until=now - 300)
    attestation = heartbeat.verify_heartbeat_attestation(
        value, expected_license_id="lic_test", license_expires_at=_expires(), now=now,
    )
    with pytest.raises(heartbeat.LicenseExpiredAtServer):
        heartbeat.enforce_attestation(
            attestation, license_expires_at=_expires(), now=now,
        )


def test_cached_attestation_rejects_more_than_five_minutes_future(keys):
    _, current = keys
    now = int(time.time())
    value = _signed(current, issued_at=now + 301, valid_until=now + 3600)
    with pytest.raises(heartbeat.HeartbeatProtocolError, match="future clock skew"):
        heartbeat.verify_heartbeat_attestation(
            value, expected_license_id="lic_test", license_expires_at=_expires(), now=now,
        )


def test_cached_attestation_accepts_exact_future_boundary_and_legitimately_old(keys):
    _, current = keys
    now = int(time.time())
    at_boundary = _signed(current, issued_at=now + 300, valid_until=now + 3600)
    assert heartbeat.verify_heartbeat_attestation(
        at_boundary, expected_license_id="lic_test", license_expires_at=_expires(), now=now,
    ).issued_at == now + 300

    old_but_valid = _signed(current, issued_at=now - 3600, valid_until=now + 60)
    assert heartbeat.verify_heartbeat_attestation(
        old_but_valid, expected_license_id="lic_test", license_expires_at=_expires(), now=now,
    ).valid_until == now + 60
