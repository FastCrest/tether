"""Tests for src/tether/pro/signature.py.

Generates a fresh Ed25519 keypair per test, monkey-patches the bundled
public key, then exercises the verify path end-to-end including tamper
detection and key-id mismatch detection.
"""
from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tether.pro import signature


def _gen_keypair() -> tuple[Ed25519PrivateKey, str]:
    priv = Ed25519PrivateKey.generate()
    pub_raw = priv.public_key().public_bytes_raw()
    return priv, base64.b64encode(pub_raw).decode("ascii")


def _sign(priv: Ed25519PrivateKey, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(priv.sign(canonical)).decode("ascii")


def _make_license(priv: Ed25519PrivateKey, key_id: str = "key_test", **overrides) -> dict:
    payload = {
        "license_version": 2,
        "license_id": "lic_test",
        "customer_id": "alice@bigco.com",
        "tier": "pro",
        "issued_at": "2026-05-01T12:00:00.000Z",
        "expires_at": "2026-06-01T12:00:00.000Z",
        "max_seats": 1,
        "hardware_binding": None,
    }
    payload.update(overrides)
    sig = _sign(priv, payload)
    return {**payload, "signature": sig, "key_id": key_id}


def test_verify_passes_for_valid_license(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", pub_b64)
    monkeypatch.setattr(signature, "BUNDLED_KEY_ID", "")  # accept any key_id
    license = _make_license(priv)
    signature.verify_license_signature(license)  # must not raise


def test_verify_fails_when_bundled_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", "")
    priv, _ = _gen_keypair()
    license = _make_license(priv)
    with pytest.raises(signature.LicenseSignatureError, match="not been deployed"):
        signature.verify_license_signature(license)


def test_verify_fails_for_tampered_license(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", pub_b64)
    monkeypatch.setattr(signature, "BUNDLED_KEY_ID", "")
    license = _make_license(priv)
    license["customer_id"] = "mallory@evil.com"  # tamper after signing
    with pytest.raises(signature.LicenseSignatureError, match="Signature verification failed"):
        signature.verify_license_signature(license)


def test_verify_fails_when_signature_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", pub_b64)
    license = _make_license(priv)
    del license["signature"]
    with pytest.raises(signature.LicenseSignatureError, match="no signature field"):
        signature.verify_license_signature(license)


def test_verify_fails_for_required_field_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", pub_b64)
    monkeypatch.setattr(signature, "BUNDLED_KEY_ID", "")
    license = _make_license(priv)
    del license["expires_at"]
    with pytest.raises(signature.LicenseSignatureError, match="missing required field: expires_at"):
        signature.verify_license_signature(license)


def test_verify_fails_for_key_id_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", pub_b64)
    monkeypatch.setattr(signature, "BUNDLED_KEY_ID", "key_bundled_in_release")
    license = _make_license(priv, key_id="key_from_other_deployment")
    with pytest.raises(signature.LicenseSignatureError, match="different deployment"):
        signature.verify_license_signature(license)


def test_verify_fails_for_malformed_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", pub_b64)
    monkeypatch.setattr(signature, "BUNDLED_KEY_ID", "")
    license = _make_license(priv)
    license["signature"] = "this is not base64@@@!!!"
    with pytest.raises(signature.LicenseSignatureError):
        signature.verify_license_signature(license)


def test_verify_fails_for_malformed_public_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signature, "BUNDLED_PUBLIC_KEY_B64", "definitely_not_a_valid_ed25519_public_key")
    monkeypatch.setattr(signature, "BUNDLED_KEY_ID", "")
    priv, _ = _gen_keypair()
    license = _make_license(priv)
    with pytest.raises(signature.LicenseSignatureError, match="malformed"):
        signature.verify_license_signature(license)


def test_signed_payload_matches_worker_canonicalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fields signed by the worker MUST match what verify_license_signature reconstructs.

    If the worker adds/removes/renames a signed field, this test catches the drift.
    Lock the field set so a refactor doesn't silently break verification.
    """
    expected = (
        "customer_id",
        "expires_at",
        "hardware_binding",
        "issued_at",
        "license_id",
        "license_version",
        "max_seats",
        "tier",
    )
    assert signature._signed_payload_fields() == expected
