"""Tests for src/tether/pro/signature.py.

Generates a fresh Ed25519 keypair per test, monkey-patches the bundled
public key, then exercises the verify path end-to-end including tamper
detection and key-id mismatch detection.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

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
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_test": pub_b64})
    license = _make_license(priv)
    signature.verify_license_signature(license)  # must not raise


def test_verify_fails_when_bundled_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {})
    priv, _ = _gen_keypair()
    license = _make_license(priv)
    with pytest.raises(signature.LicenseSignatureError, match="not been deployed"):
        signature.verify_license_signature(license)


def test_verify_fails_for_tampered_license(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_test": pub_b64})
    license = _make_license(priv)
    license["customer_id"] = "mallory@evil.com"  # tamper after signing
    with pytest.raises(signature.LicenseSignatureError, match="Signature verification failed"):
        signature.verify_license_signature(license)


def test_verify_fails_when_signature_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_test": pub_b64})
    license = _make_license(priv)
    del license["signature"]
    with pytest.raises(signature.LicenseSignatureError, match="no signature field"):
        signature.verify_license_signature(license)


def test_verify_fails_for_required_field_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_test": pub_b64})
    license = _make_license(priv)
    del license["expires_at"]
    with pytest.raises(signature.LicenseSignatureError, match="missing required field: expires_at"):
        signature.verify_license_signature(license)


def test_verify_fails_for_key_id_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_bundled_in_release": pub_b64})
    license = _make_license(priv, key_id="key_from_other_deployment")
    with pytest.raises(signature.LicenseSignatureError, match="unknown or retired"):
        signature.verify_license_signature(license)


def test_verify_fails_for_malformed_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub_b64 = _gen_keypair()
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_test": pub_b64})
    license = _make_license(priv)
    license["signature"] = "this is not base64@@@!!!"
    with pytest.raises(signature.LicenseSignatureError):
        signature.verify_license_signature(license)


def test_verify_fails_for_malformed_public_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", {"key_test": "definitely_not_a_valid_ed25519_public_key"})
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


def test_fixed_non_ascii_cross_language_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    vector = json.loads(
        (Path(__file__).parent / "fixtures" / "license_non_ascii_vector.json").read_text()
    )
    canonical = signature._canonical_json(vector["payload"])
    assert canonical == base64.b64decode(vector["canonical_utf8_b64"])
    assert "客户".encode() in canonical
    license_value = {
        **vector["payload"],
        "key_id": vector["key_id"],
        "signature": vector["signature_b64"],
    }
    monkeypatch.setattr(
        signature,
        "TRUSTED_PUBLIC_KEYS_B64",
        {vector["key_id"]: vector["public_key_b64"]},
    )
    signature.verify_license_signature(license_value)


def test_actual_worker_non_ascii_activation_verifies_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).parents[1]
    emitted = subprocess.run(
        ["node", "infra/license-worker/test/emit-non-ascii-activation.mjs"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(emitted.stdout)
    license_value = value["response"]["license"]
    monkeypatch.setattr(
        signature,
        "TRUSTED_PUBLIC_KEYS_B64",
        {license_value["key_id"]: value["public_key_b64"]},
    )
    assert license_value["customer_id"] == "客户 É"
    assert license_value["hardware_binding"]["gpu_name"] == "Éclair GPU"
    signature.verify_license_signature(license_value)
