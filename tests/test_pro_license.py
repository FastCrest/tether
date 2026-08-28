"""Security-focused tests for signed v2 license loading."""
from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tether.pro import heartbeat, signature
from tether.pro.heartbeat import HeartbeatAttestation, HeartbeatNetworkError
from tether.pro.license import (
    LICENSE_VERSION,
    HardwareFingerprintLite,
    LicenseCorrupt,
    LicenseExpired,
    LicenseHardwareMismatch,
    LicenseHeartbeatStale,
    LicenseMissing,
    ProLicense,
    issue_dev_license,
    load_license,
)


def _mk_hw(**overrides) -> HardwareFingerprintLite:
    values = {"gpu_uuid": "GPU-abc-123", "gpu_name": "NVIDIA A10G", "cpu_count": 8}
    values.update(overrides)
    return HardwareFingerprintLite(**values)


@pytest.fixture
def signing_key(monkeypatch: pytest.MonkeyPatch) -> Ed25519PrivateKey:
    private = Ed25519PrivateKey.generate()
    public_b64 = base64.b64encode(private.public_key().public_bytes_raw()).decode("ascii")
    trusted = {"key_test": public_b64}
    monkeypatch.setattr(signature, "TRUSTED_PUBLIC_KEYS_B64", trusted)
    monkeypatch.setattr(heartbeat, "TRUSTED_PUBLIC_KEYS_B64", trusted)
    return private


def _signed_license(private: Ed25519PrivateKey, **overrides) -> dict:
    now = datetime.now(timezone.utc)
    payload = {
        "license_version": 2,
        "license_id": "lic_test",
        "customer_id": "acme",
        "tier": "pro",
        "issued_at": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(days=30)).isoformat().replace("+00:00", "Z"),
        "max_seats": 1,
        "hardware_binding": {
            "gpu_uuid": "GPU-abc-123", "gpu_name": "NVIDIA A10G", "cpu_count": 8,
        },
    }
    payload.update(overrides)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "signature": base64.b64encode(private.sign(canonical)).decode("ascii"),
        "key_id": "key_test",
    }


def _active_attestation(*, license_id: str = "lic_test", valid_until: int | None = None):
    now = int(time.time())
    return HeartbeatAttestation(
        domain="tether.license.heartbeat",
        issued_at=now,
        key_id="key_test",
        license_id=license_id,
        request_nonce="AAAAAAAAAAAAAAAAAAAAAA",
        status="active",
        v=1,
        valid_until=valid_until or now + 3600,
        signature="A" * 86,
    )


def _signed_attestation(private: Ed25519PrivateKey, *, valid_until: int) -> dict:
    now = int(time.time())
    payload = {
        "domain": "tether.license.heartbeat",
        "issued_at": now,
        "key_id": "key_test",
        "license_id": "lic_test",
        "request_nonce": base64.urlsafe_b64encode(b"0123456789abcdef").decode().rstrip("="),
        "status": "active",
        "v": 1,
        "valid_until": valid_until,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "signature": base64.urlsafe_b64encode(private.sign(canonical)).decode().rstrip("="),
    }


def _write(path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _mock_online(monkeypatch: pytest.MonkeyPatch, attestation=None) -> None:
    monkeypatch.setattr(
        heartbeat,
        "send_heartbeat",
        lambda **_kwargs: attestation or _active_attestation(),
    )


def test_hardware_matching_is_exact():
    assert _mk_hw().matches(_mk_hw())
    assert not _mk_hw(gpu_uuid="A").matches(_mk_hw(gpu_uuid="B"))
    assert not _mk_hw(cpu_count=8).matches(_mk_hw(cpu_count=16))


def test_v2_round_trip_shape():
    license = ProLicense(
        license_version=LICENSE_VERSION,
        license_id="lic_test",
        customer_id="acme",
        tier="pro",
        issued_at="2026-04-25T00:00:00Z",
        expires_at="2027-04-25T00:00:00Z",
        max_seats=1,
        hardware_binding=_mk_hw(),
        signature="sig",
        key_id="key_test",
    )
    assert ProLicense.from_dict(license.to_dict()) == license


def test_load_missing_and_corrupt(tmp_path):
    with pytest.raises(LicenseMissing):
        load_license(path=tmp_path / "missing", current_hardware=_mk_hw())
    path = tmp_path / "bad"
    path.write_text("not json")
    with pytest.raises(LicenseCorrupt):
        load_license(path=path, current_hardware=_mk_hw())


def test_unsigned_v1_is_always_rejected_even_with_dev_env(tmp_path, monkeypatch):
    path = tmp_path / "legacy.license"
    issue_dev_license(customer_id="acme", hardware=_mk_hw(), path=path)
    monkeypatch.setenv("TETHER_DEV", "1")
    with pytest.raises(LicenseCorrupt, match="require signed version 2"):
        load_license(path=path, current_hardware=_mk_hw())


def test_valid_signed_v2_requires_server_and_never_rewrites_license(
    tmp_path, monkeypatch, signing_key,
):
    path = tmp_path / "pro.license"
    value = _signed_license(signing_key)
    _write(path, value)
    before = path.read_bytes()
    _mock_online(monkeypatch)

    loaded = load_license(path=path, current_hardware=_mk_hw())

    assert loaded.license_id == "lic_test"
    assert loaded.attestation_valid_until > int(time.time())
    assert path.read_bytes() == before


def test_tampered_or_unsigned_v2_is_rejected(tmp_path, signing_key):
    path = tmp_path / "pro.license"
    value = _signed_license(signing_key)
    value["tier"] = "enterprise"
    _write(path, value)
    with pytest.raises(LicenseCorrupt, match="signature verification"):
        load_license(path=path, current_hardware=_mk_hw())


def test_expired_and_hardware_mismatch_fail_before_heartbeat(tmp_path, signing_key):
    expired_path = tmp_path / "expired"
    _write(
        expired_path,
        _signed_license(
            signing_key,
            expires_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        ),
    )
    with pytest.raises(LicenseExpired):
        load_license(path=expired_path, current_hardware=_mk_hw())

    mismatch_path = tmp_path / "mismatch"
    _write(mismatch_path, _signed_license(signing_key))
    with pytest.raises(LicenseHardwareMismatch):
        load_license(path=mismatch_path, current_hardware=_mk_hw(gpu_uuid="other"))


def test_first_use_network_failure_fails_closed_without_cache(
    tmp_path, monkeypatch, signing_key,
):
    path = tmp_path / "pro.license"
    _write(path, _signed_license(signing_key))
    monkeypatch.setattr(
        heartbeat,
        "send_heartbeat",
        lambda **_kwargs: (_ for _ in ()).throw(HeartbeatNetworkError("offline")),
    )
    with pytest.raises(LicenseHeartbeatStale, match="no usable signed heartbeat cache"):
        load_license(path=path, current_hardware=_mk_hw())


def test_network_failure_uses_only_a_verified_cache_inside_grace(
    tmp_path, monkeypatch, signing_key,
):
    path = tmp_path / "pro.license"
    _write(path, _signed_license(signing_key))
    cache_path = path.with_suffix(".license.heartbeat")
    _write(cache_path, _signed_attestation(signing_key, valid_until=int(time.time()) + 60))
    monkeypatch.setattr(
        heartbeat,
        "send_heartbeat",
        lambda **_kwargs: (_ for _ in ()).throw(HeartbeatNetworkError("offline")),
    )
    loaded = load_license(path=path, current_hardware=_mk_hw())
    assert loaded.attestation_valid_until > int(time.time())


def test_network_failure_rejects_verified_cache_after_grace(
    tmp_path, monkeypatch, signing_key,
):
    path = tmp_path / "pro.license"
    _write(path, _signed_license(signing_key))
    cache_path = path.with_suffix(".license.heartbeat")
    _write(cache_path, _signed_attestation(signing_key, valid_until=int(time.time()) - 301))
    monkeypatch.setattr(
        heartbeat,
        "send_heartbeat",
        lambda **_kwargs: (_ for _ in ()).throw(HeartbeatNetworkError("offline")),
    )
    with pytest.raises(LicenseHeartbeatStale):
        load_license(path=path, current_hardware=_mk_hw())


def test_legacy_dev_file_has_private_permissions_but_cannot_unlock(tmp_path):
    path = tmp_path / "legacy.license"
    issue_dev_license(customer_id="acme", hardware=_mk_hw(), path=path)
    assert os.stat(path).st_mode & 0o777 == 0o600
    with pytest.raises(LicenseCorrupt):
        load_license(path=path, current_hardware=_mk_hw())
