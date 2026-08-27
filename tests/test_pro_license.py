"""Tests for src/tether/pro/license.py — Phase 1 self-distilling-serve Day 2.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #5: HW-bound
JWT at ~/.tether/pro.license + 24h heartbeat. Phase 1 ships substrate
(format + validation) without cryptographic verification — Phase 1.5 wires
actual signing. License absence = exit 1, NEVER silent degrade.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tether.pro.license import (
    DEFAULT_LICENSE_PATH,
    HEARTBEAT_FRESHNESS_S,
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
    defaults = dict(
        gpu_uuid="GPU-abc-123",
        gpu_name="NVIDIA A10G",
        cpu_count=8,
    )
    defaults.update(overrides)
    return HardwareFingerprintLite(**defaults)


# ---------------------------------------------------------------------------
# HardwareFingerprintLite.matches
# ---------------------------------------------------------------------------


def test_hw_matches_identical():
    assert _mk_hw().matches(_mk_hw())


def test_hw_mismatch_on_gpu_uuid():
    assert not _mk_hw(gpu_uuid="A").matches(_mk_hw(gpu_uuid="B"))


def test_hw_mismatch_on_cpu_count():
    assert not _mk_hw(cpu_count=8).matches(_mk_hw(cpu_count=16))


def test_hw_mismatch_on_gpu_name():
    assert not _mk_hw(gpu_name="A100").matches(_mk_hw(gpu_name="A10G"))


# ---------------------------------------------------------------------------
# ProLicense roundtrip + expiration
# ---------------------------------------------------------------------------


def test_license_to_from_dict_roundtrip():
    license = ProLicense(
        license_version=LICENSE_VERSION,
        customer_id="acme",
        tier="pro",
        issued_at="2026-04-25T00:00:00Z",
        expires_at="2027-04-25T00:00:00Z",
        hardware_binding=_mk_hw(),
        signature="sig123",
        last_heartbeat_at="2026-04-25T10:00:00Z",
    )
    d = license.to_dict()
    license2 = ProLicense.from_dict(d)
    assert license == license2


def test_license_is_expired_in_past():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    license = ProLicense(
        license_version=1, customer_id="acme", tier="pro",
        issued_at="2026-04-01T00:00:00Z", expires_at=past,
        hardware_binding=_mk_hw(),
    )
    assert license.is_expired()


def test_license_is_not_expired_in_future():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    license = ProLicense(
        license_version=1, customer_id="acme", tier="pro",
        issued_at="2026-04-01T00:00:00Z", expires_at=future,
        hardware_binding=_mk_hw(),
    )
    assert not license.is_expired()


def test_license_is_expired_on_unparseable_timestamp():
    license = ProLicense(
        license_version=1, customer_id="acme", tier="pro",
        issued_at="2026-04-01T00:00:00Z", expires_at="not-an-iso-date",
        hardware_binding=_mk_hw(),
    )
    assert license.is_expired()


def test_license_heartbeat_age_inf_when_unset():
    license = ProLicense(
        license_version=1, customer_id="acme", tier="pro",
        issued_at="2026-04-01T00:00:00Z", expires_at="2027-04-25T00:00:00Z",
        hardware_binding=_mk_hw(),
        last_heartbeat_at="",
    )
    assert license.heartbeat_age_s() == float("inf")


def test_license_heartbeat_age_recent_when_just_now():
    license = ProLicense(
        license_version=1, customer_id="acme", tier="pro",
        issued_at="2026-04-01T00:00:00Z", expires_at="2027-04-25T00:00:00Z",
        hardware_binding=_mk_hw(),
        last_heartbeat_at=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
    )
    assert license.heartbeat_age_s() < 60


def test_license_heartbeat_stale_after_25h():
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    license = ProLicense(
        license_version=1, customer_id="acme", tier="pro",
        issued_at="2026-04-01T00:00:00Z", expires_at="2027-04-25T00:00:00Z",
        hardware_binding=_mk_hw(),
        last_heartbeat_at=old,
    )
    assert license.is_heartbeat_stale()


# ---------------------------------------------------------------------------
# load_license — error paths
# ---------------------------------------------------------------------------


def test_load_raises_missing_when_file_absent(tmp_path):
    with pytest.raises(LicenseMissing):
        load_license(
            path=tmp_path / "no_such.license",
            current_hardware=_mk_hw(),
        )


def test_load_raises_corrupt_on_bad_json(tmp_path):
    path = tmp_path / "pro.license"
    path.write_text("not json")
    with pytest.raises(LicenseCorrupt):
        load_license(path=path, current_hardware=_mk_hw())


def test_load_raises_corrupt_on_missing_required_fields(tmp_path):
    path = tmp_path / "pro.license"
    path.write_text(json.dumps({"license_version": 1}))  # missing other fields
    with pytest.raises(LicenseCorrupt):
        load_license(path=path, current_hardware=_mk_hw())


def test_load_raises_corrupt_on_future_license_version(tmp_path):
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    # Tamper with the file to bump the license_version
    data = json.loads(path.read_text())
    data["license_version"] = LICENSE_VERSION + 99
    path.write_text(json.dumps(data))
    with pytest.raises(LicenseCorrupt, match="license version"):
        load_license(path=path, current_hardware=_mk_hw())


def test_load_raises_expired_on_past_expires_at(tmp_path):
    path = tmp_path / "pro.license"
    # Issue with valid_for_days but immediately tamper to past
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    data = json.loads(path.read_text())
    data["expires_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(data))
    with pytest.raises(LicenseExpired):
        load_license(path=path, current_hardware=_mk_hw())


def test_load_raises_hardware_mismatch_on_different_gpu(tmp_path):
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(gpu_uuid="GPU-A"),
        valid_for_days=30, path=path,
    )
    with pytest.raises(LicenseHardwareMismatch):
        load_license(
            path=path,
            current_hardware=_mk_hw(gpu_uuid="GPU-B"),
        )


def test_load_raises_heartbeat_stale_after_25h(tmp_path):
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    # Tamper the heartbeat to 25h ago
    data = json.loads(path.read_text())
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    data["last_heartbeat_at"] = old
    path.write_text(json.dumps(data))
    with pytest.raises(LicenseHeartbeatStale):
        load_license(path=path, current_hardware=_mk_hw())


def test_load_skip_heartbeat_check_succeeds_with_stale_heartbeat(tmp_path):
    """skip_heartbeat_check=True bypasses the freshness gate (used in tests
    + first-run scenarios)."""
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    data = json.loads(path.read_text())
    data["last_heartbeat_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(data))
    license = load_license(
        path=path, current_hardware=_mk_hw(), skip_heartbeat_check=True,
    )
    assert license.customer_id == "acme"


# ---------------------------------------------------------------------------
# load_license — happy path + heartbeat refresh
# ---------------------------------------------------------------------------


def test_load_succeeds_on_fresh_license(tmp_path):
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    license = load_license(path=path, current_hardware=_mk_hw())
    assert license.customer_id == "acme"
    assert license.tier == "pro"


def test_load_refreshes_heartbeat_on_success(tmp_path):
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    # Tamper the heartbeat to be within tolerance but old
    data = json.loads(path.read_text())
    old_hb = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    data["last_heartbeat_at"] = old_hb
    path.write_text(json.dumps(data))
    # Load — should succeed AND refresh heartbeat to now
    load_license(path=path, current_hardware=_mk_hw())
    new_data = json.loads(path.read_text())
    assert new_data["last_heartbeat_at"] != old_hb
    # New heartbeat is recent
    new_hb = datetime.fromisoformat(
        new_data["last_heartbeat_at"].replace("Z", "+00:00")
    )
    assert (datetime.now(timezone.utc) - new_hb).total_seconds() < 60


def test_heartbeat_rewrite_preserves_unmodelled_signed_fields(tmp_path):
    """The heartbeat refresh must NOT drop signed-envelope fields.

    ProLicense models only a subset of the on-disk license. The old code
    rewrote the file from ProLicense.to_dict(), silently dropping license_id,
    max_seats, and key_id — which are part of the signed payload, so the NEXT
    load_license would fail signature verification (a v2 license that locks
    itself out on the second startup). Persisting the raw envelope (only
    bumping last_heartbeat_at) fixes it; last_heartbeat_at is not signed.
    """
    path = tmp_path / "pro.license"
    expires = (datetime.now(timezone.utc) + timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    old_hb = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    # A v1 (legacy unsigned) license — skips the signature gate — but carrying
    # the same extra envelope fields a real signed v2 license would have.
    data = {
        "license_version": 1,
        "customer_id": "acme",
        "tier": "pro",
        "issued_at": old_hb,
        "expires_at": expires,
        "hardware_binding": {
            "gpu_uuid": "GPU-abc-123",
            "gpu_name": "NVIDIA A10G",
            "cpu_count": 8,
        },
        "signature": "",
        "last_heartbeat_at": old_hb,
        # Signed-payload fields the ProLicense dataclass does not model:
        "license_id": "lic_preserve_me",
        "max_seats": 5,
        "key_id": "key_abc123",
    }
    path.write_text(json.dumps(data))

    load_license(path=path, current_hardware=_mk_hw())

    persisted = json.loads(path.read_text())
    # The extra signed fields survive the heartbeat rewrite...
    assert persisted["license_id"] == "lic_preserve_me"
    assert persisted["max_seats"] == 5
    assert persisted["key_id"] == "key_abc123"
    # ...and the heartbeat was still refreshed.
    assert persisted["last_heartbeat_at"] != old_hb

    # A second load must also succeed (no LicenseCorrupt from dropped fields).
    load_license(path=path, current_hardware=_mk_hw())
    again = json.loads(path.read_text())
    assert again["license_id"] == "lic_preserve_me"
    assert again["max_seats"] == 5


# ---------------------------------------------------------------------------
# issue_dev_license
# ---------------------------------------------------------------------------


def test_issue_dev_writes_file_at_path(tmp_path):
    path = tmp_path / "pro.license"
    license = issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    assert path.exists()
    # Permissions are 0o600
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600
    assert license.customer_id == "acme"
    assert license.signature == ""  # Phase 1 dev license is unsigned


def test_issue_dev_round_trip_via_load_license(tmp_path):
    path = tmp_path / "pro.license"
    issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        valid_for_days=30, path=path,
    )
    loaded = load_license(path=path, current_hardware=_mk_hw())
    assert loaded.customer_id == "acme"
    assert loaded.tier == "pro"


def test_issue_dev_supports_custom_tier(tmp_path):
    path = tmp_path / "pro.license"
    license = issue_dev_license(
        customer_id="acme", hardware=_mk_hw(),
        tier="enterprise", valid_for_days=30, path=path,
    )
    assert license.tier == "enterprise"
