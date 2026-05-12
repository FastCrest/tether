"""Tests for src/reflex/curate/consent.py — opt-in receipt round-trip + revoke."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from reflex.curate import consent as curate_consent
from reflex.curate.consent import (
    CONSENT_VERSION,
    TERMS_VERSION,
    ConsentCorrupted,
    ConsentNotFound,
    ConsentScope,
    CurateConsent,
    derive_contributor_id,
    is_opted_in,
    load,
    revoke,
    save,
)


# ── derive_contributor_id ────────────────────────────────────────────────────


def test_derive_contributor_id_format() -> None:
    cid = derive_contributor_id()
    parts = cid.split("_")
    assert parts[0] == "free"
    assert len(parts) == 3
    # 16-hex hardware hash
    assert len(parts[1]) == 16
    int(parts[1], 16)
    # 8-hex uuid suffix
    assert len(parts[2]) == 8
    int(parts[2], 16)


def test_derive_contributor_id_hardware_piece_stable() -> None:
    """Same machine = same first 16 hex chars across calls (the suffix differs)."""
    a = derive_contributor_id()
    b = derive_contributor_id()
    assert a.split("_")[1] == b.split("_")[1]
    assert a.split("_")[2] != b.split("_")[2]  # uuid4 differs


# ── ConsentScope ─────────────────────────────────────────────────────────────


def test_consent_scope_default() -> None:
    s = ConsentScope.default()
    assert "recorded_episodes" in s.data_types
    assert "raw_images_unless_explicit_opt_in" in s.excludes


def test_consent_scope_round_trip() -> None:
    s = ConsentScope(
        data_types=["recorded_episodes", "synthetic_data"],
        excludes=["bad_thing"],
    )
    d = s.to_dict()
    s2 = ConsentScope.from_dict(d)
    assert s == s2


# ── CurateConsent ────────────────────────────────────────────────────────────


def test_curate_consent_validates_tier() -> None:
    with pytest.raises(ValueError, match="tier must be"):
        CurateConsent(
            version=1,
            contributor_id="free_abc",
            tier="bogus",
            opted_in_at="2026-05-05T00:00:00Z",
            accepted_terms_version=TERMS_VERSION,
            scope=ConsentScope.default(),
            privacy_mode="hash_only",
            geographic_residency="auto-detect",
        )


def test_curate_consent_validates_privacy_mode() -> None:
    with pytest.raises(ValueError, match="privacy_mode"):
        CurateConsent(
            version=1,
            contributor_id="free_abc",
            tier="free",
            opted_in_at="2026-05-05T00:00:00Z",
            accepted_terms_version=TERMS_VERSION,
            scope=ConsentScope.default(),
            privacy_mode="bogus",
            geographic_residency="auto-detect",
        )


def test_curate_consent_validates_contributor_id() -> None:
    with pytest.raises(ValueError, match="contributor_id"):
        CurateConsent(
            version=1,
            contributor_id="",
            tier="free",
            opted_in_at="2026-05-05T00:00:00Z",
            accepted_terms_version=TERMS_VERSION,
            scope=ConsentScope.default(),
            privacy_mode="hash_only",
            geographic_residency="auto-detect",
        )


def test_curate_consent_round_trip() -> None:
    receipt = CurateConsent(
        version=CONSENT_VERSION,
        contributor_id="free_deadbeef_cafef00d",
        tier="free",
        opted_in_at="2026-05-05T00:00:00Z",
        accepted_terms_version=TERMS_VERSION,
        scope=ConsentScope.default(),
        privacy_mode="hash_only",
        geographic_residency="auto-detect",
    )
    d = receipt.to_dict()
    rebuilt = CurateConsent.from_dict(d)
    assert rebuilt == receipt


# ── save / load / revoke ─────────────────────────────────────────────────────


def test_save_then_load_returns_same_receipt(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    saved = save(tier="free", path=path)
    loaded = load(path)
    assert loaded.contributor_id == saved.contributor_id
    assert loaded.tier == "free"
    assert loaded.privacy_mode == "hash_only"


def test_save_uses_anonymous_id_when_none(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    receipt = save(tier="free", path=path)
    assert receipt.contributor_id.startswith("free_")
    assert len(receipt.contributor_id.split("_")) == 3


def test_save_uses_provided_id(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    receipt = save(tier="pro", contributor_id="acme-corp-123", path=path)
    assert receipt.contributor_id == "acme-corp-123"
    assert receipt.tier == "pro"


def test_save_writes_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    save(tier="free", path=path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_load_raises_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    with pytest.raises(ConsentNotFound):
        load(path)


def test_load_raises_on_corrupted_json(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    path.write_text("not json {")
    with pytest.raises(ConsentCorrupted):
        load(path)


def test_load_raises_on_version_drift(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    save(tier="free", path=path)
    # Mutate version on disk
    data = json.loads(path.read_text())
    data["version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ConsentCorrupted, match="version drift"):
        load(path)


def test_is_opted_in_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    assert not is_opted_in(path)
    save(tier="free", path=path)
    assert is_opted_in(path)


def test_revoke_removes_file(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    save(tier="free", path=path)
    assert path.exists()
    assert revoke(path) is True
    assert not path.exists()
    assert not is_opted_in(path)


def test_revoke_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "consent.json"
    assert revoke(path) is False
    save(tier="free", path=path)
    assert revoke(path) is True
    assert revoke(path) is False
