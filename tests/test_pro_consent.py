"""Tests for src/tether/pro/consent.py — Phase 1 self-distilling-serve Day 2.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #1: data
collection is EXPLICIT opt-in via TTY prompt; non-interactive contexts
without an existing receipt fail loud.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tether.pro.consent import (
    CONSENT_VERSION,
    TERMS_VERSION,
    ConsentMismatch,
    ConsentReceipt,
    ConsentRequired,
    PIIOptions,
    ProConsent,
)


def _mk_pii(**overrides) -> PIIOptions:
    defaults = dict(
        face_blur_mode="blur",
        instruction_mode="hashed",
        state_mode="raw",
    )
    defaults.update(overrides)
    return PIIOptions(**defaults)


# ---------------------------------------------------------------------------
# PIIOptions
# ---------------------------------------------------------------------------


def test_pii_rejects_unknown_face_blur_mode():
    with pytest.raises(ValueError, match="face_blur_mode"):
        PIIOptions(face_blur_mode="unknown", instruction_mode="hashed", state_mode="raw")


def test_pii_rejects_unknown_instruction_mode():
    with pytest.raises(ValueError, match="instruction_mode"):
        PIIOptions(face_blur_mode="blur", instruction_mode="encoded", state_mode="raw")


def test_pii_rejects_unknown_state_mode():
    with pytest.raises(ValueError, match="state_mode"):
        PIIOptions(face_blur_mode="blur", instruction_mode="hashed", state_mode="encrypted")


def test_pii_accepts_all_valid_combinations():
    PIIOptions(face_blur_mode="blur", instruction_mode="hashed", state_mode="raw")
    PIIOptions(face_blur_mode="raw", instruction_mode="raw", state_mode="hashed")
    PIIOptions(face_blur_mode="skip", instruction_mode="hashed", state_mode="raw")


# ---------------------------------------------------------------------------
# ConsentReceipt
# ---------------------------------------------------------------------------


def test_receipt_to_from_dict_roundtrip():
    r = ConsentReceipt(
        consent_version=CONSENT_VERSION,
        customer_id="acme",
        accepted_at="2026-04-25T10:00:00.000000Z",
        accepted_terms_version=TERMS_VERSION,
        pii_options=_mk_pii(),
    )
    d = r.to_dict()
    r2 = ConsentReceipt.from_dict(d)
    assert r2 == r


def test_receipt_schema_version_is_one():
    assert CONSENT_VERSION == 1
    assert ConsentReceipt.SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# ProConsent — load missing receipt
# ---------------------------------------------------------------------------


def test_consent_raises_required_when_missing_in_non_interactive(tmp_path):
    """No receipt + interactive=False → ConsentRequired."""
    with pytest.raises(ConsentRequired):
        ProConsent.load_or_prompt(
            customer_id="acme",
            pii_options=_mk_pii(),
            path=tmp_path / "consent.json",
            interactive=False,
        )


def test_consent_prompts_when_missing_in_interactive(tmp_path):
    """No receipt + interactive=True → calls prompt_fn."""
    prompted = {"called": 0}

    def prompt(text):
        prompted["called"] += 1
        return True

    consent = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=prompt,
    )
    assert prompted["called"] == 1
    assert consent.has_consent
    assert consent.receipt.customer_id == "acme"


def test_consent_raises_required_on_decline(tmp_path):
    """Operator says no → ConsentRequired (don't write a 'declined' receipt;
    next start re-prompts cleanly)."""
    with pytest.raises(ConsentRequired, match="declined"):
        ProConsent.load_or_prompt(
            customer_id="acme",
            pii_options=_mk_pii(),
            path=tmp_path / "consent.json",
            interactive=True,
            prompt_fn=lambda text: False,
        )


def test_consent_writes_receipt_on_accept(tmp_path):
    consent = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=lambda text: True,
    )
    assert (tmp_path / "consent.json").exists()
    # Permissions are 0o600 (customer-private)
    mode = os.stat(tmp_path / "consent.json").st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# ProConsent — load existing receipt
# ---------------------------------------------------------------------------


def test_consent_silent_pass_on_existing_valid_receipt(tmp_path):
    """First call writes receipt; second call silently loads it (no prompt)."""
    prompted = {"called": 0}

    def prompt(text):
        prompted["called"] += 1
        return True

    # First call — prompt fires
    consent1 = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=prompt,
    )
    assert prompted["called"] == 1

    # Second call — silent
    consent2 = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=False,  # even non-interactive succeeds when receipt exists
        prompt_fn=prompt,
    )
    assert prompted["called"] == 1  # no second prompt
    assert consent2.has_consent


def test_consent_mismatch_on_different_customer_id(tmp_path):
    ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=lambda text: True,
    )
    with pytest.raises(ConsentMismatch, match="customer_id"):
        ProConsent.load_or_prompt(
            customer_id="other-customer",
            pii_options=_mk_pii(),
            path=tmp_path / "consent.json",
            interactive=False,
        )


def test_consent_mismatch_on_pii_options_change(tmp_path):
    """Same customer changes PII options → re-prompt required."""
    ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(face_blur_mode="blur"),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=lambda text: True,
    )
    with pytest.raises(ConsentMismatch, match="PII options"):
        ProConsent.load_or_prompt(
            customer_id="acme",
            pii_options=_mk_pii(face_blur_mode="raw"),
            path=tmp_path / "consent.json",
            interactive=False,
        )


def test_consent_mismatch_on_corrupted_receipt(tmp_path):
    path = tmp_path / "consent.json"
    path.write_text("not valid json")
    with pytest.raises(ConsentMismatch, match="corrupted"):
        ProConsent.load_or_prompt(
            customer_id="acme",
            pii_options=_mk_pii(),
            path=path,
            interactive=False,
        )


def test_consent_mismatch_on_consent_version_drift(tmp_path):
    """Old-version receipt → re-prompt required."""
    path = tmp_path / "consent.json"
    path.write_text(json.dumps({
        "consent_version": 99,  # future version
        "customer_id": "acme",
        "accepted_at": "2026-04-25T10:00:00Z",
        "accepted_terms_version": "old",
        "pii_options": {
            "face_blur_mode": "blur",
            "instruction_mode": "hashed",
            "state_mode": "raw",
        },
    }))
    with pytest.raises(ConsentMismatch, match="consent_version"):
        ProConsent.load_or_prompt(
            customer_id="acme",
            pii_options=_mk_pii(),
            path=path,
            interactive=False,
        )


# ---------------------------------------------------------------------------
# Revoke (GDPR/CCPA)
# ---------------------------------------------------------------------------


def test_revoke_removes_receipt_file(tmp_path):
    consent = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=lambda text: True,
    )
    assert (tmp_path / "consent.json").exists()
    consent.revoke()
    assert not (tmp_path / "consent.json").exists()
    assert not consent.has_consent


def test_revoke_idempotent_when_already_revoked(tmp_path):
    consent = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=lambda text: True,
    )
    consent.revoke()
    consent.revoke()  # no error


def test_revoke_with_data_dir_wipes_collected_data(tmp_path):
    consent = ProConsent.load_or_prompt(
        customer_id="acme",
        pii_options=_mk_pii(),
        path=tmp_path / "consent.json",
        interactive=True,
        prompt_fn=lambda text: True,
    )
    data_dir = tmp_path / "pro-data"
    data_dir.mkdir()
    (data_dir / "2026-04-25.jsonl").write_text("{}\n")
    consent.revoke(also_wipe_data_dir=data_dir)
    assert not data_dir.exists()
