"""Tests for src/tether/pro/fingerprint.py.

Covers: compute round-trip, customer-tag separation, tamper detection,
algo version handling, free-tier guard.
"""
from __future__ import annotations

import json

import pytest

from tether.pro.fingerprint import (
    ALGO,
    Fingerprint,
    compute_fingerprint,
    verify_fingerprint,
)


def _content() -> bytes:
    return b"VERIFICATION.md content for testing"


def test_compute_round_trip_succeeds() -> None:
    """Computing then verifying with the same customer_id passes."""
    fp = compute_fingerprint(_content(), "cust_alice")
    assert verify_fingerprint(_content(), fp, "cust_alice") is True


def test_compute_returns_correct_algo() -> None:
    fp = compute_fingerprint(_content(), "cust_alice")
    assert fp.algo == ALGO == "hmac-sha256-v1"


def test_customer_tag_is_anonymized() -> None:
    """Tag is a SHA256 prefix — doesn't leak the raw customer_id."""
    fp = compute_fingerprint(_content(), "cust_alice")
    assert "alice" not in fp.customer_tag
    assert len(fp.customer_tag) == 16  # 16-hex-char prefix


def test_different_customers_produce_different_signatures() -> None:
    """HMAC keys differ per customer → signatures differ for same content."""
    fp_a = compute_fingerprint(_content(), "cust_alice")
    fp_b = compute_fingerprint(_content(), "cust_bob")
    assert fp_a.signature != fp_b.signature
    assert fp_a.customer_tag != fp_b.customer_tag


def test_verify_fails_for_wrong_customer() -> None:
    """A signature from one customer doesn't validate for another."""
    fp = compute_fingerprint(_content(), "cust_alice")
    assert verify_fingerprint(_content(), fp, "cust_bob") is False


def test_verify_fails_for_tampered_content() -> None:
    """Modifying the canonical content breaks the signature."""
    fp = compute_fingerprint(_content(), "cust_alice")
    tampered = _content() + b" extra"
    assert verify_fingerprint(tampered, fp, "cust_alice") is False


def test_verify_fails_for_unknown_algo() -> None:
    """Unrecognized algo string is rejected."""
    fp = compute_fingerprint(_content(), "cust_alice")
    bad = Fingerprint(algo="other-v999", customer_tag=fp.customer_tag, signature=fp.signature)
    assert verify_fingerprint(_content(), bad, "cust_alice") is False


def test_verify_accepts_dict_form() -> None:
    """verify_fingerprint accepts both Fingerprint and dict."""
    fp = compute_fingerprint(_content(), "cust_alice")
    assert verify_fingerprint(_content(), fp.to_dict(), "cust_alice") is True


def test_verify_rejects_malformed_dict() -> None:
    """Missing fields in a dict-form fingerprint return False, not raise."""
    assert verify_fingerprint(_content(), {"algo": "hmac-sha256-v1"}, "cust_alice") is False


def test_compute_rejects_empty_customer_id() -> None:
    """Free-tier callers must NOT call compute_fingerprint."""
    with pytest.raises(ValueError, match="non-empty customer_id"):
        compute_fingerprint(_content(), "")


def test_fingerprint_is_deterministic() -> None:
    """Same input twice produces identical fingerprint."""
    fp_1 = compute_fingerprint(_content(), "cust_alice")
    fp_2 = compute_fingerprint(_content(), "cust_alice")
    assert fp_1 == fp_2


def test_fingerprint_serializes_cleanly() -> None:
    """to_dict / from_dict round-trip preserves the fingerprint."""
    fp = compute_fingerprint(_content(), "cust_alice")
    d = fp.to_dict()
    json.dumps(d)  # must be JSON-serializable
    fp2 = Fingerprint.from_dict(d)
    assert fp == fp2
