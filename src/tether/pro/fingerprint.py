"""HMAC fingerprinting for Pro-tier output artifacts.

Embeds a per-customer signature in artifacts produced under a Pro license
(VERIFICATION.md, JSONL traces, distilled model metadata) so commercial
redistribution can be traced back to the source customer.

Threat model
------------
This module is NOT cryptographic protection against forgery. The HMAC key
is derived deterministically from the license `customer_id` + a project
salt (both visible in the open-source binary). Anyone who reads the source
can compute valid fingerprints. Phase 1.5 wires the signing endpoint and
swaps the key derivation for one that uses server-issued secret material
embedded in the JWT — at that point the fingerprint becomes forgery-proof.

What Phase 1 fingerprinting actually buys you
---------------------------------------------
1. Presence-of-fingerprint signal — Pro outputs carry a marker that free-tier
   outputs don't. If a competing service publishes artifacts WITH a Tether
   Pro fingerprint, that's evidence the artifact came through Tether Pro
   (because no honest free-tier user would manually inject a fake fingerprint).

2. Customer-tag traceability — the fingerprint embeds an
   ``HMAC-SHA256(customer_id)[:16]`` tag. If artifacts with that tag appear in
   a competitor's product, you know which customer's deployment was the source.
   (The customer_id itself is not in the artifact — only a hash of it — so
   customer privacy is preserved unless you cross-reference your billing DB.)

3. Tamper-evidence on intra-deployment artifacts — within a single customer's
   deployment, the fingerprint detects whether an artifact was modified after
   Tether generated it (the HMAC over canonicalized content fails verification).

What Phase 1 does NOT buy you (Phase 1.5 trigger):
- Cryptographic forgery resistance (anyone can compute a valid fingerprint
  given a customer_id + the open source).
- Server-side proof of authenticity (no signing endpoint exists yet).

API
---
Two functions, plus a typed return:

- ``compute_fingerprint(canonical_bytes, license)`` → ``Fingerprint``
- ``verify_fingerprint(canonical_bytes, fingerprint, license)`` → ``bool``

Free-tier callers (no license) should NOT call this — fingerprints only
appear on Pro outputs. Callers gate on ``server.pro_license is not None``.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass
from typing import Any

# Bumped if the canonical-form rules or HMAC construction change. Old
# fingerprints stay verifiable via reading the algo field.
ALGO = "hmac-sha256-v1"

# Project salt — public and intentionally so. It's a domain-separation
# string, not a secret. Phase 1.5 will pair this with a server-issued
# customer-secret stored alongside the license.
_PROJECT_SALT = b"tether-vla:pro-fingerprint:v1"


@dataclass(frozen=True)
class Fingerprint:
    """HMAC-signed marker embedded in a Pro output artifact.

    Fields:
        algo: ``"hmac-sha256-v1"`` (versioned for future rotation)
        customer_tag: ``HMAC-SHA256(customer_id)[:16]`` — anonymized customer
            identifier; reverse-mapping requires the billing DB
        signature: hex HMAC-SHA256 of the canonical content bytes
    """

    algo: str
    customer_tag: str
    signature: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fingerprint":
        return cls(
            algo=str(d["algo"]),
            customer_tag=str(d["customer_tag"]),
            signature=str(d["signature"]),
        )


def _derive_key(customer_id: str) -> bytes:
    """Derive the HMAC key for a given customer.

    Phase 1: deterministic from ``customer_id`` + ``_PROJECT_SALT``. Anyone
    with the customer_id and the open source can compute the same key.

    Phase 1.5: this function will accept a ``ProLicense`` and use the
    server-issued secret material from the signed JWT. The interface
    stays stable — the fingerprint format doesn't change, only the key
    becomes forgery-resistant.
    """
    return hmac.new(_PROJECT_SALT, customer_id.encode("utf-8"), hashlib.sha256).digest()


def _customer_tag(customer_id: str) -> str:
    """Anonymized 16-char hex tag for the customer."""
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:16]


def compute_fingerprint(
    canonical_bytes: bytes,
    customer_id: str,
) -> Fingerprint:
    """Compute the fingerprint for a Pro artifact.

    Parameters
    ----------
    canonical_bytes : bytes
        The canonicalized artifact content. Caller is responsible for
        ensuring identical inputs produce identical canonicalization
        (e.g., ``json.dumps(d, sort_keys=True, separators=(",", ":")).encode()``
        for JSON; raw markdown bytes WITHOUT the fingerprint line itself
        for VERIFICATION.md).
    customer_id : str
        The ``customer_id`` field from a valid ``ProLicense``.

    Returns
    -------
    Fingerprint
        The signed marker. Caller embeds ``fingerprint.to_dict()`` into the
        artifact in a field named ``"tether_fingerprint"`` (or as a markdown
        comment for `.md` artifacts).
    """
    if not customer_id:
        raise ValueError(
            "compute_fingerprint requires a non-empty customer_id. "
            "Free-tier outputs should not be fingerprinted."
        )
    key = _derive_key(customer_id)
    sig = hmac.new(key, canonical_bytes, hashlib.sha256).hexdigest()
    return Fingerprint(algo=ALGO, customer_tag=_customer_tag(customer_id), signature=sig)


def verify_fingerprint(
    canonical_bytes: bytes,
    fingerprint: Fingerprint | dict[str, Any],
    customer_id: str,
) -> bool:
    """Verify that a fingerprint matches the canonical content for a customer.

    Returns True iff the algo is recognized AND the customer_tag matches the
    tag derived from ``customer_id`` AND the HMAC signature matches.

    Used internally by tests and by the leak-investigation runbook
    (``reflex_context/04_product/oss_leak_monitoring.md``).
    """
    if isinstance(fingerprint, dict):
        try:
            fingerprint = Fingerprint.from_dict(fingerprint)
        except (KeyError, ValueError):
            return False
    if fingerprint.algo != ALGO:
        return False
    if fingerprint.customer_tag != _customer_tag(customer_id):
        return False
    expected_sig = hmac.new(
        _derive_key(customer_id), canonical_bytes, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(fingerprint.signature, expected_sig)


__all__ = [
    "ALGO",
    "Fingerprint",
    "compute_fingerprint",
    "verify_fingerprint",
]
