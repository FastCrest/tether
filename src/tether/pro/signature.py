"""Ed25519 license signature verification using the bundled public key.

License signatures are produced by the deployed license worker
(``infra/license-worker/worker.js``) using its master Ed25519 private key.
Customers verify signatures offline using the public key bundled in
``tether.pro._public_key.BUNDLED_PUBLIC_KEY_B64``.

The canonical-bytes format must match the worker's ``canonicalJson()``
function exactly: object keys sorted lexicographically, no whitespace,
JSON-serialized scalars. Any deviation breaks verification.

Failure modes (all raise ``LicenseSignatureError``):
- Bundled public key is empty (deploy not yet completed)
- License is missing required fields
- Signature is malformed or doesn't match
- License was signed with a key whose key_id doesn't match the bundled one
"""
from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tether.pro._public_key import BUNDLED_KEY_ID, BUNDLED_PUBLIC_KEY_B64


class LicenseSignatureError(Exception):
    """Raised when license signature verification fails."""


def _canonical_json(obj: Any) -> bytes:
    """Match the worker's canonicalJson() exactly: sorted keys, no whitespace.

    Equivalent to ``json.dumps(obj, sort_keys=True, separators=(',', ':'))``
    encoded as UTF-8.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signed_payload_fields() -> tuple[str, ...]:
    """Fields that go into the signed canonical payload, in alphabetical order.

    These must match the worker's adminIssue() payload composition exactly.
    Any mismatch (extra field, missing field, different field name) breaks
    signature verification.
    """
    return (
        "customer_id",
        "expires_at",
        "hardware_binding",
        "issued_at",
        "license_id",
        "license_version",
        "max_seats",
        "tier",
    )


def verify_license_signature(license_dict: dict[str, Any]) -> None:
    """Verify the Ed25519 signature on a license dict. Raises on failure.

    Parameters
    ----------
    license_dict : dict
        The license JSON loaded from disk (or from the activation endpoint).
        Must contain at minimum: license_id, customer_id, tier, issued_at,
        expires_at, hardware_binding, max_seats, license_version, signature,
        key_id.

    Raises
    ------
    LicenseSignatureError
        If verification fails for any reason. The error message identifies
        the specific failure mode for operator debugging.
    """
    if not BUNDLED_PUBLIC_KEY_B64:
        raise LicenseSignatureError(
            "BUNDLED_PUBLIC_KEY_B64 is empty in src/tether/pro/_public_key.py. "
            "The license worker has not been deployed yet, or the public key "
            "wasn't pasted in after running POST /admin/init. See "
            "infra/license-worker/README.md for deploy steps."
        )

    sig_b64 = license_dict.get("signature", "")
    if not sig_b64:
        raise LicenseSignatureError("License has no signature field.")

    license_key_id = license_dict.get("key_id", "")
    if BUNDLED_KEY_ID and license_key_id and license_key_id != BUNDLED_KEY_ID:
        raise LicenseSignatureError(
            f"License signed with key_id={license_key_id!r} but bundled key is "
            f"{BUNDLED_KEY_ID!r}. Either the license was issued by a different "
            f"deployment, or the bundled key is stale (run tether pro upgrade)."
        )

    # Reconstruct the canonical payload that the worker signed.
    payload: dict[str, Any] = {}
    for field in _signed_payload_fields():
        if field not in license_dict:
            raise LicenseSignatureError(f"License missing required field: {field}")
        payload[field] = license_dict[field]
    canonical = _canonical_json(payload)

    try:
        pub_raw = base64.b64decode(BUNDLED_PUBLIC_KEY_B64)
        pubkey = Ed25519PublicKey.from_public_bytes(pub_raw)
    except (ValueError, Exception) as exc:
        raise LicenseSignatureError(
            f"Bundled public key is malformed: {exc}. "
            f"Re-paste from the worker's GET /v1/pubkey response."
        ) from exc

    try:
        sig = base64.b64decode(sig_b64)
    except (ValueError, Exception) as exc:
        raise LicenseSignatureError(f"Signature is not valid base64: {exc}") from exc

    try:
        pubkey.verify(sig, canonical)
    except InvalidSignature as exc:
        raise LicenseSignatureError(
            "Signature verification failed. The license file has been "
            "tampered with, or was signed by a different deployment than "
            "the bundled public key recognizes."
        ) from exc


__all__ = ["LicenseSignatureError", "verify_license_signature"]
