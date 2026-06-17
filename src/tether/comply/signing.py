"""Signing helpers for Tether Comply bundles."""

from __future__ import annotations

import base64
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tether.parity_cert import ParityCertError, canonical_json_bytes, load_ed25519_private_key


def sign_payload(payload: dict[str, Any], *, signing_key: str, key_id: str = "") -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if k != "signature"}
    private_key = load_ed25519_private_key(signing_key)
    signature = private_key.sign(canonical_json_bytes(body))
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed = dict(body)
    signed["signature"] = {
        "alg": "Ed25519",
        "key_id": key_id,
        "sig": base64.b64encode(signature).decode("ascii"),
        "public_key_b64": base64.b64encode(public_key).decode("ascii"),
    }
    return signed


def verify_payload_signature(payload: dict[str, Any]) -> None:
    sig_block = payload.get("signature")
    if not isinstance(sig_block, dict):
        raise ParityCertError("conformity bundle has no signature block")
    if sig_block.get("alg") != "Ed25519":
        raise ParityCertError(f"unsupported signature alg: {sig_block.get('alg')!r}")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(str(sig_block["public_key_b64"]), validate=True)
        )
        signature = base64.b64decode(str(sig_block["sig"]), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ParityCertError(f"malformed conformity signature block: {exc}") from exc
    body = {k: v for k, v in payload.items() if k != "signature"}
    try:
        public_key.verify(signature, canonical_json_bytes(body))
    except InvalidSignature as exc:
        raise ParityCertError("conformity bundle signature verification failed") from exc


__all__ = ["sign_payload", "verify_payload_signature"]
