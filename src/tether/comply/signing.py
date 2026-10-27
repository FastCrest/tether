"""Signing helpers for Tether Comply bundles."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from typing import Any

from cryptography.hazmat.primitives import serialization

from tether.parity_cert import (
    canonical_json_bytes,
    load_ed25519_private_key,
    verify_ed25519_signature_block,
)


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


def verify_payload_signature(payload: dict[str, Any], *, trusted_keys: Iterable[bytes]) -> None:
    verify_ed25519_signature_block(payload, trusted_keys=trusted_keys, label="Conformity bundle")


__all__ = ["sign_payload", "verify_payload_signature"]
