"""Shared ZMQ transport security helpers."""
from __future__ import annotations

from pathlib import Path

import zmq.auth


def load_curve_key(value: str | bytes | Path, *, secret: bool) -> bytes:
    """Load a Z85 CURVE key from a raw value or pyzmq certificate file."""
    if isinstance(value, bytes):
        return value

    raw_value = str(value)
    path = Path(raw_value).expanduser()
    if path.exists():
        public_key, secret_key = zmq.auth.load_certificate(path)
        key = secret_key if secret else public_key
        if key is None:
            kind = "secret" if secret else "public"
            raise ValueError(f"CURVE certificate {path} does not contain a {kind} key")
        return key

    return raw_value.encode("ascii")


__all__ = ["load_curve_key"]
