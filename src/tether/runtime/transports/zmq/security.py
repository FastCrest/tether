"""Shared ZMQ transport security helpers."""
from __future__ import annotations

import ipaddress
from pathlib import Path

import zmq.auth


def load_curve_key(value: str | bytes | Path, *, secret: bool) -> bytes:
    """Load a Z85 CURVE key from a raw value or pyzmq certificate file."""
    if isinstance(value, bytes):
        return _validate_curve_key(value, secret=secret)

    raw_value = str(value)
    path = Path(raw_value).expanduser()
    if path.exists():
        public_key, secret_key = zmq.auth.load_certificate(path)
        key = secret_key if secret else public_key
        if key is None:
            kind = "secret" if secret else "public"
            raise ValueError(f"CURVE certificate {path} does not contain a {kind} key")
        return _validate_curve_key(key, secret=secret)

    return _validate_curve_key(raw_value.encode("ascii"), secret=secret)


def is_loopback_bind(host: str) -> bool:
    """Return True when a ZMQ bind host is local-only."""
    candidate = host.strip().strip("[]")
    if candidate in {"localhost"}:
        return True
    if candidate in {"", "*", "0.0.0.0", "::"}:
        return False
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def validate_zmq_bind_security(
    *,
    host: str,
    curve_enabled: bool,
    control_auth_enabled: bool,
    allow_insecure: bool = False,
) -> None:
    """Reject externally reachable ZMQ binds unless transport security is complete."""
    if is_loopback_bind(host):
        return
    if curve_enabled and control_auth_enabled:
        return
    if allow_insecure:
        return

    missing: list[str] = []
    if not curve_enabled:
        missing.append("CURVE certificates")
    if not control_auth_enabled:
        missing.append("a ZMQ control token")
    missing_text = " and ".join(missing)
    raise ValueError(
        f"Refusing insecure ZMQ bind on host {host!r}: configure {missing_text}, "
        "bind to 127.0.0.1, or pass --zmq-insecure-ok for an isolated lab network."
    )


def _validate_curve_key(key: bytes, *, secret: bool) -> bytes:
    if len(key) != 40:
        kind = "secret" if secret else "public"
        raise ValueError(f"CURVE {kind} key must be 40 Z85 bytes, got {len(key)}")
    return key


__all__ = ["is_loopback_bind", "load_curve_key", "validate_zmq_bind_security"]
