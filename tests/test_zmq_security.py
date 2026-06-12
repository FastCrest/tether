"""Tests for ZMQ transport security helpers."""
from __future__ import annotations

import pytest

from tether.runtime.transports.zmq.security import (
    is_loopback_bind,
    load_curve_key,
    validate_zmq_bind_security,
)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "[::1]", "localhost"])
def test_is_loopback_bind_accepts_local_only_hosts(host: str) -> None:
    assert is_loopback_bind(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "*", "192.168.1.10"])
def test_is_loopback_bind_rejects_network_hosts(host: str) -> None:
    assert not is_loopback_bind(host)


def test_validate_zmq_bind_security_allows_loopback_without_auth() -> None:
    validate_zmq_bind_security(
        host="127.0.0.1",
        curve_enabled=False,
        control_auth_enabled=False,
    )


def test_validate_zmq_bind_security_requires_curve_and_control_token() -> None:
    with pytest.raises(ValueError, match="CURVE certificates.*control token"):
        validate_zmq_bind_security(
            host="0.0.0.0",
            curve_enabled=False,
            control_auth_enabled=False,
        )


def test_validate_zmq_bind_security_rejects_partial_security() -> None:
    with pytest.raises(ValueError, match="control token"):
        validate_zmq_bind_security(
            host="192.168.1.10",
            curve_enabled=True,
            control_auth_enabled=False,
        )


def test_validate_zmq_bind_security_allows_explicit_insecure_override() -> None:
    validate_zmq_bind_security(
        host="0.0.0.0",
        curve_enabled=False,
        control_auth_enabled=False,
        allow_insecure=True,
    )


def test_load_curve_key_rejects_invalid_raw_key_length() -> None:
    with pytest.raises(ValueError, match="40 Z85 bytes"):
        load_curve_key("too-short", secret=False)
