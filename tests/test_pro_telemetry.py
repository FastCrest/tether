"""Tests for src/tether/pro/telemetry.py.

Covers: payload shape lock, opt-out env var, free-tier guard, network
failure swallowing, opt-out via empty customer_id.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tether.pro.telemetry import (
    DEFAULT_TELEMETRY_ENDPOINT,
    HEARTBEAT_SCHEMA_VERSION,
    build_payload,
    emit,
)


@pytest.fixture(autouse=True)
def _isolated_tether_home(tmp_path, monkeypatch):
    """Point TETHER_HOME at a clean dir so emit()'s onboarding opt-out check
    reads a known-empty state (telemetry enabled by default) regardless of the
    developer's real ~/.tether/onboarding.json. Tests opt out explicitly."""
    monkeypatch.setenv("TETHER_HOME", str(tmp_path))
    monkeypatch.delenv("TETHER_NO_TELEMETRY", raising=False)
    return tmp_path


def test_build_payload_returns_locked_v1_shape() -> None:
    """Phase 1 payload shape is LOCKED — additive-only Phase 2."""
    p = build_payload(
        customer_id="cust_alice",
        vla_family="pi05",
        hardware_tier="a100",
        tether_version="0.8.0",
    )
    d = p.to_dict()
    assert d["schema_version"] == HEARTBEAT_SCHEMA_VERSION == 1
    assert d["license_id"] == "cust_alice"
    assert "org_hash" in d
    assert d["workload"] == {"vla_family": "pi05", "hardware_tier": "a100"}
    assert d["tether_version"] == "0.8.0"
    assert d["timestamp"].endswith("Z")


def test_org_hash_is_anonymized() -> None:
    """org_hash is a SHA256 prefix; never contains the raw customer_id."""
    p = build_payload(
        customer_id="cust_alice",
        vla_family="pi05",
        hardware_tier="a100",
        tether_version="0.8.0",
    )
    assert "alice" not in p.org_hash
    assert len(p.org_hash) == 16


def test_emit_skips_when_TETHER_NO_TELEMETRY_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-out env var prevents any HTTP call."""
    monkeypatch.setenv("TETHER_NO_TELEMETRY", "1")
    with patch("httpx.post") as mock_post:
        result = emit(customer_id="cust_alice")
        mock_post.assert_not_called()
    assert result is False


def test_emit_skips_for_empty_customer_id() -> None:
    """Free-tier (no license) calls emit with no customer_id; must no-op."""
    with patch("httpx.post") as mock_post:
        result = emit(customer_id="")
        mock_post.assert_not_called()
    assert result is False


def test_emit_honors_onboarding_opt_out(_isolated_tether_home) -> None:
    """The live emit() path must honor the onboarding/config opt-out flag.

    Regression: this path used to check only TETHER_NO_TELEMETRY, so a user who
    answered "no" at first-run (or `tether config set telemetry off`) kept
    emitting. Only the dead emit_free path honored the flag.
    """
    import json

    (_isolated_tether_home / "onboarding.json").write_text(
        json.dumps({"telemetry_enabled": False})
    )
    with patch("httpx.post") as mock_post:
        result = emit(customer_id="cust_alice", tether_version="0.12.0")
        mock_post.assert_not_called()
    assert result is False


def test_emit_proceeds_when_onboarding_opts_in(_isolated_tether_home) -> None:
    """telemetry_enabled=True (or absent) lets the heartbeat through."""
    import json

    (_isolated_tether_home / "onboarding.json").write_text(
        json.dumps({"telemetry_enabled": True})
    )
    fake_response = MagicMock()
    fake_response.status_code = 204
    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = emit(customer_id="cust_alice", tether_version="0.12.0")
    assert result is True
    mock_post.assert_called_once()


def test_emit_swallows_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry failure must NEVER raise to the caller."""
    monkeypatch.delenv("TETHER_NO_TELEMETRY", raising=False)
    with patch("httpx.post", side_effect=Exception("network down")):
        # Must not raise
        result = emit(customer_id="cust_alice")
    assert result is False


def test_emit_returns_true_on_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful 2xx returns True."""
    monkeypatch.delenv("TETHER_NO_TELEMETRY", raising=False)
    fake_response = MagicMock()
    fake_response.status_code = 204
    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = emit(customer_id="cust_alice", tether_version="0.8.0")
    assert result is True
    mock_post.assert_called_once()
    # Verify the payload shape sent to the endpoint
    _, kwargs = mock_post.call_args
    assert "json" in kwargs
    payload = kwargs["json"]
    assert payload["license_id"] == "cust_alice"
    assert payload["schema_version"] == 1


def test_emit_returns_false_on_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-2xx response returns False but doesn't raise."""
    monkeypatch.delenv("TETHER_NO_TELEMETRY", raising=False)
    fake_response = MagicMock()
    fake_response.status_code = 503
    with patch("httpx.post", return_value=fake_response):
        result = emit(customer_id="cust_alice")
    assert result is False


def test_emit_uses_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without explicit endpoint or env override, uses DEFAULT_TELEMETRY_ENDPOINT."""
    monkeypatch.delenv("TETHER_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("TETHER_TELEMETRY_ENDPOINT", raising=False)
    fake_response = MagicMock()
    fake_response.status_code = 204
    with patch("httpx.post", return_value=fake_response) as mock_post:
        emit(customer_id="cust_alice")
    args, _ = mock_post.call_args
    assert args[0] == DEFAULT_TELEMETRY_ENDPOINT


def test_emit_respects_endpoint_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """TETHER_TELEMETRY_ENDPOINT env var takes precedence over default."""
    monkeypatch.delenv("TETHER_NO_TELEMETRY", raising=False)
    monkeypatch.setenv("TETHER_TELEMETRY_ENDPOINT", "https://my.custom.endpoint/v1")
    fake_response = MagicMock()
    fake_response.status_code = 204
    with patch("httpx.post", return_value=fake_response) as mock_post:
        emit(customer_id="cust_alice")
    args, _ = mock_post.call_args
    assert args[0] == "https://my.custom.endpoint/v1"


@pytest.mark.parametrize("opt_out_value", ["1", "true", "TRUE", "yes", "on"])
def test_opt_out_env_var_accepts_common_truthy_values(
    monkeypatch: pytest.MonkeyPatch, opt_out_value: str,
) -> None:
    """Opt-out accepts any common truthy value, not just '1'."""
    monkeypatch.setenv("TETHER_NO_TELEMETRY", opt_out_value)
    with patch("httpx.post") as mock_post:
        emit(customer_id="cust_alice")
        mock_post.assert_not_called()
