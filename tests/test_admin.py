"""Tests for the admin CLI (license issuance / listing / revocation).

These exist because the prior `sys.exit`-from-library design made admin/
untestable. The refactor raises AdminError instead, so we can assert on
behavior here without spawning subprocesses.
"""
from __future__ import annotations

import httpx
import pytest

from tether.admin import issue_license, list_licenses, revoke_license
from tether.admin._client import AdminError, admin_request, get_admin_token


class _FakeResp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


# ---- _client.get_admin_token --------------------------------------------

def test_get_admin_token_missing_raises(monkeypatch):
    monkeypatch.delenv("TETHER_ADMIN_TOKEN", raising=False)
    with pytest.raises(AdminError) as ei:
        get_admin_token()
    assert ei.value.exit_code == 2
    assert "TETHER_ADMIN_TOKEN" in str(ei.value)


def test_get_admin_token_present(monkeypatch):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "  secret  ")
    assert get_admin_token() == "secret"


# ---- _client.admin_request ----------------------------------------------

def test_unsupported_method_is_not_a_network_error(monkeypatch):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")
    with pytest.raises(AdminError) as ei:
        admin_request("DELETE", "/admin/whatever")
    # The bug was this being reported as "could not reach worker".
    assert "Unsupported HTTP method" in str(ei.value)
    assert "reach worker" not in str(ei.value)


def test_success_returns_json(monkeypatch):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(200, {"licenses": []}))
    assert admin_request("GET", "/admin/list") == {"licenses": []}


def test_401_raises_auth_error(monkeypatch):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(401))
    with pytest.raises(AdminError) as ei:
        admin_request("POST", "/admin/issue", {})
    assert "401" in str(ei.value)


def test_4xx_raises(monkeypatch):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(500, {"error": "boom"}))
    with pytest.raises(AdminError):
        admin_request("POST", "/admin/issue", {})


def test_network_error_wrapped(monkeypatch):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    with pytest.raises(AdminError) as ei:
        admin_request("GET", "/admin/list")
    assert "reach worker" in str(ei.value)


def test_insecure_endpoint_warns(monkeypatch, capsys):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")
    monkeypatch.setenv("TETHER_LICENSE_ENDPOINT", "http://license.example.com")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(200, {}))
    admin_request("GET", "/admin/list")
    assert "non-HTTPS" in capsys.readouterr().err


def test_localhost_http_does_not_warn(monkeypatch, capsys):
    monkeypatch.setenv("TETHER_ADMIN_TOKEN", "t")
    monkeypatch.setenv("TETHER_LICENSE_ENDPOINT", "http://127.0.0.1:8787")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(200, {}))
    admin_request("GET", "/admin/list")
    assert "non-HTTPS" not in capsys.readouterr().err


# ---- command main() exit codes ------------------------------------------

def test_issue_license_admin_error_returns_code(monkeypatch):
    def _raise(*a, **k):
        raise AdminError("nope", exit_code=2)

    monkeypatch.setattr(issue_license, "admin_request", _raise)
    rc = issue_license.main(["--customer-id", "a@b.com"])
    assert rc == 2


def test_issue_license_success(monkeypatch, capsys):
    monkeypatch.setattr(
        issue_license, "admin_request",
        lambda *a, **k: {"license_id": "lic_1", "activation_code": "REFLEX-AAAA",
                         "license": {"expires_at": "2026-12-01"}},
    )
    rc = issue_license.main(["--customer-id", "a@b.com", "--tier", "pro"])
    assert rc == 0
    assert "lic_1" in capsys.readouterr().out


def test_issue_license_bad_max_seats(monkeypatch):
    assert issue_license.main(["--customer-id", "a@b.com", "--max-seats", "0"]) == 2


def test_list_licenses_bad_limit():
    assert list_licenses.main(["--limit", "0"]) == 2


def test_revoke_license_admin_error_returns_code(monkeypatch):
    def _raise(*a, **k):
        raise AdminError("gone", exit_code=2)

    monkeypatch.setattr(revoke_license, "admin_request", _raise)
    assert revoke_license.main(["--license-id", "lic_x"]) == 2
