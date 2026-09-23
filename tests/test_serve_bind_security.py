"""serve/go bind to localhost by default + warn on insecure public binds."""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from tether import cli


def _default(func, param: str):
    p = inspect.signature(func).parameters[param].default
    # typer.Option(...) returns an OptionInfo whose .default holds the value.
    return getattr(p, "default", p)


def test_serve_and_go_default_host_is_localhost():
    assert _default(cli.serve, "host") == "127.0.0.1"
    assert _default(cli.go, "host") == "127.0.0.1"


def test_warns_on_public_bind_without_api_key(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(cli, "err_console", mock)
    cli._warn_insecure_bind("0.0.0.0", "")
    assert mock.print.called


def test_no_warn_on_loopback(monkeypatch):
    for host in ("127.0.0.1", "localhost", "::1"):
        mock = MagicMock()
        monkeypatch.setattr(cli, "err_console", mock)
        cli._warn_insecure_bind(host, "")
        assert not mock.print.called, host


def test_no_warn_when_api_key_set(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(cli, "err_console", mock)
    cli._warn_insecure_bind("0.0.0.0", "a-secret-key")
    assert not mock.print.called


def test_warns_on_arbitrary_public_ip(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(cli, "err_console", mock)
    cli._warn_insecure_bind("192.168.1.10", "")
    assert mock.print.called
