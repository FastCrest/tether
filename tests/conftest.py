"""Pytest fixtures shared across the test suite.

CI-only behavior of Click+Rich's help renderer that broke us:
- When `CI=true` (set by GitHub Actions), Rich inserts ANSI color
  escapes BETWEEN the two `-` characters of long flag names, so the
  literal substring `"--export-mode"` is no longer present in
  `result.output` even though the rendered text looks identical.
  Setting `TERM=dumb` forces Rich into no-color plain-text mode and
  the substring search works again.
- The CI runner's terminal width defaults narrow enough that Rich
  elides long option names with `--...`. `COLUMNS=200` keeps the
  rendered panel wide enough to show all names verbatim.

Both env vars are set in `pytest_configure` (runs before any test
import), so the entire suite inherits stable, plain-text help output
matching what `--help` users see in a real shell.
"""

from __future__ import annotations

import os
import socket

import pytest


def pytest_configure(config):
    os.environ["COLUMNS"] = "200"
    os.environ["LINES"] = "80"
    os.environ["TERM"] = "dumb"


@pytest.fixture(autouse=True)
def protect_live_contribution_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real contribution services outside unit-test transport reach.

    Mock transports and loopback remain usable. A deliberate operator-owned
    integration drill must explicitly opt in; CI never sets this switch.
    Blocking DNS protects httpx, requests and urllib without forcing a client.
    """
    if os.environ.get("TETHER_TEST_ALLOW_LIVE_CONTRIBUTIONS") == "1":
        return
    from urllib.parse import urlsplit

    from tether import contributor_auth

    original = socket.getaddrinfo
    original_transport = contributor_auth._default_transport
    blocked = {
        "reflex-contributions.fastcrest.workers.dev",
        "reflex-data.fastcrest.workers.dev",
    }

    def check_host(host):
        normalized = host.decode("ascii") if isinstance(host, bytes) else str(host)
        if normalized.rstrip(".").lower() in blocked:
            raise RuntimeError(
                "Unit tests must use an injected contribution transport or loopback; "
                "real contribution services require explicit "
                "TETHER_TEST_ALLOW_LIVE_CONTRIBUTIONS=1"
            )

    def guarded(host, *args, **kwargs):
        check_host(host)
        return original(host, *args, **kwargs)

    def guarded_transport(method, url, body, headers, timeout):
        # Check the request URL before proxy resolution as well as direct DNS.
        check_host(urlsplit(url).hostname)
        return original_transport(method, url, body, headers, timeout)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)
    monkeypatch.setattr(contributor_auth, "_default_transport", guarded_transport)
