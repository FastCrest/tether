"""Shared HTTP client for admin commands."""
from __future__ import annotations

import os
import sys
from typing import Any

DEFAULT_LICENSE_ENDPOINT = "https://tether-licenses.fastcrest.workers.dev"

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


class AdminError(Exception):
    """An admin command failed (auth / network / HTTP / bad input).

    Carries the process exit code so a command's ``main()`` can ``return
    exc.exit_code`` instead of the library calling ``sys.exit`` directly.
    Raising rather than exiting is what makes the admin commands unit-testable
    and reusable as a library.
    """

    def __init__(self, message: str, *, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def get_endpoint() -> str:
    return os.environ.get("TETHER_LICENSE_ENDPOINT", DEFAULT_LICENSE_ENDPOINT).rstrip("/")


def get_admin_token() -> str:
    token = os.environ.get("TETHER_ADMIN_TOKEN", "").strip()
    if not token:
        raise AdminError(
            "TETHER_ADMIN_TOKEN env var is not set.\n"
            "  Set it to the bearer token you configured on the worker via\n"
            "  `wrangler secret put ADMIN_TOKEN`."
        )
    return token


def _warn_insecure_endpoint(url: str) -> None:
    """Warn if the admin bearer token would go out over cleartext HTTP.

    Non-HTTPS to a non-local host sends the admin secret in the clear; surface
    it loudly rather than silently leaking the token.
    """
    if url.startswith("https://"):
        return
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    if host in _LOCAL_HOSTS:
        return
    sys.stderr.write(
        f"WARNING: sending the admin token over a non-HTTPS endpoint ({url}).\n"
        "  The bearer token is transmitted in cleartext. Use an https:// "
        "TETHER_LICENSE_ENDPOINT for anything but local testing.\n"
    )


def admin_request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make an authenticated admin request to the license worker.

    Raises ``AdminError`` (exit_code 2) on auth / network / HTTP / unsupported-
    method errors so admin commands fail loudly while staying testable.
    """
    method = method.upper()
    if method not in ("GET", "POST"):
        # Raise BEFORE the network try-block so a programmer error isn't
        # misreported as "could not reach worker".
        raise AdminError(f"Unsupported HTTP method: {method!r}")

    try:
        import httpx
    except ImportError as exc:
        raise AdminError("httpx not installed. Run: pip install httpx") from exc

    url = f"{get_endpoint()}{path}"
    _warn_insecure_endpoint(url)
    headers = {"Authorization": f"Bearer {get_admin_token()}"}

    try:
        if method == "GET":
            resp = httpx.get(url, headers=headers, timeout=15.0)
        else:  # POST
            resp = httpx.post(url, headers=headers, json=body or {}, timeout=15.0)
    except httpx.HTTPError as exc:
        raise AdminError(f"Could not reach worker at {url}: {exc}") from exc

    if resp.status_code == 401:
        raise AdminError(
            "401 Unauthorized. Check that TETHER_ADMIN_TOKEN matches\n"
            "  the value set via `wrangler secret put ADMIN_TOKEN`."
        )

    try:
        body_json = resp.json()
    except Exception:  # noqa: BLE001 — worker returned non-JSON
        body_json = {"raw": resp.text}

    if resp.status_code >= 400:
        raise AdminError(f"Worker returned HTTP {resp.status_code}: {body_json}")

    return body_json


__all__ = [
    "DEFAULT_LICENSE_ENDPOINT",
    "AdminError",
    "admin_request",
    "get_admin_token",
    "get_endpoint",
]
