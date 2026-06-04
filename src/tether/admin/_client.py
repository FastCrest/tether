"""Shared HTTP client for admin commands."""
from __future__ import annotations

import os
import sys
from typing import Any

DEFAULT_LICENSE_ENDPOINT = "https://tether-licenses.fastcrest.workers.dev"


def get_endpoint() -> str:
    return os.environ.get("TETHER_LICENSE_ENDPOINT", DEFAULT_LICENSE_ENDPOINT).rstrip("/")


def get_admin_token() -> str:
    token = os.environ.get("TETHER_ADMIN_TOKEN", "").strip()
    if not token:
        sys.stderr.write(
            "ERROR: TETHER_ADMIN_TOKEN env var is not set.\n"
            "  Set it to the bearer token you configured on the worker via\n"
            "  `wrangler secret put ADMIN_TOKEN`.\n"
        )
        sys.exit(2)
    return token


def admin_request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make an authenticated admin request to the license worker.

    Exits the process with code 2 on auth/network/HTTP errors so admin scripts
    fail loudly.
    """
    try:
        import httpx
    except ImportError:
        sys.stderr.write("ERROR: httpx not installed. Run: pip install httpx\n")
        sys.exit(2)

    url = f"{get_endpoint()}{path}"
    headers = {"Authorization": f"Bearer {get_admin_token()}"}

    try:
        if method == "GET":
            resp = httpx.get(url, headers=headers, timeout=15.0)
        elif method == "POST":
            resp = httpx.post(url, headers=headers, json=body or {}, timeout=15.0)
        else:
            raise ValueError(f"Unsupported method: {method}")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: Could not reach worker at {url}: {exc}\n")
        sys.exit(2)

    if resp.status_code == 401:
        sys.stderr.write(
            "ERROR: 401 Unauthorized. Check that TETHER_ADMIN_TOKEN matches\n"
            "  the value set via `wrangler secret put ADMIN_TOKEN`.\n"
        )
        sys.exit(2)

    try:
        body_json = resp.json()
    except Exception:  # noqa: BLE001
        body_json = {"raw": resp.text}

    if resp.status_code >= 400:
        sys.stderr.write(f"ERROR: Worker returned HTTP {resp.status_code}: {body_json}\n")
        sys.exit(2)

    return body_json


__all__ = ["DEFAULT_LICENSE_ENDPOINT", "admin_request", "get_admin_token", "get_endpoint"]
