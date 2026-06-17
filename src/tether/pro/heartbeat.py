"""Daily heartbeat to the license worker.

Pro deployments POST a heartbeat to the worker's ``/v1/heartbeat``
endpoint once per 24h. The worker:

1. Checks the license is not in the revocation list. If revoked, returns
   HTTP 403 with revoked=true. The client surfaces this and refuses to
   continue serving.
2. Checks the license hasn't expired. If expired, returns HTTP 403.
3. Records the heartbeat in D1 (license_id, hardware_fingerprint,
   ip_country, tether_version, server timestamp) for sharing-detection
   queries.

Heartbeat failure modes:
- Network unreachable → log a warning, increment local fail counter.
  After ``HEARTBEAT_FRESHNESS_S`` (24h) of failures, license becomes
  stale (per ``pro/license.py``) and the server refuses to start.
- HTTP 403 revoked → log critical, raise ``LicenseRevokedError`` to halt
  the server.
- HTTP 403 expired → log critical, raise ``LicenseExpiredAtServer``.

Background thread vs sync call: this module ships a sync ``send_heartbeat()``
function. Wiring it as a background daily thread is the caller's choice
(typically a FastAPI lifespan startup task that schedules a 24h timer).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LicenseRevokedError(Exception):
    """Raised when the license worker reports revocation on heartbeat."""


class LicenseExpiredAtServer(Exception):
    """Raised when the license worker reports the license has expired."""


def send_heartbeat(
    *,
    license_id: str,
    hardware_fingerprint: str,
    tether_version: str,
    endpoint: str | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """POST a heartbeat. Returns the worker's JSON response on success.

    Raises:
        LicenseRevokedError: worker reports revocation
        LicenseExpiredAtServer: worker reports expiry
        Exception: network / HTTP failure (caller decides how to react;
                   typical pattern is "log + ignore" since cached license
                   is still valid until heartbeat-freshness window expires)
    """
    from tether.pro.activate import DEFAULT_LICENSE_ENDPOINT

    url = (endpoint or os.environ.get("TETHER_LICENSE_ENDPOINT", DEFAULT_LICENSE_ENDPOINT)).rstrip("/")
    full_url = f"{url}/v1/heartbeat"

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available; heartbeat skipped")
        return {"sent": False, "reason": "no_httpx"}

    payload = {
        "license_id": license_id,
        "hardware_fingerprint": hardware_fingerprint,
        "tether_version": tether_version,
    }

    try:
        resp = httpx.post(full_url, json=payload, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        # Network failure — caller decides what to do. The cached license
        # stays valid until HEARTBEAT_FRESHNESS_S elapses since the last
        # successful heartbeat (handled in pro/license.py).
        logger.warning("Heartbeat to %s failed: %s", url, exc)
        raise

    if resp.status_code == 403:
        body = _safe_json(resp)
        if body.get("revoked"):
            raise LicenseRevokedError(
                f"License {license_id} was revoked at "
                f"{body.get('revoked_at', 'unknown')}: "
                f"{body.get('reason', 'no reason given')}"
            )
        if body.get("expired"):
            raise LicenseExpiredAtServer(
                f"License {license_id} expired at {body.get('expires_at', 'unknown')}"
            )

    if resp.status_code != 200:
        # Treat non-2xx other than 403 as a soft failure. Caller logs + retries.
        logger.warning("Heartbeat returned HTTP %d: %s", resp.status_code, resp.text[:200])
        raise RuntimeError(f"Heartbeat HTTP {resp.status_code}")

    return _safe_json(resp)


def _safe_json(resp: Any) -> dict[str, Any]:
    try:
        out = resp.json()
        return out if isinstance(out, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "LicenseExpiredAtServer",
    "LicenseRevokedError",
    "send_heartbeat",
]
