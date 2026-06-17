"""Customer-side `tether pro activate <code>` flow.

The customer receives an activation code from the operator (via Discord,
DM, email — whatever the out-of-band channel is). Running
``tether pro activate <code>`` does the following:

1. POSTs nothing — the activation endpoint is a GET so the code can be
   shared as a URL too.
2. GETs ``<endpoint>/v1/activation/<code>`` to fetch the signed license JSON.
3. Verifies the Ed25519 signature using the bundled public key
   (``tether.pro._public_key``).
4. Captures the local hardware fingerprint and writes it into the
   ``hardware_binding`` field if the license is unbound (first activation).
5. Writes the resulting license to ``~/.tether/pro.license`` with mode 0600.
6. Prints a welcome message + telemetry-opt-out reminder.

Failure modes:
- Code expired / used / not found → 410 / 404 from the worker
- Signature verification fails → ``LicenseSignatureError`` propagated
- Disk write fails → propagated
- Endpoint unreachable → ``ActivationNetworkError``
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Override via TETHER_LICENSE_ENDPOINT for testing or custom deployments.
DEFAULT_LICENSE_ENDPOINT = "https://tether-licenses.fastcrest.workers.dev"

DEFAULT_LICENSE_PATH = "~/.tether/pro.license"


class ActivationError(Exception):
    """Base for activation-flow errors."""


class ActivationNetworkError(ActivationError):
    """The activation endpoint was unreachable."""


class ActivationCodeError(ActivationError):
    """The activation code was rejected (not found / expired / used)."""


def probe_hardware_binding() -> dict[str, Any]:
    """Probe the running host and return the dict that goes into license.hardware_binding.

    Maps from the richer ``runtime.calibration.HardwareFingerprint.current()``
    down to the ``HardwareFingerprintLite`` shape that the license validates against
    (gpu_uuid, gpu_name, cpu_count). Always returns a valid dict; unknown
    fields populated with sentinels (matches HardwareFingerprint.current()'s
    promise).
    """
    try:
        from tether.runtime.calibration import HardwareFingerprint
        fp = HardwareFingerprint.current()
        return {
            "gpu_uuid": fp.gpu_uuid,
            "gpu_name": fp.gpu_name,
            "cpu_count": fp.cpu_count,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Hardware probe failed (%s); using minimal fingerprint. "
            "License will fail to validate at server-start if hardware "
            "doesn't match — operator can rebind.", exc,
        )
        return {
            "gpu_uuid": "unknown",
            "gpu_name": "unknown",
            "cpu_count": os.cpu_count() or 1,
        }


def heartbeat_fingerprint() -> str:
    """Compute a stable 32-char hex hash of the hardware identity.

    Used as ``hardware_fingerprint`` in heartbeat payloads. Stable across
    reboots; identical for the same physical host. Hashed (not the raw
    fields) so the worker doesn't store identifying info beyond what's
    necessary for sharing-detection.
    """
    hw = probe_hardware_binding()
    canonical = f"{hw['gpu_uuid']}|{hw['gpu_name']}|{hw['cpu_count']}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def activate_license(
    code: str,
    *,
    endpoint: str | None = None,
    license_path: str | Path = DEFAULT_LICENSE_PATH,
) -> dict[str, Any]:
    """Fetch + verify + persist the license for an activation code.

    Returns the activated license dict on success. Raises one of the
    ``ActivationError`` subclasses (or ``LicenseSignatureError`` from
    ``tether.pro.signature``) on failure.
    """
    from tether.pro.signature import verify_license_signature

    code = code.strip()
    if not code.startswith("REFLEX-"):
        raise ActivationCodeError(
            f"Activation code must start with REFLEX-, got {code!r}. "
            f"Make sure you copied the entire code as sent."
        )

    url = (endpoint or os.environ.get("TETHER_LICENSE_ENDPOINT", DEFAULT_LICENSE_ENDPOINT)).rstrip("/")
    full_url = f"{url}/v1/activation/{code}"

    try:
        import httpx
    except ImportError as exc:
        raise ActivationError(
            "httpx is required for activation. Install with: pip install httpx"
        ) from exc

    try:
        resp = httpx.get(full_url, timeout=15.0)
    except Exception as exc:  # noqa: BLE001
        raise ActivationNetworkError(
            f"Could not reach license endpoint at {url}: {exc}. "
            f"Check your network and try again."
        ) from exc

    if resp.status_code == 404:
        raise ActivationCodeError(
            f"Activation code {code} was not found. Either it was mistyped, "
            f"or it expired (24h TTL). Ask the operator to issue a new one."
        )
    if resp.status_code == 410:
        body = _safe_json(resp)
        reason = body.get("error", "unknown")
        raise ActivationCodeError(
            f"Activation code {code} can no longer be used: {reason}. "
            f"Codes are single-use and expire after 24h. Ask for a new one."
        )
    if resp.status_code != 200:
        raise ActivationError(
            f"License endpoint returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    body = _safe_json(resp)
    license_dict = body.get("license")
    if not isinstance(license_dict, dict):
        raise ActivationError(f"Endpoint returned unexpected payload: {body!r}")

    # Verify signature BEFORE doing anything stateful (writing to disk).
    verify_license_signature(license_dict)

    # Bind hardware on first activation. If the license already has a
    # hardware_binding (e.g., re-activation after a worker round-trip), don't
    # overwrite — the operator will issue a new license to rebind.
    if license_dict.get("hardware_binding") is None:
        license_dict["hardware_binding"] = probe_hardware_binding()

    # Stamp the local heartbeat so license.py's heartbeat-staleness check passes.
    license_dict["last_heartbeat_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    # Write to disk with mode 0600.
    path = Path(license_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(license_dict, indent=2, sort_keys=True))
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Non-fatal: filesystems without unix perms (e.g., FAT32 on edge devices)
        pass

    logger.info(
        "Activated license %s for customer %s (tier=%s, expires %s)",
        license_dict.get("license_id"), license_dict.get("customer_id"),
        license_dict.get("tier"), license_dict.get("expires_at"),
    )
    return license_dict


def _safe_json(resp: Any) -> dict[str, Any]:
    try:
        out = resp.json()
        return out if isinstance(out, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "DEFAULT_LICENSE_ENDPOINT",
    "DEFAULT_LICENSE_PATH",
    "ActivationCodeError",
    "ActivationError",
    "ActivationNetworkError",
    "activate_license",
    "heartbeat_fingerprint",
    "probe_hardware_binding",
]
