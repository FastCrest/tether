"""Telemetry heartbeat — opt-out, anonymized. Pro and free tiers.

Sends a small JSON heartbeat to the FastCrest telemetry endpoint once per
24h. Pro-tier heartbeats include the license customer_id. Free-tier
heartbeats use license_id="free" and org_hash=SHA256(machine_fingerprint)[:16].

Privacy posture (locked Phase 1)
--------------------------------
What gets sent:

- ``license_id``   — the JWT ``customer_id`` field (Pro), or "free" (free tier).
- ``org_hash``     — ``SHA256(customer_id)[:16]`` (Pro) or
                     ``SHA256(machine_fingerprint)[:16]`` (free).
- ``workload``     — ``{vla_family, hardware_tier}``. Aggregated stats only.
- ``tether_version`` — ``__version__``. Version-distribution tracking.
- ``timestamp``    — ISO-8601 UTC. Recency / heartbeat staleness only.
- ``tier``         — "pro" or "free". Distinguishes license tiers.

Free-tier additional fields (all optional, aggregated stats only):
- ``model_name``, ``hardware_detail``, ``latency_p50/p95/p99``,
  ``error_count_24h``, ``safety_violation_count_24h``, ``episode_count_24h``,
  ``action_dim``, ``embodiment``, ``denoise_steps``, ``inference_mode``

What does NOT get sent:

- /act payloads (images, instructions, state)
- Robot trajectories or actions
- Model weights or embeddings
- Customer org name (only the SHA256 tag)
- IP addresses (Cloudflare Worker logs Cf-Connecting-IP separately; we
  do not write it to telemetry storage)

Opt-out
-------
Default behavior: telemetry ON for both Pro and free tiers.

Users can disable via:

- ``TETHER_NO_TELEMETRY=1`` environment variable
- ``--no-telemetry`` CLI flag (Phase 1.5 wiring)
- ``tether config set telemetry off``
- Onboarding prompt (free tier)
- License-level opt-out (Phase 2, Pro only)

Endpoint
--------
``https://telemetry.fastcrest.workers.dev/v1/heartbeat`` (Cloudflare
Worker; backend stub at ``infra/telemetry-worker/``). Override via
``TETHER_TELEMETRY_ENDPOINT`` env var for testing or air-gapped
deployments.

Failure mode
------------
Network unreachable / endpoint down → fail silently. Never blocks the
CLI or license heartbeat path. Telemetry failure logs at DEBUG level only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Override via TETHER_TELEMETRY_ENDPOINT for testing / air-gapped deploys.
DEFAULT_TELEMETRY_ENDPOINT = "https://tether-telemetry.fastcrest.workers.dev/v1/heartbeat"

# Two-second timeout. Telemetry is best-effort; we never block startup
# on a slow telemetry endpoint.
_REQUEST_TIMEOUT_S = 2.0

# Schema version. Bumped on a breaking field-shape change. Worker stub
# accepts v1 only and rejects future versions with HTTP 400 so old
# clients see a clean failure rather than silently wrong data.
HEARTBEAT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeartbeatPayload:
    """The exact payload sent to the telemetry endpoint. Locked Phase 1.

    See module docstring for the privacy posture and field-by-field
    explanations.
    """

    schema_version: int
    license_id: str
    org_hash: str
    workload: dict[str, str]
    tether_version: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_disabled() -> bool:
    """True iff the user has opted out of telemetry."""
    return os.environ.get("TETHER_NO_TELEMETRY", "").lower() in ("1", "true", "yes", "on")


def _org_hash(customer_id: str) -> str:
    """SHA256[:16] of customer_id. Anonymized counting tag."""
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:16]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_payload(
    *,
    customer_id: str,
    vla_family: str,
    hardware_tier: str,
    tether_version: str,
) -> HeartbeatPayload:
    """Build (but don't send) a heartbeat payload.

    Separated from emit() so tests can inspect the exact payload without
    triggering an HTTP call.
    """
    return HeartbeatPayload(
        schema_version=HEARTBEAT_SCHEMA_VERSION,
        license_id=customer_id,
        org_hash=_org_hash(customer_id),
        workload={"vla_family": vla_family, "hardware_tier": hardware_tier},
        tether_version=tether_version,
        timestamp=_utc_now_iso(),
    )


def emit(
    *,
    customer_id: str,
    vla_family: str = "unknown",
    hardware_tier: str = "unknown",
    tether_version: str = "unknown",
    endpoint: str | None = None,
) -> bool:
    """Send a heartbeat to the telemetry endpoint.

    Returns True if the heartbeat was POSTed and the endpoint returned 2xx,
    False on opt-out / network failure / non-2xx. Never raises.

    Caller is responsible for invoking this no more than once per 24h
    (the license heartbeat path in ``pro.license.load_license`` is the
    natural integration point).
    """
    if _is_disabled():
        logger.debug("Telemetry disabled via TETHER_NO_TELEMETRY env; skipping.")
        return False
    if not customer_id:
        logger.debug("Telemetry skipped: no customer_id (free tier).")
        return False

    payload = build_payload(
        customer_id=customer_id,
        vla_family=vla_family,
        hardware_tier=hardware_tier,
        tether_version=tether_version,
    )
    url = endpoint or os.environ.get("TETHER_TELEMETRY_ENDPOINT", DEFAULT_TELEMETRY_ENDPOINT)

    # Lazy httpx import — Tether's [serve] extra includes httpx; bare
    # `pip install fastcrest-tether` includes it as a base dep, so this is safe.
    try:
        import httpx
    except ImportError:
        logger.debug("Telemetry skipped: httpx not available.")
        return False

    try:
        resp = httpx.post(
            url,
            json=payload.to_dict(),
            timeout=_REQUEST_TIMEOUT_S,
            headers={"User-Agent": f"fastcrest-tether/{tether_version}"},
        )
        if 200 <= resp.status_code < 300:
            logger.debug("Telemetry heartbeat posted: %s", payload.org_hash)
            return True
        logger.debug(
            "Telemetry endpoint returned %d for %s", resp.status_code, payload.org_hash
        )
        return False
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.debug("Telemetry POST failed: %s", exc)
        return False


# ── Free-tier telemetry ────────────────────────────────────────────────
#
# Fires for ALL users (no Pro license required) if telemetry is enabled.
# Uses the same endpoint but with license_id="free" and org_hash derived
# from a machine fingerprint instead of a customer_id.
#
# Once-per-24h cache at ~/.tether/.free_telemetry_cache to avoid
# hitting the endpoint on every CLI invocation.


_FREE_TELEMETRY_CACHE_TTL = 24 * 60 * 60  # 24 hours


@dataclass(frozen=True)
class FreeHeartbeatPayload:
    """Extended heartbeat payload for free-tier telemetry.

    Includes all base fields plus deployment stats that help improve
    Tether VLA. No images, no trajectories, no PII.
    """

    schema_version: int
    license_id: str  # always "free"
    org_hash: str  # SHA256(machine_fingerprint)[:16]
    workload: dict[str, str]
    tether_version: str
    timestamp: str
    tier: str  # "free"
    model_name: str
    hardware_detail: str
    latency_p50: float | None
    latency_p95: float | None
    latency_p99: float | None
    error_count_24h: int
    safety_violation_count_24h: int
    episode_count_24h: int
    action_dim: int | None
    embodiment: str
    denoise_steps: int | None
    inference_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _machine_fingerprint() -> str:
    """Deterministic machine fingerprint for free-tier org_hash.

    Uses platform info (hostname, machine, processor) hashed together.
    No PII: the hash is one-way and truncated.
    """
    import platform
    import uuid

    parts = [
        platform.node(),
        platform.machine(),
        platform.processor(),
        platform.system(),
    ]
    # Also try MAC address for stability across reboots
    try:
        parts.append(str(uuid.getnode()))
    except Exception:
        pass
    raw = "|".join(parts)
    return raw


def _free_org_hash() -> str:
    """SHA256[:16] of machine fingerprint. Anonymized counting tag."""
    return hashlib.sha256(_machine_fingerprint().encode("utf-8")).hexdigest()[:16]


def _free_cache_path() -> Path:
    """Path to the free-tier telemetry cache file."""
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".tether"))
    home.mkdir(parents=True, exist_ok=True)
    return home / ".free_telemetry_cache"


def _free_cache_valid() -> bool:
    """True if the free-tier cache is fresh (less than 24h old)."""
    cache = _free_cache_path()
    if not cache.exists():
        return False
    try:
        data = json.loads(cache.read_text())
        return (time.time() - data.get("checked_at", 0)) < _FREE_TELEMETRY_CACHE_TTL
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _write_free_cache() -> None:
    """Write a fresh free-tier telemetry cache timestamp."""
    try:
        _free_cache_path().write_text(json.dumps({"checked_at": time.time()}))
    except OSError:
        pass


def _is_telemetry_enabled_by_onboarding() -> bool:
    """Check onboarding.json for telemetry opt-out. Returns True if enabled or missing."""
    try:
        onboarding_path = Path("~/.tether/onboarding.json").expanduser()
        if not onboarding_path.exists():
            return True  # default: telemetry on
        data = json.loads(onboarding_path.read_text())
        return bool(data.get("telemetry_enabled", True))
    except Exception:
        return True  # fail-open: telemetry on if we can't read state


def build_free_payload(
    *,
    tether_version: str = "unknown",
    model_name: str = "unknown",
    hardware_detail: str = "unknown",
    vla_family: str = "unknown",
    hardware_tier: str = "unknown",
    latency_p50: float | None = None,
    latency_p95: float | None = None,
    latency_p99: float | None = None,
    error_count_24h: int = 0,
    safety_violation_count_24h: int = 0,
    episode_count_24h: int = 0,
    action_dim: int | None = None,
    embodiment: str = "unknown",
    denoise_steps: int | None = None,
    inference_mode: str = "unknown",
) -> FreeHeartbeatPayload:
    """Build (but don't send) a free-tier heartbeat payload.

    Separated from emit_free() so tests can inspect the payload without
    triggering an HTTP call.
    """
    return FreeHeartbeatPayload(
        schema_version=HEARTBEAT_SCHEMA_VERSION,
        license_id="free",
        org_hash=_free_org_hash(),
        workload={"vla_family": vla_family, "hardware_tier": hardware_tier},
        tether_version=tether_version,
        timestamp=_utc_now_iso(),
        tier="free",
        model_name=model_name,
        hardware_detail=hardware_detail,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        latency_p99=latency_p99,
        error_count_24h=error_count_24h,
        safety_violation_count_24h=safety_violation_count_24h,
        episode_count_24h=episode_count_24h,
        action_dim=action_dim,
        embodiment=embodiment,
        denoise_steps=denoise_steps,
        inference_mode=inference_mode,
    )


def emit_free(
    *,
    tether_version: str = "unknown",
    model_name: str = "unknown",
    hardware_detail: str = "unknown",
    vla_family: str = "unknown",
    hardware_tier: str = "unknown",
    latency_p50: float | None = None,
    latency_p95: float | None = None,
    latency_p99: float | None = None,
    error_count_24h: int = 0,
    safety_violation_count_24h: int = 0,
    episode_count_24h: int = 0,
    action_dim: int | None = None,
    embodiment: str = "unknown",
    denoise_steps: int | None = None,
    inference_mode: str = "unknown",
    endpoint: str | None = None,
) -> bool:
    """Send a free-tier heartbeat to the telemetry endpoint.

    Returns True if the heartbeat was POSTed and the endpoint returned 2xx,
    False on opt-out / network failure / non-2xx / cache-fresh. Never raises.

    Does NOT require a Pro license. Fires for ALL users if telemetry enabled.
    Once-per-24h via cache file.
    """
    if _is_disabled():
        logger.debug("Free telemetry disabled via TETHER_NO_TELEMETRY env; skipping.")
        return False

    if not _is_telemetry_enabled_by_onboarding():
        logger.debug("Free telemetry disabled via onboarding; skipping.")
        return False

    if _free_cache_valid():
        logger.debug("Free telemetry cache fresh; skipping.")
        return False

    payload = build_free_payload(
        tether_version=tether_version,
        model_name=model_name,
        hardware_detail=hardware_detail,
        vla_family=vla_family,
        hardware_tier=hardware_tier,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        latency_p99=latency_p99,
        error_count_24h=error_count_24h,
        safety_violation_count_24h=safety_violation_count_24h,
        episode_count_24h=episode_count_24h,
        action_dim=action_dim,
        embodiment=embodiment,
        denoise_steps=denoise_steps,
        inference_mode=inference_mode,
    )
    url = endpoint or os.environ.get("TETHER_TELEMETRY_ENDPOINT", DEFAULT_TELEMETRY_ENDPOINT)

    try:
        import httpx
    except ImportError:
        # Fall back to stdlib urllib
        try:
            import urllib.request
            import urllib.error

            data = json.dumps(payload.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"fastcrest-tether/{tether_version}",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
            if 200 <= resp.status < 300:
                _write_free_cache()
                logger.debug("Free telemetry heartbeat posted (urllib): %s", payload.org_hash)
                return True
            return False
        except Exception as exc:
            logger.debug("Free telemetry POST failed (urllib): %s", exc)
            return False

    try:
        resp = httpx.post(
            url,
            json=payload.to_dict(),
            timeout=_REQUEST_TIMEOUT_S,
            headers={"User-Agent": f"fastcrest-tether/{tether_version}"},
        )
        if 200 <= resp.status_code < 300:
            _write_free_cache()
            logger.debug("Free telemetry heartbeat posted: %s", payload.org_hash)
            return True
        logger.debug(
            "Free telemetry endpoint returned %d for %s", resp.status_code, payload.org_hash
        )
        return False
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.debug("Free telemetry POST failed: %s", exc)
        return False


__all__ = [
    "DEFAULT_TELEMETRY_ENDPOINT",
    "HEARTBEAT_SCHEMA_VERSION",
    "FreeHeartbeatPayload",
    "HeartbeatPayload",
    "build_free_payload",
    "build_payload",
    "emit",
    "emit_free",
]
