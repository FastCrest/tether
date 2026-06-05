"""Contract helpers for Tether -> FastCrest Cloud burst validation.

The helpers in this module build and validate the dry-run handoff payload used
before online Cloud GPU burst is enabled. They do not proxy `/act`, execute a
remote model, or send raw local files.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping
from urllib import request as urllib_request
from urllib.parse import urlparse


BURST_SCHEMA_VERSION = "cloud_burst_validation.v0"
DEFAULT_TIMEOUT_S = 2.0

_SENSITIVE_KEY_PARTS = {
    "authorization",
    "api_key",
    "bearer",
    "credential",
    "device_token",
    "password",
    "plaintext",
    "secret",
    "signed_url",
    "token",
}
_RAW_KEY_PARTS = {
    "action_data",
    "action_trace",
    "camera",
    "frame",
    "image",
    "joint_positions",
    "local_path",
    "raw_action",
    "raw_observation",
    "video",
}
_SIGNED_URL_QUERY_PARTS = {
    "awsaccesskeyid",
    "expires",
    "signature",
    "token",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-signature",
}
_LOCAL_PATH_RE = re.compile(r"^(?:/Users/|/home/|/var/|/tmp/|/private/|/Volumes/|file://|~/|[A-Za-z]:\\)")


class CloudBurstContractError(ValueError):
    """Raised when a burst request/response violates the v0 contract."""


Transport = Callable[[str, dict[str, Any], dict[str, str], float], Mapping[str, Any]]


def build_burst_validation_request(
    *,
    request_id: str,
    route_decision_id: str,
    device_id: str,
    payload_hash: str,
    payload_manifest: Mapping[str, Any],
    workload_type: str = "vla_action",
    privacy_mode: str = "standard",
    cloud_allowed: bool = True,
    artifact_identity: Mapping[str, Any] | None = None,
    latency_budget_ms: float | None = None,
    max_cost_usd: float | None = None,
    cloud_backend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the bounded payload sent to Cloud's burst validator."""

    if not _required(request_id):
        raise CloudBurstContractError("missing request_id")
    if not _required(route_decision_id):
        raise CloudBurstContractError("missing route_decision_id")
    if not _required(device_id):
        raise CloudBurstContractError("missing device_id")
    if not _required(payload_hash):
        raise CloudBurstContractError("missing payload_hash")
    if _normalize_privacy(privacy_mode) == "edge_only":
        raise CloudBurstContractError("privacy_edge_only")
    if not bool(cloud_allowed):
        raise CloudBurstContractError("cloud_disallowed")

    manifest = dict(payload_manifest)
    sanitized_manifest = redact_for_burst(manifest)
    if sanitized_manifest != manifest:
        raise CloudBurstContractError("payload_manifest_unsafe")

    body: dict[str, Any] = {
        "schema_version": BURST_SCHEMA_VERSION,
        "request_id": request_id,
        "route_decision_id": route_decision_id,
        "device_id": device_id,
        "workload_type": workload_type,
        "privacy_mode": privacy_mode,
        "cloud_allowed": True,
        "payload_hash": payload_hash,
        "payload_manifest": sanitized_manifest,
        "request_hash": _hash_payload(
            {
                "request_id": request_id,
                "route_decision_id": route_decision_id,
                "device_id": device_id,
                "payload_hash": payload_hash,
                "payload_manifest": sanitized_manifest,
            }
        ),
    }
    if artifact_identity:
        body["artifact_identity"] = redact_for_burst(dict(artifact_identity))
    if latency_budget_ms is not None:
        body["latency_budget_ms"] = float(latency_budget_ms)
    if max_cost_usd is not None:
        body["max_cost_usd"] = float(max_cost_usd)
    if cloud_backend is not None:
        safe_backend = redact_for_burst(dict(cloud_backend))
        if safe_backend != dict(cloud_backend):
            raise CloudBurstContractError("cloud_backend_unsafe")
        body["cloud_backend"] = safe_backend
    return body


def post_burst_validation(
    cloud_url: str,
    request_body: Mapping[str, Any],
    *,
    api_key: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Post a dry-run burst validation request to Cloud or an injected fake."""

    if not _required(cloud_url):
        raise CloudBurstContractError("missing_cloud_url")
    if not _required(api_key):
        raise CloudBurstContractError("missing_api_key")
    url = cloud_url.rstrip("/") + "/routing/burst/validate"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    payload = dict(request_body)
    response = transport(url, payload, headers, timeout_s) if transport else _stdlib_post(url, payload, headers, timeout_s)
    return validate_burst_validation_response(response)


def validate_burst_validation_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Cloud's dry-run validation response."""

    body = dict(response)
    if body.get("schema_version") != BURST_SCHEMA_VERSION:
        raise CloudBurstContractError("burst_response_schema_mismatch")
    status = body.get("status")
    if status not in {"allow", "review", "block"}:
        raise CloudBurstContractError("burst_response_status_invalid")
    if body.get("backend_called") is not False:
        raise CloudBurstContractError("burst_validation_must_not_call_backend")
    if not isinstance(body.get("reasons"), list):
        raise CloudBurstContractError("burst_response_reasons_invalid")
    return body


def burst_trace_from_validation(request_body: Mapping[str, Any], response_body: Mapping[str, Any]) -> dict[str, Any]:
    """Return local trace/readiness-friendly metadata for a validation attempt."""

    response = validate_burst_validation_response(response_body)
    reasons = response.get("reasons") or []
    return {
        "schema_version": "tether_cloud_burst_trace.v0",
        "request_id": request_body.get("request_id"),
        "route_decision_id": request_body.get("route_decision_id"),
        "device_id": request_body.get("device_id"),
        "request_hash": request_body.get("request_hash") or response.get("request_hash"),
        "status": response.get("status"),
        "backend_called": False,
        "reason_codes": [str(reason.get("code")) for reason in reasons if isinstance(reason, Mapping)],
    }


def redact_for_burst(value: Any) -> Any:
    """Recursively remove secrets, local paths, signed URLs, and raw media/action blobs."""

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            lowered = key_s.lower()
            if _blocked_key(lowered):
                continue
            out[key_s] = redact_for_burst(item)
        return out
    if isinstance(value, list):
        return [redact_for_burst(item) for item in value]
    if isinstance(value, tuple):
        return [redact_for_burst(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _stdlib_post(url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout_s: float) -> Mapping[str, Any]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    req = urllib_request.Request(url, data=encoded, headers=dict(headers), method="POST")
    with urllib_request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - caller controls Cloud URL.
        return json.loads(resp.read().decode("utf-8"))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blocked_key(lowered_key: str) -> bool:
    return any(part in lowered_key for part in _SENSITIVE_KEY_PARTS | _RAW_KEY_PARTS)


def _redact_string(value: str) -> str:
    raw = value.strip()
    if not raw:
        return value
    if _LOCAL_PATH_RE.match(raw):
        return "[redacted_path]"
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.query:
        query = parsed.query.lower()
        if any(part in query for part in _SIGNED_URL_QUERY_PARTS):
            return "[redacted_url]"
    return value


def _normalize_privacy(value: str | None) -> str:
    raw = (value or "").strip().lower().replace("-", "_")
    if raw in {"edge", "edge_only", "local_only", "private", "privacy_edge_only"}:
        return "edge_only"
    return raw or "standard"


def _required(value: Any) -> bool:
    return bool(str(value or "").strip())


__all__ = [
    "BURST_SCHEMA_VERSION",
    "CloudBurstContractError",
    "build_burst_validation_request",
    "burst_trace_from_validation",
    "post_burst_validation",
    "redact_for_burst",
    "validate_burst_validation_response",
]
