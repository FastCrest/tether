"""Dry-run Cloud burst handoff helpers for Tether runtime.

This is not live `/act` proxying. It builds the bounded request, calls Cloud's
validation endpoint, and returns local trace metadata so operators can prove the
handoff contract before enabling online Cloud GPU burst.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .cloud_burst_contract import (
    Transport,
    build_burst_validation_request,
    burst_trace_from_validation,
    post_burst_validation,
)


@dataclass(frozen=True)
class CloudBurstDryRunResult:
    request: dict[str, Any]
    response: dict[str, Any]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "response": self.response,
            "trace": self.trace,
        }


def dry_run_cloud_burst(
    *,
    cloud_url: str,
    api_key: str,
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
    timeout_s: float = 2.0,
    transport: Transport | None = None,
) -> CloudBurstDryRunResult:
    request = build_burst_validation_request(
        request_id=request_id,
        route_decision_id=route_decision_id,
        device_id=device_id,
        workload_type=workload_type,
        privacy_mode=privacy_mode,
        cloud_allowed=cloud_allowed,
        max_cost_usd=max_cost_usd,
        payload_hash=payload_hash,
        payload_manifest=payload_manifest,
        artifact_identity=artifact_identity,
        latency_budget_ms=latency_budget_ms,
        cloud_backend=cloud_backend,
    )
    response = post_burst_validation(
        cloud_url,
        request,
        api_key=api_key,
        timeout_s=timeout_s,
        transport=transport,
    )
    trace = burst_trace_from_validation(request, response)
    return CloudBurstDryRunResult(request=request, response=response, trace=trace)


__all__ = ["CloudBurstDryRunResult", "dry_run_cloud_burst"]
