from __future__ import annotations

import pytest

from tether.runtime.cloud_burst_contract import BURST_SCHEMA_VERSION, CloudBurstContractError
from tether.runtime.cloud_fallback import dry_run_cloud_burst


def _response(status: str = "allow", request_hash: str = "abc") -> dict:
    return {
        "schema_version": BURST_SCHEMA_VERSION,
        "status": status,
        "request_id": "brq_dry",
        "route_decision_id": "rtd_dry",
        "device_id": "dev_dry",
        "request_hash": request_hash,
        "backend_called": False,
        "reasons": [{"code": "cloud_burst_allowed", "severity": "info", "message": "ok"}],
    }


def test_dry_run_cloud_burst_calls_validation_endpoint_and_records_trace() -> None:
    calls = []

    def fake_transport(url, payload, headers, timeout_s):
        calls.append((url, payload, headers, timeout_s))
        return _response(request_hash=payload["request_hash"])

    result = dry_run_cloud_burst(
        cloud_url="https://cloud.example",
        api_key="rc_test_123",
        request_id="brq_dry",
        route_decision_id="rtd_dry",
        device_id="dev_dry",
        workload_type="warehouse_packing_qa",
        payload_hash="sha256:" + "a" * 64,
        payload_manifest={"parts": [{"kind": "tensor", "sha256": "b" * 64}]},
        cloud_backend={"available": True, "backend_id": "fake-cloud", "estimated_cost_usd": 0.002},
        transport=fake_transport,
    )

    assert calls[0][0] == "https://cloud.example/routing/burst/validate"
    assert calls[0][1]["workload_type"] == "warehouse_packing_qa"
    assert calls[0][2]["X-API-Key"] == "rc_test_123"
    assert result.response["status"] == "allow"
    assert result.trace["status"] == "allow"
    assert result.trace["backend_called"] is False
    assert result.trace["reason_codes"] == ["cloud_burst_allowed"]


def test_dry_run_cloud_burst_blocks_edge_only_before_transport() -> None:
    called = False

    def fake_transport(url, payload, headers, timeout_s):
        nonlocal called
        called = True
        return _response(request_hash=payload["request_hash"])

    with pytest.raises(CloudBurstContractError, match="privacy_edge_only"):
        dry_run_cloud_burst(
            cloud_url="https://cloud.example",
            api_key="rc_test_123",
            request_id="brq_dry",
            route_decision_id="rtd_dry",
            device_id="dev_dry",
            privacy_mode="edge_only",
            payload_hash="sha256:" + "a" * 64,
            payload_manifest={"parts": [{"kind": "tensor", "sha256": "b" * 64}]},
            transport=fake_transport,
        )

    assert called is False
