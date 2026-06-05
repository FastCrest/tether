from __future__ import annotations

import json
from pathlib import Path

import pytest

from tether.runtime.cloud_burst_contract import (
    BURST_SCHEMA_VERSION,
    CloudBurstContractError,
    build_burst_validation_request,
    burst_trace_from_validation,
    post_burst_validation,
    validate_burst_validation_response,
)


FIXTURES = Path(__file__).parent / "fixtures" / "cloud_burst"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _request(**overrides):
    body = build_burst_validation_request(
        request_id="brq_1",
        route_decision_id="rtd_1",
        device_id="dev_1",
        workload_type="factory_visual_inspection",
        privacy_mode="standard",
        cloud_allowed=True,
        max_cost_usd=0.01,
        payload_hash="sha256:" + "a" * 64,
        payload_manifest={"parts": [{"kind": "tensor", "sha256": "b" * 64, "bytes": 2048}]},
        cloud_backend={"available": True, "backend_id": "fake-cloud", "estimated_cost_usd": 0.002},
    )
    body.update(overrides)
    return body


def _response(**overrides):
    body = {
        "schema_version": BURST_SCHEMA_VERSION,
        "status": "allow",
        "request_id": "brq_1",
        "route_decision_id": "rtd_1",
        "device_id": "dev_1",
        "request_hash": "abc",
        "backend_called": False,
        "reasons": [{"code": "cloud_burst_allowed", "severity": "info", "message": "ok"}],
    }
    body.update(overrides)
    return body


def test_build_burst_validation_request_is_bounded_and_deterministic() -> None:
    request = _request()

    assert request["schema_version"] == BURST_SCHEMA_VERSION
    assert request["route_decision_id"] == "rtd_1"
    assert request["payload_manifest"]["parts"][0]["kind"] == "tensor"
    assert request["request_hash"]
    assert "api_key" not in str(request)


def test_shared_cloud_burst_fixtures_cover_allow_review_and_block_shapes() -> None:
    allow_request = _fixture("allow_request.json")
    allow_response = validate_burst_validation_response(_fixture("allow_response.json"))
    review_request = _fixture("review_missing_receipt_request.json")
    block_request = _fixture("block_edge_only_request.json")

    assert allow_request["schema_version"] == BURST_SCHEMA_VERSION
    assert allow_response["status"] == "allow"
    assert review_request["route_decision_id"] == "rtd_missing_fixture"
    assert block_request["privacy_mode"] == "edge_only"


@pytest.mark.parametrize(
    ("privacy_mode", "cloud_allowed", "error"),
    [
        ("edge_only", True, "privacy_edge_only"),
        ("standard", False, "cloud_disallowed"),
    ],
)
def test_build_burst_validation_request_blocks_before_dispatch(privacy_mode, cloud_allowed, error) -> None:
    with pytest.raises(CloudBurstContractError, match=error):
        build_burst_validation_request(
            request_id="brq_1",
            route_decision_id="rtd_1",
            device_id="dev_1",
            privacy_mode=privacy_mode,
            cloud_allowed=cloud_allowed,
            payload_hash="sha256:" + "a" * 64,
            payload_manifest={"parts": []},
        )


def test_build_burst_validation_request_rejects_unsafe_manifest() -> None:
    with pytest.raises(CloudBurstContractError, match="payload_manifest_unsafe"):
        build_burst_validation_request(
            request_id="brq_1",
            route_decision_id="rtd_1",
            device_id="dev_1",
            payload_hash="sha256:" + "a" * 64,
            payload_manifest={
                "local_path": "/Users/romirjain/private/frame.png",
                "signed_url": "https://example.invalid/blob?X-Amz-Signature=secret",
                "token": "secret",
            },
        )


def test_post_burst_validation_uses_fake_transport_without_backend_execution() -> None:
    calls = []

    def fake_transport(url, payload, headers, timeout_s):
        calls.append((url, payload, headers, timeout_s))
        return _response(request_hash=payload["request_hash"])

    response = post_burst_validation(
        "https://cloud.example",
        _request(),
        api_key="rc_test_123",
        timeout_s=1.5,
        transport=fake_transport,
    )

    assert calls[0][0] == "https://cloud.example/routing/burst/validate"
    assert calls[0][2]["X-API-Key"] == "rc_test_123"
    assert response["status"] == "allow"
    assert response["backend_called"] is False


def test_validate_burst_response_rejects_backend_called() -> None:
    with pytest.raises(CloudBurstContractError, match="burst_validation_must_not_call_backend"):
        validate_burst_validation_response(_response(backend_called=True))


def test_burst_trace_from_validation_records_reason_codes() -> None:
    trace = burst_trace_from_validation(_request(), _response(status="review"))

    assert trace == {
        "schema_version": "tether_cloud_burst_trace.v0",
        "request_id": "brq_1",
        "route_decision_id": "rtd_1",
        "device_id": "dev_1",
        "request_hash": _request()["request_hash"],
        "status": "review",
        "backend_called": False,
        "reason_codes": ["cloud_burst_allowed"],
    }
