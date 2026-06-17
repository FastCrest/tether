from __future__ import annotations

import json

from tether.agent.hardware import collect_hardware_profile
from tether.agent.resources import collect_resource_pressure, collect_route_readiness


def test_hardware_profile_never_raises_and_is_json_serializable():
    profile = collect_hardware_profile()

    json.dumps(profile, sort_keys=True)
    assert isinstance(profile, dict)
    assert set(profile) >= {
        "hostname",
        "platform",
        "python",
        "agent_version",
        "tether_version",
        "gpu_name",
        "cuda",
        "jetpack",
        "tensorrt",
    }
    assert isinstance(profile["platform"], str)
    assert isinstance(profile["python"], str)


def test_resource_pressure_never_raises_and_is_json_serializable():
    pressure = collect_resource_pressure()

    json.dumps(pressure, sort_keys=True)
    assert isinstance(pressure, dict)
    assert "pressure" in pressure


def test_route_readiness_sanitizes_tokens_signed_urls_and_paths():
    def get_json(url, timeout, headers=None):
        if url.endswith("/health"):
            return 200, {
                "status": "ok",
                "state": "ready",
                "inference_mode": "onnx",
                "export_dir": "/tmp/secret/model",
            }
        return 200, {
            "artifact_version": "art_1",
            "optimized_artifact_digest": "sha256:abc",
            "target_sku": "orin",
            "signed_url": "https://example.test/model?token=secret",
            "device_token": "dvc_test_should_not_leak",
        }

    readiness = collect_route_readiness(
        None,
        observed_at=100.0,
        http_get_json=get_json,
    )
    rendered = json.dumps(readiness, sort_keys=True)

    assert readiness["serve_ready"] is True
    assert readiness["artifact_identity"]["artifact_version"] == "art_1"
    assert readiness["runtime"]["name"] == "onnx"
    assert "/tmp/secret/model" not in rendered
    assert "signed_url" not in rendered
    assert "dvc_test_should_not_leak" not in rendered
