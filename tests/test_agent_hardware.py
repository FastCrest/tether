from __future__ import annotations

import json

from tether.agent.hardware import collect_hardware_profile


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
