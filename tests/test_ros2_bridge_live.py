"""Regression marker for `ros2-bridge-live` GOALS.yaml gate.

Actual live test runs on Modal via `scripts/modal_ros2_live_test.py` —
spins an ubuntu:22.04 + ros-humble-ros-base container, installs
tether with a numpy<2.0 constraint (rclpy compiled against numpy
1.21 is forward-ABI-compatible with 1.22/1.24/1.26 but broken against
2.x), and verifies real rclpy can create the bridge node, spin_once,
and shutdown cleanly.

Receipt: `reflex_context/ros2_live_last_run.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path(__file__).parent.parent / "reflex_context" / "ros2_live_last_run.json"


def test_ros2_live_receipt_passes():
    if not RECEIPT.exists():
        pytest.skip(
            "No ros2-live receipt. Run `modal run scripts/modal_ros2_live_test.py` "
            f"and drop result at {RECEIPT}."
        )
    data = json.loads(RECEIPT.read_text())
    assert data.get("passed") is True, f"Last ros2-live run failed: {data}"
    checks = data.get("checks", {})
    for gate in ("node_created", "spin_once", "shutdown"):
        assert checks.get(gate) is True, f"{gate} did not pass: {checks}"
