"""Tests for src/tether/mcp/ros2_tools.py — ROS2-MCP bridge tool surface.

Uses a FakeROS2Context so the tests run without rclpy.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import FastMCP

from tether.mcp.ros2_tools import ROS2Context, register_ros2_tools


@dataclass
class FakeROS2Context:
    joint_state: list[float] | None = None
    image_rgb: Any | None = None  # ndarray or None
    task: str = ""
    estop_count: int = 0
    run_inference_response: dict = field(default_factory=lambda: {
        "actions": [[0.0] * 14] * 3,
        "policy_version": "pi0-libero",
    })
    run_inference_calls: list[str] = field(default_factory=list)

    def get_last_joint_state(self):
        return self.joint_state

    def get_last_image_rgb(self):
        return self.image_rgb

    def get_last_task(self) -> str:
        return self.task

    def publish_e_stop(self) -> None:
        self.estop_count += 1

    def run_inference(self, *, instruction: str) -> dict:
        self.run_inference_calls.append(instruction)
        return dict(self.run_inference_response)

    @property
    def robot_description(self) -> dict:
        return {"urdf_path": "/fake/urdf", "action_dim": 14, "base_frame": "base_link"}


def _mk_server(ctx: FakeROS2Context) -> FastMCP:
    m = FastMCP("test-ros2")
    register_ros2_tools(m, ctx)
    return m


async def _call_tool(m: FastMCP, name: str, **kwargs):
    tool = await m.get_tool(name)
    return await tool.fn(**kwargs)


@pytest.mark.asyncio
async def test_registers_all_expected_tools():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    # Sanity-check that all four tools are reachable.
    for name in ("get_joint_state", "get_camera_frame", "execute_task", "emergency_stop"):
        assert await m.get_tool(name) is not None


def test_registration_rejects_non_protocol_ctx():
    m = FastMCP("bad")
    with pytest.raises(TypeError, match="ROS2Context"):
        register_ros2_tools(m, object())  # does NOT implement the protocol


@pytest.mark.asyncio
async def test_get_joint_state_returns_stale_when_unseen():
    ctx = FakeROS2Context(joint_state=None)
    m = _mk_server(ctx)
    out = await _call_tool(m, "get_joint_state")
    assert out == {"joint_positions": [], "stale": True}


@pytest.mark.asyncio
async def test_get_joint_state_returns_positions_when_available():
    ctx = FakeROS2Context(joint_state=[0.1, 0.2, 0.3])
    m = _mk_server(ctx)
    out = await _call_tool(m, "get_joint_state")
    assert out == {"joint_positions": [0.1, 0.2, 0.3], "stale": False}


@pytest.mark.asyncio
async def test_get_joint_state_rate_limits_back_to_back_calls():
    ctx = FakeROS2Context(joint_state=[1.0])
    m = _mk_server(ctx)
    # First call burns the quota; immediate re-call should hit cooldown.
    _ = await _call_tool(m, "get_joint_state")
    out = await _call_tool(m, "get_joint_state")
    assert "error" in out
    assert out["error"]["kind"] == "RateLimited"


@pytest.mark.asyncio
async def test_execute_task_requires_confirm_true():
    ctx = FakeROS2Context(joint_state=[0.0], image_rgb=None)
    m = _mk_server(ctx)
    # Default confirm=False
    out = await _call_tool(m, "execute_task", instruction="pick up")
    assert "error" in out
    assert out["error"]["kind"] == "ConfirmationRequired"
    assert ctx.run_inference_calls == []  # no actuation


@pytest.mark.asyncio
async def test_execute_task_rejects_empty_instruction():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    out = await _call_tool(m, "execute_task", instruction="   ", confirm=True)
    assert out["error"]["kind"] == "InvalidInstruction"


@pytest.mark.asyncio
async def test_execute_task_rejects_bad_max_steps():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    out = await _call_tool(m, "execute_task", instruction="pick", confirm=True, max_steps=0)
    assert out["error"]["kind"] == "InvalidMaxSteps"
    out = await _call_tool(m, "execute_task", instruction="pick", confirm=True, max_steps=9999)
    assert out["error"]["kind"] == "InvalidMaxSteps"


@pytest.mark.asyncio
async def test_execute_task_happy_path_runs_inference():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    out = await _call_tool(
        m, "execute_task", instruction="pick up the red block", confirm=True
    )
    assert "error" not in out
    assert out["instruction"] == "pick up the red block"
    assert out["policy_version"] == "pi0-libero"
    assert len(out["actions"]) == 3
    assert ctx.run_inference_calls == ["pick up the red block"]


@pytest.mark.asyncio
async def test_execute_task_cooldown_blocks_rapid_reinvocation():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    first = await _call_tool(m, "execute_task", instruction="pick", confirm=True)
    assert "error" not in first
    second = await _call_tool(m, "execute_task", instruction="pick", confirm=True)
    assert second["error"]["kind"] == "RateLimited"
    # Second call was rejected before reaching ctx, so inference stayed at 1
    assert len(ctx.run_inference_calls) == 1


@pytest.mark.asyncio
async def test_emergency_stop_requires_confirm():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    out = await _call_tool(m, "emergency_stop")
    assert out["error"]["kind"] == "ConfirmationRequired"
    assert ctx.estop_count == 0


@pytest.mark.asyncio
async def test_emergency_stop_publishes_when_confirmed():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    out = await _call_tool(m, "emergency_stop", confirm=True)
    assert out["stopped"] is True
    assert "timestamp" in out
    assert ctx.estop_count == 1


@pytest.mark.asyncio
async def test_emergency_stop_cooldown_blocks_reinvocation():
    ctx = FakeROS2Context()
    m = _mk_server(ctx)
    _ = await _call_tool(m, "emergency_stop", confirm=True)
    out = await _call_tool(m, "emergency_stop", confirm=True)
    assert out["error"]["kind"] == "RateLimited"
    assert ctx.estop_count == 1  # second call rejected before publish


@pytest.mark.asyncio
async def test_get_camera_frame_stale_when_no_image():
    ctx = FakeROS2Context(image_rgb=None)
    m = _mk_server(ctx)
    out = await _call_tool(m, "get_camera_frame")
    assert out["stale"] is True
    assert out["image_b64"] == ""
    assert out["encoding"] == "jpeg"


@pytest.mark.asyncio
async def test_get_camera_frame_returns_base64_jpeg_when_available():
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    import numpy as np

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16, :16, 0] = 255  # a red quadrant; just needs to encode
    ctx = FakeROS2Context(image_rgb=img)
    m = _mk_server(ctx)
    out = await _call_tool(m, "get_camera_frame")
    assert out["stale"] is False
    assert out["encoding"] == "jpeg"
    assert len(out["image_b64"]) > 0
    # Should be base64 — decodable
    import base64 as b64
    raw = b64.b64decode(out["image_b64"])
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG SOI marker


@pytest.mark.asyncio
async def test_runtime_checkable_protocol_detects_missing_methods():
    # A class missing run_inference is not a valid ROS2Context.
    class Partial:
        def get_last_joint_state(self): return None
        def get_last_image_rgb(self): return None
        def get_last_task(self): return ""
        def publish_e_stop(self): pass
        # run_inference missing
        @property
        def robot_description(self): return {}

    m = FastMCP("p")
    with pytest.raises(TypeError):
        register_ros2_tools(m, Partial())
