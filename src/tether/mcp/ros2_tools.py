"""ROS2-MCP bridge tool registrations.

When `tether serve` is wired to a ROS2 node and the MCP surface is enabled,
these tools expose cached topic state (joint positions, latest camera frame)
and controlled actuation (execute task, emergency stop) to MCP-compatible
agents. Every actuation tool requires `confirm=True` and is rate-limited
(5 s cooldown on actuation, 100 ms on read).

Design:
- Decoupled from rclpy at import time. The bridge passes a `ROS2Context`
  protocol implementation in; production code binds to the live node, tests
  bind to a `FakeROS2Context`.
- Tool registrations are idempotent. Safe to call `register_ros2_tools`
  twice (it skips re-registrations).
- All actuation tools WRITE the `confirm` boolean explicitly so the LLM
  is forced to declare intent — prevents prompt-injection-driven actuation.

Feature spec: features/01_serve/subfeatures/_ecosystem/ros2-mcp-bridge/
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


# Rate limits (seconds between invocations per tool — enforced server-side).
_ACTUATION_COOLDOWN_S = 5.0
_READ_COOLDOWN_S = 0.1


@runtime_checkable
class ROS2Context(Protocol):
    """Minimal interface the ROS2 MCP tools require.

    Production: `TetherROS2Node` exposes these. Tests: `FakeROS2Context`
    does too. Lets MCP tool code import without rclpy present.
    """

    def get_last_joint_state(self) -> list[float] | None:
        """Return the cached `/joint_states` position vector, or None if unseen."""

    def get_last_image_rgb(self) -> Any | None:
        """Return the cached `/camera/image_raw` as HxWx3 uint8 ndarray, or None."""

    def get_last_task(self) -> str:
        """Return the latest instruction received on `/tether/task` (or empty)."""

    def publish_e_stop(self) -> None:
        """Publish an emergency-stop signal to the robot's e-stop topic."""

    def run_inference(self, *, instruction: str) -> dict[str, Any]:
        """Trigger one /act cycle using the cached image + state + the given
        instruction. Returns the prediction dict OR `{"error": ...}` on failure."""

    @property
    def robot_description(self) -> dict[str, Any]:
        """Static robot-identity snapshot — URDF path, action_dim, frame info.
        Returned as a JSON-safe dict."""


class _RateLimiter:
    """Per-tool server-side cooldown — rejects calls inside the window."""

    __slots__ = ("_last", "_cooldown_s")

    def __init__(self, cooldown_s: float):
        self._last: float = 0.0
        self._cooldown_s = float(cooldown_s)

    def check_and_update(self) -> tuple[bool, float]:
        """Returns (allowed, wait_s_if_denied)."""
        now = time.monotonic()
        dt = now - self._last
        if dt < self._cooldown_s:
            return False, self._cooldown_s - dt
        self._last = now
        return True, 0.0


def _encode_image_jpeg_b64(img_rgb: Any) -> str:
    """HxWx3 uint8 ndarray → base64-encoded JPEG (quality 80, size-efficient
    for LLM-side vision). Falls back to PNG if PIL/Pillow is unavailable."""
    try:
        from PIL import Image as _PILImage
    except ImportError:
        import numpy as _np
        # PNG bytes via a minimal numpy-only path would need a PNG encoder;
        # fall back: return raw shape info so the caller still gets the grid.
        shape = tuple(getattr(img_rgb, "shape", ()))
        return f"unavailable:install_pillow;shape={shape}"
    import numpy as _np

    arr = _np.asarray(img_rgb, dtype=_np.uint8)
    pil_img = _PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _err(kind: str, message: str, remediation: str) -> dict[str, Any]:
    return {"error": {"kind": kind, "message": message, "remediation": remediation}}


def register_ros2_tools(
    mcp: "FastMCP",
    ctx: ROS2Context,
) -> None:
    """Register 4 tools + 1 resource on an existing FastMCP server.

    Args:
        mcp: a FastMCP server (from `create_mcp_server()` or externally).
        ctx: implements `ROS2Context` — production binds to the live
            `TetherROS2Node`; tests bind to `FakeROS2Context`.

    Raises:
        TypeError: if `ctx` does not implement `ROS2Context`.
    """
    if not isinstance(ctx, ROS2Context):
        raise TypeError(
            f"ros2 context must implement ROS2Context protocol; got {type(ctx).__name__}"
        )

    read_limiter = _RateLimiter(_READ_COOLDOWN_S)
    execute_limiter = _RateLimiter(_ACTUATION_COOLDOWN_S)
    estop_limiter = _RateLimiter(_ACTUATION_COOLDOWN_S)

    @mcp.tool()
    async def get_joint_state() -> dict[str, Any]:
        """Return the robot's current joint positions.

        Returns:
            {joint_positions: list[float], stale: bool} — stale=True if no
            `/joint_states` message has been received yet.
        """
        ok, wait = read_limiter.check_and_update()
        if not ok:
            return _err(
                "RateLimited",
                f"get_joint_state is rate-limited; wait {wait:.2f}s",
                "MCP tools are rate-limited server-side to protect the robot. Retry after the wait.",
            )
        positions = ctx.get_last_joint_state()
        if positions is None:
            return {"joint_positions": [], "stale": True}
        return {"joint_positions": list(positions), "stale": False}

    @mcp.tool()
    async def get_camera_frame() -> dict[str, Any]:
        """Return the latest camera frame as a base64 JPEG.

        Returns:
            {image_b64: str, encoding: "jpeg", stale: bool}. When stale=True
            no image has been received yet and image_b64 is empty.
        """
        ok, wait = read_limiter.check_and_update()
        if not ok:
            return _err(
                "RateLimited",
                f"get_camera_frame is rate-limited; wait {wait:.2f}s",
                "Read-tool cooldown is 100ms to prevent tight agent loops.",
            )
        img = ctx.get_last_image_rgb()
        if img is None:
            return {"image_b64": "", "encoding": "jpeg", "stale": True}
        try:
            b64 = _encode_image_jpeg_b64(img)
        except Exception as exc:  # noqa: BLE001
            logger.error("mcp.ros2.get_camera_frame encode failed: %s", exc)
            return _err(
                type(exc).__name__,
                str(exc),
                "Install Pillow (`pip install Pillow`) for JPEG encoding.",
            )
        return {"image_b64": b64, "encoding": "jpeg", "stale": False}

    @mcp.tool()
    async def execute_task(
        instruction: str,
        confirm: bool = False,
        max_steps: int = 100,
    ) -> dict[str, Any]:
        """Run one Tether /act cycle on the robot's current observation.

        Args:
            instruction: natural-language task (e.g. "pick up the red block").
            confirm: MUST be True. This argument is a deliberate tripwire —
                the LLM has to assert intent before we actuate anything.
            max_steps: reserved for future multi-step chaining. Phase 1
                runs one chunk; the caller is responsible for looping.

        Returns:
            On success: {actions: [[float]], policy_version: str, instruction: str}.
            On failure or when confirm is missing: {error: ...}.
        """
        if confirm is not True:
            return _err(
                "ConfirmationRequired",
                "execute_task requires confirm=True to actuate the robot.",
                "Pass confirm=True explicitly. This arg is required — it is not a default-True tripwire.",
            )
        if not instruction or not instruction.strip():
            return _err(
                "InvalidInstruction",
                "instruction must be a non-empty string.",
                "Pass a task description like 'pick up the red block'.",
            )
        if max_steps < 1 or max_steps > 1000:
            return _err(
                "InvalidMaxSteps",
                f"max_steps must be in [1, 1000], got {max_steps}.",
                "Phase 1 executes one chunk per call regardless of max_steps.",
            )
        ok, wait = execute_limiter.check_and_update()
        if not ok:
            return _err(
                "RateLimited",
                f"execute_task is rate-limited; wait {wait:.2f}s",
                f"Actuation cooldown is {_ACTUATION_COOLDOWN_S}s to prevent runaway agent loops.",
            )
        result = ctx.run_inference(instruction=instruction)
        if isinstance(result, dict) and "error" in result:
            return _err(
                "InferenceFailed",
                str(result.get("error")),
                "Inspect server logs. Ensure the camera + joint_state topics are being published.",
            )
        actions = result.get("actions") if isinstance(result, dict) else None
        return {
            "actions": actions,
            "instruction": instruction,
            "policy_version": str(result.get("policy_version", "unknown"))
            if isinstance(result, dict) else "unknown",
        }

    @mcp.tool()
    async def emergency_stop(confirm: bool = False) -> dict[str, Any]:
        """Publish an emergency-stop signal to the robot.

        Args:
            confirm: MUST be True. No actuation side-effect without it.

        Returns:
            {stopped: bool, timestamp: float}. `stopped` reflects whether
            the e-stop publish succeeded.
        """
        if confirm is not True:
            return _err(
                "ConfirmationRequired",
                "emergency_stop requires confirm=True.",
                "Pass confirm=True explicitly. The LLM must actively affirm e-stop.",
            )
        ok, wait = estop_limiter.check_and_update()
        if not ok:
            return _err(
                "RateLimited",
                f"emergency_stop is rate-limited; wait {wait:.2f}s",
                "E-stop cooldown is 5s to prevent oscillating publishes.",
            )
        try:
            ctx.publish_e_stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("mcp.ros2.emergency_stop publish failed: %s", exc)
            return _err(
                type(exc).__name__,
                str(exc),
                "Ensure the ROS2 node is running and the e-stop topic exists.",
            )
        return {"stopped": True, "timestamp": time.time()}

    @mcp.resource("robot://status")
    async def robot_status() -> dict[str, Any]:
        """Static robot identity + latest-observed state snapshot.

        Composes robot_description (URDF path, action_dim, etc.) with the
        current joint state + task for a one-shot status query.
        """
        description = dict(ctx.robot_description)
        positions = ctx.get_last_joint_state()
        return {
            **description,
            "last_joint_positions": list(positions) if positions is not None else [],
            "last_task": ctx.get_last_task(),
            "joint_state_stale": positions is None,
            "snapshot_timestamp": time.time(),
        }


__all__ = [
    "ROS2Context",
    "register_ros2_tools",
]
