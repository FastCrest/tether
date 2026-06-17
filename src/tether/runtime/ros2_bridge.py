"""ROS2 bridge for tether serve.

Runs a ROS2 node that subscribes to image + state + task topics, runs
inference via TetherServer, and publishes action chunks to a topic at a
configurable rate.

rclpy is NOT pip-installable. Install ROS2 (humble/iron/jazzy) via apt or
robostack before running:

    source /opt/ros/humble/setup.bash
    tether ros2-serve <export_dir>

Default topic layout (override via CLI flags):
    subs:
      /camera/image_raw      sensor_msgs/msg/Image (rgb8)
      /joint_states          state topic (default JointState — see below)
      /tether/task           std_msgs/msg/String (text instruction)
    pub:
      /tether/actions        std_msgs/msg/Float32MultiArray (flat chunk × action_dim)

State extractors (selected via --state-msg-type):
    joint_state  sensor_msgs/msg/JointState   .position           (arms; default)
    imu          sensor_msgs/msg/Imu          .orientation (quat) (drone, partial)
    odom         nav_msgs/msg/Odometry        pose + linear twist (drone, full state)

Recommended drone topic: /mavros/local_position/odom (nav_msgs/Odometry) —
gives the policy position + orientation + linear velocity in one message,
matching the 10-DOF state shape used by the shipped quadcopter preset.
The IMU path only yields 4 DOF (orientation only) — useful as a fallback
when full odometry isn't available, but expect reduced control quality.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _require_rclpy():
    """Import rclpy + always-needed message modules or raise a helpful ImportError.

    Returns the core types every bridge configuration needs. State-specific
    message classes (Imu, Odometry) are resolved lazily via
    `_resolve_state_msg` so a missing optional package doesn't break arm
    deployments that only need JointState.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float32MultiArray, String
        return rclpy, Node, Image, String, Float32MultiArray
    except ImportError as exc:
        raise ImportError(
            "rclpy not available. The tether ROS2 bridge requires a ROS2 install "
            "(humble, iron, or jazzy) via apt or robostack — rclpy is NOT "
            "pip-installable. Run:\n"
            "    source /opt/ros/humble/setup.bash  # or iron / jazzy\n"
            "    tether ros2-serve <export_dir>\n"
            f"Underlying error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# State extractors — one per supported state message type. Each takes a
# duck-typed message object and returns a flat list[float] state vector.
#
# These are deliberately decoupled from any rclpy/ROS2 imports so they can
# be unit-tested with simple SimpleNamespace mocks on machines without ROS2.
# ---------------------------------------------------------------------------


def _extract_joint_state(msg: Any) -> list[float]:
    """sensor_msgs/JointState → joint positions (arm convention)."""
    return [float(x) for x in msg.position]


def _extract_imu(msg: Any) -> list[float]:
    """sensor_msgs/Imu → quaternion orientation [x, y, z, w] (4 DOF).

    Partial state — no position, no velocity. Use `odom` instead when you
    have access to /mavros/local_position/odom for full drone state.
    """
    o = msg.orientation
    return [float(o.x), float(o.y), float(o.z), float(o.w)]


def _extract_odom(msg: Any) -> list[float]:
    """nav_msgs/Odometry → pos (3) + quat orient (4) + linear vel (3) = 10 DOF.

    Matches the state shape declared in the shipped quadcopter preset.
    The canonical drone topic is /mavros/local_position/odom.
    """
    pose = msg.pose.pose
    twist = msg.twist.twist
    p = pose.position
    o = pose.orientation
    v = twist.linear
    return [
        float(p.x), float(p.y), float(p.z),
        float(o.x), float(o.y), float(o.z), float(o.w),
        float(v.x), float(v.y), float(v.z),
    ]


# State-msg-type -> (lazy msg class loader, extractor). The class loader is
# called only when the bridge is actually set up — keeps the optional ROS2
# package deps out of the import path for unrelated codepaths.
_STATE_EXTRACTORS: dict[str, tuple[str, Any]] = {
    "joint_state": ("joint_state", _extract_joint_state),
    "imu": ("imu", _extract_imu),
    "odom": ("odom", _extract_odom),
}


def _resolve_state_msg_class(state_msg_type: str) -> Any:
    """Lazy-import the ROS2 message class for the requested state type.

    Raises ValueError on unknown type, ImportError if the relevant ROS2
    package isn't installed (with a hint about which apt/robostack package
    provides it).
    """
    if state_msg_type == "joint_state":
        from sensor_msgs.msg import JointState
        return JointState
    if state_msg_type == "imu":
        from sensor_msgs.msg import Imu
        return Imu
    if state_msg_type == "odom":
        try:
            from nav_msgs.msg import Odometry
        except ImportError as exc:
            raise ImportError(
                "nav_msgs not installed. Required for --state-msg-type=odom. "
                "On a ROS2 install, this ships with ros-<distro>-nav-msgs "
                "(apt) or the robostack ros-<distro>-nav-msgs conda package."
            ) from exc
        return Odometry
    raise ValueError(
        f"unknown state_msg_type {state_msg_type!r}; expected one of: "
        f"{', '.join(sorted(_STATE_EXTRACTORS))}"
    )


def create_ros2_bridge_node(
    server: Any,
    *,
    image_topic: str = "/camera/image_raw",
    state_topic: str = "/joint_states",
    task_topic: str = "/tether/task",
    action_topic: str = "/tether/actions",
    rate_hz: float = 20.0,
    node_name: str = "tether_vla",
    state_msg_type: str = "joint_state",
) -> Any:
    """Build a ROS2 node that wraps ``server.predict()`` as pub/sub.

    The returned node subscribes to image + state + task topics, caches the
    latest message from each, and at ``rate_hz`` Hz invokes
    ``server.predict(image, instruction, state)`` and publishes the action
    chunk to ``action_topic`` as a flat Float32MultiArray.

    state_msg_type selects how the state vector is extracted:
      - "joint_state": sensor_msgs/JointState.position (default; arms)
      - "imu":          sensor_msgs/Imu.orientation (drone; 4-DOF partial)
      - "odom":         nav_msgs/Odometry pose + twist (drone; 10-DOF full)
    """
    rclpy, Node, Image, String, Float32MultiArray = _require_rclpy()
    if state_msg_type not in _STATE_EXTRACTORS:
        raise ValueError(
            f"unknown state_msg_type {state_msg_type!r}; expected one of: "
            f"{', '.join(sorted(_STATE_EXTRACTORS))}"
        )
    StateMsgClass = _resolve_state_msg_class(state_msg_type)
    _, state_extractor = _STATE_EXTRACTORS[state_msg_type]

    # If the loaded server has an embodiment config, surface a one-time
    # warning when the extracted state vector length doesn't match what the
    # embodiment expects. Silent mismatches caused real drone bugs (#121).
    expected_state_dim = getattr(
        getattr(server, "embodiment_config", None), "state_dim", None
    )

    class TetherROS2Node(Node):
        """Bridge node. Also implements the `mcp.ros2_tools.ROS2Context` protocol
        so the ros2-mcp-bridge tools can bind directly to a live node."""

        def __init__(self) -> None:
            super().__init__(node_name)
            self._server = server
            self._last_image: np.ndarray | None = None
            self._last_state: list[float] | None = None
            self._last_task: str = ""
            self._inference_count = 0
            self._state_dim_mismatch_warned = False

            self.create_subscription(Image, image_topic, self._image_cb, 10)
            self.create_subscription(StateMsgClass, state_topic, self._state_cb, 10)
            self.create_subscription(String, task_topic, self._task_cb, 10)
            self._action_pub = self.create_publisher(Float32MultiArray, action_topic, 10)
            self._estop_pub = self.create_publisher(String, "/tether/e_stop", 10)
            self._timer = self.create_timer(1.0 / max(0.1, rate_hz), self._tick)

            self.get_logger().info(
                f"tether ros2 node '{node_name}' up: subs={image_topic} + "
                f"{state_topic} ({state_msg_type}) + {task_topic}, "
                f"pub={action_topic} at {rate_hz:.1f} Hz"
            )

        def _image_cb(self, msg: Any) -> None:
            """Decode sensor_msgs/Image → HxWx3 uint8 numpy array.

            Handles rgb8 and bgr8 encodings. Other encodings fall back to the
            raw reshape and may need conversion by the caller.
            """
            h, w = int(msg.height), int(msg.width)
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            encoding = getattr(msg, "encoding", "rgb8")
            if arr.size == h * w * 3:
                img = arr.reshape(h, w, 3)
                if encoding == "bgr8":
                    img = img[..., ::-1]
                self._last_image = img.copy()
            elif arr.size == h * w * 4:
                # rgba8 / bgra8 — drop alpha
                img = arr.reshape(h, w, 4)[..., :3]
                if encoding in ("bgra8", "bgr8"):
                    img = img[..., ::-1]
                self._last_image = img.copy()
            else:
                self.get_logger().warning(
                    f"unsupported image size/encoding: {arr.size} bytes, "
                    f"{h}x{w}, encoding={encoding}"
                )

        def _state_cb(self, msg: Any) -> None:
            try:
                state = state_extractor(msg)
            except (AttributeError, TypeError) as exc:
                self.get_logger().error(
                    f"failed to extract state from {state_msg_type!r} message "
                    f"on {state_topic}: {exc}. Check --state-msg-type matches "
                    f"the actual message type being published."
                )
                return
            self._last_state = state
            # One-time warning on state-dim mismatch — silent shape mismatches
            # are the most common drone deployment failure mode (silently
            # garbage actions instead of a loud error). Fire once per node.
            if (
                expected_state_dim is not None
                and not self._state_dim_mismatch_warned
                and len(state) != expected_state_dim
            ):
                self.get_logger().warning(
                    f"state vector length {len(state)} from "
                    f"{state_msg_type!r} extractor does NOT match embodiment "
                    f"state_dim {expected_state_dim}. The policy will receive "
                    f"the wrong shape and likely produce nonsense actions. "
                    f"Either change --state-msg-type or load an embodiment "
                    f"config whose state_dim matches your robot's state."
                )
                self._state_dim_mismatch_warned = True

        def _task_cb(self, msg: Any) -> None:
            self._last_task = str(msg.data)

        def _tick(self) -> None:
            if self._last_image is None or self._last_state is None:
                return
            try:
                result = self._server.predict(
                    image=self._last_image,
                    instruction=self._last_task,
                    state=self._last_state,
                )
            except Exception as exc:
                self.get_logger().error(f"predict failed: {exc}")
                return
            if isinstance(result, dict) and "error" in result:
                self.get_logger().warning(f"predict error: {result['error']}")
                return
            actions = result.get("actions") if isinstance(result, dict) else None
            if not actions:
                return
            out = Float32MultiArray()
            out.data = [float(v) for chunk in actions for v in chunk]
            self._action_pub.publish(out)
            self._inference_count += 1

        # ---- ROS2Context protocol (mcp.ros2_tools) --------------------------

        def get_last_joint_state(self) -> list[float] | None:
            return list(self._last_state) if self._last_state is not None else None

        def get_last_image_rgb(self) -> Any:
            return self._last_image

        def get_last_task(self) -> str:
            return self._last_task

        def publish_e_stop(self) -> None:
            msg = String()
            msg.data = "e_stop"
            self._estop_pub.publish(msg)
            self.get_logger().warning("ros2-mcp e_stop published")

        def run_inference(self, *, instruction: str) -> dict:
            if self._last_image is None or self._last_state is None:
                return {"error": "no observation available — camera or joint_state unseen"}
            try:
                result = self._server.predict(
                    image=self._last_image,
                    instruction=instruction,
                    state=self._last_state,
                )
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
            if isinstance(result, dict) and "actions" in result:
                # Publish the chunk so the robot actuates, matching /_tick semantics.
                chunks = result["actions"]
                if chunks:
                    out = Float32MultiArray()
                    out.data = [float(v) for chunk in chunks for v in chunk]
                    self._action_pub.publish(out)
                    self._inference_count += 1
            return {
                **(result if isinstance(result, dict) else {}),
                "policy_version": getattr(self._server, "export_dir", "unknown"),
            }

        @property
        def robot_description(self) -> dict:
            ec = getattr(self._server, "embodiment_config", None)
            action_dim = getattr(ec, "action_dim", None)
            embodiment = getattr(ec, "embodiment", None) or "unknown"
            return {
                "embodiment": embodiment,
                "action_dim": int(action_dim) if action_dim is not None else None,
                "image_topic": image_topic,
                "state_topic": state_topic,
                "action_topic": action_topic,
                "e_stop_topic": "/tether/e_stop",
            }

    return TetherROS2Node()


def run_ros2_bridge(
    export_dir: str | Path,
    *,
    device: str = "cuda",
    providers: list[str] | None = None,
    strict_providers: bool = True,
    safety_config: str | Path | None = None,
    image_topic: str = "/camera/image_raw",
    state_topic: str = "/joint_states",
    task_topic: str = "/tether/task",
    action_topic: str = "/tether/actions",
    rate_hz: float = 20.0,
    node_name: str = "tether_vla",
    state_msg_type: str = "joint_state",
    mcp: bool = False,
    mcp_transport: str = "stdio",
    mcp_port: int = 8001,
) -> None:
    """Load the model, init rclpy, spin the bridge node until shutdown.

    When ``mcp=True``, also start an MCP server that exposes the running
    ROS2 bridge as agent-callable tools (per ros2-mcp-bridge.md). The bridge
    node already implements ``ROS2Context``, so it binds directly via
    ``register_ros2_tools(mcp, node)``.

    Concurrency model:
    - mcp=False: single-threaded, ``rclpy.spin(node)`` blocks until shutdown.
    - mcp=True, transport="stdio": MCP owns stdin/stdout; rclpy spins in a
      background thread (MultiThreadedExecutor); MCP runs on the main thread
      so its stdio plumbing works. Used for Claude Desktop / Cursor.
    - mcp=True, transport="http": MCP runs in background thread on
      ``mcp_port``; ``rclpy.spin(node)`` blocks on main thread.
    """
    rclpy, _, _, _, _ = _require_rclpy()
    from tether.runtime.server import TetherServer

    if mcp and mcp_transport not in ("stdio", "http"):
        raise ValueError(
            f"mcp_transport must be 'stdio' or 'http', got {mcp_transport!r}"
        )

    server = TetherServer(
        export_dir,
        device=device,
        providers=providers,
        strict_providers=strict_providers,
        safety_config=safety_config,
    )
    server.load()

    rclpy.init()
    node = None
    mcp_srv = None
    try:
        node = create_ros2_bridge_node(
            server,
            image_topic=image_topic,
            state_topic=state_topic,
            task_topic=task_topic,
            action_topic=action_topic,
            rate_hz=rate_hz,
            node_name=node_name,
            state_msg_type=state_msg_type,
        )

        if not mcp:
            rclpy.spin(node)
            return

        # MCP-enabled path: build an MCP server with the /act tools + bind
        # ros2_tools to the live node (which implements ROS2Context).
        try:
            from tether.mcp import create_mcp_server, register_ros2_tools
        except ImportError as exc:
            raise ImportError(
                "MCP optional dep not installed. Run: pip install 'fastcrest-tether[mcp]'"
            ) from exc

        mcp_srv = create_mcp_server(server)
        register_ros2_tools(mcp_srv, node)
        logger.info(
            "ros2-mcp bridge: MCP server built with /act tools + 4 ros2_tools "
            "bound to live node %r (transport=%s)",
            node_name, mcp_transport,
        )

        if mcp_transport == "http":
            # MCP HTTP runs in a background thread; rclpy.spin owns the main
            # thread (matches the legacy mcp=False path's blocking behavior).
            import threading
            def _run_mcp_http():
                mcp_srv.run(
                    transport="streamable-http",
                    host="127.0.0.1",
                    port=mcp_port,
                )
            mcp_thread = threading.Thread(
                target=_run_mcp_http, daemon=True, name="tether-mcp-http",
            )
            mcp_thread.start()
            logger.info(
                "ros2-mcp bridge: MCP HTTP server on http://127.0.0.1:%d "
                "(streamable-http); rclpy.spin owns main thread",
                mcp_port,
            )
            rclpy.spin(node)
            return

        # stdio transport: MCP owns the main thread (needs stdin/stdout);
        # rclpy.spin runs in a background thread via MultiThreadedExecutor
        # so it doesn't conflict with MCP's stdio loop.
        from rclpy.executors import MultiThreadedExecutor
        import threading
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        def _spin_executor():
            try:
                executor.spin()
            except Exception:  # noqa: BLE001 -- background thread; log + exit
                logger.exception("ros2-mcp bridge: rclpy executor crashed")

        spin_thread = threading.Thread(
            target=_spin_executor, daemon=True, name="tether-rclpy-spin",
        )
        spin_thread.start()
        logger.info(
            "ros2-mcp bridge: rclpy spinning in background thread; "
            "MCP stdio on main thread (Claude Desktop / Cursor compatible)"
        )
        # Blocks until MCP client disconnects
        mcp_srv.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("ros2 bridge interrupted by user")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


__all__ = ["create_ros2_bridge_node", "run_ros2_bridge"]
