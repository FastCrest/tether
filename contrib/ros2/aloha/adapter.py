"""Aloha ROS2 adapter — dual-arm bimanual robot control via Reflex VLA.

Ported from FluxVLA ``aloha_operator.py`` (682 LOC, Apache-2.0, LimX Dynamics).
ROS1 (``rospy``) → ROS2 (``rclpy``) port with these changes per R-1..R-4:

- Uses HTTP or ZMQ to call ``reflex serve`` (R-1: no ROS2 transport dependency)
- 3-camera setup: ``cam_high``, ``cam_left_wrist``, ``cam_right_wrist`` (R-3)
- Bimanual joint state: 14 DOF (7 left + 7 right including grippers)
- Replan every 5 steps by default

Usage::

    # Terminal 1: GPU machine
    reflex serve ./my_export/ --transport zmq --port 5555

    # Terminal 2: Robot (ROS2)
    python3 contrib/ros2/aloha/adapter.py \
        --server tcp://gpu-server:5555 \
        --instruction "pick up the red cup"

Requires: rclpy, sensor_msgs, std_msgs, cv_bridge, numpy, pyzmq/httpx
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

# ROS2 imports — these fail gracefully if not in a ROS2 environment
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# Add contrib parent to path for shared utilities
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from _common.observation_builder import ObservationBuilder
from _common.action_publisher import ActionPublisher
from _common.reflex_client import ReflexClient


# Default Aloha topic names (Trossen ALOHA 2 convention)
DEFAULT_TOPICS = {
    "cam_high": "/cam_high/image_raw",
    "cam_left_wrist": "/cam_left_wrist/image_raw",
    "cam_right_wrist": "/cam_right_wrist/image_raw",
    "joint_states_left": "/puppet_left/joint_states",
    "joint_states_right": "/puppet_right/joint_states",
    "cmd_left": "/puppet_left/joint_group_position_controller/commands",
    "cmd_right": "/puppet_right/joint_group_position_controller/commands",
}

ALOHA_ACTION_DIM = 14  # 7 joints per arm × 2 arms
ALOHA_STATE_DIM = 14
ALOHA_IMAGE_SIZE = 224
ALOHA_CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


class AlohaAdapterNode(Node if HAS_ROS2 else object):
    """ROS2 node that bridges Aloha sensors → Reflex VLA → Aloha actuators.

    Subscribes to 3 cameras + 2 joint state topics, constructs observation,
    calls reflex serve, publishes joint commands.
    """

    def __init__(
        self,
        server_url: str = "tcp://localhost:5555",
        instruction: str = "",
        control_hz: float = 30.0,
        replan_steps: int = 5,
        image_size: int = ALOHA_IMAGE_SIZE,
    ) -> None:
        if HAS_ROS2:
            super().__init__("reflex_aloha_adapter")
        self.instruction = instruction
        self.image_size = image_size

        # Reflex client
        self.client = ReflexClient(server_url)

        # Observation builder (3 cameras)
        self.obs_builder = ObservationBuilder(
            image_keys=ALOHA_CAMERA_KEYS,
            state_dim=ALOHA_STATE_DIM,
        )

        # Action publisher
        self.action_pub = ActionPublisher(
            action_dim=ALOHA_ACTION_DIM,
            replan_steps=replan_steps,
            control_hz=control_hz,
        )

        # CV bridge for ROS Image → numpy
        if HAS_ROS2:
            self.bridge = CvBridge()
            self._setup_subscribers()
            self._setup_publishers()

        # Joint state buffers
        self._left_joints: np.ndarray | None = None
        self._right_joints: np.ndarray | None = None

        self.get_logger().info(
            f"Aloha adapter ready: server={server_url}, "
            f"instruction={instruction!r}, hz={control_hz}"
        ) if HAS_ROS2 else None

    def _setup_subscribers(self) -> None:
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Camera subscribers
        for cam_key, topic in [
            ("cam_high", DEFAULT_TOPICS["cam_high"]),
            ("cam_left_wrist", DEFAULT_TOPICS["cam_left_wrist"]),
            ("cam_right_wrist", DEFAULT_TOPICS["cam_right_wrist"]),
        ]:
            self.create_subscription(
                Image, topic,
                lambda msg, k=cam_key: self._on_image(k, msg),
                qos,
            )

        # Joint state subscribers
        self.create_subscription(
            JointState, DEFAULT_TOPICS["joint_states_left"],
            self._on_left_joints, qos,
        )
        self.create_subscription(
            JointState, DEFAULT_TOPICS["joint_states_right"],
            self._on_right_joints, qos,
        )

    def _setup_publishers(self) -> None:
        self.cmd_left_pub = self.create_publisher(
            Float64MultiArray, DEFAULT_TOPICS["cmd_left"], 10,
        )
        self.cmd_right_pub = self.create_publisher(
            Float64MultiArray, DEFAULT_TOPICS["cmd_right"], 10,
        )

    def _on_image(self, key: str, msg: Any) -> None:
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # Resize to model input size
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img).resize((self.image_size, self.image_size))
        self.obs_builder.push_image(key, np.asarray(pil))

    def _on_left_joints(self, msg: Any) -> None:
        self._left_joints = np.array(msg.position, dtype=np.float32)

    def _on_right_joints(self, msg: Any) -> None:
        self._right_joints = np.array(msg.position, dtype=np.float32)
        # When we have both arms, push combined state
        if self._left_joints is not None:
            combined = np.concatenate([self._left_joints, self._right_joints])
            self.obs_builder.push_state(combined)

    def step(self) -> bool:
        """One control loop iteration. Returns True if action was published."""
        if self.action_pub.needs_replan:
            obs = self.obs_builder.build(instruction=self.instruction)
            if obs is None:
                return False

            actions = self.client.predict_action(obs)
            self.action_pub.set_chunk(actions)

        action = self.action_pub.next_action()
        if action is None:
            return False

        # Split into left (0:7) and right (7:14) arm commands
        left_cmd = Float64MultiArray(data=action[:7].tolist())
        right_cmd = Float64MultiArray(data=action[7:14].tolist())

        if HAS_ROS2:
            self.cmd_left_pub.publish(left_cmd)
            self.cmd_right_pub.publish(right_cmd)

        return True


def main():
    parser = argparse.ArgumentParser(description="Aloha ROS2 adapter for Reflex VLA")
    parser.add_argument("--server", default="tcp://localhost:5555", help="Reflex serve URL")
    parser.add_argument("--instruction", default="", help="Task instruction")
    parser.add_argument("--hz", type=float, default=30.0, help="Control loop Hz")
    parser.add_argument("--replan-steps", type=int, default=5, help="Steps between replans")
    args = parser.parse_args()

    if not HAS_ROS2:
        print("ERROR: rclpy not found. Install ROS2 first.")
        print("This adapter requires a ROS2 environment (e.g. ros2 humble/iron).")
        sys.exit(1)

    rclpy.init()
    node = AlohaAdapterNode(
        server_url=args.server,
        instruction=args.instruction,
        control_hz=args.hz,
        replan_steps=args.replan_steps,
    )

    try:
        rate = node.create_rate(args.hz)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.step()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        node.client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
