"""UR3/UR5 ROS2 adapter — single-arm robot control via Reflex VLA.

Ported from FluxVLA ``ur_operator.py`` (623 LOC, Apache-2.0, LimX Dynamics).
Per R-2: modernized to use ``ur_robot_driver``'s canonical
``FollowJointTrajectory`` ActionClient (NOT FluxVLA's custom ``/cmd/movel``
topics which are UR-internal and deprecated).

Usage::

    # Terminal 1: GPU machine
    reflex serve ./my_export/ --transport zmq --port 5555

    # Terminal 2: Robot (ROS2 + ur_robot_driver)
    python3 contrib/ros2/ur3/adapter.py \
        --server tcp://gpu-server:5555 \
        --instruction "pick up the red cup"

Requires: rclpy, sensor_msgs, control_msgs, trajectory_msgs, cv_bridge, numpy
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, JointState
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from cv_bridge import CvBridge
    from builtin_interfaces.msg import Duration
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from _common.observation_builder import ObservationBuilder
from _common.action_publisher import ActionPublisher
from _common.reflex_client import ReflexClient


# Default UR topic names (ur_robot_driver convention)
DEFAULT_TOPICS = {
    "camera": "/camera/image_raw",
    "joint_states": "/joint_states",
    "follow_joint_trajectory": "/scaled_joint_trajectory_controller/follow_joint_trajectory",
}

UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR_ACTION_DIM = 7  # 6 joints + 1 gripper
UR_STATE_DIM = 7
UR_IMAGE_SIZE = 224


class URAdapterNode(Node if HAS_ROS2 else object):
    """ROS2 node that bridges UR sensors → Reflex VLA → UR actuators.

    Subscribes to camera + joint states, constructs observation,
    calls reflex serve, sends FollowJointTrajectory goals.
    """

    def __init__(
        self,
        server_url: str = "tcp://localhost:5555",
        instruction: str = "",
        control_hz: float = 10.0,
        replan_steps: int = 5,
        image_size: int = UR_IMAGE_SIZE,
        camera_topic: str = DEFAULT_TOPICS["camera"],
    ) -> None:
        if HAS_ROS2:
            super().__init__("reflex_ur_adapter")
        self.instruction = instruction
        self.image_size = image_size

        self.client = ReflexClient(server_url)
        self.obs_builder = ObservationBuilder(
            image_keys=["camera"],
            state_dim=UR_STATE_DIM,
        )
        self.action_pub = ActionPublisher(
            action_dim=UR_ACTION_DIM,
            replan_steps=replan_steps,
            control_hz=control_hz,
        )

        if HAS_ROS2:
            self.bridge = CvBridge()
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            self.create_subscription(
                Image, camera_topic, self._on_image, qos,
            )
            self.create_subscription(
                JointState, DEFAULT_TOPICS["joint_states"],
                self._on_joint_state, qos,
            )

            self._trajectory_client = ActionClient(
                self, FollowJointTrajectory,
                DEFAULT_TOPICS["follow_joint_trajectory"],
            )

        self._current_joints: np.ndarray | None = None
        self._gripper_state: float = 0.0

        self.get_logger().info(
            f"UR adapter ready: server={server_url}, instruction={instruction!r}"
        ) if HAS_ROS2 else None

    def _on_image(self, msg: Any) -> None:
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img).resize((self.image_size, self.image_size))
        self.obs_builder.push_image("camera", np.asarray(pil))

    def _on_joint_state(self, msg: Any) -> None:
        positions = np.array(msg.position[:6], dtype=np.float32)
        self._current_joints = positions
        state = np.concatenate([positions, [self._gripper_state]])
        self.obs_builder.push_state(state)

    def _send_joint_trajectory(self, target_joints: np.ndarray) -> None:
        """Send a FollowJointTrajectory goal to the UR driver."""
        if not HAS_ROS2:
            return

        goal = FollowJointTrajectory.Goal()
        point = JointTrajectoryPoint()
        point.positions = target_joints[:6].tolist()
        point.time_from_start = Duration(sec=0, nanosec=int(0.1 * 1e9))
        goal.trajectory.joint_names = UR_JOINT_NAMES
        goal.trajectory.points = [point]

        if self._trajectory_client.wait_for_server(timeout_sec=1.0):
            self._trajectory_client.send_goal_async(goal)

        # Handle gripper separately if action_dim includes it
        if len(target_joints) > 6:
            self._gripper_state = float(target_joints[6])

    def step(self) -> bool:
        """One control loop iteration."""
        if self.action_pub.needs_replan:
            obs = self.obs_builder.build(instruction=self.instruction)
            if obs is None:
                return False

            actions = self.client.predict_action(obs)
            self.action_pub.set_chunk(actions)

        action = self.action_pub.next_action()
        if action is None:
            return False

        self._send_joint_trajectory(action)
        return True


def main():
    parser = argparse.ArgumentParser(description="UR3/UR5 ROS2 adapter for Reflex VLA")
    parser.add_argument("--server", default="tcp://localhost:5555", help="Reflex serve URL")
    parser.add_argument("--instruction", default="", help="Task instruction")
    parser.add_argument("--hz", type=float, default=10.0, help="Control loop Hz")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--camera-topic", default=DEFAULT_TOPICS["camera"])
    args = parser.parse_args()

    if not HAS_ROS2:
        print("ERROR: rclpy not found. Install ROS2 first.")
        sys.exit(1)

    rclpy.init()
    node = URAdapterNode(
        server_url=args.server,
        instruction=args.instruction,
        control_hz=args.hz,
        replan_steps=args.replan_steps,
        camera_topic=args.camera_topic,
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
