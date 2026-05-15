"""Tests for the reflex ros2 bridge.

rclpy isn't pip-installable, so these tests mock the minimal ROS2 surface
(rclpy, rclpy.node, sensor_msgs.msg, std_msgs.msg) to verify the bridge
module's internal logic without a real ROS2 install.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestServeRos2Flag:
    """`reflex serve --ros2` short-circuits the HTTP path and hands off
    to the ROS2 bridge. Verifies the CLI wiring invokes run_ros2_bridge
    and doesn't spin up uvicorn / create_app.
    """

    def test_ros2_flag_routes_to_bridge(self, monkeypatch, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        (tmp_path / "reflex_config.json").write_text(
            '{"model_type": "smolvla", "target": "desktop"}'
        )

        calls = {"bridge": 0, "app_created": 0}

        def fake_run_bridge(*args, **kwargs):
            calls["bridge"] += 1
            assert str(args[0]) == str(tmp_path)

        import reflex.runtime.ros2_bridge as bridge_mod
        monkeypatch.setattr(bridge_mod, "run_ros2_bridge", fake_run_bridge)

        def boom_create_app(*args, **kwargs):
            calls["app_created"] += 1
            raise AssertionError("create_app should NOT run in --ros2 mode")

        import reflex.runtime.server as server_mod
        monkeypatch.setattr(server_mod, "create_app", boom_create_app)

        from typer.testing import CliRunner
        from reflex.cli import app as cli_app

        runner = CliRunner()
        result = runner.invoke(cli_app, ["serve", str(tmp_path), "--ros2"])
        assert result.exit_code == 0, result.output
        assert calls["bridge"] == 1
        assert calls["app_created"] == 0


def _install_fake_rclpy(monkeypatch):
    """Register stub rclpy + message modules in sys.modules for this test."""
    rclpy = types.ModuleType("rclpy")
    rclpy.init = MagicMock()
    rclpy.shutdown = MagicMock()
    rclpy.spin = MagicMock()
    rclpy.ok = lambda: True

    rclpy_node = types.ModuleType("rclpy.node")

    class FakeNode:
        def __init__(self, name):
            self._name = name
            self._subs: list = []
            self._pubs: list = []
            self._timers: list = []

        def create_subscription(self, *a, **k):
            sub = MagicMock()
            self._subs.append(sub)
            return sub

        def create_publisher(self, *a, **k):
            pub = MagicMock()
            self._pubs.append(pub)
            return pub

        def create_timer(self, *a, **k):
            t = MagicMock()
            self._timers.append(t)
            return t

        def destroy_node(self):
            pass

        def get_logger(self):
            lg = MagicMock()
            lg.info = lambda *a, **k: None
            lg.warning = lambda *a, **k: None
            lg.error = lambda *a, **k: None
            return lg

    rclpy_node.Node = FakeNode

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = type("Image", (), {})
    sensor_msgs_msg.JointState = type("JointState", (), {})
    sensor_msgs_msg.Imu = type("Imu", (), {})

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})

    class FakeF32Array:
        def __init__(self):
            self.data: list[float] = []
    std_msgs_msg.Float32MultiArray = FakeF32Array

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "nav_msgs", nav_msgs)
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav_msgs_msg)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)


def test_import_without_rclpy_raises_helpfully():
    # Ensure rclpy isn't accidentally in sys.modules
    for k in ("rclpy", "rclpy.node", "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg"):
        sys.modules.pop(k, None)

    from reflex.runtime.ros2_bridge import _require_rclpy
    with pytest.raises(ImportError) as ei:
        _require_rclpy()
    assert "ROS2" in str(ei.value)
    assert "humble" in str(ei.value)


def test_node_construction(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    server = MagicMock()
    node = create_ros2_bridge_node(
        server,
        image_topic="/foo/image",
        rate_hz=10.0,
        node_name="test_reflex",
    )
    assert node._name == "test_reflex"
    # 3 subs (image, state, task), 2 pubs (actions, e_stop), 1 timer.
    # e_stop publisher backs the ROS2Context.publish_e_stop used by the
    # ros2-mcp-bridge feature; present regardless of MCP wiring.
    assert len(node._subs) == 3
    assert len(node._pubs) == 2
    assert len(node._timers) == 1


def test_image_callback_rgb8(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    node = create_ros2_bridge_node(MagicMock())

    msg = MagicMock()
    msg.height = 4
    msg.width = 4
    msg.encoding = "rgb8"
    msg.data = np.zeros(4 * 4 * 3, dtype=np.uint8).tobytes()
    node._image_cb(msg)
    assert node._last_image is not None
    assert node._last_image.shape == (4, 4, 3)


def test_image_callback_bgr8_reverses(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    node = create_ros2_bridge_node(MagicMock())

    # Pattern that's distinct in rgb vs bgr
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 10   # channel 0 (R in rgb, B in bgr)
    img[..., 2] = 200  # channel 2 (B in rgb, R in bgr)
    msg = MagicMock()
    msg.height = 2
    msg.width = 2
    msg.encoding = "bgr8"
    msg.data = img.tobytes()
    node._image_cb(msg)
    # After bgr -> rgb conversion, channel 0 should be 200, channel 2 should be 10
    assert node._last_image[0, 0, 0] == 200
    assert node._last_image[0, 0, 2] == 10


def test_state_callback(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    node = create_ros2_bridge_node(MagicMock())
    msg = MagicMock()
    msg.position = [0.1, 0.2, 0.3]
    node._state_cb(msg)
    assert node._last_state == [0.1, 0.2, 0.3]


def test_tick_invokes_server_and_publishes(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    server = MagicMock()
    server.predict.return_value = {
        "actions": [[0.1, 0.2], [0.3, 0.4]],
        "latency_ms": 5.0,
    }
    node = create_ros2_bridge_node(server)
    # Prime with cached image + state
    node._last_image = np.zeros((4, 4, 3), dtype=np.uint8)
    node._last_state = [0.0, 0.1]
    node._last_task = "pick it up"

    node._tick()
    server.predict.assert_called_once()
    call_kwargs = server.predict.call_args.kwargs
    assert call_kwargs["instruction"] == "pick it up"
    assert call_kwargs["state"] == [0.0, 0.1]
    # Action published: pub.publish called with a Float32MultiArray
    node._action_pub.publish.assert_called_once()
    published = node._action_pub.publish.call_args.args[0]
    assert published.data == [0.1, 0.2, 0.3, 0.4]
    assert node._inference_count == 1


def test_tick_skips_when_no_image(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    server = MagicMock()
    node = create_ros2_bridge_node(server)
    # No image cached
    node._last_state = [0.0]
    node._tick()
    server.predict.assert_not_called()


def test_tick_handles_server_error_gracefully(monkeypatch):
    _install_fake_rclpy(monkeypatch)
    from reflex.runtime.ros2_bridge import create_ros2_bridge_node

    server = MagicMock()
    server.predict.return_value = {"error": "guard_tripped"}
    node = create_ros2_bridge_node(server)
    node._last_image = np.zeros((4, 4, 3), dtype=np.uint8)
    node._last_state = [0.0]

    node._tick()
    # predict called but no publish happened
    server.predict.assert_called_once()
    node._action_pub.publish.assert_not_called()


# ---------------------------------------------------------------------------
# State extractor unit tests — exercise the decoupled extractor functions
# directly with SimpleNamespace mocks. No rclpy install needed.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


class TestJointStateExtractor:
    def test_extracts_position_field(self):
        from reflex.runtime.ros2_bridge import _extract_joint_state
        msg = SimpleNamespace(position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        state = _extract_joint_state(msg)
        assert state == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_normalises_to_floats(self):
        from reflex.runtime.ros2_bridge import _extract_joint_state
        msg = SimpleNamespace(position=(1, 2, 3))
        state = _extract_joint_state(msg)
        assert state == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in state)


class TestImuExtractor:
    def test_extracts_quaternion_xyzw(self):
        """ROS REP-103 convention — quaternion stored as (x, y, z, w)."""
        from reflex.runtime.ros2_bridge import _extract_imu
        msg = SimpleNamespace(
            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
        )
        assert _extract_imu(msg) == [0.1, 0.2, 0.3, 0.9]

    def test_output_is_four_dof_partial_state(self):
        from reflex.runtime.ros2_bridge import _extract_imu
        msg = SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        assert len(_extract_imu(msg)) == 4


class TestOdometryExtractor:
    @staticmethod
    def _mock_odom(*, p, o, v):
        return SimpleNamespace(
            pose=SimpleNamespace(pose=SimpleNamespace(
                position=SimpleNamespace(x=p[0], y=p[1], z=p[2]),
                orientation=SimpleNamespace(x=o[0], y=o[1], z=o[2], w=o[3]),
            )),
            twist=SimpleNamespace(twist=SimpleNamespace(
                linear=SimpleNamespace(x=v[0], y=v[1], z=v[2]),
            )),
        )

    def test_extracts_pos_orient_vel_in_order(self):
        from reflex.runtime.ros2_bridge import _extract_odom
        msg = self._mock_odom(
            p=(1.0, 2.0, 3.0),
            o=(0.0, 0.0, 0.0, 1.0),
            v=(0.1, 0.2, 0.3),
        )
        assert _extract_odom(msg) == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3]

    def test_output_is_ten_dof(self):
        from reflex.runtime.ros2_bridge import _extract_odom
        msg = self._mock_odom(p=(0, 0, 0), o=(0, 0, 0, 1), v=(0, 0, 0))
        assert len(_extract_odom(msg)) == 10

    def test_matches_quadcopter_preset_state_dim(self):
        """Cross-layer pin: if the quadcopter preset state_dim ever drifts
        from the odom extractor length, this breaks at CI time at the right
        layer to catch which side moved."""
        from reflex.embodiments import EmbodimentConfig
        from reflex.runtime.ros2_bridge import _extract_odom
        cfg = EmbodimentConfig.load_preset("quadcopter")
        msg = self._mock_odom(p=(0, 0, 0), o=(0, 0, 0, 1), v=(0, 0, 0))
        assert len(_extract_odom(msg)) == cfg.state_dim


class TestRegistryAndResolution:
    def test_all_documented_types_registered(self):
        from reflex.runtime.ros2_bridge import _STATE_EXTRACTORS
        assert set(_STATE_EXTRACTORS) == {"joint_state", "imu", "odom"}

    def test_resolve_unknown_type_raises(self):
        from reflex.runtime.ros2_bridge import _resolve_state_msg_class
        with pytest.raises(ValueError, match="unknown state_msg_type"):
            _resolve_state_msg_class("not_a_real_msg_type")


# ---------------------------------------------------------------------------
# Integration tests — verify create_ros2_bridge_node dispatches the right
# extractor for each state_msg_type and surfaces mismatch warnings.
# ---------------------------------------------------------------------------


class TestStateMsgTypeDispatch:
    def test_default_is_joint_state(self, monkeypatch):
        _install_fake_rclpy(monkeypatch)
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node
        node = create_ros2_bridge_node(MagicMock())
        msg = SimpleNamespace(position=[0.1, 0.2, 0.3])
        node._state_cb(msg)
        assert node._last_state == [0.1, 0.2, 0.3]

    def test_imu_dispatch(self, monkeypatch):
        _install_fake_rclpy(monkeypatch)
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node
        node = create_ros2_bridge_node(
            MagicMock(), state_msg_type="imu", state_topic="/mavros/imu/data",
        )
        msg = SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        node._state_cb(msg)
        assert node._last_state == [0.0, 0.0, 0.0, 1.0]

    def test_odom_dispatch(self, monkeypatch):
        _install_fake_rclpy(monkeypatch)
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node
        node = create_ros2_bridge_node(
            MagicMock(),
            state_msg_type="odom",
            state_topic="/mavros/local_position/odom",
        )
        msg = TestOdometryExtractor._mock_odom(
            p=(1.0, 2.0, 3.0),
            o=(0.0, 0.0, 0.0, 1.0),
            v=(0.1, 0.2, 0.3),
        )
        node._state_cb(msg)
        assert node._last_state == [
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3,
        ]

    def test_unknown_msg_type_raises(self, monkeypatch):
        _install_fake_rclpy(monkeypatch)
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node
        with pytest.raises(ValueError, match="unknown state_msg_type"):
            create_ros2_bridge_node(MagicMock(), state_msg_type="lidar")

    def test_extractor_failure_logged_not_raised(self, monkeypatch):
        """A malformed message must not crash the node — log and skip."""
        _install_fake_rclpy(monkeypatch)
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node
        node = create_ros2_bridge_node(MagicMock(), state_msg_type="imu")
        # IMU extractor expects msg.orientation — feed a JointState-shaped msg
        bad_msg = SimpleNamespace(position=[0.1, 0.2])
        node._state_cb(bad_msg)
        # _last_state stays None, no exception
        assert node._last_state is None

    def test_state_dim_mismatch_warns_once(self, monkeypatch):
        """When a loaded embodiment expects state_dim=10 (quadcopter) but
        the extractor returns 4 (imu), surface a warning. Fires once per
        node — the same mismatch every tick would spam the log."""
        _install_fake_rclpy(monkeypatch)
        from reflex.embodiments import EmbodimentConfig
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node

        server = MagicMock()
        server.embodiment_config = EmbodimentConfig.load_preset("quadcopter")

        warnings: list[str] = []
        original_create = create_ros2_bridge_node

        node = original_create(server, state_msg_type="imu")
        # Capture warnings from the node's logger (FakeNode.get_logger returns
        # a fresh MagicMock each call, so monkey-patch at the node level).
        node.get_logger = lambda warnings_ref=warnings: SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda m, *a, **k: warnings_ref.append(str(m)),
            error=lambda *a, **k: None,
        )

        msg = SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        node._state_cb(msg)
        node._state_cb(msg)  # second tick — should NOT re-warn
        node._state_cb(msg)
        mismatch_warnings = [
            w for w in warnings if "does NOT match embodiment state_dim" in w
        ]
        assert len(mismatch_warnings) == 1, mismatch_warnings

    def test_state_dim_match_no_warning(self, monkeypatch):
        """When extractor output length matches embodiment state_dim, no
        warning fires."""
        _install_fake_rclpy(monkeypatch)
        from reflex.embodiments import EmbodimentConfig
        from reflex.runtime.ros2_bridge import create_ros2_bridge_node

        server = MagicMock()
        server.embodiment_config = EmbodimentConfig.load_preset("quadcopter")

        warnings: list[str] = []
        node = create_ros2_bridge_node(server, state_msg_type="odom")
        node.get_logger = lambda warnings_ref=warnings: SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda m, *a, **k: warnings_ref.append(str(m)),
            error=lambda *a, **k: None,
        )
        msg = TestOdometryExtractor._mock_odom(
            p=(0, 0, 0), o=(0, 0, 0, 1), v=(0, 0, 0),
        )
        node._state_cb(msg)
        mismatch_warnings = [
            w for w in warnings if "does NOT match embodiment state_dim" in w
        ]
        assert mismatch_warnings == []
