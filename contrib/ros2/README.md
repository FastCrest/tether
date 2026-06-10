# ROS2 Robot Adapter Starter Kits

**Fork-and-customize** adapters for running Reflex VLA on real robots via ROS2.
These are NOT part of the core `pip install tether` — they live in `contrib/`
and you're expected to adapt them to your specific robot setup.

## Available Adapters

| Robot | Directory | Cameras | Action Space |
|---|---|---|---|
| **Aloha** (Trossen/ALOHA 2) | `contrib/ros2/aloha/` | 3 cameras (cam_high, cam_left_wrist, cam_right_wrist) | Bimanual joint positions (14 DOF) |
| **UR3/UR5** (Universal Robots) | `contrib/ros2/ur3/` | Configurable | 6 DOF joint + gripper |

## Quick Start

```bash
# 1. Start tether serve on a GPU machine
tether serve ./my_export/ --transport zmq --port 5555

# 2. On the robot (ROS2 machine), run the adapter
cd contrib/ros2/aloha/
ros2 run reflex_aloha_adapter adapter_node \
  --ros-args -p server_url:=tcp://gpu-server:5555 \
  -p instruction:="pick up the red cup"
```

## Architecture

```
┌─────────────┐     HTTP/ZMQ      ┌──────────────┐
│ Robot (ROS2) │ ────────────────> │ GPU Server   │
│              │                   │ tether serve │
│ adapter_node │ <──────────────── │ --transport  │
│ (this code)  │     actions       │   zmq/http   │
└─────────────┘                   └──────────────┘
```

The adapter subscribes to ROS2 sensor topics, constructs the observation dict,
calls `tether serve`'s `/act` endpoint, and publishes actions to actuator topics.

## Attribution

Ported from FluxVLA Engine (LimX Dynamics, Apache-2.0):
- `aloha_operator.py` → `contrib/ros2/aloha/`
- `ur_operator.py` → `contrib/ros2/ur3/`

Tron2 adapter KILLED per Lift #8 decision K2 (vendor-specific, LimX only).

## Shared Utilities

`contrib/ros2/_common/` contains shared code used by both adapters:
- `observation_builder.py` — sensor sync + observation dict construction
- `action_publisher.py` — action chunk → joint command publishing
- `tether_client.py` — HTTP/ZMQ client wrapper for calling tether serve
