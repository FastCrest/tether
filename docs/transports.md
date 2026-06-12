# Transport Layer

Tether supports multiple wire transports for `tether serve`. The transport
decouples the wire protocol from the inference runtime — PolicyRuntime
produces actions; the transport delivers them to the robot client.

## Available Transports

| Transport | Flag | When to use | Install |
|---|---|---|---|
| **HTTP** (default) | `--transport http` | Standard REST API. Works with any HTTP client (curl, Python requests, browser). Best for prototyping + debugging. | `pip install fastcrest-tether[serve]` |
| **ZMQ** | `--transport zmq` | Low-latency binary wire. 20× lower bandwidth for multi-camera setups. 10× smaller robot-side install. Best for production robot deployments where every millisecond matters. | Server: `pip install fastcrest-tether[serve]`. Robot: `pip install pyzmq msgpack numpy opencv-python-headless` (~25 MB) |
| **ROS2** | (v1.0) | Native ROS2 action server. Reserved for v1.0. | — |

## Quick Start

### HTTP (default)

```bash
# Server
tether serve ./my_export/ --port 8000

# Client (any language)
curl -X POST http://localhost:8000/act \
  -H "Content-Type: application/json" \
  -d '{"observation": {...}, "instruction": "pick up the cup"}'
```

### ZMQ

```bash
# Local dev server. Non-loopback ZMQ binds require Secure ZMQ below.
tether serve ./my_export/ --transport zmq --host 127.0.0.1 --port 5555
```

```python
# Client (robot, 25 MB install)
from tether.runtime.transports.zmq.client import ZmqRuntimeClient
import numpy as np

client = ZmqRuntimeClient("tcp://gpu-server:5555")
obs = {
    "agentview_image": camera.read(),  # 224x224x3 uint8
    "robot0_eef_pos": robot.get_eef_pos(),
    "task": "pick up the red cup",
}
actions = client.predict_action(obs)
robot.execute(actions[0])  # first action in the chunk
```

### Secure ZMQ

Production ZMQ deployments should use CURVE authentication/encryption and a
control token for operational endpoints such as `ping` and `kill`.
`tether serve --transport zmq` refuses non-loopback binds unless both are
configured. For isolated lab networks only, operators can pass
`--zmq-insecure-ok` to make that risk explicit.

Generate one server keypair and one client keypair:

```bash
python - <<'PY'
from pathlib import Path
import zmq.auth

out = Path("zmq-certs")
(out / "clients").mkdir(parents=True, exist_ok=True)
zmq.auth.create_certificates(out, "server")
zmq.auth.create_certificates(out / "clients", "robot-1")
PY
```

Start the server with the server secret certificate and the directory of
allowed client public certificates:

```bash
tether serve ./my_export/ \
  --transport zmq \
  --port 5555 \
  --zmq-server-cert ./zmq-certs/server.key_secret \
  --zmq-client-cert-dir ./zmq-certs/clients \
  --zmq-control-token "$TETHER_ZMQ_CONTROL_TOKEN"
```

Configure the robot-side client with its secret certificate, the server public
certificate, and the same control token:

```python
import os

client = ZmqRuntimeClient(
    "tcp://gpu-server:5555",
    curve_client_cert="./zmq-certs/clients/robot-1.key_secret",
    curve_server_public_key="./zmq-certs/server.key",
    auth_token=os.environ["TETHER_ZMQ_CONTROL_TOKEN"],
)
```

## ZMQ Performance

| Metric | HTTP | ZMQ | ZMQ + JPEG |
|---|---|---|---|
| Payload (3-cam 224²) | ~1.2 MB | ~450 KB | ~60 KB |
| Tail jitter (p99-p50) | baseline | ~40% lower | ~40% lower |
| Robot-side install | ~250 MB | 25 MB | 25 MB |

## Profiling

Both transports support per-request timing decomposition:

```python
# ZMQ
actions, profile = client.predict_action(obs, with_profile=True)
print(f"serialize: {profile.serialize_ms:.1f}ms")
print(f"roundtrip: {profile.zmq_roundtrip_ms:.1f}ms")
print(f"inference: {profile.server_infer_ms:.1f}ms")
print(f"deserialize: {profile.deserialize_ms:.1f}ms")
print(f"total: {profile.total_ms:.1f}ms")
```

## JPEG Compression

ZMQ transport automatically JPEG-compresses camera images (uint8, 3-channel)
whose key is in the whitelist. This reduces bandwidth ~20× with < 0.1%
quality loss on real camera images.

**Whitelisted keys:** `agentview_image`, `robot0_eye_in_hand_image`,
`cam_high`, `cam_left_wrist`, `cam_right_wrist`, `base`, `wrist_l`,
`wrist_r`, `external`, `observation.images.image`, `observation.images.image2`.

Non-whitelisted uint8 3D arrays produce a one-time warning suggesting
you add the key to the whitelist.

## Schema Versioning

Every ZMQ message includes `schema_version: 1`. If the client and server
disagree on the version, a `WireSchemaMismatchError` is raised with an
upgrade message. This prevents silent wire-format drift when upgrading
server or client independently.
