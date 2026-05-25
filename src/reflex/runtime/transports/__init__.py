"""Transport layer for `reflex serve` (HTTP, ZMQ, future ROS2).

Lift #2 per ``features/01_serve/zmq-transport.md``. The transport layer
decouples the wire protocol from the inference runtime — PolicyRuntime
produces actions; the transport delivers them to the robot client.

Available transports:
- ``http`` (default) — FastAPI + uvicorn, the existing `reflex serve` path
- ``zmq`` (Lift #2) — ZeroMQ REP socket + msgpack + JPEG-on-wire
- ``ros2`` (v1.0, not yet implemented) — ROS2 action server
"""
from __future__ import annotations

__all__: list[str] = []
