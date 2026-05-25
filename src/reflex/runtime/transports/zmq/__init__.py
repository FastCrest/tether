"""ZMQ transport for `reflex serve --transport zmq` (Lift #2).

See ``src/reflex/runtime/transports/zmq/policy_server.py`` for Layer 1
(generic socket server) and ``factory.py`` for Layer 2 (PolicyRuntime wiring).
"""
from __future__ import annotations

__all__: list[str] = []
