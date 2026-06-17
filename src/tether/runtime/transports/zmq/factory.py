"""Layer 2: create_zmq_server — wires PolicyRuntime into a PolicyServer.

Ported from FluxVLA ``zmq_server.py:185-284`` (Apache-2.0, LimX Dynamics).
Simplified because tether's ``PolicyRuntime`` already handles preprocessing,
device placement, mixed precision, and denormalization — FluxVLA's factory
had to wire each of those manually.

Usage::

    from tether.runtime.transports.zmq.factory import create_zmq_server

    server = create_zmq_server(runtime, port=5555)
    server.run()  # blocks
"""
from __future__ import annotations

import io
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from tether.runtime.transports.zmq.policy_server import PolicyServer

logger = logging.getLogger(__name__)


def create_zmq_server(
    runtime: Any,
    *,
    host: str = "*",
    port: int = 5555,
    curve_server_cert: str | Path | None = None,
    curve_client_cert_dir: str | Path | None = None,
    control_token: str | None = None,
) -> PolicyServer:
    """Create a ZMQ server that wraps a tether PolicyRuntime.

    Registers three endpoints on the PolicyServer:

    - ``predict_action`` — runs inference via ``runtime.predict_async``
      (or ``predict_action_chunk`` depending on runtime type). Returns
      serialized actions + inference time.
    - ``reset`` — no-op acknowledgement (robot-side uses this to sync
      episode boundaries).
    - ``get_status`` — uptime + request count + average inference time.

    Args:
        runtime: A tether ``PolicyRuntime`` or any object with a
            ``predict_action_chunk`` / ``predict_async`` method.
        host: Bind address.
        port: Bind port.
        curve_server_cert: Optional pyzmq CURVE server secret certificate.
        curve_client_cert_dir: Directory of allowed client public certificates.
        control_token: Optional token required for built-in control endpoints.

    Returns:
        A configured ``PolicyServer`` ready to ``run()``.
    """
    lock = threading.Lock()
    total_requests = 0
    total_infer_time = 0.0
    start_time = time.time()

    def predict_action(**kwargs: Any) -> dict:
        nonlocal total_requests, total_infer_time

        t0 = time.perf_counter()

        if hasattr(runtime, "predict_async"):
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(runtime.predict_async(kwargs))
            finally:
                loop.close()
            actions = result.get("actions") if isinstance(result, dict) else result
        elif hasattr(runtime, "predict_action_chunk"):
            actions = runtime.predict_action_chunk(kwargs)
        else:
            raise AttributeError(
                f"runtime {type(runtime).__name__} has neither predict_async "
                f"nor predict_action_chunk"
            )

        infer_time = time.perf_counter() - t0

        # Serialize actions to numpy bytes
        if hasattr(actions, "cpu"):
            actions_np = actions.detach().cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions_np = actions
        else:
            actions_np = np.asarray(actions)

        buf = io.BytesIO()
        np.save(buf, actions_np, allow_pickle=False)
        action_bytes = buf.getvalue()

        with lock:
            total_requests += 1
            total_infer_time += infer_time
            n = total_requests
            should_log = (n % 50 == 0)
            avg = total_infer_time / n if should_log else 0.0

        if should_log:
            logger.info(
                "ZMQ req=%d infer=%.1fms avg=%.1fms",
                n, infer_time * 1000, avg * 1000,
            )

        return {
            "action_data": action_bytes,
            "infer_time_ms": round(infer_time * 1000, 2),
        }

    def reset() -> dict:
        return {"status": "ok"}

    def get_status() -> dict:
        with lock:
            n = total_requests
            avg = (total_infer_time / n) if n > 0 else 0.0
        return {
            "status": "ready",
            "uptime_s": round(time.time() - start_time, 2),
            "total_requests": n,
            "avg_infer_time_ms": round(avg * 1000, 2),
        }

    server = PolicyServer(
        host=host,
        port=port,
        curve_server_cert=curve_server_cert,
        curve_client_cert_dir=curve_client_cert_dir,
        control_token=control_token,
    )
    server.register_endpoint("predict_action", predict_action)
    server.register_endpoint("reset", reset, requires_input=False)
    server.register_endpoint("get_status", get_status, requires_input=False)

    return server


__all__ = ["create_zmq_server"]
