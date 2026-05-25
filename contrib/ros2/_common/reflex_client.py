"""Shared reflex serve client for ROS2 robot adapters.

Wraps either the ZMQ client (preferred for production) or HTTP client
(fallback) with a unified interface. Robot adapters call
``predict_action(obs)`` without caring about the transport.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ReflexClient:
    """Unified client for calling reflex serve from ROS2 adapters.

    Args:
        server_url: Server URL. Format determines transport:
            - ``tcp://host:port`` → ZMQ (preferred)
            - ``http://host:port`` → HTTP fallback
        timeout_ms: Request timeout in milliseconds.
    """

    def __init__(
        self,
        server_url: str = "tcp://localhost:5555",
        timeout_ms: int = 5000,
    ) -> None:
        self._server_url = server_url
        self._timeout_ms = timeout_ms
        self._client: Any = None
        self._transport: str = ""
        self._connect()

    def _connect(self) -> None:
        if self._server_url.startswith("tcp://"):
            try:
                from reflex.runtime.transports.zmq.client import ZmqRuntimeClient
                self._client = ZmqRuntimeClient(
                    self._server_url, timeout_ms=self._timeout_ms,
                )
                self._transport = "zmq"
                logger.info("Connected via ZMQ: %s", self._server_url)
            except ImportError:
                raise ImportError(
                    "ZMQ transport requires: pip install pyzmq msgpack opencv-python-headless"
                )
        elif self._server_url.startswith("http"):
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self._server_url,
                    timeout=self._timeout_ms / 1000,
                )
                self._transport = "http"
                logger.info("Connected via HTTP: %s", self._server_url)
            except ImportError:
                raise ImportError("HTTP transport requires: pip install httpx")
        else:
            raise ValueError(
                f"Unknown URL scheme: {self._server_url!r}. "
                "Use tcp:// for ZMQ or http:// for HTTP."
            )

    def predict_action(self, obs: dict[str, Any]) -> np.ndarray:
        """Send observation, receive action chunk.

        Args:
            obs: Observation dict with images + state + task.

        Returns:
            np.ndarray of shape [chunk_size, action_dim].
        """
        if self._transport == "zmq":
            return self._client.predict_action(obs)
        elif self._transport == "http":
            import io
            import json
            import base64

            payload: dict = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    buf = io.BytesIO()
                    np.save(buf, v, allow_pickle=False)
                    payload[k] = {"__numpy_b64__": base64.b64encode(buf.getvalue()).decode()}
                else:
                    payload[k] = v

            resp = self._client.post("/act", json=payload)
            resp.raise_for_status()
            data = resp.json()
            action_b64 = data.get("actions_b64", data.get("actions"))
            if isinstance(action_b64, str):
                return np.load(io.BytesIO(base64.b64decode(action_b64)))
            return np.array(action_b64)
        else:
            raise RuntimeError(f"Unknown transport: {self._transport}")

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "ReflexClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


__all__ = ["ReflexClient"]
