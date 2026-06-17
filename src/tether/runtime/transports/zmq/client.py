"""ZmqRuntimeClient — robot-side connector for `tether serve --transport zmq`.

Designed for the robot's onboard computer where install size matters:
``pip install pyzmq msgpack numpy opencv-python-headless`` (~25 MB) is
all you need. No torch, no onnxruntime, no FastAPI.

Usage::

    from tether.runtime.transports.zmq.client import ZmqRuntimeClient

    client = ZmqRuntimeClient("tcp://gpu-server:5555")
    obs = {
        "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "task": "pick up the red cup",
    }
    actions = client.predict_action(obs)
    # With profiling:
    actions, profile = client.predict_action(obs, with_profile=True)
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq

from tether.runtime.transports.zmq.serializers import (
    SCHEMA_VERSION,
    encode_observation,
)
from tether.runtime.transports.zmq.security import load_curve_key


@dataclass
class ProfileData:
    """Per-request timing decomposition (all values in milliseconds)."""
    serialize_ms: float
    zmq_roundtrip_ms: float
    server_infer_ms: float
    deserialize_ms: float
    total_ms: float


class ZmqRuntimeClient:
    """Persistent-socket ZMQ client for robot-side inference requests.

    Connects once, reuses the socket across calls. Reconnects on failure.

    Args:
        server_url: ZMQ endpoint (e.g. ``"tcp://gpu-server:5555"``).
        timeout_ms: Per-request timeout in milliseconds. 0 = no timeout.
        jpeg_quality: JPEG quality for whitelisted image keys (1-100).
    """

    def __init__(
        self,
        server_url: str = "tcp://localhost:5555",
        timeout_ms: int = 5000,
        jpeg_quality: int = 85,
        *,
        curve_client_cert: str | Path | None = None,
        curve_client_public_key: str | bytes | None = None,
        curve_client_secret_key: str | bytes | None = None,
        curve_server_public_key: str | bytes | Path | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._server_url = server_url
        self._timeout_ms = timeout_ms
        self._jpeg_quality = jpeg_quality
        self._curve_client_cert = curve_client_cert
        self._curve_client_public_key = curve_client_public_key
        self._curve_client_secret_key = curve_client_secret_key
        self._curve_server_public_key = curve_server_public_key
        self._auth_token = auth_token
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        if self._timeout_ms > 0:
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._configure_curve()
        self._socket.connect(self._server_url)

    def _configure_curve(self) -> None:
        if self._curve_server_public_key is None:
            if (
                self._curve_client_cert is not None
                or self._curve_client_public_key is not None
                or self._curve_client_secret_key is not None
            ):
                raise ValueError("CURVE client mode requires curve_server_public_key")
            return

        if self._socket is None:
            raise RuntimeError("ZMQ socket has not been initialized")

        if self._curve_client_cert is not None:
            if self._curve_client_public_key is not None or self._curve_client_secret_key is not None:
                raise ValueError("Pass either curve_client_cert or explicit CURVE keys, not both")
            client_public_key = load_curve_key(self._curve_client_cert, secret=False)
            client_secret_key = load_curve_key(self._curve_client_cert, secret=True)
        else:
            if self._curve_client_public_key is None or self._curve_client_secret_key is None:
                raise ValueError("CURVE client mode requires both public and secret keys")
            client_public_key = load_curve_key(self._curve_client_public_key, secret=False)
            client_secret_key = load_curve_key(self._curve_client_secret_key, secret=True)

        self._socket.curve_publickey = client_public_key
        self._socket.curve_secretkey = client_secret_key
        self._socket.curve_serverkey = load_curve_key(
            self._curve_server_public_key,
            secret=False,
        )

    def _request(self, endpoint: str, data: dict[str, Any] | None = None) -> dict:
        request: dict[str, Any] = {
            "endpoint": endpoint,
            "schema_version": SCHEMA_VERSION,
        }
        if data is not None:
            request["data"] = data
        if self._auth_token is not None:
            request["auth_token"] = self._auth_token

        self._socket.send(msgpack.packb(request, use_bin_type=True))
        return msgpack.unpackb(self._socket.recv(), raw=False)

    def predict_action(
        self,
        obs: dict[str, Any],
        *,
        with_profile: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ProfileData]:
        """Send observation, receive action chunk.

        Args:
            obs: Observation dict. Images (ndim==3, uint8) in the JPEG
                whitelist get compressed automatically.
            with_profile: If True, return ``(actions, profile)`` tuple
                with per-stage timing.

        Returns:
            ``np.ndarray`` of shape ``[chunk_size, action_dim]`` (default),
            or ``(actions, ProfileData)`` if ``with_profile=True``.

        Raises:
            zmq.Again: timeout waiting for server response.
            RuntimeError: server returned an error.
        """
        t_total_start = time.perf_counter()

        # Serialize
        t0 = time.perf_counter()
        obs_bytes = encode_observation(obs, jpeg_quality=self._jpeg_quality)
        request = {
            "endpoint": "predict_action",
            "schema_version": SCHEMA_VERSION,
            "data": {"obs_data": obs_bytes},
        }
        if self._auth_token is not None:
            request["auth_token"] = self._auth_token
        request_bytes = msgpack.packb(request, use_bin_type=True)
        serialize_ms = (time.perf_counter() - t0) * 1000

        # ZMQ round-trip
        t0 = time.perf_counter()
        self._socket.send(request_bytes)
        response_bytes = self._socket.recv()
        zmq_roundtrip_ms = (time.perf_counter() - t0) * 1000

        # Deserialize response
        t0 = time.perf_counter()
        response = msgpack.unpackb(response_bytes, raw=False)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

        action_data = response["action_data"]
        actions = np.load(io.BytesIO(action_data))
        server_infer_ms = response.get("infer_time_ms", 0.0)
        deserialize_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total_start) * 1000

        if with_profile:
            profile = ProfileData(
                serialize_ms=round(serialize_ms, 3),
                zmq_roundtrip_ms=round(zmq_roundtrip_ms, 3),
                server_infer_ms=round(server_infer_ms, 3),
                deserialize_ms=round(deserialize_ms, 3),
                total_ms=round(total_ms, 3),
            )
            return actions, profile
        return actions

    def ping(self) -> dict:
        """Health check — returns server status dict."""
        return self._request("ping")

    def reset(self) -> dict:
        """Signal episode boundary to the server."""
        return self._request("reset")

    def kill(self) -> dict:
        """Request graceful server shutdown."""
        return self._request("kill")

    def close(self) -> None:
        """Clean up socket + context."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._context.term()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "ZmqRuntimeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


__all__ = ["ZmqRuntimeClient", "ProfileData"]
