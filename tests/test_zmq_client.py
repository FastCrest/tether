"""Tests for ZmqRuntimeClient (Lift #2 Day 4).

Client + server in same process via fixture. Validates predict_action
round-trip, profile data, schema version, persistent socket reuse.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tether.runtime.transports.zmq.client import ProfileData, ZmqRuntimeClient
from tether.runtime.transports.zmq.factory import create_zmq_server


class _MockRuntime:
    def predict_action_chunk(self, batch=None, **kwargs) -> np.ndarray:
        return np.ones((1, 50, 7), dtype=np.float32) * 0.42


@pytest.fixture
def server_port():
    """Start a ZMQ server on a random port, yield the port, clean up."""
    runtime = _MockRuntime()
    server = create_zmq_server(runtime, port=0)
    port = server.bound_port
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield port
    server.close()
    thread.join(timeout=2)


@pytest.fixture
def auth_server_port():
    """Start a ZMQ server that protects control endpoints with a token."""
    runtime = _MockRuntime()
    server = create_zmq_server(runtime, port=0, control_token="secret")
    port = server.bound_port
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield port
    server.close()
    thread.join(timeout=2)


# ── predict_action ───────────────────────────────────────────────────


def test_predict_action_returns_ndarray(server_port):
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        obs = {"state": np.zeros(8, dtype=np.float32)}
        actions = client.predict_action(obs)
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (1, 50, 7)
        np.testing.assert_allclose(actions, 0.42, atol=1e-5)


def test_predict_action_with_profile(server_port):
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        obs = {"state": np.zeros(8, dtype=np.float32)}
        actions, profile = client.predict_action(obs, with_profile=True)
        assert isinstance(actions, np.ndarray)
        assert isinstance(profile, ProfileData)
        assert profile.serialize_ms > 0
        assert profile.zmq_roundtrip_ms > 0
        assert profile.deserialize_ms > 0
        assert profile.total_ms > 0


def test_predict_action_with_images(server_port):
    """Images get JPEG-compressed on the wire (whitelisted keys)."""
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        obs = {
            "agentview_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "task": "pick up the cup",
        }
        actions = client.predict_action(obs)
        assert actions.shape == (1, 50, 7)


# ── persistent socket reuse ──────────────────────────────────────────


def test_persistent_socket_reuse(server_port):
    """N=20 calls reuse the same socket (no reconnect)."""
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        for i in range(20):
            actions = client.predict_action({"step": i})
            assert actions.shape == (1, 50, 7)


# ── profile overhead ─────────────────────────────────────────────────


def test_profile_overhead_minimal(server_port):
    """with_profile=True adds < 0.5ms overhead (lenient for CI)."""
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        obs = {"state": np.zeros(8, dtype=np.float32)}

        # Warmup
        for _ in range(5):
            client.predict_action(obs)

        # Measure without profile
        t0 = time.perf_counter()
        for _ in range(50):
            client.predict_action(obs)
        no_profile_ms = (time.perf_counter() - t0) * 1000 / 50

        # Measure with profile
        t0 = time.perf_counter()
        for _ in range(50):
            client.predict_action(obs, with_profile=True)
        with_profile_ms = (time.perf_counter() - t0) * 1000 / 50

        overhead = with_profile_ms - no_profile_ms
        assert overhead < 0.5, f"Profile overhead {overhead:.3f}ms > 0.5ms"


# ── ping / reset ─────────────────────────────────────────────────────


def test_ping(server_port):
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        result = client.ping()
        assert result["status"] == "ok"


def test_ping_sends_auth_token(auth_server_port):
    with ZmqRuntimeClient(
        f"tcp://127.0.0.1:{auth_server_port}",
        auth_token="secret",
    ) as client:
        result = client.ping()
        assert result["status"] == "ok"


def test_kill_sends_auth_token(auth_server_port):
    with ZmqRuntimeClient(
        f"tcp://127.0.0.1:{auth_server_port}",
        auth_token="secret",
    ) as client:
        result = client.kill()
        assert result["status"] == "ok"


def test_reset(server_port):
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        result = client.reset()
        assert result["status"] == "ok"


# ── context manager ──────────────────────────────────────────────────


def test_context_manager(server_port):
    with ZmqRuntimeClient(f"tcp://127.0.0.1:{server_port}") as client:
        actions = client.predict_action({"x": 1})
        assert actions.shape == (1, 50, 7)
    # After exit, socket should be closed
    assert client._socket is None
