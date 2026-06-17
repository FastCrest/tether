"""Tests for Layer 2 ZMQ factory (Lift #2 Day 2).

Validates that create_zmq_server wires a mock runtime into a PolicyServer
with predict_action, reset, and get_status endpoints.
"""
from __future__ import annotations

import threading
import time

import msgpack
import numpy as np
import zmq

from tether.runtime.transports.zmq.factory import create_zmq_server


class _MockRuntime:
    """Minimal mock that looks like a PolicyRuntime."""

    def predict_action_chunk(self, batch: dict) -> np.ndarray:
        return np.zeros((1, 50, 7), dtype=np.float32)


def _start_server(runtime=None, port=0):
    if runtime is None:
        runtime = _MockRuntime()
    server = create_zmq_server(runtime, port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)
    return server, thread


def _client(port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")
    return sock


def _send_recv(sock, msg):
    sock.send(msgpack.packb(msg, use_bin_type=True))
    return msgpack.unpackb(sock.recv(), raw=False)


# ── predict_action ───────────────────────────────────────────────────


def test_predict_action_returns_action_data():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client(port)

    result = _send_recv(sock, {
        "endpoint": "predict_action",
        "data": {"obs": "test"},
    })

    assert "action_data" in result
    assert "infer_time_ms" in result
    assert isinstance(result["infer_time_ms"], (int, float))

    # Deserialize and verify shape
    import io
    actions = np.load(io.BytesIO(result["action_data"]))
    assert actions.shape == (1, 50, 7)

    server.close()
    thread.join(timeout=2)


# ── reset ────────────────────────────────────────────────────────────


def test_reset_returns_ok():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client(port)

    result = _send_recv(sock, {"endpoint": "reset"})
    assert result["status"] == "ok"

    server.close()
    thread.join(timeout=2)


# ── get_status ───────────────────────────────────────────────────────


def test_get_status_returns_stats():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client(port)

    # Make one predict request first
    _send_recv(sock, {"endpoint": "predict_action", "data": {}})
    result = _send_recv(sock, {"endpoint": "get_status"})

    assert result["status"] == "ready"
    assert result["total_requests"] >= 1
    assert "uptime_s" in result
    assert "avg_infer_time_ms" in result

    server.close()
    thread.join(timeout=2)


# ── CLI flag ─────────────────────────────────────────────────────────


def test_cli_transport_flag_exists():
    from typer.testing import CliRunner
    from tether.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert "--transport" in result.output
    assert "zmq" in result.output
    assert "--zmq-insecure-ok" in result.output


def test_cli_transport_invalid_rejected():
    from typer.testing import CliRunner
    from tether.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "/tmp/nonexistent", "--transport", "ros2"])
    assert result.exit_code != 0
