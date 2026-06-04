"""Tests for Layer 1 PolicyServer (Lift #2 Day 1).

Uses real ZMQ sockets on localhost with kernel-assigned ports (port=0).
Each test gets its own server + client pair; no shared state.
"""
from __future__ import annotations

import threading
import time

import msgpack
import pytest
import zmq

from tether.runtime.transports.zmq.policy_server import (
    SCHEMA_VERSION,
    PolicyServer,
    WireSchemaMismatchError,
)


def _start_server(port: int = 0) -> tuple[PolicyServer, threading.Thread]:
    server = PolicyServer(port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)  # let the socket bind
    return server, thread


def _client_socket(port: int) -> zmq.Socket:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")
    return sock


def _send_recv(sock: zmq.Socket, msg: dict) -> dict:
    sock.send(msgpack.packb(msg, use_bin_type=True))
    return msgpack.unpackb(sock.recv(), raw=False)


# ── Ping ─────────────────────────────────────────────────────────────


def test_ping_returns_ok():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client_socket(port)
    result = _send_recv(sock, {"endpoint": "ping"})
    assert result["status"] == "ok"
    assert "uptime_s" in result
    assert result["schema_version"] == SCHEMA_VERSION
    server.close()
    thread.join(timeout=2)


# ── Kill ─────────────────────────────────────────────────────────────


def test_kill_shuts_down():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client_socket(port)
    result = _send_recv(sock, {"endpoint": "kill"})
    assert result["status"] == "ok"
    thread.join(timeout=2)
    assert not server.running


# ── Custom endpoint ──────────────────────────────────────────────────


def test_custom_endpoint_dispatches():
    server, thread = _start_server()
    port = server.bound_port

    def echo_handler(text: str = "") -> dict:
        return {"echo": text}

    server.register_endpoint("echo", echo_handler)

    sock = _client_socket(port)
    result = _send_recv(sock, {"endpoint": "echo", "data": {"text": "hello"}})
    assert result["echo"] == "hello"
    server.close()
    thread.join(timeout=2)


def test_custom_no_input_endpoint():
    server, thread = _start_server()
    port = server.bound_port

    server.register_endpoint("status", lambda: {"ready": True}, requires_input=False)

    sock = _client_socket(port)
    result = _send_recv(sock, {"endpoint": "status"})
    assert result["ready"] is True
    server.close()
    thread.join(timeout=2)


# ── Error handling ───────────────────────────────────────────────────


def test_unknown_endpoint_returns_error():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client_socket(port)
    result = _send_recv(sock, {"endpoint": "nonexistent"})
    assert "error" in result
    assert "nonexistent" in result["error"]
    server.close()
    thread.join(timeout=2)


def test_schema_version_mismatch_returns_error():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client_socket(port)
    result = _send_recv(sock, {"endpoint": "ping", "schema_version": 999})
    assert "error" in result
    assert "mismatch" in result["error"].lower()
    server.close()
    thread.join(timeout=2)


def test_protobuf_byte_returns_error():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client_socket(port)
    sock.send(b"\x01protobuf_payload_here")
    result = msgpack.unpackb(sock.recv(), raw=False)
    assert "error" in result
    assert "protobuf" in result["error"].lower()
    server.close()
    thread.join(timeout=2)


# ── Request counter ──────────────────────────────────────────────────


def test_request_counter_increments():
    server, thread = _start_server()
    port = server.bound_port
    sock = _client_socket(port)

    r1 = _send_recv(sock, {"endpoint": "ping"})
    r2 = _send_recv(sock, {"endpoint": "ping"})
    assert r2["request_count"] == r1["request_count"] + 1
    server.close()
    thread.join(timeout=2)


# ── bound_port property ─────────────────────────────────────────────


def test_bound_port_returns_real_port():
    server, thread = _start_server()
    port = server.bound_port
    assert isinstance(port, int)
    assert port > 0
    server.close()
    thread.join(timeout=2)
