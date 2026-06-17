"""Tests for Layer 1 PolicyServer (Lift #2 Day 1).

Uses real ZMQ sockets on localhost with kernel-assigned ports (port=0).
Each test gets its own server + client pair; no shared state.
"""
from __future__ import annotations

import threading
import time

import msgpack
import zmq
import zmq.auth

from tether.runtime.transports.zmq.policy_server import (
    SCHEMA_VERSION,
    PolicyServer,
)


def _start_server(port: int = 0, **kwargs) -> tuple[PolicyServer, threading.Thread]:
    server = PolicyServer(port=port, **kwargs)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.1)  # let the socket bind
    return server, thread


def _client_socket(
    port: int,
    *,
    curve_client_cert: str | None = None,
    curve_server_public_key: str | None = None,
) -> zmq.Socket:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    if curve_client_cert is not None or curve_server_public_key is not None:
        assert curve_client_cert is not None
        assert curve_server_public_key is not None
        client_public_key, client_secret_key = zmq.auth.load_certificate(curve_client_cert)
        server_public_key, _ = zmq.auth.load_certificate(curve_server_public_key)
        sock.curve_publickey = client_public_key
        sock.curve_secretkey = client_secret_key
        sock.curve_serverkey = server_public_key
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


def test_control_token_required_for_ping_and_kill():
    server, thread = _start_server(control_token="secret")
    port = server.bound_port
    sock = _client_socket(port)

    result = _send_recv(sock, {"endpoint": "ping"})
    assert "error" in result
    assert "auth token" in result["error"]

    result = _send_recv(sock, {"endpoint": "ping", "auth_token": "secret"})
    assert result["status"] == "ok"

    result = _send_recv(sock, {"endpoint": "kill", "auth_token": "wrong"})
    assert "error" in result
    assert server.running

    result = _send_recv(sock, {"endpoint": "kill", "auth_token": "secret"})
    assert result["status"] == "ok"
    thread.join(timeout=2)
    assert not server.running


def test_curve_client_can_connect_with_allowed_certificate(tmp_path):
    server_public, server_secret = zmq.auth.create_certificates(tmp_path, "server")
    client_cert_dir = tmp_path / "clients"
    client_cert_dir.mkdir()
    _client_public, client_secret = zmq.auth.create_certificates(client_cert_dir, "robot")

    server, thread = _start_server(
        curve_server_cert=server_secret,
        curve_client_cert_dir=client_cert_dir,
    )
    port = server.bound_port
    sock = _client_socket(
        port,
        curve_client_cert=client_secret,
        curve_server_public_key=server_public,
    )

    result = _send_recv(sock, {"endpoint": "ping"})
    assert result["status"] == "ok"

    server.close()
    thread.join(timeout=2)


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
