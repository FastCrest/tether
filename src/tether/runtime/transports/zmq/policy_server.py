"""Layer 1: PolicyServer — generic ZMQ REP event loop + endpoint routing.

Ported from FluxVLA ``zmq_server.py:40-182`` (Apache-2.0, LimX Dynamics)
per the Lift #2 plan. Two changes from the FluxVLA reference:

1. Protobuf path removed (reserved first-byte ``0x01`` raises
   ``NotImplementedError`` per Z-3 from the research sidecar).
2. ``uptime_s`` added to ping response.
3. ``schema_version`` checked on incoming messages (Z-3).

Usage::

    server = PolicyServer(port=5555)
    server.register_endpoint("predict_action", my_handler)
    server.run()  # blocks
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import msgpack
import zmq
from zmq.auth.thread import ThreadAuthenticator

from tether.runtime.transports.zmq.security import load_curve_key

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
FORMAT_PROTOBUF_BYTE = 0x01


class WireSchemaMismatchError(RuntimeError):
    """Raised when client and server disagree on wire schema version."""

    def __init__(self, client_version: int, server_version: int) -> None:
        super().__init__(
            f"Wire schema mismatch: client sent v{client_version}, "
            f"server expects v{server_version}. "
            f"Upgrade the {'client' if client_version < server_version else 'server'}."
        )
        self.client_version = client_version
        self.server_version = server_version


@dataclass
class _EndpointHandler:
    handler: Callable
    requires_input: bool = True
    requires_auth: bool = False


class PolicyServer:
    """Generic ZMQ REP server with named endpoint routing.

    Synchronous request-reply over a ZMQ REP socket. Endpoints are
    registered by name; incoming msgpack messages are dispatched to the
    matching handler. 500ms poll timeout for clean Ctrl-C shutdown.

    Built-in endpoints:
    - ``ping`` — returns ``{"status": "ok", "uptime_s": float}``
    - ``kill`` — triggers graceful shutdown

    Args:
        host: Bind address (``'*'`` for all interfaces).
        port: TCP port to listen on. 0 = kernel-assigned (for testing).
    """

    def __init__(
        self,
        host: str = "*",
        port: int = 5555,
        *,
        curve_server_cert: str | Path | None = None,
        curve_server_public_key: str | bytes | None = None,
        curve_server_secret_key: str | bytes | None = None,
        curve_client_cert_dir: str | Path | None = None,
        control_token: str | None = None,
    ) -> None:
        self.running = True
        self.context = zmq.Context()
        self._authenticator: ThreadAuthenticator | None = None
        self._control_token = control_token
        self.socket = self.context.socket(zmq.REP)
        self._configure_curve(
            curve_server_cert=curve_server_cert,
            curve_server_public_key=curve_server_public_key,
            curve_server_secret_key=curve_server_secret_key,
            curve_client_cert_dir=curve_client_cert_dir,
        )
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, _EndpointHandler] = {}
        self._start_time = time.monotonic()
        self._request_count = 0

        self.register_endpoint(
            "ping",
            self._handle_ping,
            requires_input=False,
            requires_auth=self._control_token is not None,
        )
        self.register_endpoint(
            "kill",
            self._handle_kill,
            requires_input=False,
            requires_auth=self._control_token is not None,
        )

    @property
    def bound_address(self) -> str:
        return self.socket.getsockopt_string(zmq.LAST_ENDPOINT)

    @property
    def bound_port(self) -> int:
        return int(self.bound_address.rsplit(":", 1)[-1])

    def register_endpoint(
        self,
        name: str,
        handler: Callable,
        requires_input: bool = True,
        requires_auth: bool = False,
    ) -> None:
        self._endpoints[name] = _EndpointHandler(handler, requires_input, requires_auth)

    def _configure_curve(
        self,
        *,
        curve_server_cert: str | Path | None,
        curve_server_public_key: str | bytes | None,
        curve_server_secret_key: str | bytes | None,
        curve_client_cert_dir: str | Path | None,
    ) -> None:
        if curve_server_cert is None and curve_server_public_key is None and curve_server_secret_key is None:
            if curve_client_cert_dir is not None:
                raise ValueError("curve_client_cert_dir requires a CURVE server certificate or keypair")
            return

        if curve_server_cert is not None:
            if curve_server_public_key is not None or curve_server_secret_key is not None:
                raise ValueError("Pass either curve_server_cert or explicit CURVE keys, not both")
            public_key = load_curve_key(curve_server_cert, secret=False)
            secret_key = load_curve_key(curve_server_cert, secret=True)
        else:
            if curve_server_public_key is None or curve_server_secret_key is None:
                raise ValueError("CURVE server mode requires both public and secret keys")
            public_key = load_curve_key(curve_server_public_key, secret=False)
            secret_key = load_curve_key(curve_server_secret_key, secret=True)

        if curve_client_cert_dir is None:
            raise ValueError("CURVE server mode requires curve_client_cert_dir for client authentication")

        client_cert_dir = Path(curve_client_cert_dir).expanduser()
        if not client_cert_dir.is_dir():
            raise ValueError(f"CURVE client certificate directory not found: {client_cert_dir}")

        self._authenticator = ThreadAuthenticator(self.context)
        self._authenticator.start()
        self._authenticator.configure_curve(domain="*", location=client_cert_dir)
        self.socket.curve_publickey = public_key
        self.socket.curve_secretkey = secret_key
        self.socket.curve_server = True

    def _authorize_control_request(self, request: dict[str, Any]) -> None:
        if self._control_token is None:
            return
        if request.get("auth_token") != self._control_token:
            raise PermissionError("ZMQ control endpoint requires a valid auth token")

    def _handle_ping(self) -> dict:
        return {
            "status": "ok",
            "uptime_s": round(time.monotonic() - self._start_time, 2),
            "request_count": self._request_count,
            "schema_version": SCHEMA_VERSION,
        }

    def _handle_kill(self) -> dict:
        self.running = False
        return {"status": "ok", "message": "Server shutting down"}

    def run(self) -> None:
        """Start the blocking event loop.

        Polls every 500ms. Decodes msgpack, dispatches to registered
        endpoint handler, encodes + sends result. Exits when
        ``self.running`` becomes ``False``.
        """
        addr = self.bound_address
        logger.info("ZMQ server listening on %s", addr)
        print(f"ZMQ server listening on {addr}", flush=True)

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(timeout=500))
                if self.socket not in socks:
                    continue

                message = self.socket.recv()
                self._request_count += 1

                # Reserved protobuf first-byte check (Z-3)
                if len(message) > 0 and message[0] == FORMAT_PROTOBUF_BYTE:
                    raise NotImplementedError(
                        "Protobuf wire format (0x01) is reserved for v2. "
                        "Use msgpack (the default)."
                    )

                request = msgpack.unpackb(message, raw=False)

                # Schema version check (Z-3)
                client_version = request.get("schema_version", 1)
                if client_version != SCHEMA_VERSION:
                    raise WireSchemaMismatchError(client_version, SCHEMA_VERSION)

                endpoint = request.get("endpoint", "predict_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint!r}")

                handler = self._endpoints[endpoint]
                if handler.requires_auth:
                    self._authorize_control_request(request)
                if handler.requires_input:
                    result = handler.handler(**request.get("data", {}))
                else:
                    result = handler.handler()

                self.socket.send(msgpack.packb(result, use_bin_type=True))

            except Exception as e:
                logger.error("ZMQ server error: %s", e)
                try:
                    self.socket.send(
                        msgpack.packb({"error": str(e)}, use_bin_type=True)
                    )
                except Exception:
                    pass

        # Clean shutdown
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        if self._authenticator is not None:
            self._authenticator.stop()
        self.context.term()
        logger.info("ZMQ server shut down cleanly after %d requests", self._request_count)

    def close(self) -> None:
        """Signal the event loop to stop from another thread."""
        self.running = False


__all__ = ["PolicyServer", "WireSchemaMismatchError", "SCHEMA_VERSION"]
