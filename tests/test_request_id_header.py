"""Request-id middleware coverage for the real Tether FastAPI app."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class _StubServer:
    def __init__(self, export_dir, *args, **kwargs):
        self.export_dir = Path(export_dir)
        self._ready = True
        self.health_state = "ready"
        self._inference_mode = "stub"
        self._vlm_loaded = False
        self.consecutive_crash_count = 0
        self.max_consecutive_crashes = 5
        self.robot_id = ""

    @property
    def ready(self):
        return self._ready

    async def load(self):
        self._ready = True
        self.health_state = "ready"


@pytest.fixture
def app(tmp_path, monkeypatch):
    from tether.runtime import server as runtime_server

    monkeypatch.setattr(runtime_server, "TetherServer", _StubServer)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    app = runtime_server.create_app(str(export_dir), device="cpu")

    @app.get("/_test/request-id")
    async def request_id_probe():
        return {"request_id": runtime_server._request_id_var.get()}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_response_has_tether_request_id_header(client):
    response = client.get("/health")

    assert response.headers["X-Tether-Request-ID"].startswith("req-")
    assert "X-Reflex-Request-ID" not in response.headers


def test_each_request_gets_a_unique_generated_id(client):
    ids = {client.get("/health").headers["X-Tether-Request-ID"] for _ in range(5)}

    assert len(ids) == 5


def test_tether_request_id_header_is_echoed(client):
    response = client.get(
        "/health",
        headers={"X-Tether-Request-ID": "  req-user-supplied  "},
    )

    assert response.headers["X-Tether-Request-ID"] == "req-user-supplied"


def test_generic_request_id_header_is_accepted_when_tether_header_missing(client):
    response = client.get("/health", headers={"X-Request-ID": "edge-proxy-123"})

    assert response.headers["X-Tether-Request-ID"] == "edge-proxy-123"


def test_tether_header_wins_over_generic_request_id(client):
    response = client.get(
        "/health",
        headers={
            "X-Tether-Request-ID": "req-tether",
            "X-Request-ID": "proxy-request",
        },
    )

    assert response.headers["X-Tether-Request-ID"] == "req-tether"


def test_request_id_is_available_inside_route_context(app):
    client = TestClient(app)

    response = client.get(
        "/_test/request-id",
        headers={"X-Tether-Request-ID": "req-route-context"},
    )

    assert response.json()["request_id"] == "req-route-context"
    assert response.headers["X-Tether-Request-ID"] == "req-route-context"
