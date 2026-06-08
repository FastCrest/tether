"""Tests for the X-Reflex-Request-ID response header middleware.

Verifies that every HTTP response carries a unique UUID4 in the
X-Reflex-Request-ID header, and that the value is available to the
OTel span via the _request_id_var context variable.

Uses FastAPI's TestClient — no model loading, no ONNX runtime.
"""
from __future__ import annotations

import contextvars
import uuid

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _build_app() -> tuple[FastAPI, contextvars.ContextVar[str]]:
    """Minimal FastAPI app with only the request-ID middleware wired in."""
    request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
        "request_id", default=""
    )

    app = FastAPI()

    @app.middleware("http")
    async def _request_id_middleware(request, call_next):
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Reflex-Request-ID"] = req_id
        return response

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act():
        return JSONResponse({"actions": []})

    return app, request_id_var


@pytest.fixture
def client():
    app, _ = _build_app()
    return TestClient(app)


class TestRequestIDHeader:
    def test_health_response_has_header(self, client):
        assert "X-Reflex-Request-ID" in client.get("/health").headers

    def test_act_response_has_header(self, client):
        assert "X-Reflex-Request-ID" in client.post("/act").headers

    def test_header_value_is_valid_uuid(self, client):
        value = client.get("/health").headers["X-Reflex-Request-ID"]
        assert str(uuid.UUID(value)) == value  # raises ValueError if malformed

    def test_each_request_gets_a_unique_id(self, client):
        ids = {client.get("/health").headers["X-Reflex-Request-ID"] for _ in range(5)}
        assert len(ids) == 5

    def test_request_id_is_accessible_inside_route(self):
        """Context var holds the same ID that ends up in the response header."""
        app, request_id_var = _build_app()

        captured: dict[str, str] = {}

        @app.post("/act_span")
        async def act_span():
            captured["id"] = request_id_var.get()
            return JSONResponse({"actions": []})

        c = TestClient(app)
        response = c.post("/act_span")

        assert captured["id"] != ""
        assert response.headers["X-Reflex-Request-ID"] == captured["id"]
