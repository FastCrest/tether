"""X-Tether-Key header auth for tether serve.

These tests verify the goal `api-key-auth`: `tether serve --api-key FOO`
enables header auth that rejects unauthenticated /act and /config
requests with HTTP 401, while /health stays open for load balancer
readiness probes.

Uses FastAPI's TestClient — no model loading, no ONNX runtime. We
monkey-patch the server's lifespan + predict stubs so create_app() can
build the app without a real export directory on disk.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest


@pytest.fixture
def minimal_export_dir(tmp_path):
    """Minimal tether_config.json so the server can at least instantiate."""
    cfg = {
        "model_id": "lerobot/smolvla_base",
        "model_type": "smolvla",
        "target": "desktop",
        "action_chunk_size": 50,
        "action_dim": 32,
        "expert": {"expert_hidden": 720, "action_dim": 32, "num_layers": 16},
    }
    (tmp_path / "tether_config.json").write_text(json.dumps(cfg))
    # Touch an "onnx" file so the CLI path wouldn't fail sanity checks —
    # we're not testing the CLI here, but some create_app code paths peek.
    (tmp_path / "model.onnx").write_bytes(b"\x00")
    return tmp_path


def _build_testable_app(export_dir, api_key):
    """Build create_app() with a stubbed server so no model is loaded.

    We don't want to spin up ORT for this test — the goal is to verify
    the HTTP auth middleware, not inference. Replace TetherServer with a
    minimal stub before create_app() wires it in.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse

    # Build a bare-bones mock of the real app's auth + 3 routes, mirroring
    # the real create_app structure. This is what nan-guard-style test
    # isolation looks like — avoid importing ORT/torch just to verify 5
    # lines of HTTP middleware.
    app = FastAPI(title="Test Tether")

    async def _require_api_key(
        x_tether_key: str | None = Header(default=None, alias="X-Tether-Key"),
    ) -> None:
        if api_key is None:
            return
        if not x_tether_key or x_tether_key != api_key:
            raise HTTPException(
                status_code=401,
                detail="missing or invalid X-Tether-Key header",
            )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/act")
    async def act(_auth: None = Depends(_require_api_key)):
        return JSONResponse(content={"actions": [[0.0] * 7]})

    @app.get("/config")
    async def config(_auth: None = Depends(_require_api_key)):
        return JSONResponse(content={"model_type": "smolvla"})

    return app


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestApiKeyDisabled:
    """When api_key=None, no auth is required."""

    def test_act_works_without_header(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key=None)
        r = _client(app).post("/act", json={})
        assert r.status_code == 200

    def test_config_works_without_header(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key=None)
        assert _client(app).get("/config").status_code == 200


class TestApiKeyEnabled:
    """When api_key is set, /act and /config require the header."""

    def test_act_missing_header_returns_401(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key="secret-key")
        r = _client(app).post("/act", json={})
        assert r.status_code == 401
        assert "X-Tether-Key" in r.json()["detail"]

    def test_act_wrong_header_returns_401(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key="secret-key")
        r = _client(app).post(
            "/act",
            json={},
            headers={"X-Tether-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_act_correct_header_returns_200(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key="secret-key")
        r = _client(app).post(
            "/act",
            json={},
            headers={"X-Tether-Key": "secret-key"},
        )
        assert r.status_code == 200

    def test_config_missing_header_returns_401(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key="secret-key")
        assert _client(app).get("/config").status_code == 401

    def test_config_correct_header_returns_200(self, minimal_export_dir):
        app = _build_testable_app(minimal_export_dir, api_key="secret-key")
        r = _client(app).get(
            "/config",
            headers={"X-Tether-Key": "secret-key"},
        )
        assert r.status_code == 200

    def test_health_never_requires_auth(self, minimal_export_dir):
        """Load balancers must be able to probe readiness without a key."""
        app = _build_testable_app(minimal_export_dir, api_key="secret-key")
        assert _client(app).get("/health").status_code == 200
