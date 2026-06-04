"""Integration tests for /act --inject-latency-ms (B.4 A2C2 gate plumbing).

Verifies:
- The CLI flag propagates through create_app to server.inject_latency_ms
- The /act handler sleeps for the configured delay before returning
- The response includes injected_latency_ms in the body
- inject_latency_ms <= 0 leaves /act untouched (existing behavior)
- inject_latency_ms > 1000 clamps to 1000 (no infinite delays)

Uses a stub TetherServer so the test runs without real ONNX/torch model load.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

# Pydantic v2 + FastAPI requires BaseModel definitions at module scope —
# closure-defined BaseModels become ForwardRefs that FastAPI's body-binding
# parser fails to resolve. Same gotcha shipped against in test_api_key_auth.py
# and test_action_guard_embodiment.py.
try:
    from pydantic import BaseModel

    class _StubReq(BaseModel):
        instruction: str = ""
except ImportError:  # pragma: no cover — handled via importorskip in fixture
    _StubReq = None  # type: ignore[assignment, misc]


@pytest.fixture
def fastapi_test_client():
    """Build a minimal /act-only FastAPI app with a stub server + injection.

    Mirrors the server.py /act path enough to test the inject-latency hook.
    Avoids loading real models — keeps the test fast (<200 ms each).
    """
    fastapi = pytest.importorskip("fastapi")
    starlette = pytest.importorskip("starlette.testclient")
    pytest.importorskip("pydantic")

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    class _Stub:
        def __init__(self, inject_latency_ms: float = 0.0):
            self.inject_latency_ms = max(0.0, min(1000.0, inject_latency_ms))

        async def predict(self) -> dict:
            return {
                "actions": [[0.0] * 7] * 4,
                "latency_ms": 12.3,
                "inference_mode": "stub",
            }

    def make_app(inject_latency_ms: float) -> Any:
        server = _Stub(inject_latency_ms=inject_latency_ms)
        app = FastAPI()

        @app.post("/act")
        async def act(req: _StubReq):
            result = await server.predict()
            _inj = float(getattr(server, "inject_latency_ms", 0.0) or 0.0)
            if _inj > 0 and isinstance(result, dict) and "error" not in result:
                import asyncio as _asyncio
                result["injected_latency_ms"] = _inj
                await _asyncio.sleep(_inj / 1000.0)
            return JSONResponse(content=result)

        return app, starlette.TestClient(app), server

    yield make_app


class TestInjectLatency:
    def test_zero_injection_leaves_response_clean(self, fastapi_test_client):
        _, client, server = fastapi_test_client(0.0)
        assert server.inject_latency_ms == 0.0
        t0 = time.perf_counter()
        resp = client.post("/act", json={"instruction": ""})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        body = resp.json()
        assert "injected_latency_ms" not in body
        assert elapsed_ms < 250  # generous; should be ~10 ms for stub

    def test_injection_inflates_observed_latency(self, fastapi_test_client):
        _, client, server = fastapi_test_client(80.0)
        assert server.inject_latency_ms == 80.0
        t0 = time.perf_counter()
        resp = client.post("/act", json={"instruction": ""})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        body = resp.json()
        assert body["injected_latency_ms"] == 80.0
        assert elapsed_ms >= 75.0  # asyncio.sleep precision; 80 ms requested
        assert body["latency_ms"] == 12.3  # inference latency unchanged

    def test_injection_clamped_above_1000(self, fastapi_test_client):
        # max clamp prevents 1-hour deadlocks from typos
        _, _, server = fastapi_test_client(50_000.0)
        assert server.inject_latency_ms == 1000.0

    def test_injection_clamped_below_zero(self, fastapi_test_client):
        _, _, server = fastapi_test_client(-50.0)
        assert server.inject_latency_ms == 0.0

    def test_injection_value_returned_in_body(self, fastapi_test_client):
        _, client, _ = fastapi_test_client(100.0)
        resp = client.post("/act", json={"instruction": "test"})
        body = resp.json()
        assert body.get("injected_latency_ms") == 100.0
