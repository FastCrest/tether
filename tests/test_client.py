"""Tests for TetherClient + TetherAsyncClient (Phase 0.5 customer-sdk).

Uses httpx.MockTransport so tests run in-process without spinning up a real
server. Verifies the SDK consumes every server-side contract we shipped today:
- /health 6-state machine (initializing/loading/warming/ready/warmup_failed/degraded)
- /act 503 + Retry-After: 60 from circuit-broken servers
- B.6 ActionGuard guard_violations + guard_clamped pass-through
- X-Tether-Key bearer auth
- episode_id auto-generation + propagation via context manager
- Image encoding from numpy / PIL / file path / raw bytes / base64 str
- Retry semantics: 401 / 422 don't retry; 503-warming retries with backoff;
  503-degraded raises by default (operator decision); 5xx retries to max_retries
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from tether.client import (
    TetherClient,
    TetherAsyncClient,
    TetherClientError,
    TetherAuthError,
    TetherValidationError,
    TetherServerDegradedError,
    TetherServerNotReadyError,
    encode_image,
)


# ---------- helpers ----------

def _client_with_handler(handler, **client_kwargs) -> TetherClient:
    """Construct a TetherClient that uses an httpx.MockTransport for requests."""
    transport = httpx.MockTransport(handler)
    client = TetherClient("http://test.invalid", **client_kwargs)
    client._http = httpx.Client(headers=client._http.headers, transport=transport, timeout=client.timeout_s)
    return client


def _async_client_with_handler(handler, **client_kwargs) -> TetherAsyncClient:
    transport = httpx.MockTransport(handler)
    client = TetherAsyncClient("http://test.invalid", **client_kwargs)
    client._http = httpx.AsyncClient(headers=client._http.headers, transport=transport, timeout=client.timeout_s)
    return client


def _act_response(actions=None, latency_ms=10.0, **extra):
    body = {
        "actions": actions or [[0.0] * 7],
        "latency_ms": latency_ms,
        "inference_mode": "stub",
        **extra,
    }
    return httpx.Response(200, json=body)


# ---------- TetherClient.act / health / config ----------

class TestActHappyPath:
    def test_act_returns_response_dict(self):
        def handler(request):
            assert request.method == "POST"
            assert request.url.path == "/act"
            return _act_response(actions=[[0.1] * 7] * 4)

        with _client_with_handler(handler) as c:
            r = c.act(instruction="pick up the cup", state=[0.0] * 8)
        assert r["actions"] == [[0.1] * 7] * 4
        assert r["latency_ms"] == 10.0

    def test_act_passes_image_instruction_state_episode(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return _act_response()

        with _client_with_handler(handler) as c:
            c.act(image=b"\xff\xd8\xff some_jpeg", instruction="hello", state=[1.0, 2.0], episode_id="ep-1")
        body = captured["body"]
        assert body["image"]
        assert body["instruction"] == "hello"
        assert body["state"] == [1.0, 2.0]
        assert body["episode_id"] == "ep-1"

    def test_act_surfaces_guard_fields(self):
        def handler(request):
            return _act_response(
                actions=[[0.5] * 7],
                guard_violations=["joint_3 clamped to upper bound 3.07"],
                guard_clamped=True,
            )

        with _client_with_handler(handler) as c:
            r = c.act(instruction="x")
        assert r["guard_clamped"] is True
        assert "joint_3" in r["guard_violations"][0]

    def test_act_surfaces_injected_latency_field(self):
        def handler(request):
            return _act_response(injected_latency_ms=100.0)

        with _client_with_handler(handler) as c:
            r = c.act(instruction="x")
        assert r["injected_latency_ms"] == 100.0


class TestAuth:
    def test_api_key_added_to_headers(self):
        captured = {}

        def handler(request):
            captured["x_tether_key"] = request.headers.get("X-Tether-Key")
            return _act_response()

        with _client_with_handler(handler, api_key="secret-key-abc") as c:
            c.act(instruction="x")
        assert captured["x_tether_key"] == "secret-key-abc"

    def test_no_api_key_omits_header(self):
        captured = {}

        def handler(request):
            captured["x_tether_key"] = request.headers.get("X-Tether-Key")
            return _act_response()

        with _client_with_handler(handler) as c:
            c.act(instruction="x")
        assert captured["x_tether_key"] is None

    def test_401_raises_auth_error_no_retry(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            return httpx.Response(401, json={"detail": "missing or invalid X-Tether-Key header"})

        with _client_with_handler(handler, max_retries=5) as c:
            with pytest.raises(TetherAuthError):
                c.act(instruction="x")
        assert call_count["n"] == 1, "401 should NOT retry"


class TestValidationError:
    def test_422_raises_validation_error_no_retry(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            return httpx.Response(422, json={"detail": [{"msg": "bad request"}]})

        with _client_with_handler(handler, max_retries=5) as c:
            with pytest.raises(TetherValidationError):
                c.act(instruction="x")
        assert call_count["n"] == 1, "422 should NOT retry"


class TestRetryOn503Warming:
    def test_503_warming_retries_then_succeeds(self):
        # Simulate server transitioning from warming → ready: first 2 calls 503,
        # third succeeds
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return httpx.Response(503, json={"state": "warming"})
            return _act_response()

        with _client_with_handler(handler, max_retries=3, initial_backoff_s=0.001) as c:
            r = c.act(instruction="x")
        assert call_count["n"] == 3
        assert r["actions"]

    def test_503_warming_respects_retry_after_header(self):
        import time as _t
        call_count = {"n": 0}
        timing = {"first_attempt_at": None, "retry_at": None}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] == 1:
                timing["first_attempt_at"] = _t.perf_counter()
                return httpx.Response(503, json={"state": "warming"}, headers={"Retry-After": "0.05"})
            timing["retry_at"] = _t.perf_counter()
            return _act_response()

        with _client_with_handler(handler, max_retries=2, initial_backoff_s=0.001) as c:
            c.act(instruction="x")
        assert call_count["n"] == 2
        assert timing["retry_at"] - timing["first_attempt_at"] >= 0.04

    def test_503_warming_exhausts_max_retries(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            return httpx.Response(503, json={"state": "warming"})

        with _client_with_handler(handler, max_retries=3, initial_backoff_s=0.001) as c:
            with pytest.raises(TetherServerNotReadyError) as exc_info:
                c.act(instruction="x")
        assert call_count["n"] == 4, "max_retries=3 → 1 initial + 3 retries = 4 total"
        assert exc_info.value.state == "warming"

    def test_503_with_warmup_failed_state_treated_as_not_ready(self):
        def handler(request):
            return httpx.Response(503, json={"state": "warmup_failed"})

        with _client_with_handler(handler, max_retries=1, initial_backoff_s=0.001) as c:
            with pytest.raises(TetherServerNotReadyError) as exc_info:
                c.act(instruction="x")
        assert exc_info.value.state == "warmup_failed"


class TestRetryOn503Degraded:
    def test_degraded_raises_by_default_no_retry(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            return httpx.Response(
                503,
                json={
                    "error": "server-degraded",
                    "consecutive_crashes": 5,
                    "max_consecutive_crashes": 5,
                    "hint": "circuit breaker tripped; restart server to clear",
                },
                headers={"Retry-After": "60"},
            )

        with _client_with_handler(handler, max_retries=5) as c:
            with pytest.raises(TetherServerDegradedError) as exc_info:
                c.act(instruction="x")
        assert call_count["n"] == 1, "degraded should NOT retry by default (operator decision)"
        assert exc_info.value.retry_after_s == 60.0

    def test_degraded_retries_when_retry_on_degraded_true(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return httpx.Response(
                    503,
                    json={"error": "server-degraded"},
                    headers={"Retry-After": "0.05"},
                )
            return _act_response()

        with _client_with_handler(handler, max_retries=2, retry_on_degraded=True, initial_backoff_s=0.001) as c:
            r = c.act(instruction="x")
        assert call_count["n"] == 2
        assert r["actions"]


class TestEpisode:
    def test_episode_auto_generates_id_and_propagates(self):
        episode_ids = []

        def handler(request):
            body = json.loads(request.content)
            episode_ids.append(body.get("episode_id"))
            return _act_response()

        with _client_with_handler(handler) as c:
            with c.episode() as ep:
                ep.act(instruction="x")
                ep.act(instruction="y")
                ep.act(instruction="z")
        assert len(episode_ids) == 3
        assert all(eid == ep.episode_id for eid in episode_ids)
        assert ep.call_count == 3

    def test_episode_explicit_id_used(self):
        episode_ids = []

        def handler(request):
            body = json.loads(request.content)
            episode_ids.append(body.get("episode_id"))
            return _act_response()

        with _client_with_handler(handler) as c:
            with c.episode(episode_id="my-custom-ep") as ep:
                ep.act(instruction="x")
        assert episode_ids == ["my-custom-ep"]
        assert ep.episode_id == "my-custom-ep"

    def test_episode_call_after_close_raises(self):
        def handler(request):
            return _act_response()

        with _client_with_handler(handler) as c:
            with c.episode() as ep:
                ep.act(instruction="x")
            with pytest.raises(TetherClientError, match="closed"):
                ep.act(instruction="x")


class TestHealth:
    def test_health_returns_state_field(self):
        def handler(request):
            assert request.url.path == "/health"
            return httpx.Response(200, json={
                "status": "ok", "state": "ready", "model_loaded": True,
                "consecutive_crashes": 0, "max_consecutive_crashes": 5,
            })

        with _client_with_handler(handler) as c:
            h = c.health()
        assert h["state"] == "ready"
        assert h["status"] == "ok"

    def test_health_does_not_raise_on_503(self):
        # Health is informational; 503 body has the diagnostic info
        def handler(request):
            return httpx.Response(503, json={"status": "not_ready", "state": "warming"})

        with _client_with_handler(handler) as c:
            h = c.health()
        assert h["state"] == "warming"


class TestImageEncoding:
    def test_encode_none_returns_empty_string(self):
        assert encode_image(None) == ""
        assert encode_image("") == ""

    def test_encode_jpeg_bytes(self):
        # Valid JPEG SOI
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"x" * 100
        out = encode_image(jpeg)
        # Round-trip
        assert base64.b64decode(out)[:3] == b"\xff\xd8\xff"

    def test_encode_png_bytes(self):
        png = b"\x89PNG\r\n\x1a\n" + b"y" * 100
        out = encode_image(png)
        assert base64.b64decode(out)[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_pathlib_file(self, tmp_path):
        p = tmp_path / "frame.bin"
        p.write_bytes(b"\xff\xd8\xff" + b"x" * 50)
        out = encode_image(p)
        assert base64.b64decode(out)[:3] == b"\xff\xd8\xff"

    def test_encode_path_str_existing(self, tmp_path):
        p = tmp_path / "frame.bin"
        p.write_bytes(b"\xff\xd8\xff" + b"x" * 50)
        out = encode_image(str(p))
        # Should b64-encode the file contents
        assert base64.b64decode(out)[:3] == b"\xff\xd8\xff"

    def test_encode_numpy_array(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[10:20, 10:20] = [255, 0, 0]
        out = encode_image(arr)
        # Decoded should be a JPEG
        assert base64.b64decode(out)[:3] == b"\xff\xd8\xff"

    def test_encode_pil_image(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (32, 32), color=(255, 0, 0))
        out = encode_image(img)
        assert base64.b64decode(out)[:3] == b"\xff\xd8\xff"

    def test_encode_unsupported_type_raises(self):
        with pytest.raises(TetherClientError, match="unsupported"):
            encode_image(12345)


class TestNetworkErrors:
    def test_network_error_retries_then_raises(self):
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            raise httpx.ConnectError("connection refused")

        with _client_with_handler(handler, max_retries=2, initial_backoff_s=0.001) as c:
            with pytest.raises(TetherClientError, match="network"):
                c.act(instruction="x")
        # 1 initial + 2 retries = 3 total attempts
        assert call_count["n"] == 3


# ---------- Async client mirror tests ----------

class TestAsyncClient:
    def test_async_act_basic(self):
        async def run():
            def handler(request):
                return _act_response(actions=[[0.5] * 7] * 2)

            client = _async_client_with_handler(handler)
            try:
                r = await client.act(instruction="x")
                assert r["actions"] == [[0.5] * 7] * 2
            finally:
                await client.close()

        asyncio.run(run())

    def test_async_retries_on_503(self):
        async def run():
            call_count = {"n": 0}

            def handler(request):
                call_count["n"] += 1
                if call_count["n"] <= 2:
                    return httpx.Response(503, json={"state": "warming"})
                return _act_response()

            client = _async_client_with_handler(handler, max_retries=3, initial_backoff_s=0.001)
            try:
                r = await client.act(instruction="x")
                assert r["actions"]
                assert call_count["n"] == 3
            finally:
                await client.close()

        asyncio.run(run())

    def test_async_episode_propagates_id(self):
        async def run():
            episode_ids = []

            def handler(request):
                body = json.loads(request.content)
                episode_ids.append(body.get("episode_id"))
                return _act_response()

            client = _async_client_with_handler(handler)
            try:
                async with client.episode() as ep:
                    await ep.act(instruction="x")
                    await ep.act(instruction="y")
                assert len(episode_ids) == 2
                assert all(eid == ep.episode_id for eid in episode_ids)
            finally:
                await client.close()

        asyncio.run(run())
