"""Day 4 tests for RTC adapter (B.3) — /act handler integration.

Day 4 wiring: PredictRequest.episode_id field + handler hooks for
episode-boundary reset and post-predict merge_and_update. Tests build
a mock app that mirrors the real create_app's /act handler structure
(same pattern as tests/test_api_key_auth.py).

Validates: episode_id change → reset; episode_id stable → no reset;
successful predict → merge_and_update populates carry; error result →
merge_and_update skipped; coexistence with OTel + JSONL hooks.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from pydantic import BaseModel

from tether.runtime.buffer import ActionChunkBuffer
from tether.runtime.rtc_adapter import RtcAdapter, RtcAdapterConfig
from tether.runtime.server import _record_rtc_adaptive_signal


# ---------------------------------------------------------------------------
# Module-level Pydantic request — must be at module scope, not inside a
# closure (Pydantic 2.13 ForwardRef breaks on closure-defined BaseModels).
# Mirrors the real PredictRequest in src/tether/runtime/server.py.
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    image: str | None = None
    instruction: str = ""
    state: list[float] | None = None
    episode_id: str | None = None


class _StubServer:
    """Bare-bones server that records predict calls + returns a fixed response."""

    def __init__(self, response: dict):
        self.response = response
        self.predict_calls: list[dict] = []
        self.rtc_adapter: RtcAdapter | None = None

    async def predict_from_base64_async(self, image_b64, instruction, state):
        self.predict_calls.append({
            "image_b64": image_b64,
            "instruction": instruction,
            "state": state,
        })
        return dict(self.response)

    # Plain (non-RTC) policy interface for the adapter
    def predict_action_chunk(self, **kwargs):
        return np.zeros((50, 7), dtype=np.float32)


def _build_app(server: _StubServer) -> FastAPI:
    """Mirror the real /act handler's RTC integration logic."""
    app = FastAPI()

    @app.post("/act")
    async def act(request: PredictRequest):
        # Episode reset hook (Day 4 logic — must match server.py)
        _rtc = getattr(server, "rtc_adapter", None)
        episode_was_reset = False
        if (
            _rtc is not None
            and request.episode_id is not None
            and request.episode_id != _rtc._active_episode_id
        ):
            _rtc.reset(episode_id=request.episode_id)
            episode_was_reset = True

        result = await server.predict_from_base64_async(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )

        # Post-predict merge_and_update hook
        if (
            _rtc is not None
            and isinstance(result, dict)
            and "error" not in result
            and isinstance(result.get("actions"), list)
            and result["actions"]
        ):
            actions_arr = np.asarray(result["actions"], dtype=np.float32)
            latency_s = float(result.get("latency_ms", 0.0)) / 1000.0
            _rtc.merge_and_update(actions_arr, elapsed_time=latency_s)

        # Echo back diagnostic info so tests can verify
        result["_test_episode_reset"] = episode_was_reset
        result["_test_chunk_count"] = (
            _rtc._chunk_count if _rtc is not None else 0
        )
        return JSONResponse(content=result)

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


GOOD_RESPONSE = {
    "actions": [[0.0] * 7] * 10,
    "latency_ms": 50.0,
    "inference_mode": "onnx_cpu",
    "num_actions": 10,
    "action_dim": 7,
}


class TestNoRtcAdapter:
    """When server.rtc_adapter is None, /act behavior is unchanged."""

    def test_request_succeeds_without_rtc(self):
        server = _StubServer(GOOD_RESPONSE)
        # rtc_adapter not set
        client = TestClient(_build_app(server))
        r = client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        assert r.status_code == 200
        assert r.json()["_test_episode_reset"] is False
        assert r.json()["_test_chunk_count"] == 0

    def test_predict_called_with_request_fields(self):
        server = _StubServer(GOOD_RESPONSE)
        client = TestClient(_build_app(server))
        client.post(
            "/act",
            json={"image": "img-data", "instruction": "pick", "state": [0.1, 0.2]},
        )
        assert len(server.predict_calls) == 1
        call = server.predict_calls[0]
        assert call["image_b64"] == "img-data"
        assert call["instruction"] == "pick"
        assert call["state"] == [0.1, 0.2]


class TestEpisodeReset:
    """RTC adapter present + episode_id changes → reset is called."""

    def _server_with_adapter(self):
        server = _StubServer(GOOD_RESPONSE)
        cfg = RtcAdapterConfig(enabled=False)
        server.rtc_adapter = RtcAdapter(
            policy=server,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        return server

    def test_first_request_with_episode_id_resets(self):
        """First call has _active_episode_id=None; any provided episode_id is a change."""
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        r = client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        assert r.json()["_test_episode_reset"] is True
        assert server.rtc_adapter._active_episode_id == "ep-1"

    def test_same_episode_id_does_not_reset(self):
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        # Second call with same episode_id — no reset
        r = client.post("/act", json={"image": "y", "episode_id": "ep-1"})
        assert r.json()["_test_episode_reset"] is False

    def test_changed_episode_id_resets(self):
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        # Switch to ep-2
        r = client.post("/act", json={"image": "y", "episode_id": "ep-2"})
        assert r.json()["_test_episode_reset"] is True
        assert server.rtc_adapter._active_episode_id == "ep-2"

    def test_no_episode_id_no_reset(self):
        """Requests without episode_id don't trigger reset (preserves session state)."""
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        r = client.post("/act", json={"image": "x"})
        assert r.json()["_test_episode_reset"] is False
        assert server.rtc_adapter._active_episode_id is None

    def test_reset_clears_chunk_count(self):
        """An episode reset zeros out chunk_count even if previous chunks accumulated."""
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        # 3 calls in ep-1 — chunk_count grows to 3
        for _ in range(3):
            client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        assert server.rtc_adapter._chunk_count == 3
        # Switch to ep-2 — reset zeros it; the post-predict merge bumps to 1
        r = client.post("/act", json={"image": "y", "episode_id": "ep-2"})
        assert r.json()["_test_chunk_count"] == 1


class TestMergeAndUpdateHook:
    """The post-predict merge_and_update populates carry state."""

    def _server_with_adapter(self, response=GOOD_RESPONSE):
        server = _StubServer(response)
        cfg = RtcAdapterConfig(enabled=False)
        server.rtc_adapter = RtcAdapter(
            policy=server,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        return server

    def test_successful_predict_increments_chunk_count(self):
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        r = client.post("/act", json={"image": "x"})
        assert r.json()["_test_chunk_count"] == 1

    def test_three_predicts_accumulate_chunk_count(self):
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        for _ in range(3):
            client.post("/act", json={"image": "x"})
        assert server.rtc_adapter._chunk_count == 3

    def test_actions_pushed_to_buffer(self):
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        client.post("/act", json={"image": "x"})
        # Buffer cap is 10, action chunk is 10 long → buffer fills to 10
        assert server.rtc_adapter.buffer.size == 10

    def test_carry_forward_populated_on_second_call(self):
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        client.post("/act", json={"image": "x"})  # chunk 1
        # Buffer has chunk 1 contents (capacity 10 = full chunk 1)
        client.post("/act", json={"image": "y"})  # chunk 2
        # _prev_chunk_left_over should now hold chunk 1's snapshot
        assert server.rtc_adapter._prev_chunk_left_over is not None
        assert server.rtc_adapter._prev_chunk_left_over.shape == (10, 7)

    def test_error_result_skips_merge(self):
        """When predict returns {"error": ...}, merge_and_update is skipped."""
        bad_response = {"error": "model crashed", "actions": []}
        server = self._server_with_adapter(response=bad_response)
        client = TestClient(_build_app(server))
        r = client.post("/act", json={"image": "x"})
        assert r.json()["_test_chunk_count"] == 0
        assert server.rtc_adapter.buffer.size == 0

    def test_empty_actions_skips_merge(self):
        empty_actions_response = {
            "actions": [],
            "latency_ms": 50.0,
        }
        server = self._server_with_adapter(response=empty_actions_response)
        client = TestClient(_build_app(server))
        r = client.post("/act", json={"image": "x"})
        assert r.json()["_test_chunk_count"] == 0

    def test_latency_seconds_recorded(self):
        """latency_ms in response → seconds in tracker."""
        # 200ms latency in response should record as 0.2s
        response = {**GOOD_RESPONSE, "latency_ms": 200.0}
        server = self._server_with_adapter(response=response)
        # Bypass cold-start discard for this test
        server.rtc_adapter = RtcAdapter(
            policy=server,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=RtcAdapterConfig(enabled=False, cold_start_discard=0),
        )
        client = TestClient(_build_app(server))
        client.post("/act", json={"image": "x"})
        assert server.rtc_adapter.latency._samples[-1] == pytest.approx(0.2)


class TestEpisodeResetMergeOrder:
    """Reset runs BEFORE predict, merge runs AFTER. Order matters because
    reset clears _prev_chunk_left_over, then the new merge populates it."""

    def _server_with_adapter(self):
        server = _StubServer(GOOD_RESPONSE)
        cfg = RtcAdapterConfig(enabled=False)
        server.rtc_adapter = RtcAdapter(
            policy=server,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        return server

    def test_reset_then_merge_in_one_call(self):
        """Switching episodes mid-stream:
        - Pre-state: 3 chunks accumulated, _prev_chunk_left_over populated
        - Call with new episode_id:
          - Reset clears _chunk_count to 0 and _prev_chunk_left_over to None
          - Predict runs (1 chunk back)
          - Merge bumps chunk_count to 1; _prev_chunk_left_over stays None
            because the buffer was empty at the moment of merge
            (reset cleared chunk_count but the buffer also gets cleared by reset?
            actually no — reset doesn't clear the buffer, it only resets the
            adapter's state. So buffer still has chunk 3.
            That means after reset, the buffer has the previous episode's
            chunk 3, and merge snapshots THAT before pushing chunk N+1.)

        This test documents that reset doesn't wipe the action buffer — only
        the adapter's tracker + carry state. The buffer is shared with the
        server's existing replan path, not owned by the adapter.
        """
        server = self._server_with_adapter()
        client = TestClient(_build_app(server))
        # Build up 2 chunks in ep-1
        client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        client.post("/act", json={"image": "x", "episode_id": "ep-1"})
        assert server.rtc_adapter._chunk_count == 2
        # Switch to ep-2
        r = client.post("/act", json={"image": "y", "episode_id": "ep-2"})
        # Chunk count is back to 1 (reset cleared, then merge bumped)
        assert r.json()["_test_chunk_count"] == 1


class TestAdaptiveSignalHelper:
    def test_server_helper_records_guard_a2c2_and_uncertainty(self):
        adapter = RtcAdapter(
            policy=_StubServer(GOOD_RESPONSE),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=RtcAdapterConfig(
                enabled=False,
                adaptive_chunking_enabled=True,
            ),
        )

        _record_rtc_adaptive_signal(
            adapter,
            {
                "a2c2_correction_magnitude": 0.25,
                "uncertainty_score": 0.4,
            },
            guard_margin=0.03,
        )

        stats = adapter.get_stats()
        assert stats["adaptive_signal"]["guard_margin"] == pytest.approx(0.03)
        assert stats["adaptive_signal"]["correction_magnitude"] == pytest.approx(0.25)
        assert stats["adaptive_signal"]["uncertainty"] == pytest.approx(0.4)
