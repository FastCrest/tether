"""RTC streams vs the serving safety gates (re-review #2, 2026-10-28).

Ported from the reviewer's throwaway repros (scratchpad/rr2). A guard trip,
guard reset, circuit-breaker degrade or license change must invalidate every
stream's replay memo and carry-forward; a refused request must not claim a
stream; idle streams free up within seconds, not minutes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("lerobot.policies.rtc")
pytest.importorskip("torch")

import tether.runtime.server as srv  # noqa: E402
from tests import _rtc_smolvla_tiny as tiny  # noqa: E402
from tests.test_rtc_act import _act, _client, _rtc, _smolvla_export  # noqa: E402

BODY = {"instruction": "pick", "state": [0.0] * 8}


@pytest.fixture(scope="module")
def tiny_policy():
    return tiny.build_policy(seed=5)


def _trip(server):
    server._action_guard = SimpleNamespace(tripped=True, trip_reason="joint limit", reset=lambda: None)


def _committed(client):
    first = _act(client, episode_id="ep").json()
    report = {"episode_id": "ep", "actions_executed": 2, "chunk_id": first["chunk_id"]}
    original = _act(client, **report)
    assert original.status_code == 200
    return report, original


class TestSafetyInvalidatesStreams:
    def test_replay_after_guard_reset_runs_fresh_inference(self, tmp_path, monkeypatch, tiny_policy):
        """RR2-B1, reviewer's repro: trip -> refused retry -> reset -> retry
        with a new observation must not replay the pre-trip chunk."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            server = client.app.state.tether_server
            report, original = _committed(client)
            _trip(server)
            assert _act(client, **report).status_code == 503
            server._action_guard = None  # operator reset
            again = _act(client, **{**report, "state": [9.0] * 8})
        assert again.headers.get("X-Tether-RTC-Replay") is None
        # The pre-trip chunk is gone with the carry-forward: a fresh start.
        assert again.status_code == 409 or again.json()["rtc"]["leftover_source"] == "none"

    def test_guard_reset_endpoint_invalidates_without_a_request_in_between(
        self, tmp_path, monkeypatch, tiny_policy
    ):
        """Trip and reset with no /act while latched: /guard/reset itself
        must drop every stream's memo and leftover."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            server = client.app.state.tether_server
            report, _ = _committed(client)
            state = {"tripped": True}
            server._action_guard = SimpleNamespace(
                tripped=True, trip_reason="x",
                reset=lambda: state.update(tripped=False),
            )
            r = client.post("/guard/reset")
            assert r.status_code == 200
            server._action_guard = None
            again = _act(client, **report)
        assert again.headers.get("X-Tether-RTC-Replay") is None

    def test_degrade_and_license_change_invalidate(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            server = client.app.state.tether_server
            report, _ = _committed(client)
            server._pro_license_deadline = 12345  # license renewed/changed
            again = _act(client, **report)
            assert again.headers.get("X-Tether-RTC-Replay") is None
            report2, _ = _committed(client)
            server.health_state = "degraded"
            assert _act(client, **report2).status_code == 503
            server.health_state = "ready"  # operator restart path in-process
            assert _act(client, **report2).headers.get("X-Tether-RTC-Replay") is None

    def test_a_response_produced_while_the_guard_latched_is_not_cached(
        self, tmp_path, monkeypatch, tiny_policy
    ):
        """The guard can latch inside check() and still return the chunk."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            server = client.app.state.tether_server
            first = _act(client, episode_id="ep").json()
            report = {"episode_id": "ep", "actions_executed": 1, "chunk_id": first["chunk_id"]}
            orig = server.predict

            def latch_during_predict(*a, **k):
                out = orig(*a, **k)
                _trip(server)
                return out

            server.predict = latch_during_predict
            assert _act(client, **report).status_code == 200
            server.predict = orig
            server._action_guard = None
            again = _act(client, **report)
        assert again.headers.get("X-Tether-RTC-Replay") is None


class TestStreamTable:
    def test_refused_requests_do_not_claim_streams(self, tmp_path, monkeypatch, tiny_policy):
        """RR2-B2(b), reviewer's repro."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc(max_streams=4)) as client:
            server = client.app.state.tether_server
            _trip(server)
            codes = [_act(client, episode_id=f"j{i}").status_code for i in range(4)]
            server._action_guard = None
            legit = _act(client, episode_id="robot")
            assert len(server.rtc_streams) == 1
        assert codes == [503] * 4
        assert legit.status_code == 200, legit.text

    def test_default_idle_age_is_seconds_derived_from_chunk_timing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TETHER_NATIVE", "1")
        app = srv.create_app(str(_smolvla_export(tmp_path)), device="cpu",
                             rtc_config=_rtc(execute_hz=100.0), prewarm=False)
        streams = app.state.tether_server.rtc_streams
        # max(2 s, 4 x chunk duration); the export here declares no chunk size,
        # so the 50-action default applies: 4 x 0.5 s = 2 s.
        assert streams.min_idle_s == pytest.approx(2.0)

    def test_full_table_503_carries_retry_after(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc(max_streams=1, stream_min_idle_s=30.0)) as client:
            assert _act(client, episode_id="a").status_code == 200
            full = _act(client, episode_id="b")
        assert full.status_code == 503
        assert full.json()["error"] == "rtc_streams_full"
        assert 1 <= int(full.headers["Retry-After"]) <= 30

    def test_rtc_without_api_key_warns_loudly(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("TETHER_NATIVE", "1")
        monkeypatch.delenv("TETHER_API_KEY", raising=False)
        with caplog.at_level("WARNING"):
            srv.create_app(str(_smolvla_export(tmp_path)), device="cpu", rtc_config=_rtc(), prewarm=False)
        assert "unauthenticated" in caplog.text and "--api-key" in caplog.text


class TestValidationAndSdk:
    def test_episode_id_with_trailing_newline_is_refused(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            r = _act(client, episode_id="robot\n")
        assert r.status_code == 422
        assert r.json()["error"] == "rtc_episode_invalid"

    def test_sdk_episode_sends_episode_start_on_its_first_request(self):
        import json

        from tether.client.client import TetherClient

        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"actions": [[0.0]], "chunk_id": f"c{len(bodies)}"})

        client = TetherClient("http://test")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        with client.episode(episode_id="fixed") as ep:
            ep.act(instruction="x")
            ep.act(instruction="x", actions_executed=2)
        assert bodies[0]["episode_start"] is True
        assert "episode_start" not in bodies[1]


class TestAsyncGates:
    """Reviewer's async repro: a guard trip while a request waits on the
    stream lock is honoured by the waiter."""

    def test_waiter_sees_a_trip_that_happened_while_it_queued(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TETHER_NATIVE", "1")
        app = srv.create_app(str(_smolvla_export(tmp_path)), device="cpu", rtc_config=_rtc(), prewarm=False)
        tiny.install(app.state.tether_server, tiny.build_policy(seed=5))

        async def main():
            async with app.router.lifespan_context(app):
                server = app.state.tether_server
                slow_target = server.predict_from_base64_async

                async def slow(*a, **k):
                    await asyncio.sleep(0.4)
                    return await slow_target(*a, **k)

                server.predict_from_base64_async = slow
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
                    a = asyncio.create_task(ac.post("/act", json={**BODY, "episode_id": "ep"}))
                    await asyncio.sleep(0.1)
                    b = asyncio.create_task(ac.post("/act", json={**BODY, "episode_id": "ep"}))
                    await asyncio.sleep(0.1)
                    _trip(server)
                    return await a, await b

        _, waiter = asyncio.run(asyncio.wait_for(main(), 60))
        assert waiter.status_code == 503
