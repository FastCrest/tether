"""RTC state is per stream (episode_id), serialised, replay-safe and validated.

Regression tests for the feat/rtc-guidance-act review (R-B2, R-B3, R-N1..R-N4).
The /act cases run the real create_app path with the tiny random SmolVLA.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest

pytest.importorskip("lerobot.policies.rtc")
pytest.importorskip("torch")

from tests.test_rtc_act import _act, _client, _rtc  # noqa: E402
from tests import _rtc_smolvla_tiny as tiny  # noqa: E402


@pytest.fixture(scope="module")
def tiny_policy():
    return tiny.build_policy(seed=5)


# ---------------------------------------------------------------------------
# R-B2: no cross-stream contamination
# ---------------------------------------------------------------------------


class TestPerStreamState:
    def test_rtc_requires_an_episode_id(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            r = _act(client)
        assert r.status_code == 422
        assert r.json()["error"] == "rtc_episode_required"

    def test_two_robots_never_share_a_leftover(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            a1 = _act(client, episode_id="robot-a").json()
            b1 = _act(client, episode_id="robot-b").json()
            # B's first chunk is unguided: A's chunk is not B's leftover.
            assert b1["rtc"]["applied"] is False
            # A's exact report still matches A's stream after B ran.
            a2 = _act(client, episode_id="robot-a", actions_executed=3, chunk_id=a1["chunk_id"])
            assert a2.status_code == 200, a2.text
            assert a2.json()["rtc"]["leftover_source"] == "client"
            # B's report against A's chunk id is refused, not cross-matched.
            cross = _act(client, episode_id="robot-b", actions_executed=1, chunk_id=a2.json()["chunk_id"])
            assert cross.status_code == 409

    def test_returning_to_an_episode_keeps_only_its_own_state(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            a1 = _act(client, episode_id="a").json()
            _act(client, episode_id="b")
            a2 = _act(client, episode_id="a", actions_executed=2, chunk_id=a1["chunk_id"]).json()
        assert a2["rtc"]["applied"] is True
        assert a2["rtc"]["leftover_len"] == tiny.CHUNK - 2


# ---------------------------------------------------------------------------
# R-B3 + R-N1: serialised per stream; an exact replay is idempotent
# ---------------------------------------------------------------------------


class TestSerialisationAndReplay:
    def test_concurrent_duplicate_reports_get_one_chunk(self, tmp_path, monkeypatch, tiny_policy):
        """Two in-flight requests with the same report: one inference, both
        answered with the same chunk, and the next real report still works."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            first = _act(client, episode_id="ep").json()
            report = {"episode_id": "ep", "actions_executed": 4, "chunk_id": first["chunk_id"]}
            with ThreadPoolExecutor(2) as pool:
                r1, r2 = pool.map(lambda _: _act(client, **report), range(2))
            assert r1.status_code == r2.status_code == 200
            assert r1.json()["chunk_id"] == r2.json()["chunk_id"]
            assert r1.json()["actions"] == r2.json()["actions"]
            nxt = _act(client, episode_id="ep", actions_executed=1, chunk_id=r1.json()["chunk_id"])
            assert nxt.status_code == 200, nxt.text

    def test_retry_after_a_lost_response_replays_it(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            first = _act(client, episode_id="ep").json()
            report = {"episode_id": "ep", "actions_executed": 3, "chunk_id": first["chunk_id"]}
            sent = _act(client, **report)
            again = _act(client, **report)
            assert again.status_code == 200
            assert again.headers.get("X-Tether-RTC-Replay") == "true"
            assert again.json() == sent.json()
            # A different count against the old chunk is stale, not a replay.
            stale = _act(client, **{**report, "actions_executed": 5})
            assert stale.status_code == 409


# ---------------------------------------------------------------------------
# R-N2: strict field validation
# ---------------------------------------------------------------------------


class TestFieldValidation:
    @pytest.mark.parametrize("bad", [True, "3", 2.0, -1, 10**6])
    def test_actions_executed_must_be_a_small_non_bool_int(self, tmp_path, monkeypatch, tiny_policy, bad):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            first = _act(client, episode_id="ep").json()
            r = _act(client, episode_id="ep", actions_executed=bad, chunk_id=first["chunk_id"])
        assert r.status_code == 422

    @pytest.mark.parametrize("bad", ["x" * 65, "not-hex!", ""])
    def test_chunk_id_has_a_length_cap_and_charset(self, tmp_path, monkeypatch, tiny_policy, bad):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            r = _act(client, episode_id="ep", actions_executed=0, chunk_id=bad)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# R-N1: client does not retry a 409 and drops a stale report
# ---------------------------------------------------------------------------


class TestClientConflictHandling:
    def test_409_is_not_retried_and_the_proxy_drops_the_stale_chunk(self):
        from tether.client.client import TetherClient, TetherClientError

        bodies = []

        def handler(request):
            import json

            body = json.loads(request.content)
            bodies.append(body)
            if body.get("chunk_id") == "c1" and len(bodies) == 2:
                return httpx.Response(409, json={"error": "rtc_chunk_mismatch"})
            return httpx.Response(200, json={"actions": [[0.0]], "chunk_id": f"c{len(bodies)}"})

        client = TetherClient("http://test", max_retries=3, initial_backoff_s=0.0)
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        with client.episode() as ep:
            ep.act(instruction="x")
            with pytest.raises(TetherClientError) as exc:
                ep.act(instruction="x", actions_executed=2)
            assert exc.value.status_code == 409
            assert len(bodies) == 2  # not retried
            ep.act(instruction="x", actions_executed=2)
        assert "chunk_id" not in bodies[2]  # stale report dropped, server estimates


# ---------------------------------------------------------------------------
# R-N3 + R-N4: pi0.5 startup refusals and the IOBinding device
# ---------------------------------------------------------------------------


class TestPi05Startup:
    def _server(self, tmp_path, **kwargs):
        from tests.export_config_factory import write_test_export_config
        from tether.runtime.decomposed_server import Pi05DecomposedServer

        write_test_export_config(
            tmp_path, model_type="pi05_decomposed", export_kind="decomposed_onnx",
            artifacts=["vlm_prefix.onnx", "expert_denoise.onnx"],
            decomposed={"per_step_expert": True},
        )
        return Pi05DecomposedServer(tmp_path, device="cpu", **kwargs)

    def test_similarity_fast_path_is_refused_by_the_capability_check(self, tmp_path):
        reason = self._server(tmp_path, action_similarity_threshold=0.05).rtc_guidance_support()
        assert reason and "fast path" in reason

    def test_per_step_export_without_conflicts_is_supported(self, tmp_path):
        assert self._server(tmp_path).rtc_guidance_support() is None

    def test_a2c2_or_bid_set_after_the_gate_is_refused(self, tmp_path):
        server = self._server(tmp_path)
        server._rtc_required = True
        with pytest.raises(RuntimeError, match="RTC"):
            server.set_a2c2_hook(object())

    @pytest.mark.parametrize("providers, device", [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
        (["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
        (["CPUExecutionProvider"], "cpu"),
        (["CoreMLExecutionProvider", "CPUExecutionProvider"], "cpu"),
    ])
    def test_iobinding_keeps_past_kv_on_the_gpu_for_cuda_and_tensorrt(self, providers, device):
        from tether.runtime.pi05_decomposed_server import iobinding_device

        assert iobinding_device(providers) == device
