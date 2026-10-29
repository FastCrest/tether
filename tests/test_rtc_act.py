"""HTTP /act with --rtc: capability gate, leftover contract, LeRobot pass-through.

The SmolVLA cases run the real create_app + /act path with a tiny random
SmolVLA (``tests/_rtc_smolvla_tiny.py``) whose denoise loop is LeRobot's own
RTC-aware ``euler_integrate``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.export_config_factory import write_test_export_config

pytest.importorskip("lerobot.policies.rtc")
torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from tether.runtime.rtc_adapter import RtcAdapterConfig  # noqa: E402

from tests import _rtc_smolvla_tiny as tiny  # noqa: E402


def _rtc(**overrides):
    cfg = dict(enabled=True, execute_hz=100.0, rtc_execution_horizon=8, cold_start_discard=0)
    cfg.update(overrides)
    return RtcAdapterConfig(**cfg)


def _smolvla_export(tmp_path, num_steps=tiny.NUM_STEPS):
    write_test_export_config(
        tmp_path,
        model_type="smolvla",
        export_kind="decomposed_onnx",
        artifacts=["expert_stack.onnx"],
        num_denoising_steps=num_steps,
    )
    return tmp_path


@pytest.fixture(scope="module")
def tiny_policy():
    return tiny.build_policy()


def _client(tmp_path, monkeypatch, policy, rtc_config):
    from tether.runtime.server import create_app

    monkeypatch.setenv("TETHER_NATIVE", "1")
    app = create_app(
        str(_smolvla_export(tmp_path)), device="cpu", rtc_config=rtc_config, prewarm=False
    )
    tiny.install(app.state.tether_server, policy)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------


class TestCapabilityGate:
    def test_baked_loop_runtime_is_refused(self, tmp_path, monkeypatch):
        from tether.runtime.server import create_app

        monkeypatch.delenv("TETHER_NATIVE", raising=False)
        write_test_export_config(
            tmp_path, model_type="smolvla", export_kind="monolithic_onnx",
            artifacts=["model.onnx"],
        )
        with pytest.raises(RuntimeError, match="per-step velocity access"):
            create_app(str(tmp_path), device="cpu", rtc_config=_rtc())

    def test_pi05_baked_export_is_refused(self, tmp_path, monkeypatch):
        from tether.runtime.server import create_app

        monkeypatch.delenv("TETHER_NATIVE", raising=False)
        write_test_export_config(
            tmp_path, model_type="pi05_decomposed", export_kind="decomposed_onnx",
            artifacts=["vlm_prefix.onnx", "expert_denoise.onnx"],
        )
        with pytest.raises(RuntimeError, match="per_step_expert=false"):
            create_app(str(tmp_path), device="cpu", rtc_config=_rtc())

    def test_pi05_per_step_export_is_admitted(self, tmp_path, monkeypatch):
        from tether.runtime.decomposed_server import Pi05DecomposedServer
        from tether.runtime.server import create_app

        monkeypatch.delenv("TETHER_NATIVE", raising=False)
        write_test_export_config(
            tmp_path, model_type="pi05_decomposed", export_kind="decomposed_onnx",
            artifacts=["vlm_prefix.onnx", "expert_denoise.onnx"],
            decomposed={"per_step_expert": True},
        )
        app = create_app(str(tmp_path), device="cpu", rtc_config=_rtc())
        assert isinstance(app.state.tether_server, Pi05DecomposedServer)
        assert app.state.tether_server.rtc_streams is not None

    def test_smolvla_native_is_admitted(self, tmp_path, monkeypatch):
        from tether.runtime.server import create_app

        monkeypatch.setenv("TETHER_NATIVE", "1")
        app = create_app(str(_smolvla_export(tmp_path)), device="cpu", rtc_config=_rtc())
        assert app.state.tether_server.rtc_streams is not None
        assert app.state.tether_server._rtc_required is True

    @pytest.mark.parametrize("kwargs, flag", [
        ({"max_batch": 2}, "--max-batch"),
        ({"bid_n_candidates": 4}, "--bid-n-candidates"),
        ({"a2c2_checkpoint": "head.npz"}, "--a2c2-checkpoint"),
        ({"action_similarity_threshold": 0.05}, "--action-similarity-threshold"),
    ])
    def test_conflicting_wedges_are_refused(self, tmp_path, kwargs, flag):
        from tether.runtime.server import create_app

        with pytest.raises(RuntimeError, match=flag):
            create_app(str(tmp_path / "unused"), device="cpu", rtc_config=_rtc(), **kwargs)

    def test_one_step_denoise_is_refused(self, tmp_path, monkeypatch):
        from tether.runtime.server import create_app

        monkeypatch.setenv("TETHER_NATIVE", "1")
        with pytest.raises(ValueError, match="num_steps=1"):
            create_app(str(_smolvla_export(tmp_path, num_steps=1)), device="cpu", rtc_config=_rtc())

    def test_rtc_off_has_no_adapter(self, tmp_path, monkeypatch):
        from tether.runtime.server import create_app

        monkeypatch.setenv("TETHER_NATIVE", "1")
        app = create_app(str(_smolvla_export(tmp_path)), device="cpu")
        assert app.state.tether_server.rtc_streams is None


# ---------------------------------------------------------------------------
# /act wiring on SmolVLA native (LeRobot RTC)
# ---------------------------------------------------------------------------


def _act(client, **body):
    r = client.post("/act", json={"instruction": "pick", "state": [0.0] * 8, **body})
    return r


class TestActWiring:
    def test_first_chunk_is_unguided_and_carries_chunk_id(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            body = _act(client, episode_id="ep").json()
        assert len(body["actions"]) == tiny.CHUNK
        assert isinstance(body["chunk_id"], str) and len(body["chunk_id"]) == 16
        assert body["rtc"]["applied"] is False
        assert body["rtc"]["leftover_source"] == "none"
        assert body["rtc"]["chunk_id"] == body["chunk_id"]

    def test_client_report_drives_the_leftover(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            first = _act(client, episode_id="ep").json()
            second = _act(client, episode_id="ep", actions_executed=3,
                          chunk_id=first["chunk_id"]).json()
        rtc = second["rtc"]
        assert rtc["applied"] is True
        assert rtc["leftover_source"] == "client"
        assert rtc["actions_executed"] == 3
        assert rtc["leftover_len"] == tiny.CHUNK - 3
        assert rtc["guidance"] == "lerobot_identity_jacobian"
        assert second["chunk_id"] != first["chunk_id"]

    def test_client_without_fields_gets_labelled_estimate(self, tmp_path, monkeypatch, tiny_policy):
        """Backward compatibility: an RTC-unaware client still works."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            _act(client, episode_id="ep")
            body = _act(client, episode_id="ep").json()
        assert body["rtc"]["leftover_source"] == "estimated"
        assert "actions" in body

    def test_rtc_off_server_ignores_the_fields(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, None) as client:
            body = _act(client, actions_executed=2, chunk_id="abc123").json()
        assert "actions" in body and "rtc" not in body and "chunk_id" not in body

    @pytest.mark.parametrize("extra, status, code", [
        ({"actions_executed": 2}, 422, "rtc_report_incomplete"),
        ({"chunk_id": "ab"}, 422, "rtc_report_incomplete"),
        ({"actions_executed": 2, "chunk_id": "deadbeef"}, 409, "rtc_chunk_mismatch"),
        ({"actions_executed": tiny.CHUNK + 1, "chunk_id": None}, 422, "rtc_actions_executed_invalid"),
    ])
    def test_bad_reports_are_refused(self, tmp_path, monkeypatch, tiny_policy, extra, status, code):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            first = _act(client, episode_id="ep").json()
            if "chunk_id" in extra and extra["chunk_id"] is None:
                extra = {**extra, "chunk_id": first["chunk_id"]}
            r = _act(client, episode_id="ep", **extra)
            assert r.status_code == status
            assert r.json()["error"] == code
            # A refused report does not advance the session.
            ok = _act(client, episode_id="ep", actions_executed=1, chunk_id=first["chunk_id"])
            assert ok.status_code == 200

    def test_new_episode_drops_the_leftover(self, tmp_path, monkeypatch, tiny_policy):
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            first = _act(client, episode_id="a").json()
            body = _act(client, episode_id="b").json()
            stale = _act(client, episode_id="b", actions_executed=1, chunk_id=first["chunk_id"])
        assert body["rtc"]["applied"] is False
        assert stale.status_code == 409

    # execute_hz=10000 forces inference_delay far past the 8-row leftover, the
    # case that made the old slicing flaky under load (review R-B1).
    @pytest.mark.parametrize("execute_hz", [100.0, 10000.0])
    def test_guidance_pulls_the_prefix_toward_the_leftover(self, tmp_path, monkeypatch, tiny_policy, execute_hz):
        """Matched seeds: RTC-on vs RTC-off second chunk against the leftover."""
        def second_chunk(rtc_config, seed):
            with _client(tmp_path, monkeypatch, tiny_policy, rtc_config) as client:
                torch.manual_seed(seed)
                first = _act(client, episode_id="ep").json()
                torch.manual_seed(seed + 1)
                extra = {} if rtc_config is None else {
                    "actions_executed": 2, "chunk_id": first["chunk_id"]}
                second = _act(client, episode_id="ep", **extra).json()
            return np.asarray(first["actions"]), np.asarray(second["actions"]), second

        gaps_on, gaps_off = [], []
        for seed in range(4):
            prev, on, body = second_chunk(_rtc(execute_hz=execute_hz), 10 * seed)
            prev_off, off, _ = second_chunk(None, 10 * seed)
            np.testing.assert_allclose(prev, prev_off, atol=1e-5)  # same first chunk
            d = body["rtc"]["inference_delay"]
            assert d > 0
            left = prev[2:]
            k = min(d, len(left))  # as production: the prefix is capped by the leftover
            gaps_on.append(np.abs(on[:k] - left[:k]).max())
            gaps_off.append(np.abs(off[:k] - left[:k]).max())
            assert body["rtc"]["prefix_max_abs"] == pytest.approx(gaps_on[-1], abs=1e-5)
        assert max(gaps_on) < min(gaps_off)

    def test_guard_clamp_is_what_the_leftover_follows(self, tmp_path, monkeypatch, tiny_policy):
        """The stored chunk is the response's actions after every guard."""
        with _client(tmp_path, monkeypatch, tiny_policy, _rtc()) as client:
            body = _act(client, episode_id="ep").json()
            adapter = client.app.state.tether_server.rtc_streams.acquire("ep").adapter
            np.testing.assert_allclose(adapter._last_chunk, np.asarray(body["actions"]), atol=1e-6)


# ---------------------------------------------------------------------------
# Leftover units: processed -> the policy's normalized space
# ---------------------------------------------------------------------------


class TestNormalizeActions:
    def test_smolvla_normalize_inverts_the_postprocessor(self, tmp_path, tiny_policy):
        from tether.runtime.smolvla_native import SmolVLANativeServer

        server = SmolVLANativeServer(_smolvla_export(tmp_path), device="cpu")
        tiny.install(server, tiny_policy)
        server.load()
        norm = np.random.default_rng(0).standard_normal((tiny.CHUNK, tiny.ACTION_DIM)).astype(np.float32)
        processed = server.postprocessor(torch.from_numpy(norm)).numpy()
        np.testing.assert_allclose(processed, norm * tiny.ACTION_STD + tiny.ACTION_MEAN, atol=1e-5)
        np.testing.assert_allclose(server.normalize_actions(processed), norm, atol=1e-5)

    def test_smolvla_refuses_a_postprocessor_it_cannot_invert(self, tmp_path, tiny_policy):
        from tether.runtime.smolvla_native import SmolVLANativeServer

        server = SmolVLANativeServer(_smolvla_export(tmp_path), device="cpu")
        tiny.install(server, tiny_policy)
        server._rtc_required = True
        server.load()
        server.postprocessor = SimpleNamespace(steps=[object()])
        with pytest.raises(RuntimeError, match="one action unnormalizer"):
            server.normalize_actions(np.zeros((2, tiny.ACTION_DIM), np.float32))

    def test_pi05_normalize_inverts_denorm(self, tmp_path):
        from tether.runtime.decomposed_server import Pi05DecomposedServer

        server = Pi05DecomposedServer(tmp_path, device="cpu")
        server._action_mean = np.array([0.1, -0.3, 2.0], np.float32)
        server._action_std = np.array([0.5, 2.0, 0.1], np.float32)
        norm = np.random.default_rng(1).standard_normal((5, 3)).astype(np.float32)
        processed = norm * server._action_std + server._action_mean
        np.testing.assert_allclose(server.normalize_actions(processed), norm, atol=1e-5)

    def test_pi05_without_stats_is_identity(self, tmp_path):
        from tether.runtime.decomposed_server import Pi05DecomposedServer

        server = Pi05DecomposedServer(tmp_path, device="cpu")
        x = np.ones((3, 7), np.float32)
        np.testing.assert_array_equal(server.normalize_actions(x), x)


class TestFrozenPolicy:
    def test_load_leaves_no_parameter_requiring_grad(self, tmp_path):
        from tether.runtime.smolvla_native import SmolVLANativeServer

        server = SmolVLANativeServer(_smolvla_export(tmp_path), device="cpu")
        tiny.install(server, tiny.build_policy(seed=3))
        server.load()
        assert not any(p.requires_grad for p in server.policy.parameters())
        assert not server.policy.training

    def test_rtc_output_is_bit_identical_with_frozen_parameters(self, tmp_path):
        """Freezing removes LeRobot's parameter graph, not any arithmetic."""
        from tether.runtime.rtc_guidance import RtcGuidance
        from tether.runtime.smolvla_native import SmolVLANativeServer

        server = SmolVLANativeServer(_smolvla_export(tmp_path), device="cpu")
        tiny.install(server, tiny.build_policy(seed=4))
        server.load()
        left = np.random.default_rng(0).standard_normal((7, tiny.ACTION_DIM)).astype(np.float32)
        g = RtcGuidance(3, 8, left)

        def run(trainable):
            server.policy.requires_grad_(trainable)
            torch.manual_seed(11)
            return np.asarray(server.predict(state=[0.0] * 8, instruction="pick", rtc=g)["actions"])

        trainable, frozen = run(True), run(False)
        np.testing.assert_array_equal(trainable, frozen)
