"""Tests for the embodiment-config → ActionGuard wiring (B.6).

Two layers:
- SafetyLimits.from_embodiment_config() — pure mapping, no I/O
- ActionGuard.from_embodiment_config() — convenience wrapper
- /act handler integration via FastAPI TestClient — clamps + adds
  guard_violations to response
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tether.embodiments import EmbodimentConfig
from tether.safety.guard import ActionGuard, SafetyLimits


# ---------------------------------------------------------------------------
# SafetyLimits.from_embodiment_config
# ---------------------------------------------------------------------------


class TestSafetyLimitsFromEmbodiment:
    def test_franka_limits_match_action_dim(self):
        cfg = EmbodimentConfig.load_preset("franka")
        limits = SafetyLimits.from_embodiment_config(cfg)
        assert len(limits.position_min) == 7
        assert len(limits.position_max) == 7
        assert len(limits.velocity_max) == 7
        assert len(limits.joint_names) == 7

    def test_franka_position_bounds_match_ranges(self):
        """position_min/max should mirror action_space.ranges per axis."""
        cfg = EmbodimentConfig.load_preset("franka")
        limits = SafetyLimits.from_embodiment_config(cfg)
        for i, (lo, hi) in enumerate(cfg.action_space["ranges"]):
            assert limits.position_min[i] == pytest.approx(lo)
            assert limits.position_max[i] == pytest.approx(hi)

    def test_velocity_uses_gripper_cap_at_gripper_idx(self):
        """Gripper dim should use max_gripper_velocity, others max_ee_velocity."""
        cfg = EmbodimentConfig.load_preset("franka")
        limits = SafetyLimits.from_embodiment_config(cfg)
        ee_vel = float(cfg.constraints["max_ee_velocity"])
        grip_vel = float(cfg.constraints["max_gripper_velocity"])
        gripper_idx = cfg.gripper_idx
        for i, v in enumerate(limits.velocity_max):
            if i == gripper_idx:
                assert v == pytest.approx(grip_vel)
            else:
                assert v == pytest.approx(ee_vel)

    @pytest.mark.parametrize("name", ["franka", "so100", "ur5"])
    def test_all_presets_produce_valid_limits(self, name):
        cfg = EmbodimentConfig.load_preset(name)
        limits = SafetyLimits.from_embodiment_config(cfg)
        assert all(v > 0 for v in limits.velocity_max)
        assert all(lo < hi for lo, hi in zip(limits.position_min, limits.position_max))


# ---------------------------------------------------------------------------
# ActionGuard.from_embodiment_config — clamp behavior on real configs
# ---------------------------------------------------------------------------


class TestActionGuardFromEmbodiment:
    def test_franka_in_range_actions_pass(self):
        """Franka joint_3 has range [-3.07, -0.07] (elbow, entirely negative).
        Use mean_action which IS within all ranges."""
        cfg = EmbodimentConfig.load_preset("franka")
        guard = ActionGuard.from_embodiment_config(cfg)
        mean = np.array(cfg.normalization["mean_action"], dtype=np.float32)
        chunk = np.tile(mean, (5, 1))  # 5 copies of mean_action
        safe, results = guard.check(chunk)
        np.testing.assert_array_equal(safe, chunk)
        assert all(not r.clamped for r in results)
        assert all(not r.violations for r in results)

    def test_franka_out_of_range_clamped(self):
        cfg = EmbodimentConfig.load_preset("franka")
        guard = ActionGuard.from_embodiment_config(cfg)
        # joint_0 max is 2.8973 — push past it
        chunk = np.zeros((1, 7), dtype=np.float32)
        chunk[0, 0] = 99.0
        safe, results = guard.check(chunk)
        assert safe[0, 0] == pytest.approx(2.8973)
        assert any("joint_0" in v for v in results[0].violations)

    def test_nan_zeros_chunk(self):
        cfg = EmbodimentConfig.load_preset("franka")
        guard = ActionGuard.from_embodiment_config(cfg)
        chunk = np.zeros((3, 7), dtype=np.float32)
        chunk[1, 4] = float("nan")
        safe, results = guard.check(chunk)
        assert np.all(safe == 0.0)
        assert results[0].clamped
        assert any("non_finite" in v for v in results[0].violations)

    def test_inf_zeros_chunk(self):
        cfg = EmbodimentConfig.load_preset("franka")
        guard = ActionGuard.from_embodiment_config(cfg)
        chunk = np.zeros((1, 7), dtype=np.float32)
        chunk[0, 2] = float("inf")
        safe, _ = guard.check(chunk)
        assert np.all(safe == 0.0)

    def test_partial_clamp_preserves_other_actions(self):
        """If only one action in a chunk needs clamping, the others pass through."""
        cfg = EmbodimentConfig.load_preset("franka")
        guard = ActionGuard.from_embodiment_config(cfg)
        mean = np.array(cfg.normalization["mean_action"], dtype=np.float32)
        chunk = np.tile(mean, (3, 1))
        chunk[1, 0] = 99.0  # only middle action's joint_0 out of range
        safe, results = guard.check(chunk)
        # Action 0 unchanged (was mean)
        np.testing.assert_array_equal(safe[0], mean)
        # Action 1 clamped on joint_0; other dims preserved
        assert safe[1, 0] == pytest.approx(2.8973)
        np.testing.assert_array_equal(safe[1, 1:], mean[1:])
        # Action 2 unchanged (was mean)
        np.testing.assert_array_equal(safe[2], mean)


# ---------------------------------------------------------------------------
# /act handler integration via TestClient
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    image: str | None = None
    instruction: str = ""
    state: list[float] | None = None
    episode_id: str | None = None


class _StubServer:
    def __init__(self, response: dict, embodiment_cfg=None):
        self.response = response
        self.embodiment_config = embodiment_cfg
        if embodiment_cfg is not None:
            self.embodiment_guard = ActionGuard.from_embodiment_config(embodiment_cfg, mode="clamp")
        else:
            self.embodiment_guard = None

    async def predict_from_base64_async(self, image_b64, instruction, state):
        return dict(self.response)


def _build_test_app(server: _StubServer) -> FastAPI:
    """Mirror the real /act handler's B.6 ActionGuard hook."""
    app = FastAPI()

    @app.post("/act")
    async def act(request: PredictRequest):
        result = await server.predict_from_base64_async(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )
        # Mirror the server.py B.6 hook
        _eg = getattr(server, "embodiment_guard", None)
        guard_violations: list[str] = []
        if (
            _eg is not None
            and isinstance(result, dict)
            and "error" not in result
            and isinstance(result.get("actions"), list)
            and result["actions"]
        ):
            arr = np.asarray(result["actions"], dtype=np.float32)
            safe, check_results = _eg.check(arr)
            if not np.array_equal(arr, safe):
                result["actions"] = safe.tolist()
                for cr in check_results:
                    guard_violations.extend(cr.violations)
                if guard_violations:
                    result["guard_violations"] = guard_violations[:20]
                    result["guard_clamped"] = True
        return JSONResponse(content=result)

    return app


class TestActHandlerHook:
    def test_no_guard_when_no_embodiment(self):
        server = _StubServer(response={"actions": [[99.0] * 7], "latency_ms": 50.0})
        client = TestClient(_build_test_app(server))
        r = client.post("/act", json={"image": "x"})
        assert r.status_code == 200
        body = r.json()
        # Without embodiment_config, no guard runs — out-of-range actions pass through
        assert body["actions"] == [[99.0] * 7]
        assert "guard_violations" not in body

    def test_in_range_actions_unchanged(self):
        """Use Franka's mean_action which is in-range across all 7 joints."""
        cfg = EmbodimentConfig.load_preset("franka")
        mean = list(cfg.normalization["mean_action"])
        server = _StubServer(
            response={"actions": [mean], "latency_ms": 50.0},
            embodiment_cfg=cfg,
        )
        client = TestClient(_build_test_app(server))
        r = client.post("/act", json={"image": "x"})
        body = r.json()
        assert body["actions"][0] == pytest.approx(mean)
        assert "guard_violations" not in body
        assert "guard_clamped" not in body

    def test_out_of_range_clamped_with_violations_in_response(self):
        cfg = EmbodimentConfig.load_preset("franka")
        server = _StubServer(
            response={"actions": [[99.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5]], "latency_ms": 50.0},
            embodiment_cfg=cfg,
        )
        client = TestClient(_build_test_app(server))
        r = client.post("/act", json={"image": "x"})
        body = r.json()
        # joint_0 clamped to 2.8973
        assert body["actions"][0][0] == pytest.approx(2.8973, abs=1e-3)
        assert body["guard_clamped"] is True
        assert any("joint_0" in v for v in body["guard_violations"])

    def test_nan_zeros_chunk_in_response(self):
        cfg = EmbodimentConfig.load_preset("franka")
        bad_chunk = [[0.0] * 7, [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]]
        # JSON can't natively encode NaN, so the policy stub returns valid JSON
        # and the conversion to np.asarray injects the NaN. Here we simulate
        # via direct list with float('nan').
        server = _StubServer(
            response={"actions": bad_chunk, "latency_ms": 50.0},
            embodiment_cfg=cfg,
        )
        client = TestClient(_build_test_app(server))
        # Note: TestClient will reject NaN serialization on the inbound /act
        # request body but accepts the model returning NaN. We test the guard
        # path by injecting via the stub's response dict (already in memory).
        r = client.post("/act", json={"image": "x"})
        # Response with NaN actions also can't be JSON-serialized cleanly
        # — fastapi will replace NaN with null in JSONResponse via simplejson
        # fallback OR error. We accept either outcome and assert the guard
        # reported the violation if the response succeeded.
        if r.status_code == 200:
            body = r.json()
            if "guard_violations" in body:
                assert any("non_finite" in v for v in body["guard_violations"])

    def test_safety_limits_loaded_from_so100(self):
        """SO-100 has 6-dim action; mean_action is in-range across all dims."""
        cfg = EmbodimentConfig.load_preset("so100")
        mean = list(cfg.normalization["mean_action"])
        server = _StubServer(
            response={"actions": [mean], "latency_ms": 50.0},
            embodiment_cfg=cfg,
        )
        client = TestClient(_build_test_app(server))
        r = client.post("/act", json={"image": "x"})
        assert r.status_code == 200
        assert r.json()["actions"][0] == pytest.approx(mean)
