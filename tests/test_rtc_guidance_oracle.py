"""Differential oracle: Tether's numpy RTC guidance vs LeRobot 0.6.1.

``tether.runtime.rtc_guidance`` ports ``RTCProcessor.denoise_step`` for the
pi0.5 per-step ONNX loop. This runs both on matched seeds — per step and over
whole Euler loops — across every schedule, delays 0..H and NFE {2, 5, 10}, and
fails on any disagreement above float32 rounding. ``error_distribution`` is
also what the verification doc reports.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.rtc")

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402

from tether.runtime.rtc_guidance import (  # noqa: E402
    SCHEDULES,
    RtcGuidance,
    guide_velocity,
    prefix_weights,
)

T, A, A_PREV = 10, 6, 4  # leftover narrower than the chunk exercises padding
TOL = 1e-5


class _Field:
    """The same nonlinear velocity field in torch and numpy (float32)."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = rng.standard_normal((A, 16)).astype(np.float32) * 0.5
        self.w2 = rng.standard_normal((16, A)).astype(np.float32) * 0.5
        self.b = rng.standard_normal((T, 16)).astype(np.float32) * 0.1

    def np(self, x, t):
        h = np.tanh(x @ self.w1 + self.b + np.float32(t))
        return (h @ self.w2).astype(np.float32)

    def torch(self, x, t):
        w1, w2, b = (torch.from_numpy(a) for a in (self.w1, self.w2, self.b))
        tt = t.reshape(-1, 1, 1) if torch.is_tensor(t) else t
        return torch.tanh(x @ w1 + b + tt) @ w2


def _processor(schedule: str, beta: float = 5.0) -> RTCProcessor:
    return RTCProcessor(
        RTCConfig(
            prefix_attention_schedule=RTCAttentionSchedule(schedule),
            max_guidance_weight=beta,
            execution_horizon=8,
        )
    )


def _np_loop(field, noise, num_steps, guidance):
    dt = -1.0 / num_steps
    x = noise.copy()
    for step in range(num_steps):
        time = 1.0 + step * dt
        v = guide_velocity(x, field.np(x, time), time, guidance)
        x = x + np.float32(dt) * v
    return x


def error_distribution(*, seeds=range(3), steps=(2, 5, 10), max_delay=8) -> dict[str, float]:
    """Per-element |numpy - LeRobot| over complete Euler loops."""
    errs = []
    for seed, schedule, num_steps, delay in itertools.product(
        seeds, SCHEDULES, steps, range(max_delay + 1)
    ):
        rng = np.random.default_rng(1000 + seed)
        field = _Field(seed)
        noise = rng.standard_normal((1, T, A)).astype(np.float32)
        prev = rng.standard_normal((T - 2, A_PREV)).astype(np.float32)
        g = RtcGuidance(delay, 8, prev, schedule, 5.0)
        ours = _np_loop(field, noise, num_steps, g)
        ref = euler_integrate(
            field.torch,
            torch.from_numpy(noise),
            num_steps,
            rtc_processor=_processor(schedule),
            rtc_enabled=True,
            inference_delay=delay,
            prev_chunk_left_over=torch.from_numpy(prev),
            execution_horizon=8,
        ).numpy()
        errs.append(np.abs(ours - ref).ravel())
    e = np.concatenate(errs)
    return {
        "n": int(e.size),
        "mean": float(e.mean()),
        "p50": float(np.percentile(e, 50)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "max": float(e.max()),
        "frac_above_1e-6": float((e > 1e-6).mean()),
    }


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_prefix_weights_match_lerobot(schedule):
    proc = _processor(schedule)
    worst = 0.0
    for total in (1, 4, 10, 50):
        for start, end in itertools.product(range(total + 2), range(total + 2)):
            ref = proc.get_prefix_weights(start, end, total).numpy()
            ours = prefix_weights(start, end, total, schedule)
            assert ours.shape == ref.shape, (start, end, total)
            if ref.size:
                worst = max(worst, float(np.abs(ours - ref).max()))
    assert worst <= 2e-7  # torch float32 linspace vs float64-then-round


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("time", [1.0, 0.8, 0.5, 0.1])
@pytest.mark.parametrize("delay", [0, 1, 3, 8, 12])
def test_single_step_matches_denoise_step(schedule, time, delay):
    rng = np.random.default_rng(delay)
    field = _Field(delay)
    x = rng.standard_normal((1, T, A)).astype(np.float32)
    prev = rng.standard_normal((T - 3, A_PREV)).astype(np.float32)
    ref = (
        _processor(schedule, beta=10.0)
        .denoise_step(
            x_t=torch.from_numpy(x),
            prev_chunk_left_over=torch.from_numpy(prev),
            inference_delay=delay,
            time=time,
            original_denoise_step_partial=lambda xt: field.torch(xt, time),
            execution_horizon=8,
        )
        .numpy()
    )
    ours = guide_velocity(x, field.np(x, time), time, RtcGuidance(delay, 8, prev, schedule, 10.0))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=TOL)


def test_no_leftover_is_identity():
    v = np.ones((1, T, A), np.float32)
    g = RtcGuidance(3, 8, None)
    assert guide_velocity(np.zeros_like(v), v, 0.5, g) is v


def test_full_loops_match_lerobot():
    dist = error_distribution()
    print("rtc numpy-vs-lerobot |err|:", dist)
    assert dist["max"] <= TOL, dist


def test_guidance_changes_the_prefix():
    """Sanity: guidance is not a no-op, and pulls the prefix toward Y."""
    rng = np.random.default_rng(7)
    field = _Field(7)
    noise = rng.standard_normal((1, T, A)).astype(np.float32)
    prev = rng.standard_normal((T, A)).astype(np.float32)
    plain = _np_loop(field, noise, 10, RtcGuidance(4, 8, None))
    guided = _np_loop(field, noise, 10, RtcGuidance(4, 8, prev))
    assert np.abs(guided[0, :4] - prev[:4]).max() < np.abs(plain[0, :4] - prev[:4]).max()
