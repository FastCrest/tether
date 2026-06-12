from __future__ import annotations

import numpy as np
import pytest

from tether.runtime.adaptive_chunking import (
    AdaptiveChunkConfig,
    AdaptiveChunkController,
    AdaptiveChunkSignal,
    overlapping_action_delta,
)
from tether.runtime.buffer import ActionChunkBuffer


def test_disabled_controller_returns_base_horizon():
    controller = AdaptiveChunkController(
        AdaptiveChunkConfig(enabled=False, base_horizon=5, max_horizon=10)
    )
    decision = controller.decide(AdaptiveChunkSignal(uncertainty=1.0), capacity=10)
    assert decision.horizon == 5
    assert decision.reason == "disabled"
    assert decision.replan_threshold_ratio == pytest.approx(0.5)


def test_high_uncertainty_shortens_horizon():
    controller = AdaptiveChunkController(
        AdaptiveChunkConfig(enabled=True, min_horizon=2, base_horizon=5, max_horizon=10)
    )
    decision = controller.decide(AdaptiveChunkSignal(uncertainty=0.9), capacity=10)
    assert decision.horizon == 2
    assert decision.reason == "uncertainty"
    assert decision.replan_threshold_ratio == pytest.approx(0.8)


def test_stable_high_latency_extends_horizon():
    controller = AdaptiveChunkController(
        AdaptiveChunkConfig(
            enabled=True,
            min_horizon=2,
            base_horizon=5,
            max_horizon=10,
            high_latency_ms=100,
        )
    )
    decision = controller.decide(
        AdaptiveChunkSignal(uncertainty=0.0, latency_ms=150),
        capacity=10,
    )
    assert decision.horizon == 10
    assert decision.reason == "stable_high_latency"
    assert decision.replan_threshold_ratio == pytest.approx(0.0)


def test_low_guard_margin_shortens_horizon():
    controller = AdaptiveChunkController(
        AdaptiveChunkConfig(enabled=True, min_horizon=1, base_horizon=5, max_horizon=10)
    )
    decision = controller.decide(AdaptiveChunkSignal(guard_margin=0.0), capacity=10)
    assert decision.horizon == 1
    assert decision.reason == "guard_margin"


def test_buffer_accepts_adaptive_decision_threshold():
    buf = ActionChunkBuffer(capacity=10)
    buf.push_chunk(np.ones((10, 1), dtype=np.float32))
    controller = AdaptiveChunkController(
        AdaptiveChunkConfig(enabled=True, min_horizon=2, base_horizon=5, max_horizon=10)
    )
    decision = controller.decide(AdaptiveChunkSignal(uncertainty=0.9), capacity=10)
    assert buf.should_replan(adaptive_decision=decision) is False
    for _ in range(2):
        buf.pop_next()
    assert buf.should_replan(adaptive_decision=decision) is True


def test_overlapping_action_delta_uses_tail_and_head():
    previous = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    current = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    assert overlapping_action_delta(previous, current, window=1) == pytest.approx(0.0)
    assert overlapping_action_delta(previous, current, window=2) == pytest.approx(
        np.linalg.norm(np.array([[-1.0], [-1.0]], dtype=np.float32)) / 2
    )

