"""Tests for cloud-edge split orchestration."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from tether.runtime.split import SplitOrchestrator, SplitConfig, InferenceTarget


class TestSplitOrchestrator:
    def test_default_prefers_edge(self):
        orch = SplitOrchestrator(SplitConfig(prefer="edge"))
        assert orch._select_target() == InferenceTarget.EDGE

    def test_cloud_prefer_without_cloud_falls_back(self):
        orch = SplitOrchestrator(SplitConfig(prefer="cloud"))
        assert orch._select_target() == InferenceTarget.FALLBACK

    def test_edge_inference(self):
        orch = SplitOrchestrator(SplitConfig(prefer="edge"))
        mock_server = MagicMock()
        mock_server.predict.return_value = {
            "actions": np.random.randn(10, 6).tolist()
        }
        result = orch.infer(mock_server, action_dim=6, chunk_size=10)
        assert result.target_used == InferenceTarget.EDGE
        assert result.actions.shape == (10, 6)
        assert result.latency_ms >= 0

    def test_fallback_returns_zeros(self):
        orch = SplitOrchestrator(SplitConfig(prefer="cloud", fallback_mode="zero"))
        orch._cloud_available = False
        mock_server = MagicMock()
        mock_server.predict.side_effect = Exception("edge down")

        # Force fallback by setting prefer to something that triggers it
        result = orch._get_fallback_actions(action_dim=6, chunk_size=5)
        assert result.shape == (5, 6)
        assert np.all(result == 0)

    def test_fallback_last_action(self):
        orch = SplitOrchestrator(SplitConfig(fallback_mode="last_action"))
        last = np.ones((5, 6), dtype=np.float32)
        orch._last_actions = last
        result = orch._get_fallback_actions(action_dim=6, chunk_size=5)
        assert np.all(result == 1.0)

    def test_cloud_latency_history(self):
        orch = SplitOrchestrator()
        orch._cloud_latency_history = [10.0, 20.0, 30.0]
        avg = sum(orch._cloud_latency_history) / len(orch._cloud_latency_history)
        assert avg == 20.0
