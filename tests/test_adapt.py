"""Tests for cross-embodiment transfer."""

import json
import numpy as np
import pytest
from pathlib import Path

from tether.models.adapt import EmbodimentAdapter, EmbodimentConfig, ActionMapping


class TestEmbodimentConfig:
    def test_default_creates(self):
        adapter = EmbodimentAdapter.default("test_robot", 7)
        assert adapter.config.num_joints == 7
        assert adapter.config.action_dim == 7
        assert len(adapter.config.joint_names) == 7

    def test_save_and_load(self, tmp_path):
        adapter = EmbodimentAdapter.default("test", 4)
        path = tmp_path / "config.json"
        adapter.config.save(path)
        loaded = EmbodimentConfig.from_json(path)
        assert loaded.num_joints == 4
        assert loaded.name == "test"


class TestActionMapping:
    def test_direct_mapping(self):
        adapter = EmbodimentAdapter.default("robot", 6)
        mapping = adapter.create_mapping(source_dim=6)
        assert mapping.mapping_type == "direct"
        actions = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        result = mapping.apply(actions)
        np.testing.assert_array_equal(result, actions)

    def test_pad_mapping(self):
        adapter = EmbodimentAdapter.default("robot", 8)
        mapping = adapter.create_mapping(source_dim=6)
        assert mapping.mapping_type == "pad"
        actions = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        result = mapping.apply(actions)
        assert result.shape == (1, 8)
        assert result[0, 6] == 0.0
        assert result[0, 7] == 0.0

    def test_truncate_mapping(self):
        adapter = EmbodimentAdapter.default("robot", 4)
        mapping = adapter.create_mapping(source_dim=7)
        assert mapping.mapping_type == "truncate"
        actions = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
        result = mapping.apply(actions)
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 4])

    def test_batch_mapping(self):
        adapter = EmbodimentAdapter.default("robot", 6)
        mapping = adapter.create_mapping(source_dim=6)
        actions = np.random.randn(10, 6).astype(np.float32)
        result = mapping.apply(actions)
        assert result.shape == (10, 6)


class TestFrameworkConfig:
    def test_lerobot_config(self):
        adapter = EmbodimentAdapter.default("my_arm", 7)
        config = adapter.generate_framework_config("lerobot")
        assert config["action_dim"] == 7
        assert config["max_action_dim"] == 32
        assert config["chunk_size"] == 50

    def test_openpi_config(self):
        adapter = EmbodimentAdapter.default("my_arm", 7)
        config = adapter.generate_framework_config("openpi")
        assert config["action_dim"] == 7
        assert "normalization" in config

    def test_gr00t_config(self):
        adapter = EmbodimentAdapter.default("my_arm", 7)
        config = adapter.generate_framework_config("gr00t")
        assert config["embodiment_tag"] == "my_arm"
