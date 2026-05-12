"""Byte tracking + Prometheus gauge for EpisodeCache.

Covers the four mutation paths: insert grows the byte total, eviction
shrinks it, reset zeros it, lookup leaves it unchanged. Also asserts the
Prometheus Gauge value matches `bytes_resident()` when labels are wired.
"""
from __future__ import annotations

import numpy as np
import pytest

from reflex.runtime.episode_cache import (
    EpisodeCache,
    _compute_entry_bytes,
)


def _fixture(seq_len: int = 4, hidden: int = 8, n_layers: int = 2):
    """Build (past_kv, prefix_pad_masks, lang_tokens) with predictable nbytes."""
    past_kv: list[np.ndarray] = []
    for _ in range(n_layers):
        past_kv.append(np.zeros((1, seq_len, hidden), dtype=np.float32))  # K
        past_kv.append(np.zeros((1, seq_len, hidden), dtype=np.float32))  # V
    prefix_pad_masks = np.ones((1, seq_len), dtype=np.bool_)
    lang_tokens = np.array([1, 2, 3, 4], dtype=np.int64)
    return past_kv, prefix_pad_masks, lang_tokens


def _expected_bytes(past_kv, prefix_pad_masks):
    return sum(a.nbytes for a in past_kv) + prefix_pad_masks.nbytes


class TestByteTotal:
    def test_insert_increments_total_bytes(self):
        cache = EpisodeCache(max_episodes=4)
        past_kv, masks, lang = _fixture()
        cache.insert("ep0", lang, past_kv, masks)
        assert cache.bytes_resident() == _expected_bytes(past_kv, masks)

    def test_compute_entry_bytes_matches_helper(self):
        past_kv, masks, _ = _fixture()
        assert _compute_entry_bytes(past_kv, masks) == _expected_bytes(past_kv, masks)

    def test_prefix_dataclass_carries_bytes_size(self):
        cache = EpisodeCache(max_episodes=2)
        past_kv, masks, lang = _fixture()
        prefix = cache.insert("ep0", lang, past_kv, masks)
        assert prefix.bytes_size == _expected_bytes(past_kv, masks)

    def test_eviction_decrements_total_bytes(self):
        cache = EpisodeCache(max_episodes=2)
        # Three inserts with distinct episode IDs — first one gets evicted.
        past_kv_a, masks_a, lang_a = _fixture()
        past_kv_b, masks_b, lang_b = _fixture()
        past_kv_c, masks_c, lang_c = _fixture()
        cache.insert("ep0", lang_a, past_kv_a, masks_a)
        cache.insert("ep1", lang_b, past_kv_b, masks_b)
        cache.insert("ep2", lang_c, past_kv_c, masks_c)
        # Two entries retained; first one evicted.
        assert len(cache) == 2
        assert cache.stats.evictions == 1
        expected = _expected_bytes(past_kv_b, masks_b) + _expected_bytes(past_kv_c, masks_c)
        assert cache.bytes_resident() == expected

    def test_reset_zeros_total_bytes(self):
        cache = EpisodeCache(max_episodes=4)
        past_kv, masks, lang = _fixture()
        cache.insert("ep0", lang, past_kv, masks)
        assert cache.bytes_resident() > 0
        cache.reset()
        assert cache.bytes_resident() == 0
        assert len(cache) == 0

    def test_lookup_does_not_change_bytes(self):
        cache = EpisodeCache(max_episodes=4)
        past_kv, masks, lang = _fixture()
        cache.insert("ep0", lang, past_kv, masks)
        before = cache.bytes_resident()
        cache.lookup("ep0", lang)
        cache.lookup("ep0", lang)
        assert cache.bytes_resident() == before


class TestPrometheusEmission:
    """Gauge value tracks bytes_resident when labels are wired in."""

    def _read_gauge(self, embodiment: str, model_id: str, policy_slot: str = "prod") -> float:
        from reflex.observability.prometheus import reflex_episode_cache_bytes_total
        return reflex_episode_cache_bytes_total.labels(
            embodiment=embodiment, model_id=model_id, policy_slot=policy_slot,
        )._value.get()

    def test_gauge_set_on_insert(self):
        cache = EpisodeCache(
            max_episodes=4,
            embodiment="franka",
            model_id="pi05_base_test_insert",
        )
        past_kv, masks, lang = _fixture()
        cache.insert("ep0", lang, past_kv, masks)
        assert self._read_gauge("franka", "pi05_base_test_insert") == cache.bytes_resident()

    def test_gauge_decremented_on_eviction(self):
        cache = EpisodeCache(
            max_episodes=1,
            embodiment="franka",
            model_id="pi05_base_test_evict",
        )
        past_kv_a, masks_a, lang_a = _fixture()
        past_kv_b, masks_b, lang_b = _fixture()
        cache.insert("ep0", lang_a, past_kv_a, masks_a)
        cache.insert("ep1", lang_b, past_kv_b, masks_b)
        # Only ep1 retained; gauge should reflect just its bytes.
        assert self._read_gauge("franka", "pi05_base_test_evict") == _expected_bytes(
            past_kv_b, masks_b
        )

    def test_gauge_zeroed_on_reset(self):
        cache = EpisodeCache(
            max_episodes=4,
            embodiment="franka",
            model_id="pi05_base_test_reset",
        )
        past_kv, masks, lang = _fixture()
        cache.insert("ep0", lang, past_kv, masks)
        cache.reset()
        assert self._read_gauge("franka", "pi05_base_test_reset") == 0.0

    def test_no_emission_when_labels_missing(self):
        # Cache built without embodiment/model_id should not raise and should
        # not push to Prometheus. Byte tracking still works.
        cache = EpisodeCache(max_episodes=4)
        past_kv, masks, lang = _fixture()
        cache.insert("ep0", lang, past_kv, masks)
        assert cache.bytes_resident() == _expected_bytes(past_kv, masks)
