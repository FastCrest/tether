from __future__ import annotations

import numpy as np
import pytest

from tether.runtime.temporal_vla_cache import TemporalVLAReusePolicy


def _hash(byte: int) -> tuple[bytes, ...]:
    return (bytes([byte] * 8),)


def test_stable_inputs_reuse():
    policy = TemporalVLAReusePolicy(phash_hamming_threshold=2)
    decision = policy.assess(
        cached_image_phashes=_hash(0),
        current_image_phashes=_hash(0),
        cached_lang_hash=b"task",
        current_lang_hash=b"task",
        cached_state=np.array([0.0, 0.0], dtype=np.float32),
        current_state=np.array([0.01, 0.0], dtype=np.float32),
        cached_step_index=1,
        current_step_index=2,
        max_age_steps=2,
    )
    assert decision.reuse is True
    assert decision.reason == "stable"
    assert decision.steps_since == 1
    assert decision.state_delta == pytest.approx(0.01)


def test_rejects_stale_entry():
    policy = TemporalVLAReusePolicy()
    decision = policy.assess(
        cached_image_phashes=_hash(0),
        current_image_phashes=_hash(0),
        cached_lang_hash=b"task",
        current_lang_hash=b"task",
        cached_step_index=1,
        current_step_index=5,
        max_age_steps=2,
    )
    assert decision.reuse is False
    assert decision.reason == "stale"


def test_rejects_image_change():
    policy = TemporalVLAReusePolicy(phash_hamming_threshold=1)
    decision = policy.assess(
        cached_image_phashes=_hash(0),
        current_image_phashes=_hash(255),
        cached_lang_hash=b"task",
        current_lang_hash=b"task",
    )
    assert decision.reuse is False
    assert decision.reason == "image_changed"
    assert decision.image_hamming == 64


def test_rejects_state_change():
    policy = TemporalVLAReusePolicy(state_delta_threshold=0.05)
    decision = policy.assess(
        cached_image_phashes=_hash(0),
        current_image_phashes=_hash(0),
        cached_lang_hash=b"task",
        current_lang_hash=b"task",
        cached_state=np.array([0.0, 0.0], dtype=np.float32),
        current_state=np.array([0.2, 0.0], dtype=np.float32),
    )
    assert decision.reuse is False
    assert decision.reason == "state_changed"


def test_can_ignore_language_mismatch_for_state_in_language_modes():
    policy = TemporalVLAReusePolicy()
    decision = policy.assess(
        cached_image_phashes=_hash(0),
        current_image_phashes=_hash(0),
        cached_lang_hash=b"old-state-text",
        current_lang_hash=b"new-state-text",
        allow_lang_mismatch=True,
    )
    assert decision.reuse is True


def test_state_signature_flattens_and_copies():
    state = np.array([[1.0, 2.0]], dtype=np.float32)
    sig = TemporalVLAReusePolicy.state_signature(state)
    assert sig is not None
    state[0, 0] = 99.0
    np.testing.assert_array_equal(sig, np.array([1.0, 2.0], dtype=np.float32))
