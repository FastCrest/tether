"""Integration tests for `run_verify`'s distributional + embodied checks.

Uses a synthetic ``gather_fn`` returning rollout-shaped result dicts WITH the
per-step trajectories the widened tap captures (``action_chunks`` +
``eef_positions``). No GPU, no real rollout — validates that run_verify wires
MMD + embodied parity into the verdict as non-bypassable checks.
"""
from __future__ import annotations

import numpy as np

from reflex.verify import run_verify


def _make_results(n_eps, *, chunk_mean, eef_jitter, rng, n_chunks=6, steps=100):
    episodes = []
    for e in range(n_eps):
        chunks = [
            rng.normal(chunk_mean, 0.1, size=(5, 7)).tolist() for _ in range(n_chunks)
        ]
        eef = np.cumsum(np.full((40, 3), 0.01), axis=0)  # smooth linear motion
        if eef_jitter:
            eef = eef + rng.normal(0.0, eef_jitter, size=(40, 3))
        episodes.append({
            "ep": e, "success": True, "steps": steps,
            "action_chunks": chunks, "eef_positions": eef.tolist(),
        })
    return {"per_task": [{"task_idx": 0, "task_description": "t", "episodes": episodes}]}


def _gather_returning(orig, opt):
    def gather(**_kwargs):
        return orig, opt
    return gather


def test_run_verify_passes_when_distributions_and_motion_match():
    rng = np.random.default_rng(0)
    orig = _make_results(32, chunk_mean=0.0, eef_jitter=0.0, rng=rng)
    opt = _make_results(32, chunk_mean=0.0, eef_jitter=0.0, rng=rng)
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)
    assert v.two_sample is not None and v.two_sample.distributions_differ is False
    assert v.embodied is not None and v.embodied.regressed() is False
    assert v.passed is True


def test_run_verify_fails_on_action_distribution_shift():
    rng = np.random.default_rng(0)
    orig = _make_results(32, chunk_mean=0.0, eef_jitter=0.0, rng=rng)
    opt = _make_results(32, chunk_mean=2.0, eef_jitter=0.0, rng=rng)  # shifted chunks
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)
    assert v.two_sample.distributions_differ is True
    assert v.passed is False  # non-bypassable, even though success parity matched


def test_run_verify_fails_on_embodied_regression():
    rng = np.random.default_rng(1)
    orig = _make_results(32, chunk_mean=0.0, eef_jitter=0.0, rng=rng)
    opt = _make_results(32, chunk_mean=0.0, eef_jitter=0.5, rng=rng)  # jittery motion
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)
    assert v.embodied.regressed() is True
    assert v.passed is False


def test_run_verify_without_trajectories_is_backward_compatible():
    # Older results / tap off: no action_chunks / eef_positions => checks no-op.
    def bare(n):
        return {"per_task": [{"task_idx": 0, "episodes": [
            {"ep": i, "success": True, "steps": 100} for i in range(n)
        ]}]}
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(bare(32), bare(32)), num_episodes=32)
    assert v.two_sample is None and v.embodied is None
    assert v.passed is True  # falls back to success-rate parity only
