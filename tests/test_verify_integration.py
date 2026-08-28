"""Integration tests for `run_verify`'s distributional + embodied checks.

Uses a synthetic ``gather_fn`` returning rollout-shaped result dicts WITH the
per-step trajectories the widened tap captures (``actions`` = executed per-step
actions + ``eef_positions``). No GPU, no real rollout — validates that
run_verify wires MMD + embodied parity into the verdict as non-bypassable
checks.
"""
from __future__ import annotations

import copy
import itertools
import numpy as np

from tether.verify import run_verify


_MEMORY_PID = itertools.count(20_000)


def _memory_evidence(value=100_000_000):
    pid = next(_MEMORY_PID)
    samples = [
        {
            "scheduled_monotonic_s": timestamp,
            "captured_monotonic_s": timestamp,
            "process_rss_bytes": value,
            "device_allocated_bytes": 0,
        }
        for timestamp in (1.0, 1.1, 1.2)
    ]
    summary = {
        "sample_hz": 10.0,
        "backend": "cpu",
        "process_identity": {"pid": pid, "capture_id": f"capture-{pid}"},
        "window": {
            "started_monotonic_s": 1.0,
            "ended_monotonic_s": 1.2,
            "duration_s": 0.2,
            "expected_samples": 3,
            "captured_samples": 3,
            "max_gap_s": 0.1,
        },
        "samples": samples,
        "process_rss": {"peak_bytes": value, "p95_bytes": value},
        "device_allocated": {"peak_bytes": 0, "p95_bytes": 0},
        "combined": {"peak_bytes": value, "p95_bytes": value},
    }
    return {"status": "available", "value": summary}


def _complete_results(episodes):
    return {
        "per_task": [{"task_idx": 0, "task_description": "t", "episodes": episodes}],
        "verification_device": "cpu",
        "safety_evidence": {"status": "available", "value": {"config": "test"}},
        "memory_evidence": _memory_evidence(),
    }


def _make_results(n_eps, *, action_mean, eef_jitter, rng, n_actions=30, steps=100):
    episodes = []
    for e in range(n_eps):
        # Per-step executed actions: one 7-dim row per control step.
        actions = rng.normal(action_mean, 0.1, size=(n_actions, 7)).tolist()
        eef = np.cumsum(np.full((40, 3), 0.01), axis=0)  # smooth linear motion
        if eef_jitter:
            eef = eef + rng.normal(0.0, eef_jitter, size=(40, 3))
        episodes.append({
            "ep": e, "success": True, "steps": steps,
            "actions": actions, "eef_positions": eef.tolist(),
            "action_timestamps_s": list(np.arange(n_actions, dtype=float) + 1.0),
            "inference_latency_ms": [10.0, 11.0],
            "safety_clamp_count": 0,
        })
    return _complete_results(episodes)


def _gather_returning(orig, opt):
    original_identity = orig.get("memory_evidence", {}).get("value", {}).get(
        "process_identity"
    )
    optimized_identity = opt.get("memory_evidence", {}).get("value", {}).get(
        "process_identity"
    )
    if original_identity and original_identity == optimized_identity:
        opt = copy.deepcopy(opt)
        identity = opt["memory_evidence"]["value"]["process_identity"]
        identity["pid"] += 1
        identity["capture_id"] += "-candidate"

    def gather(**_kwargs):
        return orig, opt
    return gather


def test_run_verify_passes_when_distributions_and_motion_match():
    rng = np.random.default_rng(0)
    orig = _make_results(32, action_mean=0.0, eef_jitter=0.0, rng=rng)
    opt = copy.deepcopy(orig)
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)
    assert v.two_sample is not None and v.two_sample.distributions_differ is False
    assert v.embodied is not None and v.embodied.regressed() is False
    assert v.passed is True


def test_run_verify_fails_on_action_distribution_shift():
    rng = np.random.default_rng(0)
    orig = _make_results(32, action_mean=0.0, eef_jitter=0.0, rng=rng)
    opt = _make_results(32, action_mean=2.0, eef_jitter=0.0, rng=rng)  # shifted actions
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)
    assert v.two_sample.distributions_differ is True
    assert v.passed is False  # non-bypassable, even though success parity matched


def test_run_verify_fails_on_embodied_regression():
    rng = np.random.default_rng(1)
    orig = _make_results(32, action_mean=0.0, eef_jitter=0.0, rng=rng)
    opt = _make_results(32, action_mean=0.0, eef_jitter=0.5, rng=rng)  # jittery motion
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)
    assert v.embodied.regressed() is True
    assert v.passed is False


def test_run_verify_conditions_distributional_test_on_commonly_succeeded():
    # Outcome confound: if one arm fails more episodes, those failures inject
    # flailing actions that make the POOLED distributions differ for a reason
    # unrelated to per-step policy fidelity. The gate must condition on episodes
    # BOTH arms succeeded, so the distributional test sees only the policy shift.
    rng = np.random.default_rng(0)

    def arm(fail_from):
        # eps [0, fail_from) succeed (mean 0); [fail_from, 32) fail (mean 5 flail)
        eps = []
        for e in range(32):
            ok = e < fail_from
            acts = rng.normal(0.0 if ok else 5.0, 0.1, size=(30, 7)).tolist()
            eps.append({
                "ep": e,
                "success": ok,
                "steps": 100 if ok else 200,
                "actions": acts,
                "action_timestamps_s": list(np.arange(30, dtype=float) + 1.0),
                "inference_latency_ms": [10.0],
                "safety_clamp_count": 0,
            })
        return _complete_results(eps)

    orig = arm(16)  # 16 succeed, 16 flailing failures
    opt = arm(30)   # 30 succeed, 2 flailing failures (candidate is *better*)
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(orig, opt), num_episodes=32)

    # Conditioned on the 16 commonly-succeeded episodes (both mean 0): no shift.
    assert v.two_sample is not None
    assert v.two_sample_episodes == 16
    assert v.two_sample.distributions_differ is False

    # Contrast: pooling ALL actions (the old, confounded way) WOULD flag a diff,
    # purely because orig carries more mean-5 flailing actions than opt.
    from tether.verify import _collect_step_actions
    from tether.verify_metrics import two_sample_test
    ba, bg = _collect_step_actions(orig)
    ca, cg = _collect_step_actions(opt)
    unconditioned = two_sample_test(ba, ca, baseline_groups=bg, candidate_groups=cg)
    assert unconditioned.distributions_differ is True


def test_run_verify_without_trajectories_fails_closed():
    # Older results / tap off: missing required evidence must never green-light.
    def bare(n):
        return {"per_task": [{"task_idx": 0, "episodes": [
            {"ep": i, "success": True, "steps": 100} for i in range(n)
        ]}], "verification_device": "cpu"}
    v = run_verify(optimized_ref="exp", gather_fn=_gather_returning(bare(32), bare(32)), num_episodes=32)
    assert v.two_sample is None and v.embodied is None
    assert v.passed is False
    assert v.first_failing_gate_id == "S1"
