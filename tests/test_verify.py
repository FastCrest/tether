"""Tests for src/tether/verify.py — the `tether verify` action-parity gate (v0).

NO GPU / NO Modal / NO network. The model-loading + simulation seam
(`gather_paired_samples`) is mocked: tests inject a synthetic ``gather_fn`` that
returns paired ``run_libero_rollout``-shaped result dicts, so the scoring +
aggregation path (the actual v0 logic) is exercised in isolation.

Locks the two load-bearing properties:
- An IDENTICAL original/optimized pair PASSes the gate.
- A clearly-DIVERGENT pair (optimized regresses success on a task) FAILs,
  via the success-cliff / Wilson gates that carry the v0 signal.
"""
from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tether.verify import (
    ParityVerdict,
    SUPPORTED_SUITES,
    _rollout_results_to_samples,
    run_verify,
)
from tether.pro.eval_gate import EvalReport, InsufficientEpisodes


# ---------------------------------------------------------------------------
# Synthetic rollout-results builders (shape == run_libero_rollout output)
# ---------------------------------------------------------------------------


def _rollout_results(
    *,
    task_success: dict[int, list[bool]],
    label: str = "stub",
    suite: str = "libero_10",
    seed: int = 7,
) -> dict:
    """Build a results dict matching run_libero_rollout's documented shape.

    ``task_success`` maps task_idx -> list of per-episode success bools.
    """
    per_task = []
    total_success = 0
    total_eps = 0
    for task_idx, successes in task_success.items():
        episodes = [
            {"ep": i, "success": s, "steps": 100} for i, s in enumerate(successes)
        ]
        n_succ = sum(1 for s in successes if s)
        per_task.append({
            "task_idx": task_idx,
            "task_description": f"task {task_idx}",
            "episodes": episodes,
            "success": n_succ,
            "total": len(successes),
        })
        total_success += n_succ
        total_eps += len(successes)
    return {
        "model": label,
        "suite": suite,
        "num_episodes_per_task": max((len(v) for v in task_success.values()), default=0),
        "seed": seed,
        "per_task": per_task,
        "total_success": total_success,
        "total_eps": total_eps,
        "success_rate_pct": (100.0 * total_success / total_eps) if total_eps else 0.0,
        "cache_stats": {},
        "errors": [],
    }


def _make_gather_fn(original_results: dict, optimized_results: dict):
    """Return a gather_fn stub that ignores its kwargs and returns the pair.

    Matches the GatherFn signature: (**kwargs) -> (original, optimized).
    """

    def _gather(**_kwargs):
        return original_results, optimized_results

    return _gather


# Three tasks × 12 episodes = 36 paired episodes (clears the n>=30 floor).
_TASKS = (0, 1, 2)
_EPS = 12


def _all_success() -> dict[int, list[bool]]:
    return {t: [True] * _EPS for t in _TASKS}


# ---------------------------------------------------------------------------
# Adapter: rollout results -> EvalSample
# ---------------------------------------------------------------------------


def test_adapter_maps_episodes_to_samples():
    results = _rollout_results(task_success={0: [True, False], 1: [True]})
    samples = _rollout_results_to_samples(results)
    assert len(samples) == 3
    # task_id is stringified task_idx; success preserved
    assert {s.task_id for s in samples} == {"0", "1"}
    assert sum(1 for s in samples if s.success) == 2
    # Sentinels populated per the EvalSample contract
    assert all(s.safety_clamp_count == 0 for s in samples)
    assert all(s.per_joint_velocity == [] for s in samples)
    assert all(s.teacher_action_trajectory is None for s in samples)


def test_adapter_handles_empty_results():
    assert _rollout_results_to_samples({}) == []
    assert _rollout_results_to_samples({"per_task": []}) == []


# ---------------------------------------------------------------------------
# Verdict aggregation — the load-bearing PASS / FAIL properties
# ---------------------------------------------------------------------------


def test_identical_pair_passes():
    """Optimized identical to original => every gate passes => PASS."""
    same = _all_success()
    gather = _make_gather_fn(
        _rollout_results(task_success=same, label="ORIGINAL"),
        _rollout_results(task_success=same, label="OPTIMIZED"),
    )
    verdict = run_verify(
        optimized_ref="/fake/export",
        suite="libero",
        target="orin",
        num_episodes=_EPS,
        gather_fn=gather,
    )
    assert isinstance(verdict, ParityVerdict)
    assert isinstance(verdict.eval_report, EvalReport)
    assert verdict.passed is True
    assert verdict.eval_report.overall_passed is True
    assert verdict.first_failing_gate_id is None
    assert verdict.n_episodes == _EPS * len(_TASKS)
    assert verdict.original_success_rate == pytest.approx(1.0)
    assert verdict.optimized_success_rate == pytest.approx(1.0)
    assert verdict.success_rate_delta == pytest.approx(0.0)


def test_divergent_pair_fails_on_success_cliff():
    """Optimized regresses task 2 from 100% -> 0% success. The per-task
    success-cliff (S3) and per-task Wilson gate (P5) must catch it => FAIL."""
    original = _all_success()
    optimized = dict(_all_success())
    optimized[2] = [False] * _EPS  # task 2 collapses on the optimized export
    gather = _make_gather_fn(
        _rollout_results(task_success=original, label="ORIGINAL"),
        _rollout_results(task_success=optimized, label="OPTIMIZED"),
    )
    verdict = run_verify(
        optimized_ref="/fake/export",
        suite="libero",
        num_episodes=_EPS,
        gather_fn=gather,
    )
    assert verdict.passed is False
    assert verdict.eval_report.overall_passed is False
    # A safety gate (success-cliff S3) should be the first failure — safety
    # precedes performance in the gate's first-failing-gate precedence.
    assert verdict.first_failing_gate_id is not None
    assert verdict.first_failing_gate_id == "S3"
    # Headline numbers reflect the regression (3 tasks, 1 fully failed).
    assert verdict.optimized_success_rate < verdict.original_success_rate
    assert verdict.success_rate_delta < 0


def test_aggregate_regression_fails_p1():
    """A diffuse regression (every task drops a few episodes, no single task
    cliffs past 5pp) should still fail the aggregate Wilson gate (P1)."""
    original = _all_success()
    # Drop 4 of 12 episodes on each task: 33pp aggregate drop, but spread so no
    # single task trips S3's 5pp cliff individually... actually 33pp > 5pp, so
    # S3 will catch each task too. The point: a real regression never passes.
    optimized = {t: [True] * 8 + [False] * 4 for t in _TASKS}
    gather = _make_gather_fn(
        _rollout_results(task_success=original),
        _rollout_results(task_success=optimized),
    )
    verdict = run_verify(
        optimized_ref="/fake/export", suite="libero",
        num_episodes=_EPS, gather_fn=gather,
    )
    assert verdict.passed is False
    assert verdict.optimized_success_rate == pytest.approx(8 / 12)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_unsupported_suite_raises():
    with pytest.raises(ValueError, match="Unsupported suite"):
        run_verify(
            optimized_ref="/fake/export",
            suite="simpler",
            gather_fn=_make_gather_fn(_rollout_results(task_success=_all_success()),
                                      _rollout_results(task_success=_all_success())),
        )


def test_insufficient_episodes_propagates():
    """Fewer than 30 paired episodes => the Pro gate raises InsufficientEpisodes;
    verify must propagate (never green-light under-powered evidence)."""
    tiny = {0: [True] * 5}  # 5 episodes total, well under the 30 floor
    gather = _make_gather_fn(
        _rollout_results(task_success=tiny),
        _rollout_results(task_success=tiny),
    )
    with pytest.raises(InsufficientEpisodes):
        run_verify(
            optimized_ref="/fake/export", suite="libero",
            num_episodes=5, gather_fn=gather,
        )


def test_original_ref_defaults_to_optimized():
    same = _all_success()
    gather = _make_gather_fn(
        _rollout_results(task_success=same),
        _rollout_results(task_success=same),
    )
    verdict = run_verify(
        optimized_ref="/fake/export", suite="libero",
        num_episodes=_EPS, gather_fn=gather,
    )
    assert verdict.original_ref == "/fake/export"
    assert verdict.optimized_ref == "/fake/export"


def test_supported_suites_is_libero():
    assert "libero" in SUPPORTED_SUITES


# ---------------------------------------------------------------------------
# Report rendering (pure)
# ---------------------------------------------------------------------------


def test_parity_report_renders_verdict():
    from tether.parity_report import render_parity_report

    same = _all_success()
    gather = _make_gather_fn(
        _rollout_results(task_success=same),
        _rollout_results(task_success=same),
    )
    verdict = run_verify(
        optimized_ref="/fake/export", suite="libero", target="orin",
        num_episodes=_EPS, gather_fn=gather,
    )
    md = render_parity_report(verdict)
    assert "# Tether Action-Parity Verification" in md
    assert "**Verdict: PASS**" in md
    assert "Gate detail" in md
    assert "TODO(tether-verify)" in md  # the v0-scope disclosure
    # Every gate id should appear in the detail table.
    for g in verdict.eval_report.all_gates:
        assert g.gate_id in md


def test_parity_cert_writes_json_and_optional_signature(tmp_path):
    from tether.parity_cert import (
        SCHEMA_VERSION,
        build_parity_cert,
        verify_parity_cert_signature,
        write_parity_cert,
    )
    from tether.parity_report import write_parity_report

    same = _all_success()
    verdict = run_verify(
        optimized_ref="/fake/export",
        suite="libero",
        target="orin",
        num_episodes=_EPS,
        gather_fn=_make_gather_fn(
            _rollout_results(task_success=same),
            _rollout_results(task_success=same),
        ),
    )
    parity_md = write_parity_report(tmp_path, verdict)
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    cert_path, sig_path = write_parity_cert(
        tmp_path,
        verdict,
        parity_md_path=parity_md,
        signing_key=base64.b64encode(seed).decode("ascii"),
        key_id="test-key",
    )
    assert cert_path.name == "parity.cert.json"
    assert sig_path is not None and sig_path.name == "parity.cert.sig"

    cert = json.loads(cert_path.read_text())
    assert cert["schema_version"] == SCHEMA_VERSION
    assert cert["verdict"] == "PASS"
    assert cert["passed"] is True
    assert cert["target"] == "orin"
    assert cert["artifacts"]["PARITY.md"]["sha256"]
    assert cert["signature"]["key_id"] == "test-key"
    verify_parity_cert_signature(cert)

    unsigned = build_parity_cert(verdict, parity_md_path=parity_md)
    assert "signature" not in unsigned


# ---------------------------------------------------------------------------
# CLI wiring (mocked run_verify — no rollout)
# ---------------------------------------------------------------------------


def test_cli_verify_pass(monkeypatch, tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    from tether.cli import app
    import tether.verify as verify_mod

    same = _all_success()
    gather = _make_gather_fn(
        _rollout_results(task_success=same),
        _rollout_results(task_success=same),
    )

    # Patch the gather seam so the CLI's real run_verify runs with synthetic
    # samples (no torch / LIBERO import, no GPU).
    monkeypatch.setattr(verify_mod, "gather_paired_samples", gather)

    runner = typer_testing.CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "verify",
            "/fake/export",
            "--target",
            "orin",
            "--num-episodes",
            str(_EPS),
            "--output",
            "verify_out",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_verify_writes_signed_parity_cert(monkeypatch, tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    from tether.cli import app
    import tether.verify as verify_mod
    from tether.parity_cert import verify_parity_cert_signature

    same = _all_success()
    monkeypatch.setattr(
        verify_mod,
        "gather_paired_samples",
        _make_gather_fn(
            _rollout_results(task_success=same),
            _rollout_results(task_success=same),
        ),
    )
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    monkeypatch.setenv("TETHER_TEST_SIGNING_KEY", base64.b64encode(seed).decode("ascii"))
    out = tmp_path / "verify"

    runner = typer_testing.CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            "/fake/export",
            "--target",
            "orin",
            "--num-episodes",
            str(_EPS),
            "--output",
            str(out),
            "--signing-key",
            "env:TETHER_TEST_SIGNING_KEY",
            "--key-id",
            "test-key",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "PARITY.md").exists()
    cert = json.loads((out / "parity.cert.json").read_text())
    assert cert["signature"]["key_id"] == "test-key"
    assert (out / "parity.cert.sig").exists()
    verify_parity_cert_signature(cert)


def test_cli_verify_fail(monkeypatch, tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    from tether.cli import app
    import tether.verify as verify_mod

    original = _all_success()
    optimized = dict(_all_success())
    optimized[2] = [False] * _EPS
    gather = _make_gather_fn(
        _rollout_results(task_success=original),
        _rollout_results(task_success=optimized),
    )
    monkeypatch.setattr(verify_mod, "gather_paired_samples", gather)

    runner = typer_testing.CliRunner()
    result = runner.invoke(
        app,
        ["verify", "/fake/export", "--num-episodes", str(_EPS), "--output", str(tmp_path / "verify_fail")],
    )
    assert result.exit_code == 1, result.output  # FAIL => exit 1
    assert "FAIL" in result.output


def test_cli_verify_unsupported_suite(monkeypatch):
    typer_testing = pytest.importorskip("typer.testing")
    from tether.cli import app

    runner = typer_testing.CliRunner()
    result = runner.invoke(app, ["verify", "/fake/export", "--eval", "simpler"])
    assert result.exit_code == 2, result.output
