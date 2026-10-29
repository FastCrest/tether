"""`tether verify` with a pre-declared paired non-inferiority margin.

Reuses the synthetic rollout builders from test_verify.py (no GPU, no LIBERO).
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tether.noninferiority import NotCertifiable
from tether.parity_cert import build_parity_cert, verify_parity_cert_signature, write_parity_cert
from tether.parity_report import render_parity_report
from tether.verify import run_verify

from tests.test_verify import _EPS, _TASKS, _all_success, _make_gather_fn, _rollout_results


def _verify(original, optimized, **kw):
    return run_verify(
        optimized_ref="/fake/export",
        suite="libero",
        target="rtx4090",
        num_episodes=_EPS,
        gather_fn=_make_gather_fn(
            _rollout_results(task_success=original), _rollout_results(task_success=optimized)
        ),
        **kw,
    )


def _with_failures(per_task: int) -> dict[int, list[bool]]:
    return {t: [i >= per_task for i in range(_EPS)] for t in _TASKS}


def test_no_margin_keeps_existing_behaviour_and_records_not_evaluated():
    v = _verify(_all_success(), _all_success())
    assert v.passed
    assert v.non_inferiority is None
    assert v.to_dict()["non_inferiority"] == {
        "status": "NOT_EVALUATED",
        "reason": "no_margin_declared",
    }


@pytest.mark.parametrize("interval", ["wald_central95", "tango_one_sided95"])
def test_identical_arms_pass_at_declared_margin(interval):
    v = _verify(
        _all_success(),
        _all_success(),
        non_inferiority_margin=-0.10,
        non_inferiority_interval=interval,
    )
    assert v.passed and v.non_inferiority.passed
    block = v.non_inferiority_block()
    assert block["status"] == "EVALUATED" and block["margin_declared"] == -0.10
    assert block["certificate"]["n_pairs"] == len(_TASKS) * _EPS
    assert block["certificate"]["task_collapse_gate"] is True


def test_regression_fails_the_verdict_even_if_other_gates_would_not():
    # 4/12 failures per task in the optimized arm: a 33-point paired drop.
    v = _verify(_all_success(), _with_failures(4), non_inferiority_margin=-0.05)
    assert not v.non_inferiority.passed
    assert "regression detected" in v.non_inferiority.verdict
    assert not v.passed


def test_underpowered_run_fails_as_insufficient_evidence():
    v = _verify(_all_success(), _with_failures(1), non_inferiority_margin=-0.05)
    assert not v.passed
    assert "insufficient evidence" in v.non_inferiority.verdict
    assert any("underpowered" in n for n in v.non_inferiority.notes)


def test_bad_declaration_is_refused_before_any_rollout():
    def gather(**_):
        raise AssertionError("rollout must not run for an invalid declaration")

    with pytest.raises(NotCertifiable):
        run_verify(
            optimized_ref="/fake/export",
            num_episodes=_EPS,
            gather_fn=gather,
            non_inferiority_margin=+0.05,
        )
    with pytest.raises(NotCertifiable):
        run_verify(
            optimized_ref="/fake/export",
            num_episodes=_EPS,
            gather_fn=gather,
            non_inferiority_margin=-0.05,
            non_inferiority_interval="wilson",
        )


def test_unpairable_results_fail_closed_with_reason():
    original = _rollout_results(task_success=_all_success())
    optimized = _rollout_results(task_success=_all_success(), seed=8)
    v = run_verify(
        optimized_ref="/fake/export",
        num_episodes=_EPS,
        gather_fn=_make_gather_fn(original, optimized),
        non_inferiority_margin=-0.05,
    )
    assert not v.passed
    block = v.non_inferiority_block()
    assert block["status"] == "NOT_CERTIFIABLE" and "not paired" in block["reason"]


def test_missing_success_evidence_is_not_certifiable():
    optimized = _rollout_results(task_success=_all_success())
    del optimized["per_task"][0]["episodes"][0]["success"]
    original = _rollout_results(task_success=_all_success())
    v = run_verify(
        optimized_ref="/fake/export",
        num_episodes=_EPS,
        gather_fn=_make_gather_fn(original, optimized),
        non_inferiority_margin=-0.05,
    )
    assert not v.passed and v.non_inferiority_block()["status"] == "NOT_CERTIFIABLE"


def test_certificate_is_inside_the_signed_parity_cert(tmp_path):
    v = _verify(
        _all_success(),
        _all_success(),
        non_inferiority_margin=-0.05,
        non_inferiority_interval="tango_one_sided95",
    )
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    cert_path, _ = write_parity_cert(tmp_path, v, signing_key=base64.b64encode(seed).decode())
    cert = json.loads(cert_path.read_text())
    ni = cert["non_inferiority"]
    assert ni["status"] == "EVALUATED" and ni["certificate"]["margin_declared"] == -0.05
    assert ni["certificate"]["ci_method"].startswith("Tango")
    trusted = [Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()]
    verify_parity_cert_signature(cert, trusted_keys=trusted)

    # Loosening the recorded margin after the fact must break the signature.
    from tether.parity_cert import ParityCertError

    cert["non_inferiority"]["certificate"]["margin_declared"] = -0.50
    with pytest.raises(ParityCertError):
        verify_parity_cert_signature(cert, trusted_keys=trusted)


def test_unsigned_cert_and_report_carry_the_block():
    v = _verify(_all_success(), _with_failures(4), non_inferiority_margin=-0.05)
    assert build_parity_cert(v)["non_inferiority"]["passed"] is False
    md = render_parity_report(v)
    assert "## Paired non-inferiority" in md and "margin declared BEFORE the run" in md


def _cli(monkeypatch, tmp_path, original, optimized, *flags):
    typer_testing = pytest.importorskip("typer.testing")
    import tether.verify as verify_mod
    from tether.cli import app

    monkeypatch.setattr(
        verify_mod,
        "gather_paired_samples",
        _make_gather_fn(
            _rollout_results(task_success=original), _rollout_results(task_success=optimized)
        ),
    )
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    monkeypatch.setenv("TETHER_TEST_NI_KEY", base64.b64encode(seed).decode("ascii"))
    out = tmp_path / "verify"
    result = typer_testing.CliRunner().invoke(
        app,
        [
            "verify",
            "/fake/export",
            "--num-episodes",
            str(_EPS),
            "--output",
            str(out),
            "--signing-key",
            "env:TETHER_TEST_NI_KEY",
            *flags,
        ],
    )
    trusted = [Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()]
    return result, out, trusted


# 36 concordant pairs: Wald collapses to [0, 0]; Tango's one-sided bound is ~-0.070, so only
# a -0.10 margin is decidable there. Both are what the statistics should say.
@pytest.mark.parametrize(
    "interval,margin", [("wald_central95", "-0.05"), ("tango_one_sided95", "-0.1")]
)
def test_cli_flags_round_trip_into_signed_cert(monkeypatch, tmp_path, interval, margin):
    result, out, trusted = _cli(
        monkeypatch,
        tmp_path,
        _all_success(),
        _all_success(),
        "--non-inferiority-margin",
        margin,
        "--non-inferiority-interval",
        interval,
    )
    assert result.exit_code == 0, result.output
    cert = json.loads((out / "parity.cert.json").read_text())
    verify_parity_cert_signature(cert, trusted_keys=trusted)
    ni = cert["non_inferiority"]
    assert ni["status"] == "EVALUATED" and ni["passed"] is True
    assert ni["margin_declared"] == float(margin) and ni["interval"] == interval
    assert ni["certificate"]["margin_declared"] == float(margin)
    assert ("ci_method" in ni["certificate"]) == (interval == "tango_one_sided95")


def test_cli_margin_fails_a_regression(monkeypatch, tmp_path):
    result, out, _ = _cli(
        monkeypatch,
        tmp_path,
        _all_success(),
        _with_failures(4),
        "--non-inferiority-margin",
        "-0.05",
    )
    assert result.exit_code == 1, result.output
    assert json.loads((out / "parity.cert.json").read_text())["non_inferiority"]["passed"] is False


def test_cli_tango_at_tight_margin_is_insufficient_evidence(monkeypatch, tmp_path):
    result, out, _ = _cli(
        monkeypatch,
        tmp_path,
        _all_success(),
        _all_success(),
        "--non-inferiority-margin",
        "-0.05",
        "--non-inferiority-interval",
        "tango_one_sided95",
    )
    assert result.exit_code == 1, result.output
    verdict = json.loads((out / "parity.cert.json").read_text())["non_inferiority"]["certificate"][
        "verdict"
    ]
    assert "insufficient evidence" in verdict


def test_cli_without_margin_records_not_evaluated(monkeypatch, tmp_path):
    result, out, _ = _cli(monkeypatch, tmp_path, _all_success(), _all_success())
    assert result.exit_code == 0, result.output
    cert = json.loads((out / "parity.cert.json").read_text())
    assert cert["non_inferiority"] == {"status": "NOT_EVALUATED", "reason": "no_margin_declared"}


@pytest.mark.parametrize(
    "flags",
    [
        ("--non-inferiority-margin", "0.05"),
        ("--non-inferiority-margin", "-0.05", "--non-inferiority-interval", "wilson"),
    ],
)
def test_cli_bad_declaration_exits_2_before_rollout(monkeypatch, tmp_path, flags):
    typer_testing = pytest.importorskip("typer.testing")
    import tether.verify as verify_mod
    from tether.cli import app

    def gather(**_):
        raise AssertionError("rollout must not run")

    monkeypatch.setattr(verify_mod, "gather_paired_samples", gather)
    result = typer_testing.CliRunner().invoke(
        app, ["verify", "/fake/export", "--output", str(tmp_path / "v"), *flags]
    )
    assert result.exit_code == 2, result.output
    assert "non-inferiority" in result.output
