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
