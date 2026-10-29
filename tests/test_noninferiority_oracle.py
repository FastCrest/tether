"""Differential oracles for :mod:`tether.noninferiority`.

The shipped Tango bounds use the closed-form constrained MLE; the oracle here
maximizes the constrained multinomial likelihood numerically and inverts the
score by bisection, so the two share only the definition of the statistic.
McNemar is checked against an exact binomial test written the "sum every
outcome no more likely than the observed one" way (scipy's method).
"""

from __future__ import annotations

import math
import random

import pytest

from tether.noninferiority import (
    NotCertifiable,
    Outcome,
    _mcnemar_exact,
    _paired_delta_ci,
    _tango_paired_score_bounds,
    certify,
    check_declaration,
)

Z = 1.6448536269514722


def _pairs(n, b, c):
    """n pairs with b teacher-only and c student-only wins; the rest split evenly concordant."""
    both = (n - b - c) // 2
    return (
        [(True, False)] * b
        + [(False, True)] * c
        + [(True, True)] * both
        + [(False, False)] * (n - b - c - both)
    )


def _oracle_score(n, b, c, d):
    lo, hi = max(0.0, -d), (1.0 - d) / 2.0
    if hi <= lo:
        return math.copysign(math.inf, c - b - n * d)

    def loglik(p10):
        p01, p_con = p10 + d, 1.0 - 2.0 * p10 - d
        total = 0.0
        for count, p in ((b, p10), (c, p01), (n - b - c, p_con)):
            if count:
                if p <= 0:
                    return -math.inf
                total += count * math.log(p)
        return total

    a, z = lo, hi
    for _ in range(300):  # ternary search; the constrained log-likelihood is concave
        m1, m2 = a + (z - a) / 3, z - (z - a) / 3
        if loglik(m1) < loglik(m2):
            a = m1
        else:
            z = m2
    p10 = (a + z) / 2
    var = n * (2 * p10 + d - d * d)
    if var <= 0:
        return math.copysign(math.inf, c - b - n * d)
    return (c - b - n * d) / math.sqrt(var)


def _oracle_bounds(n, b, c, z=Z):
    est = (c - b) / n

    def solve(lo, hi, target):  # score is decreasing in d
        for _ in range(200):
            mid = (lo + hi) / 2
            if _oracle_score(n, b, c, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = -1.0 if b == n else solve(-1.0, est, z)
    upper = 1.0 if c == n else solve(est, 1.0, -z)
    return lower, upper


CASES = [
    (20, 0, 0),
    (30, 1, 0),
    (30, 0, 1),
    (50, 3, 1),
    (100, 8, 1),
    (100, 1, 8),
    (500, 16, 0),
    (500, 17, 0),
    (500, 16, 17),
    (36, 12, 12),
    (40, 20, 0),
    (12, 0, 12),
]


@pytest.mark.parametrize("n,b,c", CASES)
def test_tango_matches_independent_mle_oracle(n, b, c):
    got = _tango_paired_score_bounds(_pairs(n, b, c))
    want = _oracle_bounds(n, b, c)
    assert got[0] == pytest.approx(want[0], abs=5e-6)
    assert got[1] == pytest.approx(want[1], abs=5e-6)


def test_tango_oracle_random_sweep_error_distribution():
    rng = random.Random(20261028)
    errors = []
    for _ in range(150):
        n = rng.randint(5, 400)
        b = rng.randint(0, n)
        c = rng.randint(0, n - b)
        got = _tango_paired_score_bounds(_pairs(n, b, c))
        want = _oracle_bounds(n, b, c)
        errors += [abs(got[0] - want[0]), abs(got[1] - want[1])]
    errors.sort()
    assert errors[len(errors) // 2] < 1e-7, f"p50={errors[len(errors) // 2]}"
    assert errors[-1] < 5e-6, f"max={errors[-1]}"


def _binom_two_sided_minlike(k, n):
    pmf = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-7)))


@pytest.mark.parametrize("b,c", [(0, 0), (1, 0), (5, 5), (12, 3), (0, 9), (40, 31), (2, 17)])
def test_mcnemar_exact_matches_binomial_test(b, c):
    expected = 1.0 if b + c == 0 else _binom_two_sided_minlike(b, b + c)
    assert _mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-12)


def test_wald_interval_is_symmetric_and_collapses_without_discordance():
    assert _paired_delta_ci(_pairs(40, 0, 0)) == (0.0, 0.0)
    lo, hi = _paired_delta_ci(_pairs(100, 8, 2))
    assert (lo + hi) / 2 == pytest.approx(-0.06)


@pytest.mark.parametrize("margin", [0.01, float("nan"), float("inf"), True, "-0.05"])
def test_declaration_rejects_non_margins(margin):
    with pytest.raises(NotCertifiable):
        check_declaration(margin)


def test_declaration_rejects_unknown_interval():
    with pytest.raises(NotCertifiable):
        check_declaration(-0.05, "wilson")


def test_rerun_is_deterministic():
    t = [Outcome(f"e{i}", 7, "t", i % 9 != 0) for i in range(60)]
    s = [Outcome(f"e{i}", 7, "t", i % 7 != 0) for i in range(60)]
    a = certify(t, s, margin=-0.05, interval="tango_one_sided95").to_json()
    b = certify(
        list(reversed(t)), list(reversed(s)), margin=-0.05, interval="tango_one_sided95"
    ).to_json()
    assert a == b
