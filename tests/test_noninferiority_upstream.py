# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in src/tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28.
#!/usr/bin/env python3
"""Certification must be able to FAIL, and must refuse inputs it cannot honestly analyse.

A gate that only says yes is decoration, so most of this file is about the ways `certify` should
say no.

Ported unchanged (apart from the import path) from the upstream ``tests/test_certify.py``;
see ``docs/provenance/instinct-port.md``.
"""

from __future__ import annotations

from tether.noninferiority import NotCertifiable, Outcome, certify

results = []


def arm(successes, tag="ep"):
    return [
        Outcome(f"{tag}{i}", 10000 * (1 + i), "adjust_bottle", bool(s))
        for i, s in enumerate(successes)
    ]


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  -- ' + extra) if extra else ''}")
    results.append(cond)


def main() -> int:
    print("=== 1. a real regression must FAIL ===")
    # teacher 92/100, student 85/100, and the student loses the discordant pairs
    t = arm([1] * 92 + [0] * 8)
    s = arm([1] * 85 + [0] * 15)
    c = certify(t, s, margin=-0.05, harness="robotwin-2.0", recipe="test")
    print("   ", str(c).replace("\n", "\n    "))
    check("7-point drop fails a 5-point margin", not c.passed)
    check("delta is negative", c.delta < 0, f"delta={c.delta:+.3f}")

    print("\n=== 2. a genuinely equivalent student must PASS ===")
    t2 = arm([1] * 92 + [0] * 8)
    s2 = arm([1] * 92 + [0] * 8)
    c2 = certify(t2, s2, margin=-0.05)
    check("identical outcomes pass", c2.passed, c2.verdict[:46])
    check("zero discordant pairs is noted, not hidden", any("discordant" in n for n in c2.notes))
    check(
        "default mode carries no tango fields (byte-stable certificate)",
        c2.ci_method is None
        and c2.tango_central90 is None
        and '"tango_central90"' not in c2.to_json(),
    )

    print("\n=== 2b. the opt-in tango_one_sided95 mode is explicitly named, never ci95 ===")
    c2t = certify(t2, s2, margin=-0.05, interval="tango_one_sided95")
    check("identical outcomes pass under the tango rule", c2t.passed, c2t.verdict[:60])
    check(
        "paired score interval stays non-degenerate at zero discordance",
        c2t.tango_central90[0] < 0 < c2t.tango_central90[1],
        f"tango_central90={c2t.tango_central90}",
    )
    check(
        "NI decides on Tango while McNemar is descriptive",
        "Tango" in c2t.ci_method and "McNemar" in c2t.p_value_kind,
    )
    check("ci95 keeps its Wald meaning in both modes", c2t.ci95 == c2.ci95, f"ci95={c2t.ci95}")
    check(
        "the decision bound is serialized under its own name",
        '"tango_central90"' in c2t.to_json()
        and c2t.lower_confidence_bound == c2t.tango_central90[0],
    )

    print("\n=== 2c. the secondary task-collapse gate fails a collapsed task ===")
    t2c = arm([1, 1, 0, 1], "x") + [Outcome("y0", 7, "other_task", True)]
    s2c = arm([1, 1, 0, 1], "x") + [Outcome("y0", 7, "other_task", False)]
    c2c = certify(t2c, s2c, margin=-0.99, fail_on_task_collapse=True)
    check(
        "collapse gate overrides an aggregate pass",
        not c2c.passed and "task-collapse" in c2c.verdict,
        c2c.verdict[:60],
    )
    check("collapsed task is named", c2c.collapsed_tasks == ("other_task",))
    c2n = certify(t2c, s2c, margin=-0.99)
    check("gate off by default", c2n.passed and c2n.collapsed_tasks is None)

    print("\n=== 3. a small drop inside the margin passes ===")
    t3 = arm([1] * 92 + [0] * 8)
    s3 = arm([1] * 90 + [0] * 10)
    c3 = certify(t3, s3, margin=-0.10)
    check("2-point drop passes a 10-point margin", c3.passed, f"delta={c3.delta:+.3f}")

    print("\n=== 4. refusals: inputs that cannot support a certificate ===")
    for name, fn in [
        (
            "unpaired arms (different episodes)",
            lambda: certify(arm([1, 1, 0], "a"), arm([1, 0, 1], "b"), margin=-0.05),
        ),
        ("partial overlap", lambda: certify(arm([1] * 5), arm([1] * 3), margin=-0.05)),
        (
            "duplicate episode in an arm",
            lambda: certify(arm([1, 1]) + arm([1, 1]), arm([1, 1]), margin=-0.05),
        ),
        (
            "positive margin (would certify a worse model)",
            lambda: certify(arm([1, 0]), arm([1, 0]), margin=+0.05),
        ),
        (
            "task mismatch on the same episode id",
            lambda: certify(
                [Outcome("e0", 1, "taskA", True)], [Outcome("e0", 1, "taskB", True)], margin=-0.05
            ),
        ),
    ]:
        try:
            fn()
            check(name, False, "did NOT refuse")
        except NotCertifiable as e:
            check(name, True, str(e).split(".")[0][:58])

    print("\n=== 5. the margin is recorded, so it cannot be chosen afterwards ===")
    c5 = certify(arm([1] * 90 + [0] * 10), arm([1] * 88 + [0] * 12), margin=-0.03)
    check("margin appears in the certificate", c5.margin_declared == -0.03)
    check("margin appears in the JSON", '"margin_declared": -0.03' in c5.to_json())

    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} checks")
    return 0 if all(results) else 1


def test_upstream_certify_checks():
    results.clear()
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
