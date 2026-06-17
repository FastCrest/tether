"""Check — LIBERO importable (eval-as-a-service Day 5).

Per ADR 2026-04-25-eval-as-a-service-architecture decision #9:
3 additive doctor checks for the eval feature; this one verifies
the LIBERO sim package can be imported in the local Python env.

Pass if `import libero` works. Warn (not fail) if ImportError — the
customer may not need local eval, but `tether eval --runtime local`
will refuse to run until installed.
"""
from __future__ import annotations

from . import Check, CheckResult, register

CHECK_ID = "check_libero_importable"
DOCS_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO"


def _run(model_path: str, **kwargs) -> CheckResult:
    try:
        import libero  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            check_id=CHECK_ID,
            name="LIBERO importable (for `tether eval --runtime local`)",
            status="warn",
            expected="`import libero` succeeds",
            actual=f"ImportError: {exc}",
            remediation=(
                f"Install via `pip install 'fastcrest-tether[eval-local]'`. "
                f"Required only for `tether eval --runtime local`; "
                f"--runtime modal ships LIBERO in the bundled image. "
                f"Phase 1 local fallback is Linux x86_64 only."
            ),
            duration_ms=0.0,
            github_issue=DOCS_URL,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id=CHECK_ID,
            name="LIBERO importable (for `tether eval --runtime local`)",
            status="warn",
            expected="`import libero` succeeds",
            actual=f"{type(exc).__name__}: {exc}",
            remediation=(
                "LIBERO imported but raised — likely a dep version conflict. "
                "Pin: robosuite==1.4.1, bddl==1.0.1, mujoco==3.3.2 per ADR "
                "2026-04-25-eval-as-a-service-architecture."
            ),
            duration_ms=0.0,
            github_issue=DOCS_URL,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="LIBERO importable (for `tether eval --runtime local`)",
        status="pass",
        expected="`import libero` succeeds",
        actual="imported successfully",
        remediation="",
        duration_ms=0.0,
        github_issue=DOCS_URL,
    )


register(Check(
    check_id=CHECK_ID,
    name="LIBERO importable",
    severity="warn",
    github_issue=DOCS_URL,
    run_fn=_run,
))
