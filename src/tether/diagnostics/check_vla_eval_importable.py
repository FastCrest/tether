"""Check — vla_eval adapter importable (eval-as-a-service Day 5).

Per ADR 2026-04-25-eval-as-a-service-architecture decision #9:
3 additive doctor checks for the eval feature; this one verifies
the Tether internal vla_eval adapter (PredictModelServer) can be
imported.

The adapter lives at src/tether/runtime/adapters/vla_eval.py and
ships unchanged from the production modal_libero_*.py recipe (per ADR
decision #2 -- wrap, not rebuild).

Pass if the import works. Warn (not fail) if ImportError — likely
indicates a missing optional dep transitively required by the adapter.
"""
from __future__ import annotations

from . import Check, CheckResult, register

CHECK_ID = "check_vla_eval_importable"
DOCS_URL = "https://github.com/FastCrest/tether-vla/blob/main/src/tether/runtime/adapters/vla_eval.py"


def _run(model_path: str, **kwargs) -> CheckResult:
    try:
        from tether.runtime.adapters import vla_eval  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            check_id=CHECK_ID,
            name="vla_eval adapter importable (for `tether eval`)",
            status="warn",
            expected="`from tether.runtime.adapters import vla_eval` succeeds",
            actual=f"ImportError: {exc}",
            remediation=(
                "Adapter import failed. Likely a missing transitive dep. "
                "Install the eval extra: `pip install 'tether[eval]'`. "
                "If reproducible after install, file a bug at "
                f"{DOCS_URL}."
            ),
            duration_ms=0.0,
            github_issue=DOCS_URL,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id=CHECK_ID,
            name="vla_eval adapter importable (for `tether eval`)",
            status="warn",
            expected="`from tether.runtime.adapters import vla_eval` succeeds",
            actual=f"{type(exc).__name__}: {exc}",
            remediation=(
                f"Adapter raised on import. File a bug at {DOCS_URL} "
                f"with the full traceback."
            ),
            duration_ms=0.0,
            github_issue=DOCS_URL,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="vla_eval adapter importable (for `tether eval`)",
        status="pass",
        expected="`from tether.runtime.adapters import vla_eval` succeeds",
        actual="imported successfully",
        remediation="",
        duration_ms=0.0,
        github_issue=DOCS_URL,
    )


register(Check(
    check_id=CHECK_ID,
    name="vla_eval adapter importable",
    severity="warn",
    github_issue=DOCS_URL,
    run_fn=_run,
))
