"""Check — Modal auth detected (eval-as-a-service Day 5).

Per ADR 2026-04-25-eval-as-a-service-architecture decision #9:
3 additive doctor checks for the eval feature; this one verifies
Modal CLI auth is configured.

Pass if `~/.modal.toml` exists OR `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`
are set. Warn (not fail) if neither — the customer may not need eval,
but `tether eval --runtime modal` will refuse to run until configured.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import Check, CheckResult, register

CHECK_ID = "check_modal_auth"
DOCS_URL = "https://modal.com/docs/guide/setup"


def _run(model_path: str, **kwargs) -> CheckResult:
    home_config = Path.home() / ".modal.toml"
    has_config_file = home_config.exists()
    has_env_tokens = bool(
        os.environ.get("MODAL_TOKEN_ID")
        and os.environ.get("MODAL_TOKEN_SECRET")
    )

    if has_config_file or has_env_tokens:
        source = "~/.modal.toml" if has_config_file else "MODAL_TOKEN_* env"
        return CheckResult(
            check_id=CHECK_ID,
            name="Modal auth (for `tether eval --runtime modal`)",
            status="pass",
            expected="~/.modal.toml OR MODAL_TOKEN_ID+SECRET env",
            actual=f"detected via {source}",
            remediation="",
            duration_ms=0.0,
            github_issue=DOCS_URL,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Modal auth (for `tether eval --runtime modal`)",
        status="warn",
        expected="~/.modal.toml OR MODAL_TOKEN_ID+SECRET env",
        actual="neither detected",
        remediation=(
            f"Install Modal CLI + run `modal token new` to authenticate. "
            f"Optional unless you plan to run `tether eval --runtime modal`. "
            f"See {DOCS_URL}."
        ),
        duration_ms=0.0,
        github_issue=DOCS_URL,
    )


register(Check(
    check_id=CHECK_ID,
    name="Modal auth",
    severity="warn",
    github_issue=DOCS_URL,
    run_fn=_run,
))
