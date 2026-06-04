"""Result shapes for pre-flight validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreflightCheck:
    """One check's result.

    Severity:
      - `ok`    — check passed, no action needed
      - `warn`  — proceeds but user should know (e.g. small dataset)
      - `fail`  — refuses the run; user must fix before proceeding
    """

    name: str
    severity: str  # "ok" | "warn" | "fail"
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.severity == "ok"

    @property
    def is_blocking(self) -> bool:
        return self.severity == "fail"


@dataclass
class PreflightReport:
    """Aggregated result of all checks. Passed to the caller as the
    decision point: proceed, proceed-with-warnings, or abort.
    """

    checks: list[PreflightCheck] = field(default_factory=list)

    def add(self, check: PreflightCheck) -> None:
        self.checks.append(check)

    @property
    def has_failures(self) -> bool:
        return any(c.is_blocking for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.severity == "warn" for c in self.checks)

    def render(self) -> str:
        """Human-readable multi-line summary, e.g. for stdout."""
        lines = []
        for c in self.checks:
            badge = {"ok": "✓", "warn": "⚠", "fail": "✗"}.get(c.severity, "?")
            lines.append(f"  {badge} [{c.name}] {c.summary}")
            for k, v in c.detail.items():
                lines.append(f"      {k}: {v}")
        return "\n".join(lines)

    def error_message(self) -> str:
        """One-line error summarizing all failures (for run_finetune's
        FinetuneResult.error field)."""
        fails = [c for c in self.checks if c.is_blocking]
        if not fails:
            return ""
        return "preflight failed:\n  " + "\n  ".join(
            f"- [{c.name}] {c.summary}" for c in fails
        )


__all__ = ["PreflightCheck", "PreflightReport"]
