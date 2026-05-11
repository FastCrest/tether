"""Reflex serve deployment diagnostics — `reflex doctor`.

Runs 10 falsifiable checks against an export + embodiment config to catch
the most common deployment failures before users hit them. Each check
maps to a known-broken LeRobot GitHub issue or a systemic VLA deploy
failure mode.

Day 1 ships 5 checks (LeRobot async pain points). Day 2 adds 5 more
(systemic VLA deploy failures + hardware compat).

Plan: features/01_serve/subfeatures/_dx_gaps/reflex-doctor_plan.md
Canonical: features/01_serve/subfeatures/_dx_gaps/reflex-doctor.md
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

CheckStatus = Literal["pass", "fail", "warn", "skip"]
CheckSeverity = Literal["error", "warn", "info"]


@dataclass
class CheckResult:
    """One check's verdict. Stable contract — JSON-serialized via format_json()."""

    check_id: str
    name: str
    status: CheckStatus
    expected: str
    actual: str
    remediation: str  # Empty only when status == "pass" or "skip"
    duration_ms: float
    github_issue: str | None = None

    def __post_init__(self) -> None:
        # Falsifiability gate — every fail must have a remediation
        if self.status == "fail" and not self.remediation:
            raise ValueError(
                f"Check {self.check_id!r} has status=fail but empty remediation. "
                f"Every fail MUST have a remediation pointing to a GitHub issue or docs URL."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "remediation": self.remediation,
            "duration_ms": round(self.duration_ms, 2),
            "github_issue": self.github_issue,
        }


@dataclass
class Check:
    """A diagnostic check definition. Subclass this and implement run().

    Subclasses set class attributes (id, name, severity, github_issue) and
    implement run(model_path, embodiment_name, **kwargs) -> CheckResult.
    """

    check_id: str
    name: str
    severity: CheckSeverity
    github_issue: str | None
    run_fn: Callable[..., CheckResult]

    def execute(self, **kwargs) -> CheckResult:
        """Time + run. Catches exceptions and converts to CheckResult.fail."""
        t0 = time.monotonic()
        try:
            result = self.run_fn(**kwargs)
            # Backfill duration on the result (run_fn doesn't time itself)
            if result.duration_ms == 0.0:
                result = CheckResult(
                    check_id=result.check_id,
                    name=result.name,
                    status=result.status,
                    expected=result.expected,
                    actual=result.actual,
                    remediation=result.remediation,
                    duration_ms=(time.monotonic() - t0) * 1000.0,
                    github_issue=result.github_issue,
                )
            return result
        except Exception as e:  # noqa: BLE001 — checks must never crash doctor
            return CheckResult(
                check_id=self.check_id,
                name=self.name,
                status="fail",
                expected="check completes without exception",
                actual=f"raised {type(e).__name__}: {e}",
                remediation=(
                    f"Check {self.check_id} crashed — file a bug at "
                    f"https://github.com/FastCrest/reflex-vla/issues with the "
                    f"full traceback above."
                ),
                duration_ms=(time.monotonic() - t0) * 1000.0,
                github_issue=self.github_issue,
            )


# Registry — populated lazily on first run_all_checks() call
_REGISTRY: list[Check] = []
_REGISTRY_LOADED: bool = False


def register(check: Check) -> None:
    """Append a check to the registry. Called by check modules at import."""
    _REGISTRY.append(check)


def _ensure_registry_loaded() -> None:
    """Import every check_*.py module so they self-register. Idempotent.

    Uses a separate flag rather than `if _REGISTRY:` so a partial-load
    (e.g. when a single check_*.py was imported directly elsewhere) still
    triggers the full sweep. Re-imports are cheap — sys.modules caches them
    and `register()` is at module top-level so it only fires once per
    interpreter regardless of how many times you import the module.
    """
    global _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return
    # Day 1 checks
    from . import (  # noqa: F401
        check_image_dims,
        check_model_load,
        check_onnx_provider,
        check_rtc_chunks,
        check_vlm_tokenization,
    )
    # Day 2 checks
    from . import (  # noqa: F401
        check_action_denorm,
        check_gpu_memory,
        check_gripper,
        check_hardware_compat,
        check_state_proprio,
    )
    # Phase 1 cuda-graphs check — added 2026-04-24 per ADR
    # 2026-04-24-cuda-graphs-architecture (surfaces A10G tier-aware semantics)
    from . import check_cuda_graphs  # noqa: F401
    # Phase 1 eval-as-a-service checks — added 2026-04-25 per ADR
    # 2026-04-25-eval-as-a-service-architecture decision #9
    from . import (  # noqa: F401
        check_libero_importable,
        check_modal_auth,
        check_vla_eval_importable,
    )
    _REGISTRY_LOADED = True


def run_all_checks(
    model_path: str,
    embodiment_name: str = "custom",
    *,
    rtc: bool = False,
    skip: list[str] | None = None,
) -> list[CheckResult]:
    """Run every registered check. Returns one CheckResult per check.

    Args:
        model_path: path to the export directory.
        embodiment_name: preset name from configs/embodiments/.
        rtc: if True, RTC-related checks validate config; otherwise they skip.
        skip: list of check_ids to skip (returns CheckResult.skip).
    """
    _ensure_registry_loaded()
    skip_set = set(skip or [])
    results: list[CheckResult] = []
    for check in _REGISTRY:
        if check.check_id in skip_set:
            results.append(CheckResult(
                check_id=check.check_id,
                name=check.name,
                status="skip",
                expected="(skipped via --skip)",
                actual="",
                remediation="",
                duration_ms=0.0,
                github_issue=check.github_issue,
            ))
            continue
        results.append(check.execute(
            model_path=model_path,
            embodiment_name=embodiment_name,
            rtc=rtc,
        ))
    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


_STATUS_GLYPH = {"pass": "✓", "fail": "✗", "warn": "!", "skip": "⊘"}


def format_human(results: list[CheckResult]) -> str:
    """Plain-text table. ASCII glyphs (Unicode-safe but not emoji)."""
    if not results:
        return "(no checks ran)"
    # Sort: fails first, warns second, passes third, skips last (most-actionable first)
    order = {"fail": 0, "warn": 1, "pass": 2, "skip": 3}
    sorted_results = sorted(results, key=lambda r: (order.get(r.status, 9), r.check_id))

    summary = {
        "pass": sum(1 for r in results if r.status == "pass"),
        "fail": sum(1 for r in results if r.status == "fail"),
        "warn": sum(1 for r in results if r.status == "warn"),
        "skip": sum(1 for r in results if r.status == "skip"),
    }

    lines: list[str] = []
    lines.append(
        f"reflex doctor — {len(results)} checks, "
        f"{summary['pass']} pass, {summary['fail']} fail, "
        f"{summary['warn']} warn, {summary['skip']} skip"
    )
    lines.append("")

    for r in sorted_results:
        glyph = _STATUS_GLYPH.get(r.status, "?")
        lines.append(f"  [{glyph}] {r.check_id:32s} {r.name}  ({r.duration_ms:.0f}ms)")
        if r.status in ("fail", "warn"):
            lines.append(f"        expected:    {r.expected}")
            lines.append(f"        actual:      {r.actual}")
            if r.remediation:
                lines.append(f"        remediation: {r.remediation}")
            if r.github_issue:
                lines.append(f"        ref: {r.github_issue}")
            lines.append("")
    return "\n".join(lines)


def format_json(
    results: list[CheckResult],
    *,
    model_path: str = "",
    embodiment_name: str = "",
    schema_version: int = 1,
) -> str:
    """Stable JSON envelope for CI / dashboards. schema_version = 1."""
    summary = {
        "pass": sum(1 for r in results if r.status == "pass"),
        "fail": sum(1 for r in results if r.status == "fail"),
        "warn": sum(1 for r in results if r.status == "warn"),
        "skip": sum(1 for r in results if r.status == "skip"),
    }
    envelope = {
        "schema_version": schema_version,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "model": model_path,
        "embodiment": embodiment_name,
        "checks": [r.to_dict() for r in results],
        "summary": summary,
    }
    return json.dumps(envelope, indent=2)


def exit_code(results: list[CheckResult]) -> int:
    """Exit code mapping per plan §Goal item 4."""
    if any(r.status == "fail" for r in results):
        return 1
    return 0


__all__ = [
    "Check",
    "CheckResult",
    "CheckStatus",
    "CheckSeverity",
    "register",
    "run_all_checks",
    "format_human",
    "format_json",
    "exit_code",
]
