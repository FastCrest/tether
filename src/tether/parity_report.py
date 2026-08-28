"""Auto-written PARITY.md — action-parity trust receipt for `tether verify`.

Sibling of :mod:`tether.verification_report` (which writes VERIFICATION.md for
`tether validate`'s ONNX-vs-PyTorch numerical round-trip). PARITY.md is the
*behavioral* receipt: it records whether the OPTIMIZED export behaves like the
ORIGINAL native-PyTorch policy when both run the same eval suite, scored through
the Pro 9-gate evaluator.

Called by the `tether verify` CLI handler with the :class:`ParityVerdict`
produced by :func:`tether.verify.run_verify`.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tether.verify import ParityVerdict

REPORT_FILENAME = "PARITY.md"


def _tether_version() -> str:
    try:
        from tether import __version__

        return __version__
    except Exception:
        return "unknown"


def _pct(x: object) -> str:
    from tether.verification_evidence import EvidenceValue

    if isinstance(x, EvidenceValue):
        assert x.reason is not None
        return f"UNAVAILABLE ({x.reason.value})"
    if not isinstance(x, (int, float)):
        raise TypeError(f"expected numeric success rate, got {type(x).__name__}")
    return f"{100.0 * float(x):.1f}%"


def _delta_pct(x: object) -> str:
    from tether.verification_evidence import EvidenceValue

    if isinstance(x, EvidenceValue):
        assert x.reason is not None
        return f"UNAVAILABLE ({x.reason.value})"
    if not isinstance(x, (int, float)):
        raise TypeError(f"expected numeric success delta, got {type(x).__name__}")
    return f"{float(x) * 100:+.1f}pp"


def _measurement(x: object) -> str:
    from tether.verification_evidence import EvidenceValue

    if isinstance(x, EvidenceValue):
        assert x.reason is not None
        return f"UNAVAILABLE ({x.reason.value})"
    if not isinstance(x, (int, float)):
        raise TypeError(f"expected numeric measurement, got {type(x).__name__}")
    return f"{float(x):.4g}"


def render_parity_report(verdict: "ParityVerdict") -> str:
    """Render the PARITY.md body for ``verdict`` as a markdown string.

    Pure (no I/O) so it can be unit-tested + reused by callers that want the
    text without writing a file.
    """
    from tether.pro.eval_gate import MIN_EPISODES_TO_EVALUATE

    report = verdict.eval_report
    verdict_str = "PASS" if verdict.passed else "FAIL"
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    lines: list[str] = [
        "# Tether Action-Parity Verification",
        "",
        f"Generated: {now}",
        "",
        f"**Verdict: {verdict_str}**",
        "",
        "## What this checks",
        "",
        "Does the OPTIMIZED export behave like the ORIGINAL native-PyTorch "
        "policy? Both run the same eval suite, paired by task + seed; the "
        "paired outcomes are scored through the Tether Pro 9-gate evaluator "
        "(original = baseline, optimized = candidate).",
        "",
        "## Run",
        "",
        f"- **Optimized (under test):** `{verdict.optimized_ref}`",
        f"- **Original (reference):** `{verdict.original_ref}`",
        f"- **Suite:** {verdict.suite}",
        f"- **Target:** {verdict.target}",
        f"- **Paired episodes:** {verdict.n_episodes} "
        f"(min for statistical power: {MIN_EPISODES_TO_EVALUATE})",
        f"- **Tether version:** {_tether_version()}",
        f"- **Platform:** {platform.platform()}",
        "",
        "## Headline",
        "",
        "| Metric | Original | Optimized | Delta |",
        "|---|---|---|---|",
        f"| Success rate | {_pct(verdict.original_success_rate)} | "
        f"{_pct(verdict.optimized_success_rate)} | "
        f"{_delta_pct(verdict.success_rate_delta)} |",
        "",
    ]

    if verdict.first_failing_gate_id is not None and report.first_failing_gate:
        g = report.first_failing_gate
        lines += [
            f"**First failing gate:** `{g.gate_id}` ({g.gate_class}) — {g.message}",
            "",
        ]

    lines += [
        "## Gate detail",
        "",
        "| Gate | Class | Result | Measured | Threshold | Detail |",
        "|---|---|---|---|---|---|",
    ]
    for g in report.all_gates:
        lines.append(
            f"| {g.gate_id} | {g.gate_class} | "
            f"{'PASS' if g.passed else 'FAIL'} | "
            f"{_measurement(g.measured)} | {g.threshold:.4g} | {g.message} |"
        )

    lines += [
        "",
        "## Optional diagnostics",
        "",
        f"- Two-sample distribution test: "
        f"{'EVALUATED' if verdict.two_sample is not None else 'NOT_EVALUATED (missing_field)'}",
        f"- Embodied-motion comparison: "
        f"{'EVALUATED' if verdict.embodied is not None else 'NOT_EVALUATED (missing_field)'}",
        "",
        "## Evidence scope",
        "",
        "This receipt uses Verification Evidence v1. Required rollout channels "
        "are measured explicitly and unavailable data fails its first affected "
        "gate instead of being formatted as a numeric sentinel.",
        "",
        "Safety evidence is available only when an explicit ActionGuard config "
        "was applied to both arms. Memory evidence contains 10 Hz process RSS "
        "and process-attributed device allocation samples with peak and "
        "nearest-rank p95 summaries.",
        "",
        "_Auto-generated by `tether verify`. Re-run to refresh this file._",
        "",
    ]
    return "\n".join(lines)


def write_parity_report(
    output_dir: str | Path,
    verdict: "ParityVerdict",
) -> Path:
    """Write ``<output_dir>/PARITY.md`` for ``verdict`` and return its path.

    ``output_dir`` is created if missing (it may be an export dir or a
    dedicated verify-output dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / REPORT_FILENAME
    out.write_text(render_parity_report(verdict) + "\n")
    return out


__all__ = ["REPORT_FILENAME", "render_parity_report", "write_parity_report"]
