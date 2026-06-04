"""Pro-tier weekly report — CLI-only default; email/Slack opt-in.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #8:
- CLI-only default (`tether report last-week`)
- Reasoning: SES adds billing + spam-filter + TLS cert surface;
  Slack assumes Slack; CLI is zero-dep + scriptable + air-gap-friendly
- Email + Slack are opt-in: `--report-channel email:admin@customer.com`
  or `--report-channel slack:<webhook>`

Phase 1 ships the report-shape primitive + CLI channel rendering. Email
+ Slack adapters are stubs that raise NotImplementedError loudly when
invoked (no silent skip — operators must know they need Phase 1.5
wiring).

Content per ADR:
- Headline metric (% faster, +pp success)
- Per-task delta
- Rollback-if-any (with audit_id link)
- Next-distill-ETA
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)


ReportChannel = Literal["cli", "email", "slack"]


@dataclass(frozen=True)
class TaskDelta:
    """One task's success-rate delta over the report window."""

    task_id: str
    n_episodes: int
    success_rate_pp: float  # current
    delta_pp: float  # current - prior_window


@dataclass(frozen=True)
class RollbackEntry:
    """Audit-linked summary of one rollback during the report window."""

    audit_id: str
    timestamp: str  # ISO 8601
    trigger: str  # auto|cli|endpoint
    reason: str  # T1|T2|T3|operator-cli|operator-endpoint|forced
    from_slot: str
    to_slot: str
    elapsed_s: float


@dataclass(frozen=True)
class WeeklyReport:
    """The frozen weekly snapshot. Built once per report invocation;
    rendered to one or more channels."""

    customer_id: str
    period_start: str  # ISO 8601 (UTC)
    period_end: str    # ISO 8601 (UTC)

    # Headline metrics
    headline_speedup_pct: float       # >0 means current faster than prior
    headline_success_delta_pp: float  # >0 means current better than prior

    # Per-task deltas
    task_deltas: tuple[TaskDelta, ...]

    # Rollbacks within the period
    rollbacks: tuple[RollbackEntry, ...]

    # Forward-looking
    next_distill_eta: str | None  # ISO 8601 OR None when manual mode

    # Pro-tier budget tracking
    distill_runs_used: int
    distill_runs_limit: int
    eval_runs_used: int
    eval_runs_limit: int
    modal_spend_usd: float
    modal_spend_cap_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "headline_speedup_pct": self.headline_speedup_pct,
            "headline_success_delta_pp": self.headline_success_delta_pp,
            "task_deltas": [asdict(t) for t in self.task_deltas],
            "rollbacks": [asdict(r) for r in self.rollbacks],
            "next_distill_eta": self.next_distill_eta,
            "distill_runs_used": self.distill_runs_used,
            "distill_runs_limit": self.distill_runs_limit,
            "eval_runs_used": self.eval_runs_used,
            "eval_runs_limit": self.eval_runs_limit,
            "modal_spend_usd": self.modal_spend_usd,
            "modal_spend_cap_usd": self.modal_spend_cap_usd,
        }

    @classmethod
    def empty(cls, *, customer_id: str, period_start: str, period_end: str) -> "WeeklyReport":
        """Construct an empty report — used when no activity in the window."""
        return cls(
            customer_id=customer_id,
            period_start=period_start,
            period_end=period_end,
            headline_speedup_pct=0.0,
            headline_success_delta_pp=0.0,
            task_deltas=(),
            rollbacks=(),
            next_distill_eta=None,
            distill_runs_used=0, distill_runs_limit=4,
            eval_runs_used=0, eval_runs_limit=8,
            modal_spend_usd=0.0, modal_spend_cap_usd=60.0,
        )


def render_cli(report: WeeklyReport) -> str:
    """Render the report as a human-readable terminal string. Used by
    `tether report last-week` AND by the email/Slack adapters as the
    body when they ship."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Tether Pro — Weekly Report")
    lines.append("=" * 72)
    lines.append(f"Customer: {report.customer_id}")
    lines.append(f"Period:   {report.period_start} → {report.period_end}")
    lines.append("")

    # Headline
    speedup_arrow = "▲" if report.headline_speedup_pct > 0 else "▼" if report.headline_speedup_pct < 0 else "—"
    success_arrow = "▲" if report.headline_success_delta_pp > 0 else "▼" if report.headline_success_delta_pp < 0 else "—"
    lines.append("Headline:")
    lines.append(
        f"  Inference speed: {speedup_arrow} {abs(report.headline_speedup_pct):.1f}% "
        f"{'faster' if report.headline_speedup_pct >= 0 else 'slower'}"
    )
    lines.append(
        f"  Task success:    {success_arrow} {abs(report.headline_success_delta_pp):.1f}pp "
        f"{'higher' if report.headline_success_delta_pp >= 0 else 'lower'}"
    )
    lines.append("")

    # Per-task
    if report.task_deltas:
        lines.append("Per-task (sorted by improvement):")
        sorted_deltas = sorted(
            report.task_deltas, key=lambda d: d.delta_pp, reverse=True,
        )
        for d in sorted_deltas:
            arrow = "▲" if d.delta_pp > 0 else "▼" if d.delta_pp < 0 else "—"
            lines.append(
                f"  {d.task_id:24s}  n={d.n_episodes:5d}  "
                f"success={d.success_rate_pp:5.1f}%  "
                f"{arrow} {abs(d.delta_pp):4.1f}pp"
            )
        lines.append("")

    # Rollbacks
    if report.rollbacks:
        lines.append(f"Rollbacks ({len(report.rollbacks)}):")
        for r in report.rollbacks:
            lines.append(
                f"  {r.timestamp}  audit={r.audit_id}  trigger={r.trigger}  "
                f"reason={r.reason}  {r.from_slot}→{r.to_slot}  ({r.elapsed_s:.1f}s)"
            )
        lines.append("")
    else:
        lines.append("Rollbacks: none ✓")
        lines.append("")

    # Pro-tier budget
    lines.append("Pro-tier budget:")
    distill_pct = report.distill_runs_used / max(1, report.distill_runs_limit) * 100
    eval_pct = report.eval_runs_used / max(1, report.eval_runs_limit) * 100
    spend_pct = report.modal_spend_usd / max(1, report.modal_spend_cap_usd) * 100
    lines.append(
        f"  Distill runs: {report.distill_runs_used}/{report.distill_runs_limit} "
        f"({distill_pct:.0f}%)  {_render_bar(distill_pct)}"
    )
    lines.append(
        f"  Eval runs:    {report.eval_runs_used}/{report.eval_runs_limit} "
        f"({eval_pct:.0f}%)  {_render_bar(eval_pct)}"
    )
    lines.append(
        f"  Modal spend:  ${report.modal_spend_usd:.2f}/${report.modal_spend_cap_usd:.2f} "
        f"({spend_pct:.0f}%)  {_render_bar(spend_pct)}"
    )
    lines.append("")

    # Next-distill ETA
    if report.next_distill_eta:
        lines.append(f"Next distill: {report.next_distill_eta}")
    else:
        lines.append("Next distill: manual mode (not scheduled)")
    lines.append("=" * 72)
    return "\n".join(lines)


def render_json(report: WeeklyReport) -> str:
    """Render as JSON for scripting / external consumption."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def send_email(report: WeeklyReport, *, to: str) -> None:
    """Phase 1.5 channel — raises NotImplementedError loudly. Operators
    must wire SES / SMTP themselves, NOT silently no-op."""
    raise NotImplementedError(
        f"Email channel is Phase 1.5; can't send to {to}. "
        f"Use --report-channel cli (default) for now, OR wire your own "
        f"SES/SMTP via the WeeklyReport.to_dict() output."
    )


def send_slack(report: WeeklyReport, *, webhook: str) -> None:
    """Phase 1.5 channel — raises NotImplementedError loudly."""
    raise NotImplementedError(
        f"Slack channel is Phase 1.5; can't send to {webhook[:30]}... "
        f"Use --report-channel cli (default) for now, OR wire your own "
        f"Slack incoming webhook via the WeeklyReport.to_dict() output."
    )


def _render_bar(pct: float, width: int = 20) -> str:
    """Tiny ASCII progress bar. Caps at width."""
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


__all__ = [
    "RollbackEntry",
    "TaskDelta",
    "WeeklyReport",
    "render_cli",
    "render_json",
    "send_email",
    "send_slack",
]
