"""Tests for src/tether/pro/weekly_report.py — Phase 1 self-distilling-serve Day 9."""
from __future__ import annotations

import json

import pytest

from tether.pro.weekly_report import (
    RollbackEntry,
    TaskDelta,
    WeeklyReport,
    render_cli,
    render_json,
    send_email,
    send_slack,
)


def _mk_report(**overrides) -> WeeklyReport:
    defaults = dict(
        customer_id="acme",
        period_start="2026-04-18T00:00:00Z",
        period_end="2026-04-25T00:00:00Z",
        headline_speedup_pct=5.2,
        headline_success_delta_pp=2.3,
        task_deltas=(
            TaskDelta(task_id="pick_block", n_episodes=80, success_rate_pp=88.0, delta_pp=3.0),
            TaskDelta(task_id="stack_blocks", n_episodes=40, success_rate_pp=72.0, delta_pp=-1.5),
        ),
        rollbacks=(),
        next_distill_eta="2026-04-26T03:00:00Z",
        distill_runs_used=2, distill_runs_limit=4,
        eval_runs_used=4, eval_runs_limit=8,
        modal_spend_usd=18.5, modal_spend_cap_usd=60.0,
    )
    defaults.update(overrides)
    return WeeklyReport(**defaults)


# ---------------------------------------------------------------------------
# WeeklyReport shape
# ---------------------------------------------------------------------------


def test_report_to_dict_round_trip():
    r = _mk_report()
    d = r.to_dict()
    s = json.dumps(d)
    restored = json.loads(s)
    assert restored["customer_id"] == "acme"
    assert restored["headline_speedup_pct"] == 5.2
    assert len(restored["task_deltas"]) == 2


def test_report_empty_factory():
    r = WeeklyReport.empty(
        customer_id="acme",
        period_start="2026-04-18T00:00:00Z",
        period_end="2026-04-25T00:00:00Z",
    )
    assert r.task_deltas == ()
    assert r.rollbacks == ()
    assert r.headline_speedup_pct == 0.0
    assert r.distill_runs_limit == 4  # Pro tier default


def test_report_is_frozen():
    r = _mk_report()
    with pytest.raises(AttributeError):
        r.customer_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# render_cli
# ---------------------------------------------------------------------------


def test_render_cli_includes_customer_id():
    r = _mk_report()
    out = render_cli(r)
    assert "acme" in out


def test_render_cli_includes_headline_metrics():
    r = _mk_report(headline_speedup_pct=8.0, headline_success_delta_pp=3.0)
    out = render_cli(r)
    assert "8.0%" in out
    assert "3.0pp" in out


def test_render_cli_shows_speedup_arrow_for_improvement():
    r = _mk_report(headline_speedup_pct=5.0)
    out = render_cli(r)
    assert "▲" in out
    assert "faster" in out


def test_render_cli_shows_speedup_arrow_for_regression():
    r = _mk_report(headline_speedup_pct=-3.0)
    out = render_cli(r)
    assert "▼" in out
    assert "slower" in out


def test_render_cli_shows_per_task_deltas():
    r = _mk_report()
    out = render_cli(r)
    assert "pick_block" in out
    assert "stack_blocks" in out


def test_render_cli_per_task_sorted_by_improvement():
    """Tasks sorted by delta_pp descending — best improvement first."""
    r = _mk_report(task_deltas=(
        TaskDelta(task_id="task_low", n_episodes=10, success_rate_pp=50.0, delta_pp=-5.0),
        TaskDelta(task_id="task_high", n_episodes=10, success_rate_pp=90.0, delta_pp=10.0),
        TaskDelta(task_id="task_mid", n_episodes=10, success_rate_pp=70.0, delta_pp=2.0),
    ))
    out = render_cli(r)
    pos_high = out.index("task_high")
    pos_mid = out.index("task_mid")
    pos_low = out.index("task_low")
    assert pos_high < pos_mid < pos_low


def test_render_cli_no_rollbacks_shows_clean():
    r = _mk_report(rollbacks=())
    out = render_cli(r)
    assert "none ✓" in out


def test_render_cli_with_rollbacks_lists_each():
    r = _mk_report(rollbacks=(
        RollbackEntry(
            audit_id="rb-123-abcd",
            timestamp="2026-04-22T15:30:00Z",
            trigger="auto",
            reason="T1",
            from_slot="a", to_slot="b",
            elapsed_s=12.5,
        ),
    ))
    out = render_cli(r)
    assert "rb-123-abcd" in out
    assert "T1" in out
    assert "a→b" in out


def test_render_cli_shows_budget_bars():
    r = _mk_report(distill_runs_used=2, distill_runs_limit=4)
    out = render_cli(r)
    # 50% of 20-char bar = 10 filled blocks
    assert "█" in out


def test_render_cli_next_distill_eta_when_set():
    r = _mk_report(next_distill_eta="2026-04-26T03:00:00Z")
    out = render_cli(r)
    assert "2026-04-26T03:00:00Z" in out


def test_render_cli_next_distill_manual_mode_when_none():
    r = _mk_report(next_distill_eta=None)
    out = render_cli(r)
    assert "manual mode" in out


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------


def test_render_json_is_valid_json():
    r = _mk_report()
    payload = render_json(r)
    parsed = json.loads(payload)
    assert parsed["customer_id"] == "acme"


def test_render_json_includes_all_fields():
    r = _mk_report()
    parsed = json.loads(render_json(r))
    expected_keys = {
        "customer_id", "period_start", "period_end",
        "headline_speedup_pct", "headline_success_delta_pp",
        "task_deltas", "rollbacks", "next_distill_eta",
        "distill_runs_used", "distill_runs_limit",
        "eval_runs_used", "eval_runs_limit",
        "modal_spend_usd", "modal_spend_cap_usd",
    }
    assert expected_keys.issubset(parsed.keys())


# ---------------------------------------------------------------------------
# Phase 1.5 stubs — must raise loudly, NEVER silent
# ---------------------------------------------------------------------------


def test_send_email_raises_not_implemented():
    """Phase 1 ships CLI default; email is opt-in Phase 1.5. Operators
    must know they need to wire SES themselves — silent skip is wrong."""
    r = _mk_report()
    with pytest.raises(NotImplementedError, match="Email"):
        send_email(r, to="admin@customer.com")


def test_send_slack_raises_not_implemented():
    r = _mk_report()
    with pytest.raises(NotImplementedError, match="Slack"):
        send_slack(r, webhook="https://hooks.slack.com/services/...")


# ---------------------------------------------------------------------------
# TaskDelta + RollbackEntry shape
# ---------------------------------------------------------------------------


def test_task_delta_is_frozen():
    t = TaskDelta(task_id="x", n_episodes=10, success_rate_pp=80.0, delta_pp=3.0)
    with pytest.raises(AttributeError):
        t.task_id = "y"  # type: ignore[misc]


def test_rollback_entry_is_frozen():
    r = RollbackEntry(
        audit_id="rb-1", timestamp="2026-04-22T15:30:00Z",
        trigger="auto", reason="T1",
        from_slot="a", to_slot="b", elapsed_s=10.0,
    )
    with pytest.raises(AttributeError):
        r.reason = "T2"  # type: ignore[misc]
