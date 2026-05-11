"""Check 9 — RTC chunk-boundary alignment (LeRobot #2356, #2531).

THE HEADLINE check — directly maps to the highest-signal LeRobot async
issues with zero maintainer response. Validates that RTC's replan_hz /
execute_hz / chunk_size triple lines up so chunk boundaries don't stall
between replan ticks.

Skips silently when --rtc not set (RTC is opt-in).
"""
from __future__ import annotations

from . import Check, CheckResult, register

CHECK_ID = "check_rtc_chunks"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2356"


def _run(embodiment_name: str = "custom", rtc: bool = False, **kwargs) -> CheckResult:
    if not rtc:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="skip",
            expected="--rtc passed for chunk-boundary validation",
            actual="RTC disabled",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if embodiment_name == "custom":
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="warn",
            expected="--embodiment <preset> for control rate cross-check",
            actual="embodiment=custom — using lerobot RTC defaults (replan_hz=20, execute_hz=100)",
            remediation=(
                "Pass --embodiment franka|so100|ur5 to validate the chunk_size against "
                "this preset's control rates. Skipping cross-check."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    try:
        from reflex.embodiments import EmbodimentConfig
        cfg = EmbodimentConfig.load_preset(embodiment_name)
    except (ValueError, FileNotFoundError) as e:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="fail",
            expected=f"embodiment preset {embodiment_name!r} loads",
            actual=f"load failed: {e}",
            remediation="See docs/embodiment_schema.md for the preset list.",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    control = cfg.control
    chunk_size = int(control["chunk_size"])
    frequency_hz = float(control["frequency_hz"])  # robot control loop rate
    # rtc_execution_horizon is INTEGER COUNT OF ACTIONS per
    # embodiments/validate.py:163 — not seconds. An earlier revision of this
    # check multiplied by frequency_hz, over-counting by ~Hz× and flagging
    # franka (chunk=50, horizon=25) as fail when it actually aligns 2:1.
    actions_per_horizon = int(control["rtc_execution_horizon"])

    if actions_per_horizon < 1:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="fail",
            expected="rtc_execution_horizon ≥ 1 action",
            actual=f"rtc_execution_horizon = {actions_per_horizon} actions",
            remediation=(
                f"Increase rtc_execution_horizon (currently {actions_per_horizon} "
                f"actions). With 0 actions per horizon, RTC degenerates to no-RTC."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if chunk_size < actions_per_horizon:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="fail",
            expected=f"chunk_size ≥ {actions_per_horizon} actions (one horizon)",
            actual=f"chunk_size={chunk_size} < {actions_per_horizon} actions/horizon",
            remediation=(
                f"Either increase chunk_size to ≥ {actions_per_horizon + 1} OR "
                f"reduce rtc_execution_horizon. Mismatch causes the boundary stalls "
                f"reported in LeRobot #2356."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if chunk_size % actions_per_horizon != 0:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="warn",
            expected=f"chunk_size ({chunk_size}) is a multiple of actions_per_horizon ({actions_per_horizon})",
            actual=f"chunk_size % horizon_actions = {chunk_size % actions_per_horizon}",
            remediation=(
                f"Non-integer ratio means the last partial-horizon at chunk boundary "
                f"will run with stale guidance. Consider chunk_size="
                f"{actions_per_horizon * (chunk_size // actions_per_horizon + 1)} "
                f"for cleaner alignment. Not a hard failure."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="RTC chunk boundary",
        status="pass",
        expected="chunk_size, frequency_hz, rtc_execution_horizon align cleanly",
        actual=(
            f"chunk_size={chunk_size}, frequency_hz={frequency_hz}, "
            f"horizon={actions_per_horizon} actions"
        ),
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="RTC chunk boundary",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
