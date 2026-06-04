"""Check 5 — Action denormalization (LeRobot #414, #2210).

Validates that the embodiment config's normalization stats match the
action_space dim and contain no NaN/Inf. Catches the silent class of
bug where mean/std arrays are wrong-length and the runtime produces
zero or NaN actions without a clear error.
"""
from __future__ import annotations

import math

from . import Check, CheckResult, register

CHECK_ID = "check_action_denorm"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2210"


def _has_bad_value(arr) -> bool:
    return any(math.isnan(x) or math.isinf(x) for x in arr)


def _run(embodiment_name: str = "custom", **kwargs) -> CheckResult:
    if embodiment_name == "custom":
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="skip",
            expected="--embodiment <preset> for normalization-stats cross-check",
            actual="embodiment=custom — no preset to validate against",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    try:
        from tether.embodiments import EmbodimentConfig
        cfg = EmbodimentConfig.load_preset(embodiment_name)
    except (ValueError, FileNotFoundError) as e:
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="fail",
            expected=f"embodiment preset {embodiment_name!r} loads",
            actual=f"load failed: {e}",
            remediation=(
                f"Use a shipped preset (franka/so100/ur5) or pass "
                f"--custom-embodiment-config <path>. See docs/embodiment_schema.md."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    action_dim = cfg.action_dim
    norm = cfg.normalization
    mean = norm.get("mean_action", [])
    std = norm.get("std_action", [])

    # Length checks
    if len(mean) != action_dim:
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="fail",
            expected=f"mean_action has {action_dim} elements (= action_space.dim)",
            actual=f"mean_action has {len(mean)} elements",
            remediation=(
                f"Fix {embodiment_name}.json: normalization.mean_action must be "
                f"length {action_dim}. Mismatched length silently produces zero/NaN "
                f"actions in the runtime."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if len(std) != action_dim:
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="fail",
            expected=f"std_action has {action_dim} elements",
            actual=f"std_action has {len(std)} elements",
            remediation=(
                f"Fix {embodiment_name}.json: normalization.std_action must be "
                f"length {action_dim}."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # NaN/Inf checks
    if _has_bad_value(mean):
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="fail",
            expected="no NaN/Inf in mean_action",
            actual=f"mean_action contains NaN or Inf: {mean}",
            remediation=(
                "Replace NaN/Inf values in normalization.mean_action with finite "
                "floats. Likely cause: stale calibration that included a divide-by-zero."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )
    if _has_bad_value(std):
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="fail",
            expected="no NaN/Inf in std_action",
            actual=f"std_action contains NaN or Inf: {std}",
            remediation="Replace NaN/Inf in normalization.std_action with finite floats.",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # std must be positive (already enforced by schema's exclusiveMinimum:0,
    # but doctor cross-checks for runtime safety)
    if any(s <= 0 for s in std):
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="fail",
            expected="all std_action values > 0",
            actual=f"std_action contains non-positive values: {std}",
            remediation="std_action values must be > 0 (used as divisor in denorm).",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Sanity warn: huge std suggests miscalibration
    if any(s > 10.0 for s in std):
        return CheckResult(
            check_id=CHECK_ID,
            name="Action denormalization",
            status="warn",
            expected="std_action values < 10.0 (typical robot joint range)",
            actual=f"std_action has values > 10: {[s for s in std if s > 10]}",
            remediation=(
                "std_action > 10 is unusual — verify the calibration was run on "
                "data in the right units (radians vs degrees, normalized vs raw)."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Action denormalization",
        status="pass",
        expected=f"mean+std arrays match action_dim={action_dim}, all finite, std>0",
        actual=f"validated {action_dim}-D normalization for {embodiment_name}",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="Action denormalization",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
