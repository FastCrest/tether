"""Check 6 — Gripper config sanity (LeRobot #2210, #2531).

Validates gripper.component_idx is inside action space, close_threshold
is in [0, 1], and the inverted flag is consistent with embodiment defaults.
Inverted gripper bug is one of the most-reported "robot does the opposite"
issues — silent failure mode worth catching pre-deploy.
"""
from __future__ import annotations

from . import Check, CheckResult, register

CHECK_ID = "check_gripper"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2531"

# Embodiments where inverted=True is suspicious (defaults are non-inverted).
# Customers can override and ignore the warn if they really mean it.
_NON_INVERTED_DEFAULTS = {"franka", "ur5"}


def _run(embodiment_name: str = "custom", **kwargs) -> CheckResult:
    if embodiment_name == "custom":
        return CheckResult(
            check_id=CHECK_ID,
            name="Gripper config",
            status="skip",
            expected="--embodiment <preset> for gripper config validation",
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
            name="Gripper config",
            status="fail",
            expected=f"embodiment preset {embodiment_name!r} loads",
            actual=f"load failed: {e}",
            remediation="See docs/embodiment_schema.md for the preset list.",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    gripper = cfg.gripper
    action_dim = cfg.action_dim
    idx = gripper.get("component_idx", -1)
    threshold = gripper.get("close_threshold", -1.0)
    inverted = gripper.get("inverted", False)

    # component_idx within action vector
    if not 0 <= idx < action_dim:
        return CheckResult(
            check_id=CHECK_ID,
            name="Gripper config",
            status="fail",
            expected=f"gripper.component_idx in [0, {action_dim})",
            actual=f"component_idx={idx}, action_dim={action_dim}",
            remediation=(
                f"Fix {embodiment_name}.json: gripper.component_idx must point "
                f"at a valid action dimension (0..{action_dim - 1})."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # close_threshold in [0, 1] (normalized action range)
    if not 0.0 <= threshold <= 1.0:
        return CheckResult(
            check_id=CHECK_ID,
            name="Gripper config",
            status="fail",
            expected="gripper.close_threshold in [0.0, 1.0]",
            actual=f"close_threshold={threshold}",
            remediation=(
                f"close_threshold must be in [0, 1] (normalized gripper action range). "
                f"Below 0 or above 1 means 'never close' or 'always close' — neither is intentional."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Suspicious inverted flag for known non-inverted defaults
    if inverted and embodiment_name in _NON_INVERTED_DEFAULTS:
        return CheckResult(
            check_id=CHECK_ID,
            name="Gripper config",
            status="warn",
            expected=(
                f"{embodiment_name} grippers default to inverted=False (high → close)"
            ),
            actual=f"{embodiment_name}.json has inverted=True",
            remediation=(
                f"Inverted gripper for {embodiment_name} is unusual — the default RG2/RG6 "
                f"is non-inverted. Verify your hardware actually inverts (high values "
                f"= open). Ignore if intentional. See "
                f"https://github.com/huggingface/lerobot/issues/2210 for the most-common "
                f"version of this bug."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Gripper config",
        status="pass",
        expected=f"gripper at idx {idx} of {action_dim}-D action, threshold {threshold}, inverted={inverted}",
        actual=f"all gripper fields valid for {embodiment_name}",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="Gripper config",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
