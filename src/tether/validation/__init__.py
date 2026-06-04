"""Dataset validation — pre-flight checks for LeRobot training datasets.

`tether validate-dataset <path>` parses a LeRobot v3.0 dataset, runs a registry
of falsifiable checks (schema completeness, shape consistency, action-finite,
embodiment match, episode count, timing monotonicity), and emits a structured
report. Catches the common "trained 2 hours, then crashed because action_dim
mismatch" footgun BEFORE customers spend Modal credits on a doomed run.

Pairs with `tether doctor` (which validates model + runtime). Together they
cover both halves of "will this dataset → model → serve work?"
"""

from tether.validation.dataset_checks import (
    Decision,
    CheckResult,
    DatasetContext,
    register_dataset_check,
    REGISTERED_CHECKS,
    run_all_checks,
    format_human,
    format_json,
    overall_decision,
)

__all__ = [
    "Decision",
    "CheckResult",
    "DatasetContext",
    "register_dataset_check",
    "REGISTERED_CHECKS",
    "run_all_checks",
    "format_human",
    "format_json",
    "overall_decision",
]
