"""ManiSkill3 task-success evaluation plugin (v0.3 — not yet implemented)."""
from __future__ import annotations

from typing import Any

__all__ = ["run_maniskill"]


def run_maniskill(*, export_dir: str, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError(
        "ManiSkill eval ships in v0.3. Track progress at GOALS.yaml: `maniskill-eval`."
    )
