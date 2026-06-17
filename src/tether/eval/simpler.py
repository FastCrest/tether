"""SimplerEnv task-success evaluation plugin (v0.3 — not yet implemented)."""
from __future__ import annotations

from typing import Any

__all__ = ["run_simpler"]


def run_simpler(*, export_dir: str, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError(
        "SimplerEnv eval ships in v0.3. Track progress at GOALS.yaml: `simpler-eval`."
    )
