"""Task-success evaluation plugins for `tether bench --benchmark`.

Each plugin wraps a simulation benchmark (LIBERO, SimplerEnv, ManiSkill).
Heavy sim dependencies (MuJoCo, robosuite, LIBERO) live behind the `eval` extra:
    pip install 'tether[eval]'

The plugin framework is deliberately thin — each benchmark is one function
that takes an export_dir and returns a standardized result dict.
"""
from __future__ import annotations

from typing import Any

__all__ = ["run_task_benchmark"]


def run_task_benchmark(
    benchmark: str,
    *,
    export_dir: str,
    episodes_per_task: int = 10,
    device: str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch to the named benchmark plugin.

    Returns a dict with:
        benchmark: name
        success_rate: 0.0-1.0 (fraction of episodes that succeeded)
        episodes_completed: total episodes run
        per_task: [{task, success_rate, num_episodes}, ...]
        duration_s: wall-clock seconds
    """
    if benchmark.startswith("libero"):
        raise ValueError(
            "LIBERO was archived on 2026-04-17. Tether's product wedge is "
            "deployment parity + latency, not sim benchmarking. "
            "Archived scripts live at archive/scripts/ if you want to resurrect them."
        )
    if benchmark == "simpler":
        from tether.eval.simpler import run_simpler
        return run_simpler(export_dir=export_dir, **kwargs)
    if benchmark == "maniskill":
        from tether.eval.maniskill import run_maniskill
        return run_maniskill(export_dir=export_dir, **kwargs)
    raise ValueError(f"Unknown benchmark: {benchmark!r}")
