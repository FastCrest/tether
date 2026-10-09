"""CLI orchestration for local LeRobot ALOHA evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from tether.eval.aloha import AlohaEvalConfig, run_aloha_eval


def aloha_eval_command(
    *, policy: str, revision: str, task: str, device: str, seed: int,
    num_episodes: int, output: str, python: str, timeout_s: float,
) -> None:
    console = Console()
    try:
        config = AlohaEvalConfig(
            policy=policy, revision=revision or None, task=task, device=device,
            seed=seed, num_episodes=num_episodes, output=Path(output),
            python=python or sys.executable, timeout_s=timeout_s,
        )
        console.print(f"Tether Eval: ALOHA / {task}, {num_episodes} episodes, seed {seed}, {device}, local ($0 compute)")
        report = run_aloha_eval(config)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        Console(stderr=True).print(f"[red]ALOHA evaluation failed: {detail}[/red]")
        raise typer.Exit(1) from exc
    interval = report["confidence_interval"]
    console.print(
        f"Success: {report['n_success']}/{report['n_total']} = {report['success_rate']:.1%} "
        f"(Wilson 95% CI {interval['lower']:.1%}-{interval['upper']:.1%})"
    )
    console.print(f"Results: {output}/report.json; receipt: {output}/case-identity.json")
