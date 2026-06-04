"""JSON envelope for `tether eval` output — schema v1 LOCKED.

Per ADR 2026-04-25-eval-as-a-service-architecture decision #3:
schema is LOCKED at v1; Phase 2 evolution is additive-only.

Customers grep on these fields in CI scripts; renaming = breakage.
The shape mirrors src/tether/bench/report.py (env block conventions)
for cross-verb consistency.

Top-level fields (all REQUIRED at v1):
- schema_version: int = 1
- tether_version: str
- suite: str
- runtime: str
- tasks: list[str]
- num_episodes_per_task: int
- seed: int
- started_at, finished_at, wall_clock_s
- aggregate: {success_rate, n_success, n_total}
- results: list[per-task: {task_id, n_success, n_total, success_rate, episodes}]
- episodes: flat list[per-episode: {task_id, episode_index, success,
  terminal_reason, wall_clock_s, n_steps, video_path, error_message}]
- cost: nested CostEstimate dict (per cost_model.py)
- modal: nested {image_digest, provider} block (None when runtime=local)
- env: nested EvalEnvironment dict (git_sha, gpu, python_version, ...)
- video_paths: list[str] (paths to per-episode MP4s when --video set)
- notes: list[str] (free-form audit trail)
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tether.eval.cost_model import CostEstimate
from tether.eval.libero import EvalReport

logger = logging.getLogger(__name__)


# Schema version — bumped only on breaking changes; additive evolution
# (new optional fields) does NOT bump.
EVAL_ENVELOPE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalEnvironment:
    """Reproducibility envelope mirroring bench/report.py BenchEnvironment.

    Captures enough state to re-run + cross-check an eval. Frozen so
    the envelope can't drift after creation.
    """

    timestamp_utc: str
    tether_version: str
    git_sha: str
    git_dirty: bool
    python_version: str
    platform: str  # e.g. "Darwin-25.3.0-arm64"
    export_dir: str
    onnx_files: list[dict]  # [{name, sha256, bytes}]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalEnvelope:
    """Full JSON envelope wrapping EvalReport + cost + modal + env.

    schema_version is locked at v1; mutating the field name or removing
    a required field requires bumping to v2 (breaking change).
    """

    schema_version: int
    tether_version: str
    suite: str
    runtime: str
    seed: int
    started_at: str
    finished_at: str
    wall_clock_s: float
    tasks: tuple[str, ...]
    num_episodes_per_task: int
    aggregate: dict
    results: tuple[dict, ...]  # per-task dicts
    episodes: tuple[dict, ...]  # flattened per-episode dicts
    cost: dict  # CostEstimate.to_dict()
    modal: dict | None  # {image_digest, provider} when runtime=modal else None
    env: dict  # EvalEnvironment.to_dict()
    video_paths: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EVAL_ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EVAL_ENVELOPE_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if self.num_episodes_per_task < 0:
            raise ValueError(
                f"num_episodes_per_task must be >= 0, got "
                f"{self.num_episodes_per_task}"
            )
        if self.wall_clock_s < 0:
            raise ValueError(
                f"wall_clock_s must be >= 0, got {self.wall_clock_s}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tether_version": self.tether_version,
            "suite": self.suite,
            "runtime": self.runtime,
            "seed": self.seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_clock_s": self.wall_clock_s,
            "tasks": list(self.tasks),
            "num_episodes_per_task": self.num_episodes_per_task,
            "aggregate": self.aggregate,
            "results": list(self.results),
            "episodes": list(self.episodes),
            "cost": self.cost,
            "modal": self.modal,
            "env": self.env,
            "video_paths": list(self.video_paths),
            "notes": list(self.notes),
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the envelope as JSON to `path`. Returns the resolved Path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return out


def build_envelope(
    *,
    report: EvalReport,
    cost: CostEstimate,
    env: EvalEnvironment,
    num_episodes_per_task: int,
    modal_block: dict | None = None,
    video_paths: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> EvalEnvelope:
    """Compose an EvalEnvelope from the per-Day-1 EvalReport + Day 4
    cost + env blocks.

    Phase 1 wiring: cli.py calls this after LiberoSuite.run() returns,
    then writes the envelope to <output>/report.json.
    """
    # Per-task block (mirrors EvalReport.results)
    results = tuple(
        {
            "task_id": r.task_id,
            "n_success": r.n_success,
            "n_total": r.n_total,
            "success_rate": r.success_rate,
        }
        for r in report.results
    )
    # Flat per-episode list
    episodes = tuple(
        {
            "task_id": ep.task_id,
            "episode_index": ep.episode_index,
            "success": ep.success,
            "terminal_reason": ep.terminal_reason,
            "wall_clock_s": ep.wall_clock_s,
            "n_steps": ep.n_steps,
            "video_path": ep.video_path,
            "error_message": ep.error_message,
        }
        for r in report.results
        for ep in r.episodes
    )
    aggregate = {
        "success_rate": report.aggregate_success_rate,
        "n_success": report.aggregate_n_success,
        "n_total": report.aggregate_n_total,
    }

    return EvalEnvelope(
        schema_version=EVAL_ENVELOPE_SCHEMA_VERSION,
        tether_version=env.tether_version,
        suite=report.suite,
        runtime=report.runtime,
        seed=report.seed,
        started_at=report.started_at,
        finished_at=report.finished_at,
        wall_clock_s=report.wall_clock_s,
        tasks=report.tasks,
        num_episodes_per_task=num_episodes_per_task,
        aggregate=aggregate,
        results=results,
        episodes=episodes,
        cost=cost.to_dict(),
        modal=modal_block,
        env=env.to_dict(),
        video_paths=video_paths,
        notes=notes,
    )


def capture_environment(
    *,
    export_dir: str | Path,
    repo_dir: str | Path | None = None,
) -> EvalEnvironment:
    """Capture env block at eval-run time. Mirrors bench/report.py shape
    so the two verbs render comparable envelopes.
    """
    export_path = Path(export_dir)
    repo_path = Path(repo_dir) if repo_dir else Path.cwd()

    git_sha, git_dirty = _git_info(repo_path)

    return EvalEnvironment(
        timestamp_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        tether_version=_tether_version(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        export_dir=str(export_path.resolve()),
        onnx_files=_onnx_file_summary(export_path),
    )


def _git_info(repo_dir: Path) -> tuple[str, bool]:
    """Returns (sha, is_dirty). Empty sha if not in a git repo."""
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
            capture_output=True, text=True, timeout=5,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo_dir),
            capture_output=True, text=True, timeout=5,
        )
        sha = sha_result.stdout.strip()
        return sha[:12] if sha else "", bool(status_result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "", False


def _tether_version() -> str:
    try:
        from importlib.metadata import version
        return version("tether-vla")
    except Exception:  # noqa: BLE001
        return "0.1.0+dev"


def _onnx_file_summary(export_dir: Path) -> list[dict]:
    """Hash + size of each *.onnx file under the export dir. Bounded:
    only top-level glob, no recursive walk (avoids huge weights dirs)."""
    if not export_dir.exists():
        return []
    out: list[dict] = []
    for p in sorted(export_dir.glob("*.onnx")):
        try:
            data = p.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            out.append({"name": p.name, "sha256": sha, "bytes": len(data)})
        except OSError:
            # File disappeared mid-capture — skip rather than crash
            continue
    return out


__all__ = [
    "EVAL_ENVELOPE_SCHEMA_VERSION",
    "EvalEnvelope",
    "EvalEnvironment",
    "build_envelope",
    "capture_environment",
]
