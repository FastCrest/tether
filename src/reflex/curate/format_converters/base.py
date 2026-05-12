"""FormatConverter abstract base + common helpers.

Every concrete converter takes Reflex JSONL traces (one line per /act
request+response) and produces a target-format directory. The base class
defines the per-episode iteration + shared statistics; subclasses
implement the format-specific output.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass
class ConversionResult:
    """Outcome of a converter.convert() call."""

    output_dir: str
    format: str
    episode_count: int = 0
    step_count: int = 0
    bytes_written: int = 0
    skipped_episodes: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    warnings: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "format": self.format,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "bytes_written": self.bytes_written,
            "skipped_episodes": self.skipped_episodes,
            "skipped_reasons": dict(self.skipped_reasons),
            "warnings": list(self.warnings),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each non-empty JSONL row as a dict."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _group_by_episode(
    rows: Iterator[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group rows by episode_id (typically the recorder's session_id)."""
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ep_id = row.get("episode_id") or "anon"
        by_ep[ep_id].append(row)
    return dict(by_ep)


class FormatConverter:
    """Abstract base. Subclasses override `convert()`.

    Common protocol:
      - Input: list of JSONL paths or a single directory of JSONLs
      - Output: target-format directory at `output_dir`
      - Result: ConversionResult with statistics
    """

    FORMAT_NAME: str = "abstract"

    def convert(
        self,
        *,
        input_jsonl: str | Path | list[str | Path],
        output_dir: str | Path,
        min_quality: float | None = None,
        canonical_only: bool = False,
    ) -> ConversionResult:
        """Convert input JSONL(s) to the target format directory.

        Args:
            input_jsonl: path to a single JSONL file OR a list of JSONL paths
                OR a directory (all .jsonl files within are read).
            output_dir: target output directory. Created if absent.
            min_quality: drop episodes with quality_score below threshold.
                When None, all episodes pass.
            canonical_only: drop episodes where is_canonical=False (dedup
                cluster non-canonicals filtered out).
        """
        raise NotImplementedError("subclasses must implement convert()")

    # ─── shared helpers exposed to subclasses ──────────────────────────────

    @staticmethod
    def _resolve_inputs(input_jsonl: str | Path | list[str | Path]) -> list[Path]:
        if isinstance(input_jsonl, (str, Path)):
            p = Path(input_jsonl).expanduser()
            if p.is_dir():
                return sorted(p.glob("*.jsonl"))
            return [p]
        return [Path(x).expanduser() for x in input_jsonl]

    @staticmethod
    def _filter_episode(
        rows: list[dict[str, Any]],
        *,
        min_quality: float | None = None,
        canonical_only: bool = False,
    ) -> tuple[bool, str]:
        """Return (should_include, reason_when_skipped)."""
        if not rows:
            return False, "empty_episode"
        md0 = rows[0].get("metadata", {}) or {}
        if min_quality is not None:
            qs = float(md0.get("quality_score") or 0.0)
            if qs < min_quality:
                return False, f"quality_below_{min_quality}"
        if canonical_only:
            is_canonical = md0.get("is_canonical")
            if is_canonical is False:
                return False, "non_canonical_dedup_cluster_member"
        return True, ""

    @staticmethod
    def _flatten_actions_and_steps(
        rows: list[dict[str, Any]],
    ) -> tuple[list[list[float]], list[list[float] | None]]:
        """Flatten rows' action_chunks into per-step actions + per-step states.

        Returns (actions, states). Each action is a list[float]; each state
        is either a list[float] or None (when state_vec was absent for that row).
        Ordering: walks rows in input order; within each row, walks the chunk
        in chunk-index order. State for each step in the chunk uses the row's
        state_vec (replicated across all steps in the chunk — one state per
        chunk in the recorder).
        """
        actions: list[list[float]] = []
        states: list[list[float] | None] = []
        for row in rows:
            chunk = row.get("action_chunk") or []
            state = row.get("state_vec")
            for action in chunk:
                if isinstance(action, list):
                    actions.append(action)
                    states.append(state if isinstance(state, list) else None)
        return actions, states


__all__ = [
    "ConversionResult",
    "FormatConverter",
    "_group_by_episode",
    "_iter_jsonl",
    "_utc_now_iso",
]
