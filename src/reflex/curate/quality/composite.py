"""Composite quality_score() — combines the 4 signals into a [0, 1] score.

Returns a `QualityResult` carrying both the composite score and the per-signal
breakdown so contributors can see why their score is what it is (per
quality-scoring_research.md open question 4).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

QUALITY_VERSION = "rule-v1"


@dataclass(frozen=True)
class QualityWeights:
    """Per-embodiment composition weights. Must sum to 1.0 for the score
    to land in [0, 1]; non-summing weights are normalized at score time."""

    success: float
    smoothness: float
    efficiency: float
    coverage: float

    def __post_init__(self) -> None:
        for name, val in (
            ("success", self.success),
            ("smoothness", self.smoothness),
            ("efficiency", self.efficiency),
            ("coverage", self.coverage),
        ):
            if val < 0:
                raise ValueError(f"{name} weight must be non-negative, got {val}")

    @property
    def total(self) -> float:
        return self.success + self.smoothness + self.efficiency + self.coverage

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QualityResult:
    """Output of quality_score(). Carries the composite + per-signal breakdown
    + version + timestamp for embedding into the parquet record."""

    quality_score: float
    components: dict[str, float]
    weights: dict[str, float]
    quality_version: str
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "quality_components": dict(self.components),
            "quality_weights": dict(self.weights),
            "quality_version": self.quality_version,
            "computed_at": self.computed_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quality_score(
    *,
    actions: np.ndarray,
    chunk_ids: np.ndarray | None = None,
    success_flag: bool | None = None,
    embodiment: str = "*",
    task: str = "*",
    execute_hz: float = 50.0,
    gripper_dims: Sequence[int] = (),
    episode_phash: str | None = None,
    recent_phashes: Sequence[str] = (),
    weights: QualityWeights | None = None,
) -> QualityResult:
    """Compute a per-episode quality score in [0, 1].

    Composes 4 signals (success / smoothness / efficiency / coverage) with
    per-embodiment weights. See per_embodiment.py for default weights.

    Args:
        actions: shape (T, action_dim) — recorded action chunks (emitted by
            policy; per F3.1 these are NOT the executed actions).
        chunk_ids: shape (T,) — chunk_id per step. When provided, smoothness
            is computed within-chunk and averaged. Optional; falls back to
            single-pass smoothness when None.
        success_flag: harness-recorded success ground truth. None for
            production deployments without an evaluator.
        embodiment: slug for baseline + weight lookup.
        task: task type slug (pick/place/pour/insert/...) for baseline lookup.
        execute_hz: control loop frequency for velocity-normalization (F3.4).
        gripper_dims: indices of gripper channels to skip in smoothness (F3.5).
        episode_phash: perceptual hash of the episode's first frame for
            coverage signal. None if image-hash unavailable.
        recent_phashes: phashes of recent episodes for diversity comparison.
        weights: override the per-embodiment defaults.
    """
    from reflex.curate.quality.per_embodiment import weights_for
    from reflex.curate.quality.signals import (
        coverage_signal,
        efficiency_signal,
        policy_smoothness_signal,
        success_signal,
    )

    w = weights or weights_for(embodiment)

    # Compute per-signal values.
    success = success_signal(
        success_flag=success_flag,
        actions=actions,
        gripper_dims=gripper_dims,
    )
    smoothness = policy_smoothness_signal(
        actions=actions,
        chunk_ids=chunk_ids,
        execute_hz=execute_hz,
        gripper_dims=gripper_dims,
    )
    efficiency = efficiency_signal(
        episode_steps=int(actions.shape[0]) if actions.size else 0,
        embodiment=embodiment,
        task=task,
    )
    coverage = coverage_signal(
        episode_phash=episode_phash,
        recent_phashes=recent_phashes,
    )

    components = {
        "success": float(success),
        "smoothness": float(smoothness),
        "efficiency": float(efficiency),
        "coverage": float(coverage),
    }

    # Normalize weights if they don't sum to 1.0 (defensive — config can
    # specify any non-negative values).
    total = w.total
    if total <= 0:
        # Degenerate config — fall back to equal weights.
        w_dict = {"success": 0.25, "smoothness": 0.25, "efficiency": 0.25, "coverage": 0.25}
    else:
        w_dict = {
            "success": w.success / total,
            "smoothness": w.smoothness / total,
            "efficiency": w.efficiency / total,
            "coverage": w.coverage / total,
        }

    score = sum(components[k] * w_dict[k] for k in components)
    score = float(np.clip(score, 0.0, 1.0))

    return QualityResult(
        quality_score=score,
        components=components,
        weights=w_dict,
        quality_version=QUALITY_VERSION,
        computed_at=_utc_now_iso(),
    )


def quality_from_jsonl_rows(
    rows: list[dict[str, Any]],
    *,
    recent_phashes: Sequence[str] = (),
) -> QualityResult:
    """Compute quality from a list of /act event rows (uploader entry point).

    Each row is the dict written by ProDataCollector / FreeContributorCollector
    (one row per /act call, with a flattened action_chunk + episode_id +
    metadata dict containing chunk_id + contributor_id + tier + optional
    embodiment + execute_hz).

    Best-effort field extraction; falls back to defaults when the recorder
    didn't tag context. As recordings include richer metadata over time,
    scores improve.
    """
    if not rows:
        # Empty episode → degenerate result.
        return QualityResult(
            quality_score=0.0,
            components={"success": 0.0, "smoothness": 0.0, "efficiency": 0.0, "coverage": 0.0},
            weights=DEFAULT_WEIGHTS_DICT,
            quality_version=QUALITY_VERSION,
            computed_at=_utc_now_iso(),
        )

    # Flatten action_chunks across rows, tag each step with its row's chunk_id
    # so policy_smoothness can compute within-chunk.
    flat_actions: list[list[float]] = []
    flat_chunks: list[int] = []
    for row in rows:
        chunk_id = int(row.get("metadata", {}).get("chunk_id", 0) or 0)
        chunk = row.get("action_chunk") or []
        for action in chunk:
            if isinstance(action, list):
                flat_actions.append(action)
                flat_chunks.append(chunk_id)
    if not flat_actions:
        return QualityResult(
            quality_score=0.0,
            components={"success": 0.0, "smoothness": 0.0, "efficiency": 0.0, "coverage": 0.0},
            weights=DEFAULT_WEIGHTS_DICT,
            quality_version=QUALITY_VERSION,
            computed_at=_utc_now_iso(),
        )
    actions_arr = np.asarray(flat_actions, dtype=np.float32)
    chunk_ids_arr = np.asarray(flat_chunks, dtype=np.int32)

    # Best-effort context extraction. When the recorder hasn't tagged these
    # (typical for fresh JSONL streams without per-embodiment metadata
    # injection — see free_collector.py), we fall back to defaults. The
    # quality scorer ranks correctly even with embodiment="*" + execute_hz
    # 50.0 — they widen the per-embodiment thresholds, not break them.
    md0 = rows[0].get("metadata", {}) or {}
    embodiment = str(md0.get("embodiment", "*"))
    task = str(md0.get("task", "*"))
    execute_hz = float(md0.get("execute_hz", 50.0))
    gripper_dims_raw = md0.get("gripper_dims", ())
    gripper_dims = tuple(int(x) for x in gripper_dims_raw) if gripper_dims_raw else ()
    success_flag_raw = md0.get("success_flag")
    if success_flag_raw is None:
        success_flag = None
    else:
        success_flag = bool(success_flag_raw)
    episode_phash = md0.get("episode_phash")

    return quality_score(
        actions=actions_arr,
        chunk_ids=chunk_ids_arr,
        success_flag=success_flag,
        embodiment=embodiment,
        task=task,
        execute_hz=execute_hz,
        gripper_dims=gripper_dims,
        episode_phash=episode_phash if isinstance(episode_phash, str) else None,
        recent_phashes=tuple(recent_phashes),
    )


# Pre-computed default weights dict for the empty-episode short-circuit
# (avoids importing per_embodiment lazily inside that path).
DEFAULT_WEIGHTS_DICT = {
    "success": 0.4, "smoothness": 0.3, "efficiency": 0.2, "coverage": 0.1,
}


__all__ = [
    "QUALITY_VERSION",
    "DEFAULT_WEIGHTS_DICT",
    "QualityResult",
    "QualityWeights",
    "quality_from_jsonl_rows",
    "quality_score",
]
