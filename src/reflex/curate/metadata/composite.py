"""Top-level metadata enrichment — composes all 4 tag sources.

Schema versioned via `taxonomy_version` (per research sidecar open question
1). Phase 2 ML-classifier output coexists with Phase 1 rule-based tags by
emitting `taxonomy_version: ml-v2` on those records.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from reflex.curate.metadata.difficulty import difficulty_from_instruction
from reflex.curate.metadata.language import detect_language
from reflex.curate.metadata.task_classifier import (
    classify_subtype,
    classify_task,
)
from reflex.curate.metadata.trajectory_tags import (
    action_complexity,
    terminal_gripper_state,
)


@dataclass(frozen=True)
class EnrichmentResult:
    """Per-episode enrichment payload. Frozen — computed once per episode."""

    tags: dict[str, dict[str, Any]]
    taxonomy_version: str
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": dict(self.tags),
            "taxonomy_version": self.taxonomy_version,
            "computed_at": self.computed_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tag_with_source(value: Any, *, confidence: float, source: str) -> dict[str, Any]:
    """Wrap a tag value with confidence + source provenance.

    Per the spec, every tag carries provenance so consumers can re-derive
    or compare across versions. Sources used in v1:
        text-keyword-v1     keyword classifier
        text-heuristic-v1   difficulty heuristic
        langdetect-v1       script-block language detector
        action-trajectory-v1 action-derived tags
    """
    return {"value": value, "confidence": float(confidence), "source": source}


def enrich_metadata(
    *,
    instruction: str | None = None,
    actions: np.ndarray | None = None,
    gripper_dim: int | None = None,
) -> EnrichmentResult:
    """Compute the metadata bundle for a single episode.

    Each tag emits a {value, confidence, source} dict. Missing inputs
    silently produce 'unknown' tags rather than failing.
    """
    from reflex.curate.metadata import TAXONOMY_VERSION

    tags: dict[str, dict[str, Any]] = {}

    # Text-based tags
    if instruction is not None:
        task_type, task_conf = classify_task(instruction)
        tags["task_type"] = _tag_with_source(
            task_type, confidence=task_conf, source="text-keyword-v1",
        )
        subtype, subtype_conf = classify_subtype(instruction, task_type)
        tags["task_subtype"] = _tag_with_source(
            subtype, confidence=subtype_conf, source="text-keyword-v1",
        )
        difficulty, diff_conf = difficulty_from_instruction(instruction)
        tags["difficulty"] = _tag_with_source(
            difficulty, confidence=diff_conf, source="text-heuristic-v1",
        )
        lang, lang_conf = detect_language(instruction)
        tags["instruction_language"] = _tag_with_source(
            lang, confidence=lang_conf, source="langdetect-v1",
        )
    else:
        for k in ("task_type", "task_subtype", "difficulty", "instruction_language"):
            tags[k] = _tag_with_source(
                "unknown" if k != "difficulty" else 0.0,
                confidence=0.0,
                source="text-keyword-v1",
            )

    # Action-trajectory tags
    if actions is not None and actions.size > 0:
        gripper, gripper_conf = terminal_gripper_state(actions, gripper_dim=gripper_dim)
        tags["terminal_gripper_state"] = _tag_with_source(
            gripper, confidence=gripper_conf, source="action-trajectory-v1",
        )
        complexity = action_complexity(actions)
        tags["action_range_max"] = _tag_with_source(
            complexity["action_range_max"], confidence=1.0, source="action-trajectory-v1",
        )
        tags["action_variance"] = _tag_with_source(
            complexity["action_variance"], confidence=1.0, source="action-trajectory-v1",
        )
    else:
        tags["terminal_gripper_state"] = _tag_with_source(
            "unknown", confidence=0.0, source="action-trajectory-v1",
        )

    return EnrichmentResult(
        tags=tags,
        taxonomy_version=TAXONOMY_VERSION,
        computed_at=_utc_now_iso(),
    )


def enrich_from_jsonl_rows(rows: list[dict[str, Any]]) -> EnrichmentResult:
    """Convenience: extract instruction + actions from a list of /act event
    rows (uploader entry point) and run enrich_metadata.
    """
    if not rows:
        return enrich_metadata()

    # Instruction: prefer raw text; fall back to hash (which won't yield
    # useful tags, but keeps enrichment from crashing).
    md0 = rows[0].get("metadata", {}) or {}
    instruction = rows[0].get("instruction_raw")
    if not instruction:
        instruction = md0.get("instruction")  # alternative location

    # Actions: flatten chunks across rows.
    flat_actions: list[list[float]] = []
    for r in rows:
        chunk = r.get("action_chunk") or []
        for action in chunk:
            if isinstance(action, list):
                flat_actions.append(action)
    actions_arr = np.asarray(flat_actions, dtype=np.float32) if flat_actions else None

    gripper_dims_raw = md0.get("gripper_dims", ())
    gripper_dim = (
        int(gripper_dims_raw[0])
        if gripper_dims_raw and isinstance(gripper_dims_raw, (list, tuple))
        else None
    )

    return enrich_metadata(
        instruction=instruction,
        actions=actions_arr,
        gripper_dim=gripper_dim,
    )


__all__ = [
    "EnrichmentResult",
    "enrich_from_jsonl_rows",
    "enrich_metadata",
]
