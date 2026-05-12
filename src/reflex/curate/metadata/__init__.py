"""Auto-tagging episode metadata for the Curate wedge.

Per `_curation/metadata-enrichment.md` + research sidecar:
- Text-based tags (Phase 1): task_type / task_subtype / instruction_language
  / heuristic difficulty
- Action-trajectory tags (Phase 1): action_smoothness, terminal_gripper_state,
  action_complexity (re-uses signals already computed by quality scoring)
- Image-based tags (Phase 2 — deferred per ADR): scene type, lighting,
  object types, background complexity (requires VLM call)

Powers the Tier-2 "Curated Task Datasets" product — labs filter by task
type / language / difficulty when buying datasets.

Submodules:
    task_classifier    — keyword-based task type detection
    difficulty         — heuristic difficulty estimate from instruction
    language           — instruction language detection
    trajectory_tags    — action-derived tags
    composite          — top-level enrich() orchestrator
"""
from __future__ import annotations

TAXONOMY_VERSION = "rule-v1"

from reflex.curate.metadata.composite import (
    EnrichmentResult,
    enrich_from_jsonl_rows,
    enrich_metadata,
)
from reflex.curate.metadata.difficulty import difficulty_from_instruction
from reflex.curate.metadata.language import detect_language
from reflex.curate.metadata.task_classifier import (
    TASK_KEYWORDS,
    TASK_TYPES,
    classify_task,
    classify_subtype,
)
from reflex.curate.metadata.trajectory_tags import (
    action_complexity,
    terminal_gripper_state,
)

__all__ = [
    "TAXONOMY_VERSION",
    "TASK_KEYWORDS",
    "TASK_TYPES",
    "EnrichmentResult",
    "action_complexity",
    "classify_task",
    "classify_subtype",
    "detect_language",
    "difficulty_from_instruction",
    "enrich_from_jsonl_rows",
    "enrich_metadata",
    "terminal_gripper_state",
]
