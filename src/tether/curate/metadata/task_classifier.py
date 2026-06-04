"""Text-based task-type classifier.

Per the research sidecar (Lens 1 finding): no industry standard task
taxonomy exists. We ship a 24-verb controlled vocabulary picked to overlap
with the common-core verbs across RT-X / DROID / LeRobot / AgiBotWorld /
LIBERO. v1 is keyword-match only; LLM fallback (DistilBERT) deferred to
Phase 1.5 if keyword miss rate exceeds 20%.
"""
from __future__ import annotations

from typing import Mapping


# 24 verbs + unknown. Ordered loosely by frequency-of-occurrence across the
# 5 surveyed datasets so the keyword classifier picks the most common
# matching action when an instruction has multiple verbs.
TASK_TYPES: tuple[str, ...] = (
    "pick", "place", "push", "pull", "open", "close",
    "pour", "insert", "remove", "rotate", "press", "flip",
    "stack", "unstack", "fold", "wipe", "sweep", "cut",
    "slide", "lift", "lower", "attach", "detach", "reach",
    "navigate", "manipulate", "unknown",
)

# Per-verb keyword cluster. Order within each cluster matters: more specific
# keywords first so "pick up" matches "pick" before "up" matches anything else.
TASK_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "pick":     ("pick up", "pick", "grab", "grasp", "lift up", "take"),
    "place":    ("place", "put down", "put", "set down", "drop", "deposit"),
    "push":     ("push", "shove", "press into"),
    "pull":     ("pull", "drag", "draw"),
    "open":     ("open", "unlock", "unseal"),
    "close":    ("close", "shut", "seal", "lock"),
    "pour":     ("pour", "fill", "transfer liquid"),
    "insert":   ("insert", "plug in", "plug", "slot"),
    "remove":   ("remove", "extract", "unplug", "take out"),
    "rotate":   ("rotate", "turn", "spin"),
    "press":    ("press", "depress", "activate"),
    "flip":     ("flip", "invert", "overturn"),
    "stack":    ("stack on", "stack"),
    "unstack":  ("unstack", "destack"),
    "fold":     ("fold", "crease", "origami"),
    "wipe":     ("wipe", "scrub", "clean"),
    "sweep":    ("sweep", "gather"),
    "cut":      ("cut", "slice", "chop"),
    "slide":    ("slide"),
    "lift":     ("lift up", "lift", "raise", "elevate"),
    "lower":    ("lower", "descend"),
    "attach":   ("attach", "fasten", "connect"),
    "detach":   ("detach", "unfasten", "disconnect"),
    "reach":    ("reach for", "reach"),
    "navigate": ("navigate", "go to", "drive"),
    "manipulate": ("manipulate", "operate", "use"),
}

# Subtype hints — keyword → subtype label. Per the research sidecar,
# subtypes are best-effort; default "unknown" when no clear match.
SUBTYPE_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "block":      ("block", "cube", "brick"),
    "cylinder":   ("cylinder", "cup", "bottle", "can"),
    "sphere":     ("ball", "sphere", "orb"),
    "soft_object": ("cloth", "rag", "fabric", "towel", "shirt", "pillow"),
    "tool":       ("tool", "screwdriver", "hammer", "wrench"),
    "food":       ("apple", "fruit", "food", "vegetable", "banana", "carrot"),
    "drawer":     ("drawer", "cabinet"),
    "in_box":     ("in box", "into box", "in the box"),
    "in_basket":  ("in basket", "in the basket"),
    "on_shelf":   ("on shelf", "on the shelf"),
}


def classify_task(instruction: str) -> tuple[str, float]:
    """Return (task_type, confidence) from an instruction string.

    Confidence semantics:
        1.0 — single-task match, exact keyword
        0.7 — multiple matches, picked first action-verb in instruction order
        0.0 — no keyword matched (task_type='unknown')
    """
    if not instruction or not instruction.strip():
        return "unknown", 0.0
    text = instruction.lower()

    # Find the earliest position of a keyword for each task type.
    matches: list[tuple[int, str]] = []
    for task_type, keywords in TASK_KEYWORDS.items():
        if isinstance(keywords, str):
            # Tolerate single-string clusters; per-spec we use tuples but a
            # bare string in the dict still works.
            keywords = (keywords,)
        for kw in keywords:
            pos = text.find(kw)
            if pos >= 0:
                matches.append((pos, task_type))
                break  # don't double-count within the same task_type

    if not matches:
        return "unknown", 0.0

    # Pick the first action-verb in instruction order.
    matches.sort(key=lambda pair: pair[0])
    chosen = matches[0][1]
    confidence = 1.0 if len(matches) == 1 else 0.7
    return chosen, confidence


def classify_subtype(instruction: str, task_type: str) -> tuple[str, float]:
    """Return (subtype, confidence) given a known task_type.

    Best-effort match against SUBTYPE_KEYWORDS; falls back to 'unknown' with
    zero confidence when no match.
    """
    if not instruction or not instruction.strip() or task_type == "unknown":
        return "unknown", 0.0
    text = instruction.lower()
    for subtype, keywords in SUBTYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return subtype, 0.8
    return "unknown", 0.0


__all__ = [
    "SUBTYPE_KEYWORDS",
    "TASK_KEYWORDS",
    "TASK_TYPES",
    "classify_subtype",
    "classify_task",
]
