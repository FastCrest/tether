"""Heuristic difficulty estimate from instruction text.

Per the research sidecar (open question 3): ship the heuristic; flag low
confidence; signal is a sorting input, not load-bearing. Phase 2.5 ML-based
difficulty trained on labeled corpus.
"""
from __future__ import annotations


# Marker words/phrases by difficulty contribution.
PRECISION_MARKERS = ("carefully", "gently", "precisely", "softly", "slowly")
SEQUENCE_MARKERS = ("then", "after", "first", "next", "finally", "before")
MULTI_STEP_MARKERS = (" and ", " then ", " while ", " before ", " after ")


def difficulty_from_instruction(instruction: str) -> tuple[float, float]:
    """Return (difficulty, confidence) from an instruction string.

    Difficulty in [0, 1]: 0 = trivial; 1 = complex multi-step with precision.
    Confidence in [0, 1]: how sure we are this signal is meaningful.
    Long instructions get higher confidence (more text to grade); short
    "pick" instructions get lower confidence.
    """
    if not instruction or not instruction.strip():
        return 0.0, 0.0
    text = instruction.lower()
    word_count = len(instruction.split())

    score = 0.0
    if word_count >= 15:
        score += 0.3
    elif word_count >= 8:
        score += 0.15

    if any(m in text for m in MULTI_STEP_MARKERS):
        score += 0.2

    if any(m in text for m in PRECISION_MARKERS):
        score += 0.2

    if any(m in text for m in SEQUENCE_MARKERS):
        score += 0.3

    score = min(score, 1.0)

    # Confidence scales with word count — more words = more signal.
    if word_count >= 15:
        confidence = 0.8
    elif word_count >= 6:
        confidence = 0.6
    else:
        confidence = 0.3

    return float(score), float(confidence)


__all__ = [
    "PRECISION_MARKERS",
    "SEQUENCE_MARKERS",
    "MULTI_STEP_MARKERS",
    "difficulty_from_instruction",
]
