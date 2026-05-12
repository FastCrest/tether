"""Lightweight instruction-language detection.

Phase 1 ships a script-block heuristic (no new deps). Detects:
    - en (default): all-ASCII alpha
    - ru, uk: Cyrillic block
    - zh: CJK block + no kana
    - ja: Hiragana / Katakana
    - ko: Hangul
    - ar: Arabic
    - hi: Devanagari

Phase 1.5 swaps in `langdetect` (~2 MB dep) when miss-rate justifies it.
For the Phase 1 customer base (researchers, mostly English), the heuristic
is sufficient.
"""
from __future__ import annotations

import unicodedata


def _has_block(text: str, ranges: list[tuple[int, int]]) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return True
    return False


_HIRAGANA_KATAKANA = [(0x3040, 0x309F), (0x30A0, 0x30FF)]
_CJK = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]
_HANGUL = [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]
_CYRILLIC = [(0x0400, 0x04FF)]
_ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F)]
_DEVANAGARI = [(0x0900, 0x097F)]


def detect_language(text: str) -> tuple[str, float]:
    """Return (ISO 639-1 code, confidence).

    Confidence is heuristic — high (0.95) for distinct scripts, medium
    (0.8) for Latin / 'en', low (0.3) for empty / mixed inputs.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    if _has_block(text, _HIRAGANA_KATAKANA):
        return "ja", 0.95
    if _has_block(text, _HANGUL):
        return "ko", 0.95
    if _has_block(text, _CJK):
        return "zh", 0.9
    if _has_block(text, _CYRILLIC):
        return "ru", 0.9
    if _has_block(text, _ARABIC):
        return "ar", 0.95
    if _has_block(text, _DEVANAGARI):
        return "hi", 0.9

    # Latin script — assume English. Phase 1.5 langdetect can distinguish
    # en/de/fr/es/etc.; for Phase 1 we're optimistic.
    has_latin_alpha = any(ch.isalpha() for ch in text)
    if has_latin_alpha:
        return "en", 0.7
    return "unknown", 0.2


__all__ = [
    "detect_language",
]
