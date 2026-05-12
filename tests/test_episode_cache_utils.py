"""Tests for episode-cache hashing helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/reflex/runtime/episode_cache.py"
_SPEC = importlib.util.spec_from_file_location("episode_cache", _MODULE_PATH)
assert _SPEC is not None
episode_cache = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = episode_cache
_SPEC.loader.exec_module(episode_cache)
lang_hash = episode_cache.lang_hash


def _tokens(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def test_lang_hash_returns_16_byte_sha256_prefix():
    tokens = _tokens("pick up the red block")

    digest = lang_hash(tokens)
    expected = hashlib.sha256(tokens.tobytes()).digest()[:16]

    assert isinstance(digest, bytes)
    assert len(digest) == 16
    assert digest == expected


def test_lang_hash_is_stable_for_same_tokens():
    tokens = _tokens("open the drawer")

    assert lang_hash(tokens) == lang_hash(tokens.copy())


def test_lang_hash_differs_for_small_prompt_set():
    prompts = [
        "pick up the red block",
        "pick up the blue block",
        "open the drawer",
        "close the drawer",
        "move left",
        "move right",
        "",
    ]

    hashes = [lang_hash(_tokens(prompt)) for prompt in prompts]

    assert len(set(hashes)) == len(prompts)
