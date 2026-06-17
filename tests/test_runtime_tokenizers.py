from __future__ import annotations

import sys
import types

import pytest

from tether.runtime.tokenizers import (
    OfflineTokenizerMissingError,
    find_bundled_tokenizer_path,
    load_export_tokenizer,
)


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"


def test_load_export_tokenizer_prefers_bundled_local_path(tmp_path, monkeypatch):
    calls = []
    fake_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, *, local_files_only=False):
            calls.append((str(source), local_files_only))
            return _FakeTokenizer()

    fake_mod.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)

    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer_config.json").write_text("{}")
    tok = load_export_tokenizer(
        tmp_path,
        {"tokenizer_ref": "remote/ref"},
        default_ref="fallback/ref",
        set_pad_to_eos=True,
    )

    assert tok is not None
    assert tok.pad_token == "</s>"
    assert calls == [(str(tmp_path / "tokenizer"), True)]


def test_load_export_tokenizer_falls_back_to_remote_ref(tmp_path, monkeypatch):
    calls = []
    fake_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, *, local_files_only=False):
            calls.append((str(source), local_files_only))
            return _FakeTokenizer()

    fake_mod.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)

    tok = load_export_tokenizer(
        tmp_path,
        {"tokenizer_ref": "remote/ref"},
        default_ref="fallback/ref",
    )

    assert tok is not None
    assert calls == [("remote/ref", False)]


def test_find_bundled_tokenizer_path_honors_config_path(tmp_path):
    tok_dir = tmp_path / "assets" / "tok"
    tok_dir.mkdir(parents=True)
    (tok_dir / "tokenizer.json").write_text("{}")

    found = find_bundled_tokenizer_path(tmp_path, {"tokenizer_path": "assets/tok"})

    assert found == tok_dir


def test_load_export_tokenizer_offline_rejects_remote_fallback(tmp_path, monkeypatch):
    calls = []
    fake_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, *, local_files_only=False):
            calls.append((str(source), local_files_only))
            return _FakeTokenizer()

    fake_mod.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)
    monkeypatch.setenv("TETHER_OFFLINE", "1")

    with pytest.raises(OfflineTokenizerMissingError, match="offline tokenizer"):
        load_export_tokenizer(
            tmp_path,
            {"tokenizer_ref": "remote/ref"},
            default_ref="fallback/ref",
        )

    assert calls == []
