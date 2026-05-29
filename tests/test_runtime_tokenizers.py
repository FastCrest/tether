from __future__ import annotations

import sys
import types

from reflex.runtime.tokenizers import load_export_tokenizer


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
