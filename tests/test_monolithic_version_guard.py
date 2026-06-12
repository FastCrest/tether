"""The monolithic export must reject a wrong lerobot version up front (#190).

Previously only `transformers` was version-checked; `lerobot` was merely
imported, so a mismatched lerobot passed the dep check and failed later with a
confusing monkeypatch/mask error.
"""
from __future__ import annotations

import sys
import types

import pytest

from tether.exporters import monolithic


def _fake_lerobot(version: str) -> types.ModuleType:
    m = types.ModuleType("lerobot")
    m.__version__ = version
    return m


def test_wrong_lerobot_version_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", _fake_lerobot("0.4.0"))
    with pytest.raises(ImportError) as ei:
        monolithic._require_monolithic_deps()
    assert "lerobot==0.5.1 (found 0.4.0" in str(ei.value)


def test_correct_lerobot_version_not_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", _fake_lerobot("0.5.1"))
    # Other monolithic deps may be absent in this env (so it can still raise),
    # but the lerobot-version complaint must NOT be among the reasons.
    try:
        monolithic._require_monolithic_deps()
        msg = ""
    except ImportError as e:
        msg = str(e)
    assert "found 0.5.1" not in msg
    assert "lerobot==0.5.1 (found" not in msg
