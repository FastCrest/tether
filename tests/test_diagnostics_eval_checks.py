"""Tests for the 3 eval-related doctor checks added Day 5.

Per ADR 2026-04-25-eval-as-a-service-architecture decision #9:
- check_modal_auth: warns if neither ~/.modal.toml nor MODAL_TOKEN_*
  env are detected
- check_libero_importable: warns if `import libero` fails
- check_vla_eval_importable: warns if `from tether.runtime.adapters
  import vla_eval` fails

All 3 are warns (not fails) — customers not using eval shouldn't
have doctor exit non-zero.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# check_modal_auth
# ---------------------------------------------------------------------------


def test_modal_auth_passes_when_config_file_exists(monkeypatch, tmp_path):
    fake_home = tmp_path
    (fake_home / ".modal.toml").write_text("[default]\ntoken_id='t'")
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    from tether.diagnostics.check_modal_auth import _run
    result = _run(model_path="")
    assert result.status == "pass"
    assert "modal.toml" in result.actual.lower()


def test_modal_auth_passes_when_env_tokens_set(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # no config file
    monkeypatch.setenv("MODAL_TOKEN_ID", "id123")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret456")

    from tether.diagnostics.check_modal_auth import _run
    result = _run(model_path="")
    assert result.status == "pass"
    assert "env" in result.actual.lower()


def test_modal_auth_warns_when_neither(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # no config file
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    from tether.diagnostics.check_modal_auth import _run
    result = _run(model_path="")
    assert result.status == "warn"
    assert "modal token new" in result.remediation


def test_modal_auth_warns_when_only_id_set(monkeypatch, tmp_path):
    """Both ID and SECRET required."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("MODAL_TOKEN_ID", "id123")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    from tether.diagnostics.check_modal_auth import _run
    result = _run(model_path="")
    assert result.status == "warn"


# ---------------------------------------------------------------------------
# check_libero_importable
# ---------------------------------------------------------------------------


def test_libero_check_warns_when_import_fails(monkeypatch):
    """Force ImportError via monkeypatched __import__."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "libero":
            raise ImportError("no libero")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    from tether.diagnostics.check_libero_importable import _run
    result = _run(model_path="")
    assert result.status == "warn"
    assert "ImportError" in result.actual
    assert "eval-local" in result.remediation


def test_libero_check_passes_when_already_importable(monkeypatch):
    """Stub libero in sys.modules so import is a no-op."""
    import sys
    import types
    fake_libero = types.ModuleType("libero")
    monkeypatch.setitem(sys.modules, "libero", fake_libero)

    from tether.diagnostics.check_libero_importable import _run
    result = _run(model_path="")
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# check_vla_eval_importable
# ---------------------------------------------------------------------------


def test_vla_eval_check_passes_in_real_env():
    """The Tether internal adapter SHOULD be importable in dev env."""
    from tether.diagnostics.check_vla_eval_importable import _run
    result = _run(model_path="")
    assert result.status == "pass"


def test_vla_eval_check_warns_on_import_error(monkeypatch):
    """Import _run BEFORE patching (otherwise the patch blocks the test's own import)."""
    from tether.diagnostics.check_vla_eval_importable import _run

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        # Block only the adapter's submodule import — not the test's own.
        if name == "tether.runtime.adapters" or "adapters.vla_eval" in name:
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    # Force reload to skip already-cached module
    import sys
    sys.modules.pop("tether.runtime.adapters.vla_eval", None)
    sys.modules.pop("tether.runtime.adapters", None)

    result = _run(model_path="")
    assert result.status == "warn"


# ---------------------------------------------------------------------------
# Registry integration — all 3 are loaded by run_all_checks
# ---------------------------------------------------------------------------


def test_eval_checks_registered_in_doctor():
    from tether.diagnostics import _ensure_registry_loaded, _REGISTRY
    _ensure_registry_loaded()
    check_ids = {c.check_id for c in _REGISTRY}
    assert "check_modal_auth" in check_ids
    assert "check_libero_importable" in check_ids
    assert "check_vla_eval_importable" in check_ids
