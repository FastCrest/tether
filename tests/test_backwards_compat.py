"""Backwards-compat tests for the v0.12.0 ``reflex`` → ``tether`` rename.

These tests lock in the deprecation contract described in
``src/reflex/__init__.py`` and CHANGELOG.md v0.12.0. The ``reflex`` shim
forwards to ``tether`` for the v0.12.x and v0.13.x windows; it is removed
in v0.14.0. If any of these fail, the shim is broken.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings


def _fresh_reflex_import():
    """Drop any cached ``reflex`` / ``tether`` modules so the warning fires.

    The DeprecationWarning is emitted at import time. Once ``reflex`` has
    been imported into the test runner process (by any earlier test), a
    second ``import reflex`` is a no-op and the warning does NOT re-fire.
    We evict the cached module(s) before re-importing so each test gets a
    clean slate.
    """
    for name in list(sys.modules):
        if name == "reflex" or name.startswith("reflex."):
            del sys.modules[name]


def test_reflex_import_emits_deprecation_warning():
    _fresh_reflex_import()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        import reflex  # noqa: F401
        deprecation_warnings = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert any(
            "renamed to `tether`" in str(w.message)
            or "renamed to 'tether'" in str(w.message)
            for w in deprecation_warnings
        ), f"expected a 'renamed to tether' DeprecationWarning, got: {[str(w.message) for w in deprecation_warnings]}"


def test_reflex_export_matches_tether_export():
    """The shim must re-export ``__version__`` identically."""
    import reflex
    import tether
    for symbol in ["__version__"]:
        assert getattr(reflex, symbol) == getattr(tether, symbol), (
            f"reflex.{symbol} ({getattr(reflex, symbol)!r}) "
            f"!= tether.{symbol} ({getattr(tether, symbol)!r})"
        )


def test_reflex_cli_entry_point_redirects():
    """The ``reflex._redirect:main`` entry point must be importable."""
    result = subprocess.run(
        [sys.executable, "-c", "from reflex._redirect import main; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "ok" in result.stdout, (
        f"reflex._redirect import failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_reflex_submodule_aliasing_shares_module_objects():
    """``from reflex.X import Y`` and ``from tether.X import Y`` must resolve
    to the same module object — otherwise parallel module instances would
    create separate state (separate registries, separate singletons).
    """
    import reflex  # noqa: F401 — load shim first
    import tether.fixtures
    # After loading tether.fixtures, the shim's _alias_submodules() should
    # have aliased it as reflex.fixtures so that the next import reuses the
    # same module object.
    reflex_fixtures = importlib.import_module("reflex.fixtures")
    tether_fixtures = importlib.import_module("tether.fixtures")
    assert reflex_fixtures is tether_fixtures, (
        "reflex.fixtures and tether.fixtures must be the same module object "
        "(otherwise registries / singletons fork)"
    )
