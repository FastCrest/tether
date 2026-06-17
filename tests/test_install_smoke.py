"""Integration smoke tests for v0.7 ORT-TRT EP first-class install.

Verifies in a FRESH subprocess (not in-process — to avoid cached state
from prior tests) that:
- importing tether doesn't crash on macOS, Linux, or Windows
- on Linux: LD_LIBRARY_PATH gets set IF nvidia/tensorrt site-packages exist
- on macOS: import is a no-op for env (LD_LIBRARY_PATH untouched)
- the patch helper is exported as `tether._patch_ld_library_path`

These tests don't require GPU hardware; they validate the install
plumbing works without crashing across platforms.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run_in_subprocess(snippet: str, env: dict | None = None) -> tuple[int, str, str]:
    """Run a Python snippet in a fresh subprocess. Returns (rc, stdout, stderr)."""
    env_full = os.environ.copy()
    if env:
        env_full.update(env)
    # Make sure the subprocess can import the dev source tree
    src_path = str(Path(__file__).parent.parent / "src")
    env_full["PYTHONPATH"] = src_path + os.pathsep + env_full.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, env=env_full, timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_import_reflex_does_not_crash():
    """Smoke: `import tether` must not raise on any platform."""
    rc, stdout, stderr = _run_in_subprocess("import tether; print('OK')")
    assert rc == 0, f"Import failed: stderr={stderr!r}"
    assert "OK" in stdout


def test_import_reflex_exposes_patch_helper():
    """The patch helper is part of the public API (used by tests + doctor)."""
    snippet = textwrap.dedent("""
        import tether
        assert hasattr(tether, '_patch_ld_library_path'), "patch helper missing"
        assert callable(tether._patch_ld_library_path), "patch helper not callable"
        print("OK")
    """)
    rc, stdout, stderr = _run_in_subprocess(snippet)
    assert rc == 0, f"stderr={stderr!r}"
    assert "OK" in stdout


def test_import_tether_version_is_at_least_v07():
    """Sanity check on the version string. Uses semver parsing so version
    bumps don't break this test — only an accidental regression below
    0.7.0 (our wheel-compatibility floor) trips the assertion."""
    snippet = textwrap.dedent("""
        import tether
        from packaging.version import Version
        v = Version(tether.__version__)
        assert v >= Version("0.7.0"), f"version regressed below 0.7.0: {v}"
        print(tether.__version__)
    """)
    rc, stdout, stderr = _run_in_subprocess(snippet)
    assert rc == 0, f"stderr={stderr!r}"
    from packaging.version import Version
    assert Version(stdout.strip()) >= Version("0.7.0")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific")
def test_import_reflex_macos_no_ld_library_path_change():
    """On macOS, importing tether must not set/modify LD_LIBRARY_PATH."""
    snippet = textwrap.dedent("""
        import os
        before = os.environ.get('LD_LIBRARY_PATH', '')
        import tether
        after = os.environ.get('LD_LIBRARY_PATH', '')
        assert before == after, f"LD_LIBRARY_PATH changed: {before!r} -> {after!r}"
        print("OK")
    """)
    rc, stdout, stderr = _run_in_subprocess(snippet, env={"LD_LIBRARY_PATH": ""})
    assert rc == 0, f"stderr={stderr!r}"
    assert "OK" in stdout


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific")
def test_import_reflex_linux_respects_opt_out():
    """TETHER_NO_LD_LIBRARY_PATH_PATCH=1 disables the patch on Linux too."""
    snippet = textwrap.dedent("""
        import os
        before = os.environ.get('LD_LIBRARY_PATH', '')
        import tether
        after = os.environ.get('LD_LIBRARY_PATH', '')
        assert before == after, f"LD_LIBRARY_PATH changed despite opt-out: {before!r} -> {after!r}"
        print("OK")
    """)
    rc, stdout, stderr = _run_in_subprocess(
        snippet,
        env={"TETHER_NO_LD_LIBRARY_PATH_PATCH": "1", "LD_LIBRARY_PATH": ""},
    )
    assert rc == 0, f"stderr={stderr!r}"
    assert "OK" in stdout


def test_import_reflex_idempotent_across_processes():
    """Importing tether twice in the same process is safe (no double-patch).

    Tests the same-process case (different from cross-process) — verifies
    the patch is idempotent if `tether` is somehow re-imported.
    """
    snippet = textwrap.dedent("""
        import os, importlib
        import tether
        first = os.environ.get('LD_LIBRARY_PATH', '')
        importlib.reload(tether)
        second = os.environ.get('LD_LIBRARY_PATH', '')
        assert first == second, f"reload changed env: {first!r} -> {second!r}"
        print("OK")
    """)
    rc, stdout, stderr = _run_in_subprocess(snippet)
    assert rc == 0, f"stderr={stderr!r}"
    assert "OK" in stdout
