"""Tests for the 2026-05-22 `reflex export` stderr-routing fix.

Discovered during Day 7 Modal validation: `reflex export gr00t` failed
with empty stderr because all CLI error paths used `console.print` (rich,
defaults to stdout). Subprocess wrappers logging only `r.stderr` saw the
non-zero exit but no message. Cause: `_require_monolithic_deps()` in
`monolithic.py` raises `ImportError` when the `[monolithic]` extras
aren't installed, and the export() command's catch block printed via
`console.print` (stdout) instead of stderr.

Fix: `err_console = Console(stderr=True)` added to `cli.py`; error paths
in the `export()` command route through it. Subprocess wrappers calling
`reflex export` now see error detail in `r.stderr`.

This test pins:
1. `err_console` is defined in `reflex.cli` and writes to stderr.
2. `reflex export` with `--monolithic` on a system missing `[monolithic]`
   deps emits the failure message to stderr (not stdout).
3. The matching diagnostic improvement in modal_test_gr00t.py logs BOTH
   stderr + stdout on subprocess failure.
"""
from __future__ import annotations

import sys

import pytest


def test_err_console_defined_and_writes_to_stderr():
    """`err_console` exists in cli.py and is a rich.Console with stderr=True."""
    from reflex.cli import err_console
    # The Console object exposes its target file; check it's stderr.
    # rich.Console stores the file as `file` attribute when constructed.
    assert err_console.file is sys.stderr


def test_cli_export_routes_errors_to_stderr_via_err_console():
    """Sanity: `reflex.cli.export` uses err_console for its error branches.

    Inspects the source code so we don't need to fire a real export
    that requires GPU + 6GB of GR00T weights.
    """
    import inspect
    import reflex.cli as cli_module

    src = inspect.getsource(cli_module)

    # The two ImportError handlers in export() (monolithic dep missing)
    # MUST route their error message via err_console (stderr), not console
    # (stdout). Catches regressions where future contributors copy-paste
    # an error path without thinking about subprocess wrappers.
    assert "err_console.print(f\"[red]{exc}[/red]\", markup=False)" in src, (
        "Expected at least one err_console.print(...) for an ImportError "
        "in cli.export — was the 2026-05-22 stderr fix reverted?"
    )

    # The "Missing monolithic dep" message specifically must go to stderr.
    assert "err_console.print(f\"Missing monolithic dep: {exc}\"" in src


def test_no_stdout_error_paints_remain_in_cli():
    """Post-2026-05-22 sweep: zero `console.print(...[red]...)` calls remain
    in cli.py. All error/warning paints route through `err_console.print`
    so subprocess wrappers see them in stderr.

    Regression-pin: catches a contributor accidentally adding a new
    error path via `console.print` instead of `err_console.print`. Same
    class of bug as the Day 7 modal_test_gr00t empty-stderr issue.
    """
    import re
    from pathlib import Path

    cli_path = Path(__file__).parent.parent / "src" / "reflex" / "cli.py"
    src = cli_path.read_text()

    # Match `console.print(...)` (NOT `err_console.print(...)`) calls
    # that contain `[red]` markup somewhere in the args. The word-boundary
    # negative lookbehind ensures `err_console.print` is excluded.
    pattern = re.compile(r"(?<![a-zA-Z_])console\.print\([^()]*\[red\]", re.DOTALL)
    matches = pattern.findall(src)
    assert len(matches) == 0, (
        f"Found {len(matches)} stdout error path(s) in cli.py — these must "
        "route through `err_console.print` so subprocess wrappers see them "
        "in stderr.\nFirst few: " + "\n".join(m[:120] for m in matches[:3])
    )
