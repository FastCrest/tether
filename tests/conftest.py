"""Pytest fixtures shared across the test suite.

CI-only behavior of Click+Rich's help renderer that broke us:
- When `CI=true` (set by GitHub Actions), Rich inserts ANSI color
  escapes BETWEEN the two `-` characters of long flag names, so the
  literal substring `"--export-mode"` is no longer present in
  `result.output` even though the rendered text looks identical.
  Setting `TERM=dumb` forces Rich into no-color plain-text mode and
  the substring search works again.
- The CI runner's terminal width defaults narrow enough that Rich
  elides long option names with `--...`. `COLUMNS=200` keeps the
  rendered panel wide enough to show all names verbatim.

Both env vars are set in `pytest_configure` (runs before any test
import), so the entire suite inherits stable, plain-text help output
matching what `--help` users see in a real shell.
"""
from __future__ import annotations

import os


def pytest_configure(config):
    os.environ["COLUMNS"] = "200"
    os.environ["LINES"] = "80"
    os.environ["TERM"] = "dumb"
