"""Backwards-compat CLI entry point — `reflex` forwards to `tether`.

The `reflex` PyPI package and CLI binary were renamed to `tether` in v0.12.0
(2026-06-03). This module is wired as the ``reflex = "reflex._redirect:main"``
entry point in ``pyproject.toml`` so that customers who type ``reflex chat``
or run a CI pipeline with ``reflex export`` keep working — they get a one-line
DeprecationWarning to stderr telling them to migrate to ``tether``, then the
command runs normally.

Removed in v0.14.0 (~6 months from the rename).
"""
from __future__ import annotations

import sys


_DEPRECATION_BANNER = (
    "\033[33m[deprecation]\033[0m The `reflex` CLI was renamed to `tether` in "
    "v0.12.0. Please use `tether {args}` instead. "
    "The `reflex` binary is forwarding for now but will be removed in v0.14.0 (~6 months).\n"
)


def main() -> None:
    """Forward the `reflex` invocation to `tether`'s typer app.

    Behaviour-preserving: every flag, sub-command, exit code is identical.
    The only difference is the one-line deprecation banner emitted to stderr
    before the command runs.
    """
    # Print deprecation banner — to stderr so it never contaminates piped
    # stdout (matters for `reflex export --json | jq ...` patterns).
    args_for_banner = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "<command>"
    sys.stderr.write(_DEPRECATION_BANNER.format(args=args_for_banner))
    sys.stderr.flush()

    # Forward to the real CLI. Importing `tether.cli` lazily so that the
    # banner fires immediately even if tether's import is slow (good UX —
    # the user sees "you should switch to tether" before any latency).
    from tether.cli import app

    # Typer apps can be called directly; this matches what setuptools/hatch
    # do for `tether = "tether.cli:app"`. No argv manipulation needed —
    # Typer reads sys.argv natively.
    app()


if __name__ == "__main__":
    main()
