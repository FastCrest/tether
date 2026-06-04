"""Backwards-compat shim — `reflex` is now `tether`.

This package was renamed from ``reflex`` (pip: ``reflex-vla``) to ``tether``
in v0.12.0 (2026-06-03). This shim keeps ``from reflex import X`` working
for the 6-month deprecation window (through v0.13.x); the shim is removed
in v0.14.0.

The shim is a thin re-export — every public symbol resolves to the same
object as ``tether.X``::

    from reflex import TetherClient   # works, prints DeprecationWarning once
    from tether import TetherClient   # canonical, no warning

    assert reflex.TetherClient is tether.TetherClient

Migration:

    # Old
    pip install reflex-vla
    from reflex import VLAExport
    reflex --help

    # New
    pip install tether
    from tether import VLAExport
    tether --help

Env vars renamed (``REFLEX_*`` → ``TETHER_*``). Both forms are accepted
during the deprecation window — ``REFLEX_*`` emits a DeprecationWarning
on read. See ``tether/_compat_env.py``.

See CHANGELOG.md v0.12.0 for the full deprecation contract.
"""
from __future__ import annotations

import sys
import warnings

_DEPRECATION_MSG = (
    "The `reflex` package was renamed to `tether` in v0.12.0. "
    "Please update your imports: `from reflex import X` → `from tether import X`. "
    "This compatibility shim will be removed in v0.14.0 (~6 months). "
    "See https://github.com/FastCrest/tether/blob/main/CHANGELOG.md#v0120 "
    "for the migration guide."
)

# Emit the warning exactly once per process — not once per import statement.
# DeprecationWarning is suppressed by default in production; users can opt in
# via PYTHONWARNINGS=default::DeprecationWarning or python -W default.
warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

# Re-export everything from tether. Critically, we import the real `tether`
# module and alias every submodule so that:
#   `from reflex.runtime.server import create_app`
# resolves to the same object as:
#   `from tether.runtime.server import create_app`
# Without the sys.modules aliasing, the two would be parallel module instances
# (separate state, separate registries) — a subtle and dangerous bug.
import tether as _tether  # noqa: E402

# Re-export top-level public symbols.
__version__ = _tether.__version__
__all__ = list(getattr(_tether, "__all__", []))


# Alias every loaded `tether.X` module as `reflex.X` so subsequent imports
# resolve to the same module object. This is the load-bearing piece: without
# it, `from reflex.runtime import X` would create a separate module instance
# rather than re-using `tether.runtime`.
def _alias_submodules() -> None:
    for name, mod in list(sys.modules.items()):
        if name == "tether":
            sys.modules.setdefault("reflex", sys.modules[__name__])
            continue
        if name.startswith("tether."):
            alias = "reflex" + name[len("tether"):]
            sys.modules.setdefault(alias, mod)


_alias_submodules()


def __getattr__(name: str):
    """Lazy attribute access — forward to `tether.<name>`.

    Mirrors `tether.__getattr__` so the lazy public surface (ValidateRoundTrip,
    SUPPORTED_MODEL_TYPES, etc.) keeps working from `reflex`. Importing a
    submodule via `from reflex.X import Y` is handled by Python's import
    machinery + `_alias_submodules` above; this hook covers `reflex.<lazy>`
    attribute access on the top-level package.
    """
    try:
        attr = getattr(_tether, name)
    except AttributeError:
        raise AttributeError(f"module 'reflex' has no attribute {name!r}") from None
    # On lazy resolution, also alias any newly-loaded submodule into reflex.*
    _alias_submodules()
    return attr
