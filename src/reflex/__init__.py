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
    pip install fastcrest-tether
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


# Install a meta path finder so that ``from reflex.X import Y`` works EVEN
# when ``tether.X`` hasn't been imported yet. Without this, the shim only
# aliases tether submodules that were already in ``sys.modules`` at the
# moment ``reflex`` was imported — anything imported later (`from
# reflex.fixtures import load_fixtures`) would 404 because there's no
# ``src/reflex/fixtures/`` directory on disk.
#
# The finder is idempotent (Python dedupes by identity in sys.meta_path)
# and only matches the ``reflex.*`` namespace, so it never interferes with
# unrelated imports.
import importlib  # noqa: E402
import importlib.abc  # noqa: E402
import importlib.util  # noqa: E402


class _ReflexAliasLoader(importlib.abc.Loader):
    """Loader whose ``exec_module`` is a no-op — the module is pre-populated
    in ``sys.modules`` by ``_ReflexShimFinder.find_spec``.
    """

    def create_module(self, spec):
        return sys.modules[spec.name]

    def exec_module(self, module):
        return None  # already executed under the `tether.*` name


class _ReflexShimFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that maps ``reflex.X`` imports to ``tether.X``.

    Only fires for the ``reflex`` namespace (other packages are untouched).
    On a hit, it imports the matching ``tether.X`` module via the normal
    import system, then registers the SAME module object under the
    ``reflex.X`` name in ``sys.modules`` — guaranteeing both names resolve
    to one shared instance (no parallel state).
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != "reflex" and not fullname.startswith("reflex."):
            return None
        if fullname == "reflex":
            return None  # this package is already loaded — leave it alone
        tether_name = "tether" + fullname[len("reflex"):]
        try:
            tether_mod = importlib.import_module(tether_name)
        except ImportError:
            return None  # let the default machinery raise the real error
        # Pre-register the same module object under the reflex name so that
        # subsequent imports (and our own no-op loader) see it.
        sys.modules[fullname] = tether_mod
        spec = importlib.util.spec_from_loader(fullname, _ReflexAliasLoader())
        # If the underlying tether module is a package, propagate that so
        # `from reflex.fixtures.vla_fixtures import ...` works.
        if hasattr(tether_mod, "__path__"):
            spec.submodule_search_locations = list(tether_mod.__path__)
        return spec


if not any(isinstance(f, _ReflexShimFinder) for f in sys.meta_path):
    sys.meta_path.append(_ReflexShimFinder())


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
