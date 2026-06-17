"""Backwards-compatible env-var resolution.

In v0.12.0 the project renamed from ``reflex-vla`` to ``tether``. Every
``REFLEX_*`` env var has a canonical ``TETHER_*`` equivalent. To avoid
breaking customer deployments mid-deprecation, the runtime accepts BOTH
during the deprecation window (through v0.13.x; removed in v0.14.0).

Precedence: ``TETHER_X`` is checked first; if unset, ``REFLEX_X`` is read
and a one-time DeprecationWarning is emitted.

Usage from anywhere in the codebase::

    from tether._compat_env import getenv
    api_key = getenv("API_KEY")          # reads TETHER_API_KEY or REFLEX_API_KEY
    home = getenv("HOME", default="~/.tether")

This is opt-in: existing ``os.environ.get("TETHER_X")`` call sites work
exactly as before; only sites that explicitly want fallback semantics
need to switch.

For the long deprecation tail we ALSO install a process-wide os.environ
patch that mirrors REFLEX_* → TETHER_* at import time so untouched
``os.environ.get("TETHER_X")`` call sites work even when the customer
still has REFLEX_X set in their shell. Opt out via
``TETHER_NO_ENV_COMPAT=1``.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional


_WARNED: set[str] = set()


def getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a TETHER_* env var, falling back to REFLEX_* with a deprecation warning.

    Parameters
    ----------
    name : str
        The variable suffix (without the ``TETHER_`` or ``REFLEX_`` prefix).
        E.g. ``getenv("API_KEY")`` reads ``TETHER_API_KEY`` (preferred) or
        ``REFLEX_API_KEY`` (deprecated).
    default : str, optional
        Returned if neither variable is set.
    """
    tether_key = f"TETHER_{name}"
    reflex_key = f"REFLEX_{name}"
    v = os.environ.get(tether_key)
    if v is not None:
        return v
    v = os.environ.get(reflex_key)
    if v is not None:
        if reflex_key not in _WARNED:
            _WARNED.add(reflex_key)
            warnings.warn(
                f"Env var {reflex_key} is deprecated; use {tether_key} instead. "
                f"REFLEX_* env vars will be removed in v0.14.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v
    return default


def _mirror_reflex_env_to_tether() -> None:
    """Mirror REFLEX_* env vars into TETHER_* at process start.

    This makes existing ``os.environ.get("TETHER_X")`` call sites work for
    customers who still have REFLEX_X exported in their shell or systemd
    units. The mirror is one-way (REFLEX_X is the SOURCE; TETHER_X wins if
    both are set) and idempotent. Opt out via TETHER_NO_ENV_COMPAT=1.

    Called once from ``tether/__init__.py``.
    """
    if os.environ.get("TETHER_NO_ENV_COMPAT"):
        return
    warned_any = False
    for key, value in list(os.environ.items()):
        if not key.startswith("REFLEX_"):
            continue
        tether_equiv = "TETHER_" + key[len("REFLEX_"):]
        if tether_equiv in os.environ:
            continue  # TETHER_* takes precedence; don't clobber
        os.environ[tether_equiv] = value
        warned_any = True
    if warned_any:
        warnings.warn(
            "Detected REFLEX_* env vars in environment. These were renamed to "
            "TETHER_* in v0.12.0 and mirrored automatically for now. Update your "
            "deployment to use TETHER_* directly before v0.14.0. "
            "Set TETHER_NO_ENV_COMPAT=1 to disable mirroring.",
            DeprecationWarning,
            stacklevel=2,
        )
