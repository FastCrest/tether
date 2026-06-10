"""Once-per-day check for a newer tether on PyPI.

Runs on CLI startup; if a newer version exists, prints a one-line nag to stderr.
Cached for 24h in TETHER_HOME so we don't ping PyPI on every invocation.
Disable via `TETHER_NO_UPGRADE_CHECK=1`. Skipped automatically on dev installs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CACHE_TTL_SECONDS = 24 * 60 * 60
PYPI_URL = "https://pypi.org/pypi/tether/json"


def _cache_path() -> Path:
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".cache" / "tether"))
    home.mkdir(parents=True, exist_ok=True)
    return home / ".upgrade_check"


def _is_dev_install() -> bool:
    """Skip the check when running from an editable install — the user is
    presumably developing tether itself and doesn't want a nag.

    Only `.pth`-style editable installs trigger this. Regular `pip install .`
    or `pip install tether` from PyPI both proceed normally.
    """
    try:
        from importlib.metadata import distribution
        dist = distribution("tether")
        # Editable installs (pip install -e .) leave a __editable__*.pth
        # file in site-packages that re-routes to the source dir.
        files = dist.files or []
        for f in files:
            name = f.name if hasattr(f, "name") else str(f).rsplit("/", 1)[-1]
            if name.startswith("__editable__") and name.endswith(".pth"):
                return True
    except Exception:
        return False  # can't determine → don't skip; let the network check decide
    return False


def _parse_version(v: str) -> tuple[int, ...]:
    """Loose semver parse; ignores pre-release suffixes."""
    parts: list[int] = []
    for chunk in v.split("."):
        n = ""
        for c in chunk:
            if c.isdigit():
                n += c
            else:
                break
        parts.append(int(n) if n else 0)
    return tuple(parts)


def _read_cache() -> dict | None:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - data.get("checked_at", 0) > CACHE_TTL_SECONDS:
        return None
    return data


def _write_cache(latest: str) -> None:
    try:
        _cache_path().write_text(json.dumps({"latest": latest, "checked_at": time.time()}))
    except OSError:
        pass  # not load-bearing


def _fetch_latest() -> str | None:
    try:
        import httpx
        r = httpx.get(PYPI_URL, timeout=2.0)
        if r.status_code != 200:
            return None
        return r.json().get("info", {}).get("version")
    except Exception:
        return None


def maybe_nag(current_version: str) -> None:
    """Print upgrade nag if a newer version exists on PyPI. Silent otherwise.

    Honors TETHER_NO_UPGRADE_CHECK=1 and skips on dev installs.
    """
    if os.environ.get("TETHER_NO_UPGRADE_CHECK"):
        return
    if _is_dev_install():
        return

    cached = _read_cache()
    if cached:
        latest = cached.get("latest")
    else:
        latest = _fetch_latest()
        if latest:
            _write_cache(latest)

    if not latest:
        return

    if _parse_version(latest) > _parse_version(current_version):
        sys.stderr.write(
            f"\033[2m[tether] {latest} is available — upgrade: pip install -U tether "
            f"(you have {current_version}, set TETHER_NO_UPGRADE_CHECK=1 to silence)\033[0m\n"
        )
