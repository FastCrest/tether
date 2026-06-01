"""Software Bill of Materials generator (CRA Annex I) — CycloneDX 1.5 JSON.

Enumerates the installed Python distributions via the stdlib so there is NO new
dependency. The SBOM is one of the conformity-bundle evidence artifacts and
drives downstream vulnerability tracking (a deployer can feed it to any SCA tool).
"""
from __future__ import annotations

import datetime as _dt
import json
from importlib import metadata as _md
from typing import Any


def _purl(name: str, version: str) -> str:
    # Package URL (purl) for PyPI components — the standard SCA identifier.
    return f"pkg:pypi/{name.lower().replace('_', '-')}@{version}"


def generate_sbom(root_name: str = "reflex-vla") -> dict[str, Any]:
    """Build a CycloneDX 1.5 SBOM of the current Python environment."""
    components: list[dict[str, Any]] = []
    root_version = "unknown"
    seen: set[str] = set()
    for dist in _md.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # noqa: BLE001 — broken dist metadata shouldn't abort the SBOM
            continue
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        version = (dist.version or "unknown")
        if name.lower() == root_name.lower():
            root_version = version
            continue  # the root goes in metadata.component, not the list
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": _purl(name, version),
            }
        )
    components.sort(key=lambda c: c["name"].lower())

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": now,
            "tools": [{"vendor": "FastCrest", "name": "reflex-comply", "version": root_version}],
            "component": {
                "type": "application",
                "name": root_name,
                "version": root_version,
                "purl": _purl(root_name, root_version),
            },
        },
        "components": components,
    }


def sbom_json(root_name: str = "reflex-vla", indent: int = 2) -> str:
    return json.dumps(generate_sbom(root_name), indent=indent, sort_keys=False)
