"""SBOM generation for Tether Comply.

No external dependency is required; this emits a compact CycloneDX-compatible
JSON document covering the installed Python environment plus bundle artifacts.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import uuid
from pathlib import Path
from typing import Any

from tether import __version__
from tether.comply.schemas import SBOM_SCHEMA_VERSION, utc_now_iso
from tether.verification_report import _sha256


def _component_from_distribution(dist: metadata.Distribution) -> dict[str, Any] | None:
    name = dist.metadata.get("Name")
    version = dist.version
    if not name or not version:
        return None
    component: dict[str, Any] = {
        "type": "library",
        "bom-ref": f"pkg:pypi/{name.lower()}@{version}",
        "name": name,
        "version": version,
        "purl": f"pkg:pypi/{name.lower()}@{version}",
    }
    license_text = dist.metadata.get("License")
    if license_text:
        component["licenses"] = [{"license": {"name": license_text[:120]}}]
    return component


def generate_sbom(
    *,
    artifact_paths: list[str | Path] | None = None,
    include_environment: bool = True,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = [
        {
            "type": "application",
            "bom-ref": f"pkg:pypi/fastcrest-tether@{__version__}",
            "name": "fastcrest-tether",
            "version": __version__,
            "purl": f"pkg:pypi/fastcrest-tether@{__version__}",
        }
    ]

    if include_environment:
        seen = {"fastcrest-tether"}
        for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
            comp = _component_from_distribution(dist)
            if comp is None:
                continue
            key = str(comp["name"]).lower()
            if key in seen:
                continue
            seen.add(key)
            components.append(comp)

    for artifact in artifact_paths or []:
        p = Path(artifact)
        if not p.exists() or not p.is_file():
            continue
        components.append({
            "type": "file",
            "bom-ref": f"file:{p.name}:{_sha256(p)[:12]}",
            "name": p.name,
            "hashes": [{"alg": "SHA-256", "content": _sha256(p)}],
        })

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": utc_now_iso(),
            "tools": [
                {
                    "vendor": "Tether Labs",
                    "name": "tether comply sbom",
                    "version": __version__,
                }
            ],
            "component": components[0],
            "properties": [
                {"name": "tether.schema", "value": SBOM_SCHEMA_VERSION},
            ],
        },
        "components": components,
    }


def write_sbom(path: str | Path, sbom: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n")
    return out


__all__ = ["generate_sbom", "write_sbom"]
