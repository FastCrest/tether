"""Connector — install, start, query, and stop integrations."""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

import requests

from reflex.integrations.registry import Integration, get_integration

logger = logging.getLogger(__name__)

_RUNNING: dict[str, subprocess.Popen] = {}


def connect(name: str, extra_args: list[str] | None = None) -> dict[str, Any]:
    integration = get_integration(name)
    if integration is None:
        from reflex.integrations.registry import list_integrations
        available = [i.name for i in list_integrations()]
        raise ValueError(f"Unknown integration {name!r}. Available: {available}")

    if integration.health_check():
        return {
            "status": "already_running",
            "name": name,
            "url": integration.health_url,
            "mcp_tools": integration.mcp_tools,
        }

    if not integration.is_installed():
        integration.install()

    proc = integration.start(extra_args=extra_args)
    _RUNNING[name] = proc

    return {
        "status": "started",
        "name": name,
        "pid": proc.pid,
        "url": integration.health_url,
        "mcp_tools": integration.mcp_tools,
    }


def disconnect(name: str) -> dict[str, Any]:
    proc = _RUNNING.pop(name, None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
        return {"status": "stopped", "name": name, "pid": proc.pid}

    integration = get_integration(name)
    if integration and integration.health_check():
        return {"status": "external_still_running", "name": name}

    return {"status": "not_running", "name": name}


def query_objects(
    integration_url: str, endpoint: str = "/objects", **params: Any,
) -> list[dict]:
    resp = requests.get(f"{integration_url.rstrip('/')}{endpoint}", params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def semantic_search(
    integration_url: str, query: str, top_k: int = 5,
) -> list[dict]:
    resp = requests.get(
        f"{integration_url.rstrip('/')}/search/semantic",
        params={"query": query, "top_k": top_k},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


def spatial_search(
    integration_url: str, x: float, y: float, z: float, radius: float = 1.5,
) -> list[dict]:
    resp = requests.get(
        f"{integration_url.rstrip('/')}/search/spatial",
        params={"x": x, "y": y, "z": z, "radius": radius},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()
