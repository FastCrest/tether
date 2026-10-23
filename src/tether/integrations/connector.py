"""Connector — install, start, query, and stop integrations."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from tether.integrations.registry import get_integration, state_dir

logger = logging.getLogger(__name__)

# Same-process handle cache. NOT relied on for stop — `tether connect up`
# Popens a child then the CLI process exits, so by the time `tether connect
# down` runs in a NEW process this dict is empty. The pid file below is what
# makes stop work across invocations.
_RUNNING: dict[str, subprocess.Popen] = {}


def _pid_file(name: str) -> Path:
    return state_dir() / f"{name}.pid"


def _write_pid(name: str, pid: int) -> None:
    _pid_file(name).write_text(str(pid))


def _read_pid(name: str) -> int | None:
    try:
        return int(_pid_file(name).read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def connect(name: str, extra_args: list[str] | None = None) -> dict[str, Any]:
    integration = get_integration(name)
    if integration is None:
        from tether.integrations.registry import list_integrations
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
    # Persist the pid so a later `tether connect down` (a separate CLI process)
    # can actually stop this child instead of orphaning it.
    _write_pid(name, proc.pid)

    return {
        "status": "started",
        "name": name,
        "pid": proc.pid,
        "url": integration.health_url,
        "mcp_tools": integration.mcp_tools,
    }


def _signal_for(integration) -> int:
    sig_name = getattr(integration, "stop_signal", "SIGTERM") if integration else "SIGTERM"
    return getattr(signal, sig_name, signal.SIGTERM)


def disconnect(name: str) -> dict[str, Any]:
    integration = get_integration(name)

    # 1) Same-process handle (rare — only if up + down ran in one process).
    proc = _RUNNING.pop(name, None)
    if proc is not None and proc.poll() is None:
        proc.send_signal(_signal_for(integration))
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        _pid_file(name).unlink(missing_ok=True)
        return {"status": "stopped", "name": name, "pid": proc.pid}

    # 2) Cross-invocation: signal the pid we persisted at connect time.
    pid = _read_pid(name)
    if pid is not None and _pid_alive(pid):
        sig = _signal_for(integration)
        try:
            os.kill(pid, sig)
            # Wait for it to actually exit; escalate to SIGKILL if it won't.
            for _ in range(10):
                if not _pid_alive(pid):
                    break
                time.sleep(1)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _pid_file(name).unlink(missing_ok=True)
        return {"status": "stopped", "name": name, "pid": pid}

    # 3) Stale pid file (process already gone) — clean it up.
    _pid_file(name).unlink(missing_ok=True)

    # 4) Something is still answering on the port but we didn't start it.
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
