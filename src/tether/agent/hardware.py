"""Best-effort hardware profile collection for Agent heartbeats."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def collect_hardware_profile() -> JsonDict:
    profile: JsonDict = {
        "hostname": _safe(socket.gethostname),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "agent_version": _agent_version(),
        "tether_version": _package_version("tether"),
        "gpu_name": None,
        "cuda": None,
        "jetpack": None,
        "tensorrt": _package_version("tensorrt"),
    }
    _add_nvidia_smi(profile)
    _add_nvcc(profile)
    _add_jetpack(profile)
    return _json_safe(profile)


def _agent_version() -> str | None:
    try:
        from tether.agent import AGENT_VERSION

        return AGENT_VERSION
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _add_nvidia_smi(profile: JsonDict) -> None:
    output = _run(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"])
    if not output:
        return
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if parts:
        profile["gpu_name"] = parts[0] or None
    if len(parts) >= 3 and parts[2]:
        profile["cuda"] = parts[2]
    if len(parts) >= 2 and parts[1]:
        profile["nvidia_driver"] = parts[1]


def _add_nvcc(profile: JsonDict) -> None:
    if profile.get("cuda"):
        return
    output = _run(["nvcc", "--version"])
    if not output:
        return
    marker = "release "
    for line in output.splitlines():
        if marker in line:
            profile["cuda"] = line.split(marker, 1)[1].split(",", 1)[0].strip()
            return


def _add_jetpack(profile: JsonDict) -> None:
    marker_path = Path("/etc/nv_tegra_release")
    try:
        if marker_path.exists():
            profile["jetpack"] = marker_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _json_safe(value: JsonDict) -> JsonDict:
    return json.loads(json.dumps(value, sort_keys=True, default=str))
