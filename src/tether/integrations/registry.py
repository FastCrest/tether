"""Integration registry — known external tools tether can connect to."""
from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Integration:
    name: str
    description: str
    pip_package: str
    pip_extras: str = ""
    health_url: str = "http://localhost:8000/healthz"
    start_command: list[str] = field(default_factory=list)
    stop_signal: str = "SIGTERM"
    default_port: int = 8000
    mcp_tools: tuple[str, ...] = ()
    homepage: str = ""
    license: str = ""

    @property
    def pip_spec(self) -> str:
        if self.pip_extras:
            return f"{self.pip_package}[{self.pip_extras}]"
        return self.pip_package

    def is_installed(self) -> bool:
        try:
            importlib.import_module(self.pip_package.replace("-", "_").split("[")[0])
            return True
        except ImportError:
            return False

    def install(self) -> None:
        logger.info("Installing %s...", self.pip_spec)
        cmd = [
            sys.executable, "-m", "pip", "install", self.pip_spec,
            "--extra-index-url", "https://download.pytorch.org/whl/cu128",
            "-q",
        ]
        subprocess.check_call(cmd)
        logger.info("Installed %s", self.pip_spec)

    def health_check(self, timeout: float = 2.0) -> bool:
        try:
            resp = requests.get(self.health_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self, extra_args: list[str] | None = None) -> subprocess.Popen:
        cmd = list(self.start_command)
        if extra_args:
            cmd.extend(extra_args)
        logger.info("Starting %s: %s", self.name, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(30):
            if self.health_check(timeout=1.0):
                logger.info("%s is healthy at %s", self.name, self.health_url)
                return proc
            time.sleep(1)
        proc.terminate()
        raise RuntimeError(
            f"{self.name} failed to become healthy at {self.health_url} "
            f"within 30s. Check logs with: {' '.join(cmd)}"
        )


RTSM = Integration(
    name="rtsm",
    description="Real-Time Spatial Memory — persistent 3D object map from RGB-D streams",
    pip_package="rtsm",
    pip_extras="gpu",
    health_url="http://localhost:8002/healthz",
    start_command=[sys.executable, "-m", "rtsm", "demo", "--no-viz"],
    default_port=8002,
    mcp_tools=(
        "rtsm.semantic_query",
        "rtsm.spatial_query",
        "rtsm.relational_query",
        "rtsm.list_objects",
        "rtsm.get_object",
        "rtsm.status",
    ),
    homepage="https://github.com/calabi-inc/rtsm",
    license="Apache-2.0",
)


_REGISTRY: dict[str, Integration] = {
    "rtsm": RTSM,
}


def get_integration(name: str) -> Integration | None:
    return _REGISTRY.get(name)


def list_integrations() -> list[Integration]:
    return list(_REGISTRY.values())
