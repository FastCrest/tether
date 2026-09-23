"""Integration registry — known external tools tether can connect to."""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def state_dir() -> Path:
    """Per-user dir for integration runtime state (pid + log files).

    Honors TETHER_HOME so it agrees with the rest of the CLI's config home.
    """
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".tether"))
    d = home / "integrations"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class Integration:
    name: str
    description: str
    pip_package: str
    pip_extras: str = ""
    # Module name to import for the installed-check. Defaults to the pip
    # package with '-' → '_'; set explicitly when they differ (the common
    # case for any package whose import name isn't its pip name).
    import_name: str = ""
    # Optional version constraint applied at install time (e.g. ">=1.2,<2").
    # Empty = unpinned (a future release of the integration can change
    # behavior silently — set this for anything load-bearing).
    pip_version_spec: str = ""
    health_url: str = "http://localhost:8000/healthz"
    start_command: list[str] = field(default_factory=list)
    stop_signal: str = "SIGTERM"
    default_port: int = 8000
    mcp_tools: tuple[str, ...] = ()
    homepage: str = ""
    license: str = ""

    @property
    def pip_spec(self) -> str:
        spec = self.pip_package
        if self.pip_extras:
            spec = f"{spec}[{self.pip_extras}]"
        if self.pip_version_spec:
            spec = f"{spec}{self.pip_version_spec}"
        return spec

    @property
    def _import_name(self) -> str:
        return self.import_name or self.pip_package.replace("-", "_")

    @property
    def log_file(self) -> Path:
        return state_dir() / f"{self.name}.log"

    def is_installed(self) -> bool:
        # find_spec resolves the module WITHOUT importing it — so checking
        # "is it installed?" doesn't run the target package's import-time side
        # effects (rtsm[gpu] can init CUDA on import).
        try:
            return importlib.util.find_spec(self._import_name) is not None
        except (ImportError, ValueError):
            return False

    def install(self) -> None:
        logger.info("Installing %s into %s ...", self.pip_spec, sys.executable)
        cmd = [sys.executable, "-m", "pip", "install", self.pip_spec]
        # The PyTorch CUDA wheel index is only correct on Linux + CUDA. On
        # macOS, Jetson (JetPack wheels), and CPU boxes it's useless or wrong,
        # so don't force it there.
        if sys.platform.startswith("linux") and "cu" in self.pip_extras.lower():
            cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cu128"]
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
        log_path = self.log_file
        logger.info("Starting %s: %s (logs → %s)", self.name, " ".join(cmd), log_path)
        # Redirect child output to a log file, NOT PIPE: an unread PIPE fills
        # its 64KB buffer and the child blocks, looking like a hang. The log
        # file also gives `connect` a real place to point users on failure.
        log_fh = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        try:
            for _ in range(30):
                # Fail fast if the child died on launch (bad flag, missing
                # model) instead of waiting the full 30s for a health check
                # that can never pass.
                if proc.poll() is not None:
                    tail = self._log_tail(log_path)
                    raise RuntimeError(
                        f"{self.name} exited immediately (code {proc.returncode}). "
                        f"Last log lines:\n{tail}\nFull log: {log_path}"
                    )
                if self.health_check(timeout=1.0):
                    logger.info("%s is healthy at %s", self.name, self.health_url)
                    return proc
                time.sleep(1)
        finally:
            log_fh.close()
        proc.terminate()
        tail = self._log_tail(log_path)
        raise RuntimeError(
            f"{self.name} failed to become healthy at {self.health_url} "
            f"within 30s. Last log lines:\n{tail}\nFull log: {log_path}"
        )

    @staticmethod
    def _log_tail(path: Path, n: int = 20) -> str:
        try:
            return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(log unavailable)"


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
