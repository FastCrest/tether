"""Service file generation for long-running Tether Agent daemons."""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SERVICE_NAME = "tether-agent"
LAUNCHD_LABEL = "com.fastcrest.tether-agent"


@dataclass(slots=True)
class ServicePlan:
    kind: str
    path: Path
    content: str


def detect_service_kind(platform: str | None = None) -> str:
    value = platform or sys.platform
    if value == "darwin":
        return "launchd"
    if value.startswith("linux"):
        return "systemd"
    raise ValueError(f"unsupported service platform: {value}")


def resolve_tether_bin(explicit: str | os.PathLike[str] | None = None) -> str:
    if explicit:
        return str(explicit)
    return shutil.which("tether") or str(Path(sys.argv[0]).resolve())


def service_path(kind: str, *, home: str | os.PathLike[str] | None = None) -> Path:
    base = Path(home) if home is not None else Path.home()
    if kind == "systemd":
        return base / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
    if kind == "launchd":
        return base / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    raise ValueError(f"unsupported service kind: {kind}")


def render_systemd_user_service(
    *,
    tether_bin: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> str:
    argv = " ".join(
        shlex.quote(str(part))
        for part in (
            tether_bin,
            "agent",
            "start",
            "--config",
            config_path,
        )
    )
    return "\n".join(
        [
            "[Unit]",
            "Description=Tether Agent",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={argv}",
            "Restart=always",
            "RestartSec=5",
            "Environment=TETHER_NO_UPGRADE_CHECK=1",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def render_launchd_plist(
    *,
    tether_bin: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    home: str | os.PathLike[str] | None = None,
) -> str:
    base = Path(home) if home is not None else Path.home()
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [
            str(tether_bin),
            "agent",
            "start",
            "--config",
            str(config_path),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {"TETHER_NO_UPGRADE_CHECK": "1"},
        "StandardOutPath": str(base / "Library" / "Logs" / "tether-agent.log"),
        "StandardErrorPath": str(base / "Library" / "Logs" / "tether-agent.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True).decode("utf-8")


def build_service_plan(
    *,
    kind: str = "auto",
    config_path: str | os.PathLike[str],
    tether_bin: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> ServicePlan:
    resolved_kind = detect_service_kind() if kind == "auto" else kind
    resolved_tether = resolve_tether_bin(tether_bin)
    target = service_path(resolved_kind, home=home)
    if resolved_kind == "systemd":
        content = render_systemd_user_service(
            tether_bin=resolved_tether,
            config_path=config_path,
        )
    elif resolved_kind == "launchd":
        content = render_launchd_plist(
            tether_bin=resolved_tether,
            config_path=config_path,
            home=home,
        )
    else:
        raise ValueError(f"unsupported service kind: {resolved_kind}")
    return ServicePlan(kind=resolved_kind, path=target, content=content)


def write_service(plan: ServicePlan) -> Path:
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(plan.content, encoding="utf-8")
    return plan.path


def remove_service(
    *,
    kind: str = "auto",
    home: str | os.PathLike[str] | None = None,
) -> Path:
    resolved_kind = detect_service_kind() if kind == "auto" else kind
    target = service_path(resolved_kind, home=home)
    if target.exists():
        target.unlink()
    return target
