from __future__ import annotations

import plistlib

from tether.agent.service import (
    build_service_plan,
    remove_service,
    render_launchd_plist,
    render_systemd_user_service,
    service_path,
    write_service,
)


def test_render_systemd_user_service():
    content = render_systemd_user_service(
        tether_bin="/opt/tether/bin/tether",
        config_path="/home/edge/.tether/agent.json",
    )

    assert "Description=Tether Agent" in content
    assert "ExecStart=/opt/tether/bin/tether agent start --config /home/edge/.tether/agent.json" in content
    assert "Restart=always" in content
    assert "WantedBy=default.target" in content


def test_render_launchd_plist():
    content = render_launchd_plist(
        tether_bin="/opt/tether/bin/tether",
        config_path="/Users/edge/.tether/agent.json",
        home="/Users/edge",
    )
    payload = plistlib.loads(content.encode("utf-8"))

    assert payload["Label"] == "com.fastcrest.tether-agent"
    assert payload["ProgramArguments"] == [
        "/opt/tether/bin/tether",
        "agent",
        "start",
        "--config",
        "/Users/edge/.tether/agent.json",
    ]
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True
    assert payload["StandardOutPath"] == "/Users/edge/Library/Logs/tether-agent.log"


def test_build_and_write_systemd_service_to_temp_home(tmp_path):
    plan = build_service_plan(
        kind="systemd",
        config_path=tmp_path / "agent.json",
        tether_bin="/opt/tether/bin/tether",
        home=tmp_path,
    )

    assert plan.path == tmp_path / ".config" / "systemd" / "user" / "tether-agent.service"
    written = write_service(plan)
    assert written == plan.path
    assert written.read_text() == plan.content


def test_remove_service_only_deletes_target_path(tmp_path):
    target = service_path("launchd", home=tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text("payload")
    neighbor = target.with_name("other.plist")
    neighbor.write_text("keep")

    removed = remove_service(kind="launchd", home=tmp_path)

    assert removed == target
    assert not target.exists()
    assert neighbor.read_text() == "keep"
