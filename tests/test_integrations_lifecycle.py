"""Integration connect/disconnect lifecycle.

Pins the fix for `tether connect down` being a structural no-op (the running
handle lived in a module dict that died with the CLI process) by persisting the
pid, plus the registry hardening (find_spec install-check, fail-fast start,
version-pinned spec).
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from tether.integrations import connector
from tether.integrations.registry import Integration


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TETHER_HOME", str(tmp_path))
    connector._RUNNING.clear()
    yield


def test_disconnect_stops_pid_from_a_separate_process():
    """The pid persisted at connect time lets a fresh process actually stop it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        connector._write_pid("rtsm", proc.pid)
        connector._RUNNING.clear()  # simulate: `connect down` is a new CLI process
        assert connector._pid_alive(proc.pid)

        result = connector.disconnect("rtsm")

        assert result["status"] == "stopped"
        assert result["pid"] == proc.pid
        proc.wait(timeout=5)
        assert not connector._pid_alive(proc.pid)
        assert not connector._pid_file("rtsm").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_disconnect_cleans_stale_pid_file():
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    connector._write_pid("rtsm", dead.pid)
    connector._RUNNING.clear()

    result = connector.disconnect("rtsm")

    assert result["status"] in ("not_running", "external_still_running")
    assert not connector._pid_file("rtsm").exists()


def test_is_installed_uses_find_spec_no_import():
    present = Integration(name="a", description="", pip_package="os")
    assert present.is_installed() is True
    absent = Integration(name="b", description="", pip_package="totally-not-real-pkg-xyz")
    assert absent.is_installed() is False


def test_is_installed_honors_explicit_import_name():
    # pip name != import name: declare it.
    i = Integration(name="c", description="", pip_package="pip-thing", import_name="json")
    assert i.is_installed() is True


def test_start_fails_fast_on_immediate_exit():
    integ = Integration(
        name="boom",
        description="",
        pip_package="x",
        start_command=[sys.executable, "-c", "import sys; sys.exit(3)"],
        health_url="http://localhost:59999/healthz",
    )
    with pytest.raises(RuntimeError) as ei:
        integ.start()
    assert "exited immediately" in str(ei.value)
    assert integ.log_file.exists()


def test_pip_spec_includes_extras_and_version():
    i = Integration(
        name="z", description="", pip_package="rtsm",
        pip_extras="gpu", pip_version_spec=">=1.0,<2",
    )
    assert i.pip_spec == "rtsm[gpu]>=1.0,<2"


def test_connect_disconnect_exported():
    from tether.integrations import connect, disconnect  # noqa: F401
