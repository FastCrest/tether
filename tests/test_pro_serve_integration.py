"""Reachable-path and strict-deadline tests for paid HTTP serving."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from tether.pro.license import LicenseCorrupt, LicenseHeartbeatStale, issue_dev_license
from tether.runtime import server as runtime_server


def _export_dir(tmp_path):
    path = tmp_path / "export"
    path.mkdir()
    (path / "model.onnx").write_bytes(b"stub")
    (path / "tether_config.json").write_text(json.dumps({
        "model_type": "gr00t",
        "action_dim": 7,
    }))
    return path


def _loaded_license(*, valid_until: int | None = None):
    now = int(time.time())
    return SimpleNamespace(
        license_id="lic_test",
        customer_id="acme",
        tier="pro",
        expires_at=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
        attestation_valid_until=valid_until or now + 3600,
        heartbeat_cache_file="/tmp/test-heartbeat-cache",
    )


def _patch_paid_load(monkeypatch, loaded, calls):
    from tether.pro import activate, license as license_module

    monkeypatch.setattr(
        activate,
        "probe_hardware_binding",
        lambda: {"gpu_uuid": "GPU-test", "gpu_name": "Test GPU", "cpu_count": 8},
    )

    def fake_load_license(**kwargs):
        calls.append(kwargs)
        if isinstance(loaded, Exception):
            raise loaded
        return loaded

    monkeypatch.setattr(license_module, "load_license", fake_load_license)


def test_programmatic_create_app_loads_and_attaches_paid_license(tmp_path, monkeypatch):
    export = _export_dir(tmp_path)
    calls = []
    loaded = _loaded_license()
    _patch_paid_load(monkeypatch, loaded, calls)

    app = runtime_server.create_app(
        str(export), device="cpu", pro=True, pro_license="/licenses/pro.license",
    )

    assert calls[0]["path"] == "/licenses/pro.license"
    assert app.state.tether_server.pro_license is loaded
    assert app.state.tether_server._pro_license_deadline == runtime_server._paid_license_deadline(loaded)


def test_programmatic_pro_uses_environment_path(tmp_path, monkeypatch):
    export = _export_dir(tmp_path)
    calls = []
    _patch_paid_load(monkeypatch, _loaded_license(), calls)
    monkeypatch.setenv("TETHER_PRO_LICENSE", "/env/pro.license")
    runtime_server.create_app(str(export), device="cpu", pro=True)
    assert calls[0]["path"] == "/env/pro.license"


def test_cli_pro_flag_forwards_configured_license_path(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from tether.cli import app as cli_app
    import uvicorn

    export = _export_dir(tmp_path)
    observed = {}

    def fake_create_app(*args, **kwargs):
        observed.update({"args": args, "kwargs": kwargs})
        return object()

    monkeypatch.setattr(runtime_server, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    result = CliRunner().invoke(
        cli_app,
        [
            "serve", str(export), "--device", "cpu", "--pro",
            "--pro-license", "/cli/pro.license", "--no-prewarm",
        ],
        env={"TETHER_SKIP_ONBOARDING": "1"},
    )
    assert result.exit_code == 0, result.output
    assert observed["kwargs"]["pro"] is True
    assert observed["kwargs"]["pro_license"] == "/cli/pro.license"


@pytest.mark.parametrize(
    "alternate_args",
    [
        ["--ros2"],
        ["--mcp", "--mcp-transport", "stdio"],
        ["--mcp", "--mcp-transport", "http"],
    ],
)
def test_cli_paid_mode_rejects_ungated_ros2_and_mcp_before_startup(
    tmp_path, monkeypatch, alternate_args,
):
    from typer.testing import CliRunner

    from tether.cli import app as cli_app

    export = _export_dir(tmp_path)
    started = []
    monkeypatch.setattr(
        runtime_server,
        "create_app",
        lambda *_args, **_kwargs: started.append("http") or object(),
    )
    result = CliRunner().invoke(
        cli_app,
        ["serve", str(export), "--device", "cpu", "--pro", *alternate_args],
        env={"TETHER_SKIP_ONBOARDING": "1"},
    )
    assert result.exit_code == 1, result.output
    assert "does not yet enforce the signed-license lease" in result.output
    assert started == []


def test_cli_paid_mode_keeps_zmq_refusal(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tether.cli import app as cli_app

    export = _export_dir(tmp_path)
    started = []
    monkeypatch.setattr(
        runtime_server,
        "create_app",
        lambda *_args, **_kwargs: started.append("http") or object(),
    )
    result = CliRunner().invoke(
        cli_app,
        ["serve", str(export), "--device", "cpu", "--pro", "--transport", "zmq"],
        env={"TETHER_SKIP_ONBOARDING": "1"},
    )
    assert result.exit_code == 1, result.output
    assert "ZMQ transport has no signed-lease admission gate" in result.output
    assert started == []


def test_paid_startup_fails_loudly_when_first_heartbeat_cannot_validate(
    tmp_path, monkeypatch,
):
    export = _export_dir(tmp_path)
    calls = []
    _patch_paid_load(monkeypatch, LicenseHeartbeatStale("no verified cache"), calls)
    with pytest.raises(LicenseHeartbeatStale, match="no verified cache"):
        runtime_server.create_app(str(export), device="cpu", pro=True)


def test_unsigned_v1_cannot_reach_paid_serve_through_programmatic_path(
    tmp_path, monkeypatch,
):
    export = _export_dir(tmp_path)
    license_path = tmp_path / "legacy.license"
    from tether.pro import activate
    from tether.pro.license import HardwareFingerprintLite

    hardware = HardwareFingerprintLite("GPU-test", "Test GPU", 8)
    issue_dev_license(customer_id="acme", hardware=hardware, path=license_path)
    monkeypatch.setattr(activate, "probe_hardware_binding", lambda: {
        "gpu_uuid": hardware.gpu_uuid,
        "gpu_name": hardware.gpu_name,
        "cpu_count": hardware.cpu_count,
    })
    monkeypatch.setenv("TETHER_DEV", "1")
    with pytest.raises(LicenseCorrupt, match="require signed version 2"):
        runtime_server.create_app(
            str(export), device="cpu", pro=True, pro_license=license_path,
        )


def test_license_path_without_explicit_paid_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires pro=True"):
        runtime_server.create_app(
            str(_export_dir(tmp_path)), device="cpu", pro_license="pro.license",
        )


def test_paid_admission_fails_at_exact_deadline_and_act_returns_503(tmp_path, monkeypatch):
    export = _export_dir(tmp_path)
    now = int(time.time())
    calls = []
    loaded = _loaded_license(valid_until=now - 300)
    _patch_paid_load(monkeypatch, loaded, calls)
    app = runtime_server.create_app(str(export), device="cpu", pro=True)
    server = app.state.tether_server
    server._pro_license_deadline = now

    assert runtime_server._paid_license_admitted(server, now_s=now) is False

    async def request_act():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            return await client.post("/act", json={"instruction": "test"})

    response = asyncio.run(request_act())
    assert response.status_code == 503
    assert response.json()["error"] == "pro-license-lease-expired"


def test_late_renewal_cannot_advance_or_restore_paid_admission():
    previous = 1000
    server = SimpleNamespace(
        pro_license=_loaded_license(),
        _pro_license_deadline=previous,
        health_state="ready",
    )
    attestation = SimpleNamespace(valid_until=5000)
    assert runtime_server._accept_heartbeat_renewal(
        server, attestation, previous_deadline=previous, completed_at_s=previous,
    ) is False
    assert server._pro_license_deadline == previous
    assert server.health_state == "degraded"


def test_on_time_renewal_atomically_advances_deadline():
    previous = int(time.time()) + 60
    server = SimpleNamespace(
        pro_license=_loaded_license(),
        _pro_license_deadline=previous,
        health_state="ready",
    )
    attestation = SimpleNamespace(valid_until=previous + 3600)
    assert runtime_server._accept_heartbeat_renewal(
        server, attestation, previous_deadline=previous, completed_at_s=previous - 1,
    ) is True
    assert server._pro_license_deadline > previous
    assert server.health_state == "ready"


def test_refresh_schedule_preserves_safety_margin():
    assert runtime_server._heartbeat_refresh_delay(deadline_s=1000, now_s=980) == 0
    assert runtime_server._heartbeat_refresh_delay(deadline_s=1000, now_s=900) == 70
    assert runtime_server._heartbeat_refresh_delay(deadline_s=2000, now_s=1000) == 300


def test_timed_out_heartbeat_thread_cannot_mutate_cache_or_deadline(tmp_path):
    cache = tmp_path / "lease.heartbeat"
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    previous = int(time.time()) + 1
    server = SimpleNamespace(
        pro_license=_loaded_license(),
        _pro_license_deadline=previous,
        health_state="ready",
    )
    attestation = SimpleNamespace(
        valid_until=previous + 3600,
        to_dict=lambda: {"valid_until": previous + 3600},
    )

    def blocked_send_heartbeat(**kwargs):
        started.set()
        release.wait(timeout=5)
        # This models the old authority leak: a non-None cache path would let
        # the uncancellable thread write after its await had timed out.
        if kwargs["cache_path"] is not None:
            cache.write_text("late authority")
        finished.set()
        return attestation

    async def exercise_timeout():
        refresh = asyncio.create_task(runtime_server._refresh_paid_heartbeat_once(
            server,
            send_heartbeat_fn=blocked_send_heartbeat,
            license_id="lic_test",
            hardware_fingerprint="fp",
            tether_version="test",
            license_expires_at=server.pro_license.expires_at,
            cache_path=cache,
            previous_deadline=previous,
        ))
        assert await asyncio.to_thread(started.wait, 2)
        # asyncio.TimeoutError is only an alias of builtins.TimeoutError on
        # Python 3.11+; use the asyncio spelling for the supported 3.10 job.
        with pytest.raises(asyncio.TimeoutError):
            await refresh
        assert int(time.time()) >= previous
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)

    asyncio.run(exercise_timeout())
    assert server._pro_license_deadline == previous
    assert not cache.exists()
