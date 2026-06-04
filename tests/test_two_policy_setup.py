"""Tests for src/tether/runtime/two_policy_setup.py — composes the policy-
versioning substrate for the FastAPI serve runtime.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tether.runtime.two_policy_setup import (
    TwoPolicyServingState,
    _estimate_export_size_bytes,
    _probe_total_gpu_bytes,
    _safe_hash,
    setup_two_policy_serving,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubServer:
    """Minimal TetherServer stand-in for tests."""
    model_id: str
    _model_hash: str
    _action_guard: Any = None
    export_dir: str = ""
    loaded: bool = False

    async def predict_from_base64_async(self, *, image_b64, instruction, state):
        return {"actions": [[0.1] * 7]}

    def load(self):
        self.loaded = True


def _make_server_factory(model_id: str = "stub-model"):
    """Return a server_factory closure that creates _StubServer instances."""
    counter = {"n": 0}

    def _factory(*, export_dir: str, **kwargs):
        counter["n"] += 1
        srv = _StubServer(
            model_id=f"{model_id}-{counter['n']}",
            _model_hash=f"abcd{counter['n']:012x}",
            export_dir=export_dir,
        )
        srv.load()
        return srv

    return _factory


@pytest.fixture
def fake_export_a(tmp_path):
    p = tmp_path / "export_a"
    p.mkdir()
    (p / "model.onnx").write_bytes(b"x" * 1000)
    return p


@pytest.fixture
def fake_export_b(tmp_path):
    p = tmp_path / "export_b"
    p.mkdir()
    (p / "model.onnx").write_bytes(b"y" * 1000)
    return p


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_setup_rejects_no_rtc_false(fake_export_a, fake_export_b):
    """no_rtc=False must fail (per ADR: RTC carry-over is per-policy)."""
    with pytest.raises(ValueError, match="--no-rtc"):
        asyncio.run(setup_two_policy_serving(
            export_a=fake_export_a, export_b=fake_export_b,
            no_rtc=False,
            server_factory=_make_server_factory(),
            skip_memory_check=True,
        ))


def test_setup_rejects_invalid_split(fake_export_a, fake_export_b):
    with pytest.raises(ValueError, match="split_a_percent"):
        asyncio.run(setup_two_policy_serving(
            export_a=fake_export_a, export_b=fake_export_b,
            split_a_percent=150,
            server_factory=_make_server_factory(),
            skip_memory_check=True,
        ))


def test_setup_rejects_missing_export_a(tmp_path, fake_export_b):
    with pytest.raises(FileNotFoundError, match="--policy-a"):
        asyncio.run(setup_two_policy_serving(
            export_a=tmp_path / "nope-a", export_b=fake_export_b,
            server_factory=_make_server_factory(),
            skip_memory_check=True,
        ))


def test_setup_rejects_missing_export_b(fake_export_a, tmp_path):
    with pytest.raises(FileNotFoundError, match="--policy-b"):
        asyncio.run(setup_two_policy_serving(
            export_a=fake_export_a, export_b=tmp_path / "nope-b",
            server_factory=_make_server_factory(),
            skip_memory_check=True,
        ))


# ---------------------------------------------------------------------------
# Memory check
# ---------------------------------------------------------------------------


def test_setup_rejects_when_2x_memory_exceeds_safety(fake_export_a, fake_export_b, monkeypatch):
    """Stub model_size large + total_gpu small -> ValueError."""
    # Force the probes to return values that fail the safety check
    monkeypatch.setattr(
        "tether.runtime.two_policy_setup._estimate_export_size_bytes",
        lambda p: 8 * 10**9,  # 8GB per model
    )
    monkeypatch.setattr(
        "tether.runtime.two_policy_setup._probe_total_gpu_bytes",
        lambda: 16 * 10**9,  # 16GB total
    )
    # 2 * 8GB = 16GB > 0.7 * 16GB = 11.2GB -> fail
    with pytest.raises(ValueError, match="VRAM"):
        asyncio.run(setup_two_policy_serving(
            export_a=fake_export_a, export_b=fake_export_b,
            server_factory=_make_server_factory(),
        ))


def test_setup_skip_memory_check_bypasses(fake_export_a, fake_export_b):
    """skip_memory_check=True bypasses the refuse-to-load check."""
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))
    assert state.server_a.loaded
    assert state.server_b.loaded


def test_setup_proceeds_when_probes_return_zero(fake_export_a, fake_export_b, monkeypatch):
    """When probes can't determine sizes (CPU-only host), proceed
    with a warning instead of blocking."""
    monkeypatch.setattr(
        "tether.runtime.two_policy_setup._estimate_export_size_bytes",
        lambda p: 0,
    )
    monkeypatch.setattr(
        "tether.runtime.two_policy_setup._probe_total_gpu_bytes",
        lambda: 0,
    )
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
    ))
    # Loaded successfully -- check was skipped
    assert state.server_a is not None
    assert state.server_b is not None


# ---------------------------------------------------------------------------
# Happy-path composition
# ---------------------------------------------------------------------------


def test_setup_loads_both_servers(fake_export_a, fake_export_b):
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))
    assert state.server_a.loaded
    assert state.server_b.loaded
    assert state.server_a is not state.server_b  # distinct instances


def test_setup_builds_dispatcher(fake_export_a, fake_export_b):
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        split_a_percent=80,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))
    assert state.dispatcher is not None
    assert state.split_a_percent == 80
    assert state.dispatcher.split_a_percent == 80


def test_setup_policy_bundles_carry_export_meta(fake_export_a, fake_export_b):
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))
    assert state.policy_a.slot == "a"
    assert state.policy_b.slot == "b"
    assert state.policy_a.export_dir == str(fake_export_a)
    assert state.policy_b.export_dir == str(fake_export_b)
    # 16-hex hash convention
    assert len(state.policy_a.model_hash) == 16
    assert len(state.policy_b.model_hash) == 16


def test_setup_no_rtc_enforced_in_policy_bundles(fake_export_a, fake_export_b):
    """Per ADR: 2-policy mode forces rtc_adapter=None on both bundles."""
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))
    assert state.policy_a.rtc_adapter is None
    assert state.policy_b.rtc_adapter is None
    assert state.no_rtc_enforced is True


def test_setup_dispatcher_routes_to_correct_predict(fake_export_a, fake_export_b):
    """Wired predict_a/predict_b must hit the right server."""
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))

    # Stub request shape (dataclass-like with image/instruction/state attrs)
    @dataclass
    class _Req:
        image: str = "img-data"
        instruction: str = "test"
        state: list[float] | None = None

    # Force routing to a specific slot by setting split=100 -> all to A
    state2 = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        split_a_percent=100,
        server_factory=_make_server_factory(model_id="forced-a"),
        skip_memory_check=True,
    ))
    result, decision = asyncio.run(state2.dispatcher.predict(
        request=_Req(), episode_id="ep_force_a", request_id="req_1",
    ))
    assert decision.slot == "a"
    assert "actions" in result


def test_setup_runtime_factory_invoked_when_provided(fake_export_a, fake_export_b):
    """When runtime_factory is provided, it builds a per-slot runtime
    that gets attached to each Policy bundle."""
    runtime_calls = []

    def _runtime_factory(*, server, slot):
        runtime_calls.append({"slot": slot, "server": server})
        return f"runtime_for_{slot}"  # placeholder runtime

    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        runtime_factory=_runtime_factory,
        skip_memory_check=True,
    ))
    # Both slots got their own runtime built
    slots = [c["slot"] for c in runtime_calls]
    assert sorted(slots) == ["a", "b"]
    assert state.runtime_a == "runtime_for_a"
    assert state.runtime_b == "runtime_for_b"
    assert state.policy_a.runtime == "runtime_for_a"
    assert state.policy_b.runtime == "runtime_for_b"


def test_setup_default_runtime_factory_none_skips_runtimes(fake_export_a, fake_export_b):
    """When runtime_factory is None (default for tests/legacy backends),
    the runtime fields are None on the bundles."""
    state = asyncio.run(setup_two_policy_serving(
        export_a=fake_export_a, export_b=fake_export_b,
        server_factory=_make_server_factory(),
        skip_memory_check=True,
    ))
    assert state.runtime_a is None
    assert state.runtime_b is None


# ---------------------------------------------------------------------------
# Helpers (probe + hash)
# ---------------------------------------------------------------------------


def test_estimate_export_size_sums_weight_files(tmp_path):
    p = tmp_path / "export"
    p.mkdir()
    (p / "model.onnx").write_bytes(b"x" * 100)
    (p / "model.onnx.data").write_bytes(b"x" * 200)
    (p / "weights.safetensors").write_bytes(b"x" * 300)
    (p / "extra.bin").write_bytes(b"x" * 400)
    (p / "ignored.txt").write_bytes(b"x" * 9999)  # not counted
    assert _estimate_export_size_bytes(p) == 1000


def test_estimate_export_size_zero_for_missing_dir(tmp_path):
    assert _estimate_export_size_bytes(tmp_path / "nope") == 0


def test_estimate_export_size_zero_for_no_weights(tmp_path):
    p = tmp_path / "empty"
    p.mkdir()
    (p / "config.json").write_text("{}")  # not a weight file
    assert _estimate_export_size_bytes(p) == 0


def test_safe_hash_uses_server_attribute_when_set():
    server = _StubServer(model_id="x", _model_hash="deadbeefcafe1234")
    assert _safe_hash(server, Path("/tmp/x")) == "deadbeefcafe1234"


def test_safe_hash_falls_back_to_path_sha(tmp_path):
    server = object()  # no _model_hash attribute
    h = _safe_hash(server, tmp_path / "export_xyz")
    assert isinstance(h, str)
    assert len(h) == 16
    # Stable across calls for the same path
    h2 = _safe_hash(server, tmp_path / "export_xyz")
    assert h == h2


def test_probe_total_gpu_bytes_returns_zero_when_no_probe_succeeds(monkeypatch):
    """Force both torch + nvidia-smi paths to fail -> 0."""
    # Monkey-patch torch import to fail
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)  # next import errors
    # subprocess.run for nvidia-smi: simulate command-not-found
    import subprocess
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    assert _probe_total_gpu_bytes() == 0
