"""WIM's expiry contract, independent of inference hardware or wall time."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tether.runtime.buffer import ActionChunkBuffer


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


CHUNK = np.arange(5, dtype=float).reshape(-1, 1)


def test_late_inference_skips_original_indices_and_rejects_all_late():
    clock = Clock(0.025)
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4, control_hz=100)
    assert buf.push_chunk(CHUNK, t_obs=0) == 2
    np.testing.assert_array_equal(buf.pop_next(), [3])
    clock.now = 0.100
    assert buf.push_chunk(CHUNK, t_obs=0) == 0
    assert buf.pop_next() is None
    stats = buf.stats()
    assert stats.expired_actions == 8
    assert stats.late_chunks == 2
    assert stats.dropped_replans == 1
    assert stats.max_observed_lateness_ms == 100


def test_dropped_replan_discards_tail_without_synthesizing():
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4)
    buf.push_chunk(CHUNK)
    np.testing.assert_array_equal(buf.pop_next(), [0])
    # No push arrives. The next action is due at 10 ms, not at push time.
    clock.now = 0.014001
    assert buf.pop_next() is None
    assert buf.size == 0
    assert buf.pop_next() is None
    assert buf.should_replan()
    assert buf.stats().expired_actions == 4
    assert buf.stats().late_chunks == 1
    assert buf.stats().dropped_replans == 0  # no submission is observable


@pytest.mark.parametrize("hz", [50, 100, 200, 1000])
@pytest.mark.parametrize("bound", [0, 2, 5.9])
def test_per_action_deadline_ordering(hz, bound):
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=bound, control_hz=hz)
    buf.push_chunk(CHUNK)
    buf.pop_next()
    due = 1 / hz
    clock.now = due + bound / 1000
    assert buf.pop_next() is not None  # equality accepted
    clock.now = 2 / hz + (bound + 0.001) / 1000
    assert clock.now < 2 / hz + 0.006
    assert buf.pop_next() is None
    assert buf.stats().expired_actions == 3


@pytest.mark.parametrize("bound_name", ["max_lateness_ms", "max_age_ms"])
def test_exact_boundary_push_pop_and_next_float(bound_name):
    clock = Clock(10)
    buf = ActionChunkBuffer(clock=clock, **{bound_name: 4})
    clock.now = 10.004
    assert buf.push_chunk(np.ones((1, 1)), t_obs=10) == 1
    assert buf.pop_next() is not None
    clock.now = np.nextafter(10.004, np.inf)
    assert buf.push_chunk(np.ones((1, 1)), t_obs=10) == 0
    assert buf.stats().expired_actions == 1


def test_total_age_cap_is_independent_of_lateness():
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4, max_age_ms=5)
    buf.push_chunk(CHUNK)
    buf.pop_next()
    clock.now = 0.005001  # action 1 is early, but chunk is too old
    assert buf.pop_next() is None
    assert buf.stats().expired_actions == 4


def test_forward_jump_expires_and_backward_jump_never_rejuvenates():
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4)
    buf.push_chunk(CHUNK)
    clock.now = 1
    buf.stats()  # records the clock high-water mark, without expiring
    clock.now = -10
    assert buf.pop_next() is None
    assert buf.stats().expired_actions == 5
    assert buf.stats().last_replan_age_ms == 1000
    clock.now = 2
    buf.push_chunk(CHUNK)
    assert buf.pop_next() is not None  # a genuinely fresh chunk recovers


@pytest.mark.parametrize("hz", [50, 100, 1000])
def test_default_off_fifo_append_overwrite_clear(hz):
    clock = Clock()
    buf = ActionChunkBuffer(capacity=5, clock=clock, control_hz=hz)
    buf.push_chunk(CHUNK)
    clock.now = 1e6
    for expected in CHUNK[:2]:
        np.testing.assert_array_equal(buf.pop_next(), expected)
    assert buf.push_chunk(CHUNK, overwrite_stale=False) == 5  # legacy requested count
    expected = np.concatenate((CHUNK[2:], CHUNK[:2]))
    np.testing.assert_array_equal(buf.peek_all(), expected)
    buf.push_chunk(CHUNK)
    assert buf.stats().stale_overwrites == 5
    buf.clear()
    assert buf.stats().stale_overwrites == 10
    assert buf.pop_next() is None
    assert buf.stats().expired_actions == 0
    assert buf.stats().dropped_replans == 0


@pytest.mark.parametrize("kwargs", [
    {"max_lateness_ms": -1}, {"max_age_ms": float("nan")},
    {"max_lateness_ms": float("inf")}, {"control_hz": 0},
    {"control_hz": float("nan")}, {"capacity": -1},
])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        ActionChunkBuffer(**kwargs)


@pytest.mark.parametrize("obs", [float("nan"), float("inf"), 1])
def test_bad_observation_timestamp_cannot_bypass_expiry(obs):
    buf = ActionChunkBuffer(clock=Clock(), max_lateness_ms=4)
    buf.push_chunk(CHUNK)
    with pytest.raises(ValueError, match="t_obs"):
        buf.push_chunk(CHUNK, t_obs=obs)
    assert buf.size == 5


def test_concurrent_push_pop_stats_and_peeks_account_for_every_action():
    clock = Clock(1)
    buf = ActionChunkBuffer(capacity=5, clock=clock, max_lateness_ms=4)
    barrier = threading.Barrier(4)
    def push(fresh):
        barrier.wait()
        for _ in range(1000):
            buf.push_chunk(CHUNK, t_obs=1 if fresh else 0)
        return 0
    def pop():
        barrier.wait()
        consumed = 0
        for _ in range(5000):
            consumed += buf.pop_next() is not None
            buf.peek_all()
            buf.stats()
        return consumed
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(push, True), pool.submit(push, False),
                   pool.submit(pop), pool.submit(pop)]
        consumed = sum(f.result(timeout=20) for f in futures)
    stats = buf.stats()
    assert stats.replans == 2000
    assert stats.dropped_replans == 1000
    assert stats.expired_actions == 5000
    assert consumed + stats.size + stats.stale_overwrites + stats.expired_actions == 10000


# These are all create_app backend classes; no inference/RTC qualification is
# implied by configuring their common execution buffer on an unloaded instance.
BACKENDS = [
    ("server", "TetherServer"), ("decomposed_server", "Pi05DecomposedServer"),
    ("pi0_onnx_server", "Pi0OnnxServer"), ("pi05_onnx_server", "Pi05OnnxServer"),
    ("smolvla_onnx_server", "SmolVLAOnnxServer"), ("smolvla_native", "SmolVLANativeServer"),
]


@pytest.mark.parametrize("module,name", BACKENDS)
@pytest.mark.parametrize("hz", [50, 100, 200, 1000])
def test_backend_buffer_deadline_ordering_and_rtc_identity(module, name, hz):
    from tether.runtime.server import configure_action_buffer
    cls = getattr(importlib.import_module(f"tether.runtime.{module}"), name)
    server = cls.__new__(cls)
    server.config = {"chunk_size": 5}
    server.rtc_adapter = SimpleNamespace(buffer=None)
    clock = Clock()
    buf = configure_action_buffer(server, replan_hz=20, execute_hz=hz,
                                  max_lateness_ms=4, clock=clock)
    assert server.rtc_adapter.buffer is buf is server._action_buffer
    assert buf.control_hz == hz
    buf.push_chunk(CHUNK)
    buf.pop_next()
    clock.now = 1 / hz + 0.004001
    assert clock.now < 1 / hz + 0.006
    assert buf.pop_next() is None
    assert buf.stats().expired_actions == 4


def test_rtc_peeks_and_previous_leftover_remain_read_only():
    from tether.runtime.rtc_adapter import RtcAdapter, RtcAdapterConfig
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4)
    adapter = RtcAdapter(policy=SimpleNamespace(), action_buffer=buf,
                         config=RtcAdapterConfig(enabled=False))
    adapter.merge_and_update(CHUNK, elapsed_time=0.01, t_obs=0)
    clock.now = 0.005
    snapshot = buf.peek_all()
    np.testing.assert_array_equal(snapshot, CHUNK)  # no lazy expiry on peek
    snapshot[:] = -1
    np.testing.assert_array_equal(buf.peek_next(), CHUNK[0])
    adapter.merge_and_update(CHUNK + 10, elapsed_time=0.01)
    np.testing.assert_array_equal(adapter._prev_chunk_left_over, CHUNK)
    clock.now = 0.010
    assert buf.pop_next() is None
    np.testing.assert_array_equal(adapter._prev_chunk_left_over, CHUNK)
    adapter.merge_and_update(CHUNK + 20, elapsed_time=0.01)
    assert adapter._prev_chunk_left_over is None  # expired tail was discarded
    assert adapter.get_stats()["action_buffer"]["expired_actions"] == 5


def test_snapshot_prometheus_and_record(tmp_path):
    from tether.observability import render_metrics, set_action_buffer_stats
    from tether.runtime.record import RecordWriter
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4)
    buf.push_chunk(CHUNK)
    clock.now = 1
    assert buf.pop_next() is None
    snapshot = asdict(buf.stats())
    set_action_buffer_stats(snapshot)
    metrics = render_metrics().decode()
    for key in ("expired_actions", "late_chunks", "dropped_replans", "max_observed_lateness_ms"):
        assert f"tether_action_buffer_{key} {float(snapshot[key])}" in metrics
    import json
    rec = RecordWriter(
        record_dir=tmp_path, model_hash="test", config_hash="test",
        export_dir=tmp_path, model_type="gr00t", export_kind="monolithic_onnx",
        providers=[], gzip_output=False,
    )
    for kwargs in ({"action_buffer": snapshot}, {}):
        rec.write_request(chunk_id=0, image_b64=None, instruction="test", state=None,
                          actions=[], action_dim=1, latency_total_ms=0, **kwargs)
    rec.close()
    rows = [json.loads(line) for line in rec.filepath.read_text().splitlines()]
    assert rows[1]["action_buffer"] == snapshot
    assert "action_buffer" not in rows[2]


def test_server_rejects_unusable_or_invalid_expiry_before_loading(tmp_path):
    from tether.runtime.server import create_app
    with pytest.raises(ValueError, match="requires replan_hz"):
        create_app(str(tmp_path), max_lateness_ms=4)
    with pytest.raises(ValueError, match="max_lateness_ms"):
        create_app(str(tmp_path), max_lateness_ms=float("nan"))


def test_late_prefix_is_skipped_before_capacity_limit():
    clock = Clock(0.035)
    buf = ActionChunkBuffer(capacity=2, clock=clock, max_lateness_ms=4)
    assert buf.push_chunk(np.arange(10).reshape(-1, 1), t_obs=0) == 2
    np.testing.assert_array_equal(buf.peek_all(), [[4], [5]])
    assert buf.stats().expired_actions == 4
    assert buf.stats().dropped_replans == 0


@pytest.mark.parametrize("model_type,export_kind,artifacts,native", [
    ("gr00t", "monolithic_onnx", ["model.onnx"], False),
    ("pi05_decomposed", "decomposed_onnx", ["vlm_prefix.onnx", "expert_denoise.onnx"], False),
    ("pi0", "monolithic_onnx", ["model.onnx"], False),
    ("pi05", "monolithic_onnx", ["model.onnx"], False),
    ("smolvla", "monolithic_onnx", ["model.onnx"], False),
    ("smolvla", "decomposed_onnx", ["expert_stack.onnx"], False),
    ("smolvla", "decomposed_onnx", ["expert_stack.onnx"], True),
])
def test_real_app_lifespan_config_and_metrics(
    tmp_path, monkeypatch, model_type, export_kind, artifacts, native,
):
    from fastapi.testclient import TestClient
    from tests.export_config_factory import write_test_export_config
    from tether.runtime.server import create_app
    monkeypatch.delenv("TETHER_NATIVE", raising=False)
    if native:
        monkeypatch.setenv("TETHER_NATIVE", "1")
    monkeypatch.setenv("TETHER_NO_CONTRIB_NUDGE", "1")
    write_test_export_config(tmp_path, model_type=model_type, export_kind=export_kind,
                             artifacts=artifacts, action_chunk_size=5)
    app = create_app(str(tmp_path), device="cpu", prewarm=False,
                     replan_hz=20, execute_hz=100, max_lateness_ms=4)
    server = app.state.tether_server
    def load():
        server._ready = True
        server.chunk_size = 5
        server.action_dim = 1
    monkeypatch.setattr(server, "load", load)
    with TestClient(app) as client:
        buf = server._action_buffer
        assert buf.max_lateness_ms == 4
        assert buf.control_hz == 100
        clock = Clock()
        buf._clock = clock
        buf.push_chunk(CHUNK)
        clock.now = 0.004001
        assert buf.pop_next() is None
        cfg = client.get("/config").json()["action_buffer"]
        assert cfg["max_lateness_ms"] == 4
        assert cfg["http_act_consumes_buffer"] is False
        assert cfg["stats"]["expired_actions"] == 5
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "tether_action_buffer_expired_actions 5.0" in metrics.text


def test_invalid_clock_and_never_submitted_chunk():
    clock = Clock()
    buf = ActionChunkBuffer(clock=clock, max_lateness_ms=4)
    clock.now = 1000
    assert buf.pop_next() is None
    assert buf.stats().dropped_replans == 0
    assert buf.stats().expired_actions == 0
    buf.push_chunk(CHUNK)
    clock.now = float("nan")
    with pytest.raises(ValueError, match="clock"):
        buf.pop_next()
    assert buf.size == 5


def test_concurrent_clock_progress_and_expiring_pops():
    class AdvancingClock:
        now = 0.0
        def __call__(self):
            self.now += 0.001
            return self.now
    buf = ActionChunkBuffer(capacity=5, clock=AdvancingClock(), max_lateness_ms=0)
    barrier = threading.Barrier(3)
    def producer():
        barrier.wait()
        for _ in range(1000):
            buf.push_chunk(CHUNK)
        return 0
    def consumer():
        barrier.wait()
        count = 0
        for _ in range(2000):
            count += buf.pop_next() is not None
        return count
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(producer), pool.submit(consumer), pool.submit(consumer)]
        consumed = sum(f.result(timeout=20) for f in futures)
    consumed += buf.pop_next() is not None  # exercise the final producer's tail too
    stats = buf.stats()
    assert stats.replans == 1000
    assert consumed + stats.size + stats.stale_overwrites + stats.expired_actions == 5000
    assert stats.expired_actions > 0


def test_server_constructor_bounds_are_used_by_configure_replan(tmp_path):
    from tests.export_config_factory import write_test_export_config
    from tether.runtime.server import TetherServer
    write_test_export_config(tmp_path, model_type="smolvla", export_kind="decomposed_onnx",
                             artifacts=["expert_stack.onnx"])
    server = TetherServer(tmp_path, device="cpu", max_lateness_ms=4, max_age_ms=100)
    clock = Clock()
    server.configure_replan(20, 100, clock=clock)
    buf = server._action_buffer
    assert buf.max_lateness_ms == 4
    assert buf.max_age_ms == 100
    buf.push_chunk(CHUNK)
    clock.now = 0.004001
    assert buf.pop_next() is None
    server._inference_executor.shutdown()


def test_default_off_zero_capacity_preserves_legacy_empty_buffer():
    buf = ActionChunkBuffer(capacity=0, clock=Clock())
    assert buf.push_chunk(CHUNK) == 0
    assert buf.pop_next() is None
    assert buf.stats().size == 0


@pytest.mark.parametrize("bound", [None, 4])
def test_bad_replan_config_preserves_default_off_but_never_drops_requested_expiry(
    tmp_path, monkeypatch, bound,
):
    from fastapi.testclient import TestClient
    from tests.export_config_factory import write_test_export_config
    from tether.runtime.server import create_app
    monkeypatch.delenv("TETHER_NATIVE", raising=False)
    monkeypatch.setenv("TETHER_NO_CONTRIB_NUDGE", "1")
    write_test_export_config(tmp_path, model_type="gr00t", export_kind="monolithic_onnx",
                             artifacts=["model.onnx"])
    app = create_app(str(tmp_path), device="cpu", prewarm=False,
                     replan_hz=200, execute_hz=100, max_lateness_ms=bound)
    server = app.state.tether_server
    def load():
        server._ready = True
        server.chunk_size = 5
        server.action_dim = 1
    monkeypatch.setattr(server, "load", load)
    if bound is not None:
        with pytest.raises(ValueError, match="backwards"):
            with TestClient(app):
                pass
    else:
        with TestClient(app) as client:
            assert server._action_buffer is None
            assert client.get("/health").status_code == 200
            assert "action_buffer" not in client.get("/config").json()
