"""Tests for src/tether/pro/data_collection.py — Phase 1 self-distilling-serve Day 1.

Per a2u-distilling-serve execution plan B.5 Day 1 acceptance criteria:
- CollectedEvent schema-v1 invariants (validation + serialization)
- ProDataCollector lifecycle (start/stop idempotent, drain on stop)
- record() raises QueueFull at capacity (never blocks /act)
- Background flush via daily JSONL files
- 90-day retention prune
- to_parquet_file() works with + without pandas
- hash_instruction() determinism
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tether.pro.data_collection import (
    DEFAULT_RETENTION_DAYS,
    PARQUET_SCHEMA_VERSION,
    CollectedEvent,
    ProDataCollector,
    QueueFull,
    hash_instruction,
)


# ---------------------------------------------------------------------------
# CollectedEvent
# ---------------------------------------------------------------------------


def _mk_event(**overrides) -> CollectedEvent:
    defaults = dict(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        episode_id="ep_42",
        state_vec=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        action_chunk=[[0.0] * 7 for _ in range(50)],
        reward_proxy=1.0,
        image_b64=None,
        instruction_hash="abc123",
        instruction_raw=None,
        metadata={},
    )
    defaults.update(overrides)
    return CollectedEvent(**defaults)


def test_event_rejects_reward_proxy_below_zero():
    with pytest.raises(ValueError, match="reward_proxy"):
        _mk_event(reward_proxy=-0.1)


def test_event_rejects_reward_proxy_above_one():
    with pytest.raises(ValueError, match="reward_proxy"):
        _mk_event(reward_proxy=1.5)


def test_event_accepts_boundary_reward_proxies():
    _mk_event(reward_proxy=0.0)
    _mk_event(reward_proxy=1.0)


def test_event_rejects_non_list_state_vec():
    with pytest.raises(TypeError, match="state_vec"):
        _mk_event(state_vec="not a list")  # type: ignore[arg-type]


def test_event_rejects_non_list_action_chunk():
    with pytest.raises(TypeError, match="action_chunk"):
        _mk_event(action_chunk="not a list")  # type: ignore[arg-type]


def test_event_to_row_includes_schema_version():
    e = _mk_event()
    row = e.to_row()
    assert row["schema_version"] == PARQUET_SCHEMA_VERSION
    assert row["episode_id"] == "ep_42"
    assert row["reward_proxy"] == 1.0


def test_event_to_row_is_json_safe():
    """The row dict must serialize cleanly via json.dumps."""
    e = _mk_event()
    row = e.to_row()
    # Round-trip via JSON
    restored = json.loads(json.dumps(row))
    assert restored["episode_id"] == e.episode_id
    assert restored["state_vec"] == e.state_vec


def test_event_schema_version_is_one():
    """Phase 1 = v1; bump on breaking changes."""
    assert PARQUET_SCHEMA_VERSION == 1
    assert CollectedEvent.SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# hash_instruction
# ---------------------------------------------------------------------------


def test_hash_instruction_deterministic():
    h1 = hash_instruction("pick up the red block")
    h2 = hash_instruction("pick up the red block")
    assert h1 == h2


def test_hash_instruction_differs_for_different_text():
    h1 = hash_instruction("pick up the red block")
    h2 = hash_instruction("pick up the blue block")
    assert h1 != h2


def test_hash_instruction_empty_string_for_none():
    assert hash_instruction(None) == ""
    assert hash_instruction("") == ""


def test_hash_instruction_is_sha256():
    h = hash_instruction("test")
    expected = hashlib.sha256(b"test").hexdigest()
    assert h == expected


# ---------------------------------------------------------------------------
# ProDataCollector — construction validation
# ---------------------------------------------------------------------------


def test_collector_rejects_zero_max_queue(tmp_path):
    with pytest.raises(ValueError, match="max_queue"):
        ProDataCollector(data_dir=tmp_path, max_queue=0)


def test_collector_rejects_zero_flush_every_events(tmp_path):
    with pytest.raises(ValueError, match="flush_every_events"):
        ProDataCollector(data_dir=tmp_path, flush_every_events=0)


def test_collector_rejects_zero_flush_every_seconds(tmp_path):
    with pytest.raises(ValueError, match="flush_every_seconds"):
        ProDataCollector(data_dir=tmp_path, flush_every_seconds=0)


def test_collector_data_dir_resolves_user_home():
    """`~/.tether/...` should expand to the actual home dir."""
    c = ProDataCollector(data_dir="~/.tether/test-pro-data")
    assert "~" not in str(c.data_dir)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_collector_start_creates_data_dir(tmp_path):
    new_dir = tmp_path / "new_subdir"
    c = ProDataCollector(data_dir=new_dir)
    assert not new_dir.exists()
    c.start()
    try:
        assert new_dir.exists()
    finally:
        c.stop()


def test_collector_start_then_stop_idempotent(tmp_path):
    c = ProDataCollector(data_dir=tmp_path / "d")
    c.start()
    assert c.is_running
    c.start()  # idempotent
    assert c.is_running
    c.stop()
    assert not c.is_running
    c.stop()  # idempotent


def test_collector_record_appends_to_queue(tmp_path):
    c = ProDataCollector(data_dir=tmp_path / "d", max_queue=10)
    c.start()
    try:
        c.record(_mk_event())
        assert c.events_recorded == 1
        assert c.queue_depth() >= 0  # may be 0 if writer flushed already
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


def test_collector_record_raises_queue_full_at_capacity(tmp_path):
    """Use a tiny queue + slow writer to force the QueueFull path."""
    c = ProDataCollector(
        data_dir=tmp_path / "d",
        max_queue=3,
        flush_every_events=1000,  # writer won't flush during the test
        flush_every_seconds=60,
    )
    # Don't start the writer — just exercise the queue under cap
    c._data_dir.mkdir(parents=True, exist_ok=True)
    c.record(_mk_event())
    c.record(_mk_event())
    c.record(_mk_event())
    with pytest.raises(QueueFull):
        c.record(_mk_event())
    assert c.events_dropped >= 1


# ---------------------------------------------------------------------------
# Flush + JSONL writing
# ---------------------------------------------------------------------------


def test_collector_writes_jsonl_after_flush(tmp_path):
    """Force a flush + verify the daily file exists with JSONL rows."""
    c = ProDataCollector(
        data_dir=tmp_path / "d",
        flush_every_events=1,
        flush_every_seconds=60,
    )
    c.start()
    try:
        for i in range(3):
            c.record(_mk_event(episode_id=f"ep_{i}"))
        # Wait for writer thread to flush
        time.sleep(1.0)
    finally:
        c.stop()
    # Find the daily file
    files = list((tmp_path / "d").glob("*.jsonl"))
    assert len(files) >= 1
    # Parse + verify event count
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    assert len(rows) >= 3
    assert all(r["schema_version"] == 1 for r in rows)


def test_collector_drain_on_stop(tmp_path):
    """Even with high flush thresholds, stop() should drain pending events."""
    c = ProDataCollector(
        data_dir=tmp_path / "d",
        flush_every_events=10_000,
        flush_every_seconds=600,
    )
    c.start()
    try:
        c.record(_mk_event(episode_id="ep_drain"))
    finally:
        c.stop()
    files = list((tmp_path / "d").glob("*.jsonl"))
    assert len(files) >= 1


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_collector_snapshot_reports_state(tmp_path):
    c = ProDataCollector(data_dir=tmp_path / "d", max_queue=42)
    snap = c.snapshot()
    assert snap["max_queue"] == 42
    assert snap["queue_depth"] == 0
    assert snap["events_recorded"] == 0
    assert snap["events_dropped"] == 0
    assert snap["events_flushed"] == 0
    assert snap["is_running"] is False
    c.start()
    try:
        c.record(_mk_event())
        snap = c.snapshot()
        assert snap["events_recorded"] == 1
        assert snap["is_running"] is True
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# Retention prune
# ---------------------------------------------------------------------------


def test_prune_removes_old_files(tmp_path):
    c = ProDataCollector(data_dir=tmp_path / "d")
    c._data_dir.mkdir(parents=True, exist_ok=True)
    # Create three files; backdate one
    old_file = c._data_dir / "2024-01-01.jsonl"
    new_file = c._data_dir / "2026-04-25.jsonl"
    old_file.write_text("{}\n")
    new_file.write_text("{}\n")
    # Set old_file mtime to 100 days ago
    old_mtime = time.time() - 100 * 86_400
    os.utime(old_file, (old_mtime, old_mtime))
    removed = c.prune_older_than(retention_days=90)
    assert removed == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_prune_rejects_zero_retention(tmp_path):
    c = ProDataCollector(data_dir=tmp_path / "d")
    with pytest.raises(ValueError, match="retention_days"):
        c.prune_older_than(retention_days=0)


def test_prune_returns_zero_when_no_files(tmp_path):
    c = ProDataCollector(data_dir=tmp_path / "d")
    assert c.prune_older_than(retention_days=90) == 0


# ---------------------------------------------------------------------------
# to_parquet_file
# ---------------------------------------------------------------------------


def test_to_parquet_file_collects_rows(tmp_path):
    c = ProDataCollector(
        data_dir=tmp_path / "d",
        flush_every_events=1,
        flush_every_seconds=60,
    )
    c.start()
    try:
        for i in range(5):
            c.record(_mk_event(episode_id=f"ep_{i}"))
        time.sleep(1.0)
    finally:
        c.stop()
    out = tmp_path / "out.parquet"
    n = c.to_parquet_file(out)
    assert n == 5
    assert out.exists()


def test_to_parquet_file_falls_back_to_jsonl_without_pandas(tmp_path, monkeypatch):
    """When pandas isn't importable, write JSONL at the parquet path."""
    c = ProDataCollector(
        data_dir=tmp_path / "d",
        flush_every_events=1,
        flush_every_seconds=60,
    )
    c.start()
    try:
        c.record(_mk_event())
        time.sleep(1.0)
    finally:
        c.stop()
    # Force ImportError on pandas for this path
    import sys
    real_modules = sys.modules.copy()
    monkeypatch.setitem(sys.modules, "pandas", None)
    try:
        out = tmp_path / "out.parquet"
        n = c.to_parquet_file(out)
        assert n >= 1
        assert out.exists()
        # File should contain JSONL when pandas unavailable
        with open(out) as f:
            first_line = f.readline()
            assert json.loads(first_line)["schema_version"] == 1
    finally:
        sys.modules.update(real_modules)
