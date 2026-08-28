"""Tests for src/tether/pro/upload.py — episode upload client."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import pytest
import tether.pro.upload as upload_module

from tether.pro.upload import (
    DEFAULT_DATA_ENDPOINT,
    UploadClient,
    UploadManifest,
)


@pytest.fixture
def upload_dir(tmp_path):
    """Temporary upload queue directory."""
    return tmp_path / "upload-queue"


@pytest.fixture
def sample_episode(tmp_path):
    """Create a sample JSONL episode file."""
    ep = tmp_path / "episode.jsonl"
    row = {
        "schema_version": 1,
        "timestamp": "2026-05-04T12:00:00Z",
        "episode_id": "test-ep-001",
        "state_vec": [0.1, 0.2],
        "action_chunk": [[0.3, 0.4]],
        "reward_proxy": 1.0,
        "image_b64": None,
        "instruction_hash": "abc123",
        "instruction_raw": None,
        "metadata": {"anonymized": True},
    }
    ep.write_text(json.dumps(row) + "\n")
    return ep


# ── Test 1: Queue episode creates manifest ────────────────────────────

def test_queue_episode_creates_manifest(upload_dir, sample_episode):
    """Queueing an episode creates a data file + manifest in pending/."""
    client = UploadClient(queue_dir=upload_dir)
    manifest = client.queue_episode(
        sample_episode, episode_id="ep001", anonymized=True,
    )
    assert manifest is not None
    assert manifest.episode_id == "ep001"
    assert manifest.file_name == "ep001.jsonl"
    assert manifest.anonymized is True
    assert manifest.file_size > 0
    assert len(manifest.file_hash) == 64  # SHA256 hex
    assert len(manifest.contributor_hash) == 16

    # Verify files exist in pending/
    pending = upload_dir / "pending"
    assert pending.exists()
    assert (pending / "ep001.jsonl").exists()
    assert (pending / "ep001.manifest.json").exists()


# ── Test 2: Rejects non-anonymized data ───────────────────────────────

def test_rejects_non_anonymized(upload_dir, tmp_path):
    """Upload is rejected when anonymization is not verified."""
    non_anon = tmp_path / "raw.jsonl"
    non_anon.write_text(json.dumps({"metadata": {}}) + "\n")

    client = UploadClient(queue_dir=upload_dir)
    manifest = client.queue_episode(non_anon, anonymized=False)
    assert manifest is None  # rejected


# ── Test 3: Force flag bypasses anonymization check ───────────────────

def test_force_bypasses_check(upload_dir, tmp_path):
    """force=True allows queueing without anonymization verification."""
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"data": "test"}) + "\n")

    client = UploadClient(queue_dir=upload_dir)
    manifest = client.queue_episode(
        raw, episode_id="forced", anonymized=False, force=True,
    )
    assert manifest is not None
    assert manifest.episode_id == "forced"


# ── Test 4: Manifest serialization round-trip ─────────────────────────

def test_manifest_round_trip():
    """UploadManifest serializes and deserializes correctly."""
    manifest = UploadManifest(
        episode_id="ep001",
        file_name="ep001.jsonl",
        source_path="/tmp/test.jsonl",
        queued_at="2026-05-04T12:00:00Z",
        file_size=1234,
        file_hash="a" * 64,
        anonymized=True,
        contributor_hash="b" * 16,
        attempts=2,
        last_attempt_at="2026-05-04T12:01:00Z",
        completed_at=None,
        error="timeout",
    )
    d = manifest.to_dict()
    restored = UploadManifest.from_dict(d)
    assert restored.episode_id == "ep001"
    assert restored.file_name == "ep001.jsonl"
    assert restored.attempts == 2
    assert restored.error == "timeout"
    assert restored.completed_at is None


# ── Test 5: Revoke deletes all data ───────────────────────────────────

def test_revoke_deletes_all(upload_dir, sample_episode):
    """revoke_all() removes all queued and completed data."""
    client = UploadClient(queue_dir=upload_dir)
    client.queue_episode(sample_episode, episode_id="ep001", anonymized=True)
    assert client.pending_count() == 1

    removed = client.revoke_all()
    assert removed >= 2  # data file + manifest
    assert client.pending_count() == 0


# ── Test 6: Stats reporting ───────────────────────────────────────────

def test_stats_reporting(upload_dir):
    """stats() returns correct structure."""
    client = UploadClient(queue_dir=upload_dir)
    s = client.stats()
    assert "pending" in s
    assert "completed" in s
    assert "failed" in s
    assert "endpoint" in s
    assert s["pending"] == 0
    assert s["endpoint"] == DEFAULT_DATA_ENDPOINT


def test_upload_uses_contributor_reservation_flow_without_legacy_endpoint(
    upload_dir, sample_episode,
):
    class FakeAuthClient:
        def __init__(self):
            self.calls = []

        def upload(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "completed"}

    auth = FakeAuthClient()
    client = UploadClient(queue_dir=upload_dir, auth_client=auth)
    manifest = client.queue_episode(
        sample_episode, episode_id="ep-auth", anonymized=True,
    )
    assert manifest is not None
    queued = upload_dir / "pending" / "ep-auth.jsonl"
    assert client._upload_file(queued, manifest) is True
    assert len(auth.calls) == 1
    assert auth.calls[0]["file_name"] == "ep-auth.jsonl"
    assert auth.calls[0]["file_bytes"] == queued.read_bytes()
    assert auth.calls[0]["media_type"] == "application/jsonl"
    assert auth.calls[0]["state_path"] == queued.with_name(
        f".{queued.name}.upload-v1.json"
    )
    assert "tether-data.fastcrest.workers.dev" not in client.stats()["endpoint"]


def test_stale_legacy_endpoint_override_is_migrated(upload_dir):
    client = UploadClient(
        queue_dir=upload_dir,
        endpoint="https://tether-data.fastcrest.workers.dev/v1/episodes/upload",
    )
    assert client.stats()["endpoint"] == DEFAULT_DATA_ENDPOINT


def test_pro_overwrite_waits_through_upload_and_move(upload_dir, sample_episode, tmp_path):
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text('{"metadata":{"anonymized":true},"generation":"new"}\n')

    class RacingAuth:
        thread = None

        def upload(self, **kwargs):
            self.thread = threading.Thread(
                target=lambda: client.queue_episode(
                    replacement, episode_id="same", anonymized=True,
                )
            )
            self.thread.start()
            time.sleep(0.05)
            assert self.thread.is_alive()
            assert b'"generation":"new"' not in kwargs["file_bytes"]
            return {"status": "completed", "upload_id": "upl_old"}

    auth = RacingAuth()
    client = UploadClient(queue_dir=upload_dir, auth_client=auth)
    assert client.queue_episode(
        sample_episode, episode_id="same", anonymized=True,
    ) is not None
    client._process_pending()
    assert auth.thread is not None
    auth.thread.join(timeout=2)
    assert not auth.thread.is_alive()
    assert (upload_dir / "completed" / "same.jsonl").read_bytes() == sample_episode.read_bytes()
    assert (upload_dir / "pending" / "same.jsonl").read_bytes() == replacement.read_bytes()


def test_pro_terminal_attempt_move_cannot_consume_concurrent_replacement(
    upload_dir, sample_episode, tmp_path, monkeypatch,
):
    replacement = tmp_path / "terminal-replacement.jsonl"
    replacement.write_text('{"metadata":{"anonymized":true},"generation":"replacement"}\n')
    client = UploadClient(queue_dir=upload_dir, max_retries=1)
    assert client.queue_episode(sample_episode, episode_id="terminal", anonymized=True)
    manifest_path = upload_dir / "pending" / "terminal.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["attempts"] = 1
    manifest_path.write_text(json.dumps(manifest))
    replacement_thread = None
    original_read = Path.read_text

    def racing_read(path, *args, **kwargs):
        nonlocal replacement_thread
        if path == manifest_path and replacement_thread is None:
            replacement_thread = threading.Thread(target=lambda: client.queue_episode(
                replacement, episode_id="terminal", anonymized=True,
            ))
            replacement_thread.start()
            time.sleep(0.05)
            assert replacement_thread.is_alive()
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", racing_read)
    client._process_pending()
    assert replacement_thread is not None
    replacement_thread.join(timeout=2)
    assert (upload_dir / "pending" / "terminal.jsonl").read_bytes() == replacement.read_bytes()
    assert (upload_dir / "failed" / "terminal.jsonl").read_bytes() == sample_episode.read_bytes()


@pytest.mark.parametrize(
    ("old_suffix", "replacement_suffix"),
    [(".jsonl", ".parquet"), (".parquet", ".jsonl")],
)
def test_pro_format_replacement_uploads_only_manifest_bound_generation(
    upload_dir, tmp_path, old_suffix, replacement_suffix,
):
    old = tmp_path / f"old{old_suffix}"
    old.write_bytes(b"old-generation")
    replacement = tmp_path / f"replacement{replacement_suffix}"
    replacement.write_bytes(b"replacement-generation")

    class RecordingAuth:
        def __init__(self):
            self.calls = []

        def upload(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "completed", "upload_id": "upl_replacement"}

    auth = RecordingAuth()
    client = UploadClient(queue_dir=upload_dir, auth_client=auth)
    assert client.queue_episode(
        old, episode_id="same", anonymized=True,
    ) is not None
    obsolete_state = (
        upload_dir / "pending" / f".same{old_suffix}.upload-v1.json"
    )
    obsolete_state.write_text("{}")
    replacement_manifest = client.queue_episode(
        replacement, episode_id="same", anonymized=True,
    )
    assert replacement_manifest is not None
    assert replacement_manifest.file_name == f"same{replacement_suffix}"

    pending = upload_dir / "pending"
    assert not (pending / f"same{old_suffix}").exists()
    assert not obsolete_state.exists()
    client._process_pending()

    assert [call["file_name"] for call in auth.calls] == [
        f"same{replacement_suffix}"
    ]
    assert auth.calls[0]["file_bytes"] == b"replacement-generation"
    completed = upload_dir / "completed"
    assert (completed / f"same{replacement_suffix}").read_bytes() == (
        b"replacement-generation"
    )
    completed_manifest = json.loads(
        (completed / "same.manifest.json").read_text()
    )
    assert completed_manifest["file_name"] == f"same{replacement_suffix}"
    assert completed_manifest["completed_at"] is not None
    assert not (pending / "same.manifest.json").exists()
    assert not (pending / f"same{old_suffix}").exists()
    assert not (pending / f"same{replacement_suffix}").exists()


@pytest.mark.parametrize(
    ("old_suffix", "replacement_suffix"),
    [(".jsonl", ".parquet"), (".parquet", ".jsonl")],
)
def test_concurrent_format_replacement_publishes_before_processing(
    upload_dir, tmp_path, monkeypatch, old_suffix, replacement_suffix,
):
    old = tmp_path / f"old-concurrent{old_suffix}"
    old.write_bytes(b"old-concurrent-generation")
    replacement = tmp_path / f"replacement-concurrent{replacement_suffix}"
    replacement.write_bytes(b"replacement-concurrent-generation")
    copy_published = threading.Event()
    finish_copy = threading.Event()
    original_copy2 = upload_module.shutil.copy2

    class RecordingAuth:
        def __init__(self):
            self.calls = []

        def upload(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "completed", "upload_id": "upl_concurrent"}

    auth = RecordingAuth()
    client = UploadClient(queue_dir=upload_dir, auth_client=auth)
    assert client.queue_episode(
        old, episode_id="same", anonymized=True,
    ) is not None

    def blocking_copy(source, destination, *args, **kwargs):
        result = original_copy2(source, destination, *args, **kwargs)
        if Path(source) == replacement:
            copy_published.set()
            assert finish_copy.wait(timeout=2)
        return result

    monkeypatch.setattr(upload_module.shutil, "copy2", blocking_copy)
    replacement_thread = threading.Thread(target=lambda: client.queue_episode(
        replacement, episode_id="same", anonymized=True,
    ))
    replacement_thread.start()
    assert copy_published.wait(timeout=2)

    processor_thread = threading.Thread(target=client._process_pending)
    processor_thread.start()
    time.sleep(0.05)
    assert processor_thread.is_alive()
    assert auth.calls == []

    finish_copy.set()
    replacement_thread.join(timeout=2)
    processor_thread.join(timeout=2)
    assert not replacement_thread.is_alive()
    assert not processor_thread.is_alive()
    assert [call["file_name"] for call in auth.calls] == [
        f"same{replacement_suffix}"
    ]
    assert auth.calls[0]["file_bytes"] == b"replacement-concurrent-generation"
    assert not (upload_dir / "pending" / f"same{old_suffix}").exists()
    assert (
        upload_dir / "completed" / f"same{replacement_suffix}"
    ).read_bytes() == b"replacement-concurrent-generation"


@pytest.mark.parametrize("attempts", [0, 1])
def test_manifest_hash_mismatch_blocks_upload_and_terminal_move(
    upload_dir, tmp_path, attempts,
):
    source = tmp_path / "bound.jsonl"
    source.write_bytes(b"bound-generation")

    class RecordingAuth:
        def __init__(self):
            self.calls = []

        def upload(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "completed"}

    auth = RecordingAuth()
    client = UploadClient(
        queue_dir=upload_dir, auth_client=auth, max_retries=1,
    )
    assert client.queue_episode(
        source, episode_id="bound", anonymized=True,
    ) is not None
    pending = upload_dir / "pending"
    data_path = pending / "bound.jsonl"
    data_path.write_bytes(b"wrong-generation")  # same size, different SHA-256
    manifest_path = pending / "bound.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["attempts"] = attempts
    manifest_path.write_text(json.dumps(manifest))

    client._process_pending()

    assert auth.calls == []
    assert data_path.read_bytes() == b"wrong-generation"
    assert manifest_path.exists()
    assert not (upload_dir / "completed" / "bound.jsonl").exists()
    assert not (upload_dir / "failed" / "bound.jsonl").exists()
    persisted = json.loads(manifest_path.read_text())
    assert persisted["error"] == "queued data does not match manifest hash and size"
