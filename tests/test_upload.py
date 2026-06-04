"""Tests for src/tether/pro/upload.py — episode upload client."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tether.pro.upload import (
    DEFAULT_DATA_ENDPOINT,
    UploadClient,
    UploadManifest,
    _verify_anonymized,
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
