"""Tests for src/tether/curate/uploader.py — episode filter + queue management."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tether.curate.uploader import (
    MAX_DUP_IMAGE_HASH_FRAC,
    MAX_ZERO_ACTION_FRAC,
    MIN_EPISODE_STEPS,
    Uploader,
    _stats_for_episode,
    filter_episodes,
)


def _mk_row(
    *,
    episode_id: str = "ep-001",
    action_chunk: list[list[float]] | None = None,
    state_vec: list[float] | None = None,
    image_b64: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": "2026-05-05T00:00:00Z",
        "episode_id": episode_id,
        "state_vec": state_vec or [0.1, 0.2],
        "action_chunk": action_chunk or [[0.1, 0.2]],
        "reward_proxy": 1.0,
        "image_b64": image_b64,
        "instruction_hash": "abc",
        "instruction_raw": None,
        "metadata": {},
    }


# ── _stats_for_episode ───────────────────────────────────────────────────────


def test_stats_short_episode_rejects() -> None:
    rows = [_mk_row() for _ in range(5)]
    s = _stats_for_episode(rows)
    assert s.step_count == 5
    reasons = s.reject_reasons()
    assert any("length" in r for r in reasons)
    assert not s.accepted


def test_stats_long_episode_accepts() -> None:
    # MIN_EPISODE_STEPS=30, give it 35 healthy rows
    rows = [
        _mk_row(action_chunk=[[float(i % 5), float(i % 7)]])
        for i in range(MIN_EPISODE_STEPS + 5)
    ]
    s = _stats_for_episode(rows)
    assert s.accepted, f"expected accepted, got reasons {s.reject_reasons()}"


def test_stats_nan_action_rejects() -> None:
    rows = [_mk_row() for _ in range(MIN_EPISODE_STEPS + 1)]
    rows[3]["action_chunk"] = [[1.0, math.nan]]
    s = _stats_for_episode(rows)
    assert s.has_nan_or_inf_action
    assert "nan_or_inf_in_actions" in s.reject_reasons()


def test_stats_all_zero_actions_rejects() -> None:
    rows = [
        _mk_row(action_chunk=[[0.0, 0.0]]) for _ in range(MIN_EPISODE_STEPS + 5)
    ]
    s = _stats_for_episode(rows)
    assert s.zero_action_frac > MAX_ZERO_ACTION_FRAC
    assert any("all_zero" in r for r in s.reject_reasons())


def test_stats_dup_images_rejects() -> None:
    rows = [
        _mk_row(image_b64="same_hash", action_chunk=[[float(i), float(i + 1)]])
        for i in range(MIN_EPISODE_STEPS + 5)
    ]
    s = _stats_for_episode(rows)
    assert s.dup_image_hash_frac > MAX_DUP_IMAGE_HASH_FRAC
    assert any("dup_images" in r for r in s.reject_reasons())


# ── filter_episodes ──────────────────────────────────────────────────────────


def test_filter_episodes_splits_accepted_rejected() -> None:
    long = [
        _mk_row(episode_id="long", action_chunk=[[float(i), float(i + 1)]])
        for i in range(MIN_EPISODE_STEPS + 5)
    ]
    short = [_mk_row(episode_id="short") for _ in range(5)]
    rows_by_ep = {"long": long, "short": short}
    accepted, stats = filter_episodes(rows_by_ep)
    assert "long" in accepted
    assert "short" not in accepted
    assert not stats["short"].accepted
    assert stats["long"].accepted


# ── Uploader run_once dry-run mode ──────────────────────────────────────────


def test_run_once_dry_run_keeps_files(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    jsonl_path = queue_dir / "2026-05-05.jsonl"

    rows = [
        _mk_row(episode_id="ep-good", action_chunk=[[float(i), float(i + 1)]])
        for i in range(MIN_EPISODE_STEPS + 5)
    ]
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    uploader = Uploader(
        contributor_id="free_test_123",
        queue_dir=queue_dir,
        uploaded_dir=tmp_path / "uploaded",
        rejected_dir=tmp_path / "rejected",
        live=False,
    )
    outcome = uploader.run_once()

    assert outcome.files_scanned == 1
    assert outcome.episodes_inspected == 1
    assert outcome.episodes_accepted == 1
    assert outcome.episodes_rejected == 0
    assert outcome.files_kept_in_queue == 1
    assert outcome.files_uploaded == 0
    # File still in queue
    assert jsonl_path.exists()


def test_run_once_archives_all_rejected(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    jsonl_path = queue_dir / "2026-05-05.jsonl"

    short_rows = [_mk_row(episode_id="ep-short") for _ in range(5)]
    with open(jsonl_path, "w") as f:
        for row in short_rows:
            f.write(json.dumps(row) + "\n")

    uploader = Uploader(
        contributor_id="free_test_123",
        queue_dir=queue_dir,
        uploaded_dir=tmp_path / "uploaded",
        rejected_dir=tmp_path / "rejected",
        live=False,
    )
    outcome = uploader.run_once()
    assert outcome.episodes_rejected == 1
    # File moved out of queue → into rejected
    assert not jsonl_path.exists()
    assert (tmp_path / "rejected" / "2026-05-05.jsonl").exists()


def test_run_once_live_mode_raises_stub_for_now(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    jsonl_path = queue_dir / "2026-05-05.jsonl"
    rows = [
        _mk_row(episode_id="ep-good", action_chunk=[[float(i), float(i + 1)]])
        for i in range(MIN_EPISODE_STEPS + 5)
    ]
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    uploader = Uploader(
        contributor_id="free_test_123",
        queue_dir=queue_dir,
        uploaded_dir=tmp_path / "uploaded",
        rejected_dir=tmp_path / "rejected",
        live=True,
    )
    outcome = uploader.run_once()
    # Live mode + worker not deployed → falls back to keeping in queue + logs
    assert outcome.files_kept_in_queue == 1
    assert outcome.files_uploaded == 0
    assert jsonl_path.exists()


def test_kill_switch_skips_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TETHER_NO_CONTRIB_UPLOAD", "1")
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "x.jsonl").write_text(json.dumps(_mk_row()) + "\n")

    uploader = Uploader(
        contributor_id="free_test_123",
        queue_dir=queue_dir,
        live=False,
    )
    outcome = uploader.run_once()
    assert outcome.files_scanned == 0


def test_live_upload_uses_signed_reservation_capability_and_completion(tmp_path: Path) -> None:
    class FakeAuthClient:
        def __init__(self):
            self.calls = []

        def upload(self, **kwargs):
            self.calls.append(("upload", kwargs))
            return {"status": "completed", "upload_id": "upl_test"}

    auth = FakeAuthClient()
    uploader = Uploader(
        contributor_id="legacy_receipt_id_is_not_sent",
        tier="enterprise",
        queue_dir=tmp_path,
        live=True,
        auth_client=auth,
    )
    payload = b'{"metadata":{"anonymized":true}}\n'
    assert uploader._upload_one(
        file_name="episode.jsonl", file_bytes=payload, episode_count=1,
    ) == len(payload)
    assert [call[0] for call in auth.calls] == ["upload"]
    upload = auth.calls[0][1]
    assert upload["file_bytes"] == payload
    assert upload["media_type"] == "application/jsonl"
    assert "tier" not in upload and "contributor_id" not in upload


def test_curate_upload_passes_owner_queue_state_path(tmp_path: Path) -> None:
    class FakeAuthClient:
        def __init__(self):
            self.kwargs = None

        def upload(self, **kwargs):
            self.kwargs = kwargs
            return {"status": "completed", "upload_id": "upl_test"}

    auth = FakeAuthClient()
    uploader = Uploader(
        contributor_id="legacy_receipt_id_is_not_sent",
        queue_dir=tmp_path,
        live=True,
        auth_client=auth,
    )
    state_path = tmp_path / ".episode.jsonl.upload-v1.json"
    uploader._upload_one(
        file_name="episode.jsonl",
        file_bytes=b"payload",
        episode_count=1,
        state_path=state_path,
    )
    assert auth.kwargs is not None
    assert auth.kwargs["state_path"] == state_path


def test_live_upload_snapshot_preserves_concurrent_new_generation(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    queue.mkdir()
    active = queue / "today.jsonl"
    rows = [
        _mk_row(episode_id="snapshot", action_chunk=[[float(i), float(i + 1)]])
        for i in range(MIN_EPISODE_STEPS + 2)
    ]
    active.write_text("".join(json.dumps(row) + "\n" for row in rows))
    appended = _mk_row(episode_id="new-generation", action_chunk=[[1.0, 2.0]])

    class RacingAuth:
        kwargs = None

        def upload(self, **kwargs):
            self.kwargs = kwargs
            # The collector has already atomically rotated the old generation.
            active.write_text(json.dumps(appended) + "\n")
            return {"status": "completed", "upload_id": "upl_snapshot"}

    auth = RacingAuth()
    uploader = Uploader(
        contributor_id="ignored", queue_dir=queue,
        uploaded_dir=tmp_path / "uploaded", rejected_dir=tmp_path / "rejected",
        live=True, auth_client=auth,
    )
    outcome = uploader.run_once()
    assert outcome.files_uploaded == 1
    assert active.read_text() == json.dumps(appended) + "\n"
    assert (tmp_path / "uploaded" / "today.jsonl").exists()
    assert auth.kwargs is not None
    assert "upload-snapshot-" in Path(auth.kwargs["state_path"]).name
    assert auth.kwargs["file_name"] == "today.jsonl"
    assert b"new-generation" not in auth.kwargs["file_bytes"]
