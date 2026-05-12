"""Live-worker integration test for the contribution uploader.

Skipped by default. Set REFLEX_LIVE_INTEGRATION_TESTS=1 to run.

What it verifies (against the deployed worker at
https://reflex-contributions.fastcrest.workers.dev):

  1. /healthz returns 200
  2. Sign + PUT + complete round-trip with a unique synthetic contributor_id
  3. Stats endpoint reflects the uploaded episode
  4. Revoke cascade flips the contributor's revoked_at + future signs return 403
  5. Smoke contributor cleaned up afterward (DELETE not exposed; relies on
     test-side uniqueness via uuid in the contributor_id so re-runs don't collide)

CI integration: set REFLEX_LIVE_INTEGRATION_TESTS=1 in a nightly job; smoke
test takes ~3-5s. Don't run on every commit (network flakiness + load).

Manual run:
    REFLEX_LIVE_INTEGRATION_TESTS=1 pytest tests/test_curate_uploader_integration.py -v
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

REQUIRES_LIVE_WORKER = pytest.mark.skipif(
    os.environ.get("REFLEX_LIVE_INTEGRATION_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="Set REFLEX_LIVE_INTEGRATION_TESTS=1 to enable live-worker tests",
)


@pytest.fixture
def smoke_contributor_id() -> str:
    """Per-test-run unique contributor_id so concurrent CI runs don't collide."""
    return f"free_integration_smoke_{uuid.uuid4().hex[:12]}"


@REQUIRES_LIVE_WORKER
def test_healthz() -> None:
    import httpx
    from reflex.curate.uploader import _worker_url

    r = httpx.get(f"{_worker_url()}/healthz", timeout=10.0)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


@REQUIRES_LIVE_WORKER
def test_full_round_trip(smoke_contributor_id: str) -> None:
    """sign → put → complete → stats → revoke → sign-rejected."""
    import httpx
    from reflex.curate.uploader import (
        _complete_upload,
        _put_bytes,
        _request_signed_url,
        _worker_url,
    )

    file_name = "smoke.jsonl"
    payload_bytes = b'{"hello":"world"}\n' * 10
    byte_size = len(payload_bytes)

    # 1. Sign
    sign = _request_signed_url(
        contributor_id=smoke_contributor_id,
        tier="free",
        opted_in_at="2026-05-05T00:00:00Z",
        file_name=file_name,
        byte_size=byte_size,
        episode_count=1,
        privacy_mode="hash_only",
    )
    assert "upload_id" in sign
    assert "put_url" in sign
    assert sign["r2_key"].startswith(f"free-contributors/{smoke_contributor_id}/")

    # 2. PUT
    bytes_up = _put_bytes(
        put_url=sign["put_url"],
        file_bytes=payload_bytes,
        max_mbps=10.0,
    )
    assert bytes_up == byte_size

    # 3. Complete
    complete = _complete_upload(upload_id=sign["upload_id"], episode_count=1)
    assert complete.get("status") == "completed"

    # 4. Stats
    stats_resp = httpx.get(
        f"{_worker_url()}/v1/contributors/{smoke_contributor_id}/stats", timeout=10.0,
    )
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_episodes"] >= 1
    assert stats["total_uploads"] >= 1
    assert stats["total_bytes"] >= byte_size
    assert stats["revoked_at"] is None

    # 5. Revoke
    revoke = httpx.post(
        f"{_worker_url()}/v1/revoke/cascade",
        json={"contributor_id": smoke_contributor_id, "scope": "all"},
        timeout=10.0,
    )
    assert revoke.status_code == 200
    assert "request_id" in revoke.json()

    # 6. Stats after revoke shows revoked_at set
    stats2 = httpx.get(
        f"{_worker_url()}/v1/contributors/{smoke_contributor_id}/stats", timeout=10.0,
    ).json()
    assert stats2.get("revoked_at") is not None

    # 7. Future sign attempt is refused
    with pytest.raises(Exception) as exc_info:
        _request_signed_url(
            contributor_id=smoke_contributor_id,
            tier="free",
            opted_in_at="2026-05-05T00:00:00Z",
            file_name=file_name,
            byte_size=byte_size,
            episode_count=1,
            privacy_mode="hash_only",
        )
    # Either ContributorRevoked or WorkerError carrying a 403 — both are correct
    assert "revoked" in str(exc_info.value).lower() or "403" in str(exc_info.value)


@REQUIRES_LIVE_WORKER
def test_complete_without_put_returns_412(smoke_contributor_id: str) -> None:
    """The worker should refuse to mark a session 'completed' when bytes
    never landed in R2."""
    from reflex.curate.uploader import (
        _request_signed_url,
        _complete_upload,
        WorkerError,
    )

    sign = _request_signed_url(
        contributor_id=smoke_contributor_id,
        tier="free",
        opted_in_at="2026-05-05T00:00:00Z",
        file_name="nobytes.jsonl",
        byte_size=100,
        episode_count=1,
        privacy_mode="hash_only",
    )
    with pytest.raises(WorkerError) as exc_info:
        _complete_upload(upload_id=sign["upload_id"], episode_count=1)
    # 412 precondition failed — bytes weren't put
    assert exc_info.value.status == 412


@REQUIRES_LIVE_WORKER
def test_revoke_cascade_status_endpoint(smoke_contributor_id: str) -> None:
    """Verify the 5-stage cascade status endpoint returns expected shape."""
    import httpx
    from reflex.curate.uploader import _worker_url

    # Initiate revoke (no prior upload — just testing the cascade shape)
    revoke = httpx.post(
        f"{_worker_url()}/v1/revoke/cascade",
        json={"contributor_id": smoke_contributor_id, "scope": "all"},
        timeout=10.0,
    ).json()
    assert "request_id" in revoke
    request_id = revoke["request_id"]

    # Status endpoint immediately after revoke — cascade should be in_progress
    # with revoke + auto-completed Phase 1 stages 4 + 5 marked complete.
    status = httpx.get(
        f"{_worker_url()}/v1/revoke/cascade-status/{request_id}", timeout=10.0,
    ).json()
    assert status["request_id"] == request_id
    assert status["contributor_id"] == smoke_contributor_id
    assert status["overall_status"] in ("in_progress", "completed")

    # Stage shape
    stage_names = {s["name"] for s in status["stages"]}
    assert stage_names == {
        "revoke", "tombstone", "r2_purge", "derived_rebuild", "buyer_notification"
    }

    # Phase 1 simplification: derived_rebuild + buyer_notification auto-complete
    derived = next(s for s in status["stages"] if s["name"] == "derived_rebuild")
    buyer = next(s for s in status["stages"] if s["name"] == "buyer_notification")
    assert derived["status"] == "completed"
    assert buyer["status"] == "completed"


@REQUIRES_LIVE_WORKER
def test_revoke_cascade_status_404_unknown_request() -> None:
    import httpx
    from reflex.curate.uploader import _worker_url

    r = httpx.get(
        f"{_worker_url()}/v1/revoke/cascade-status/rev_does_not_exist",
        timeout=10.0,
    )
    assert r.status_code == 404
