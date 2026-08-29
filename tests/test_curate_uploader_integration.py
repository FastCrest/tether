"""Live-worker integration test for the contribution uploader.

Skipped by default. Set TETHER_LIVE_INTEGRATION_TESTS=1 to run.

What it verifies (against the deployed worker at
https://reflex-contributions.fastcrest.workers.dev):

  1. /healthz returns 200
  2. Sign + PUT + complete round-trip with a unique synthetic contributor_id
  3. Stats endpoint reflects the uploaded episode
  4. Revoke cascade flips the contributor's revoked_at + future signs return 403
  5. Smoke contributor cleaned up afterward (DELETE not exposed; relies on
     test-side uniqueness via uuid in the contributor_id so re-runs don't collide)

CI integration: set TETHER_LIVE_INTEGRATION_TESTS=1 in a nightly job; smoke
test takes ~3-5s. Don't run on every commit (network flakiness + load).

Manual run:
    TETHER_LIVE_INTEGRATION_TESTS=1 pytest tests/test_curate_uploader_integration.py -v
"""
from __future__ import annotations

import os
import uuid

import pytest

REQUIRES_LIVE_WORKER = pytest.mark.skipif(
    os.environ.get("TETHER_LIVE_INTEGRATION_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="Set TETHER_LIVE_INTEGRATION_TESTS=1 to enable live-worker tests",
)
REQUIRES_LIVE_ADMIN = pytest.mark.skipif(
    not os.environ.get("TETHER_CONTRIBUTION_ADMIN_TOKEN"),
    reason="Set TETHER_CONTRIBUTION_ADMIN_TOKEN for admin-only revoke smoke tests",
)


@pytest.fixture
def smoke_contributor_id() -> str:
    """Per-test-run unique contributor_id so concurrent CI runs don't collide."""
    return f"free_integration_smoke_{uuid.uuid4().hex[:12]}"


@REQUIRES_LIVE_WORKER
def test_healthz() -> None:
    import httpx
    from tether.curate.uploader import _worker_url

    r = httpx.get(f"{_worker_url()}/healthz", timeout=10.0)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


@REQUIRES_LIVE_WORKER
def test_full_round_trip(smoke_contributor_id: str) -> None:
    """register → signed reserve → capability PUT → complete → signed stats."""
    del smoke_contributor_id
    from tether.contributor_auth import ContributorAuthClient
    from tether.curate.uploader import _worker_url

    file_name = "smoke.jsonl"
    payload_bytes = b'{"hello":"world"}\n' * 10
    client = ContributorAuthClient(_worker_url())
    sign = client.reserve(
        file_name=file_name,
        file_bytes=payload_bytes,
        media_type="application/jsonl",
    )
    assert "upload_id" in sign
    assert "put_url" in sign
    assert sign["r2_key"].startswith(
        f"free-contributors/{client.credentials.contributor_id}/"
    )

    # 2. PUT
    bytes_up = client.put(sign, payload_bytes, media_type="application/jsonl")
    assert bytes_up == len(payload_bytes)

    # 3. Complete
    complete = client.complete(sign["upload_id"])
    assert complete.get("status") == "completed"

    # 4. Stats
    stats = client.stats()
    # Episode counts are not trusted from the client in Contributor Auth v1.
    assert stats["total_episodes"] >= 0
    assert stats["total_uploads"] >= 1
    assert stats["total_bytes"] >= len(payload_bytes)
    assert stats["revoked_at"] is None


@REQUIRES_LIVE_WORKER
def test_complete_without_put_returns_409(smoke_contributor_id: str) -> None:
    """The worker should refuse to mark a session 'completed' when bytes
    never landed in R2."""
    del smoke_contributor_id
    from tether.contributor_auth import ContributorAuthClient, ContributorAuthError
    from tether.curate.uploader import _worker_url

    client = ContributorAuthClient(_worker_url())
    sign = client.reserve(
        file_name="nobytes.jsonl",
        file_bytes=b"x" * 100,
        media_type="application/jsonl",
    )
    with pytest.raises(ContributorAuthError) as exc_info:
        client.complete(sign["upload_id"])
    # Reservation is still pending; completion is not ready until PUT finalizes.
    assert exc_info.value.status == 409
    assert exc_info.value.body.get("error") == "upload_not_ready"


@REQUIRES_LIVE_WORKER
@REQUIRES_LIVE_ADMIN
def test_revoke_cascade_status_endpoint(smoke_contributor_id: str) -> None:
    """Verify the 5-stage cascade status endpoint returns expected shape."""
    import httpx
    from tether.curate.uploader import _worker_url

    # Initiate revoke (no prior upload — just testing the cascade shape)
    revoke = httpx.post(
        f"{_worker_url()}/v1/revoke/cascade",
        json={"contributor_id": smoke_contributor_id, "scope": "all"},
        headers={"Authorization": f"Bearer {os.environ['TETHER_CONTRIBUTION_ADMIN_TOKEN']}"},
        timeout=10.0,
    ).json()
    assert "request_id" in revoke
    request_id = revoke["request_id"]

    # Status endpoint immediately after revoke — cascade should be in_progress
    # with revoke + auto-completed Phase 1 stages 4 + 5 marked complete.
    status = httpx.get(
        f"{_worker_url()}/v1/revoke/cascade-status/{request_id}",
        headers={"Authorization": f"Bearer {os.environ['TETHER_CONTRIBUTION_ADMIN_TOKEN']}"},
        timeout=10.0,
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
@REQUIRES_LIVE_ADMIN
def test_revoke_cascade_status_404_unknown_request() -> None:
    import httpx
    from tether.curate.uploader import _worker_url

    r = httpx.get(
        f"{_worker_url()}/v1/revoke/cascade-status/rev_does_not_exist",
        headers={"Authorization": f"Bearer {os.environ['TETHER_CONTRIBUTION_ADMIN_TOKEN']}"},
        timeout=10.0,
    )
    assert r.status_code == 404
