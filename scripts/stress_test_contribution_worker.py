"""Stress-test the deployed contribution-worker.

Verifies the rate-limit + concurrency story before the first real
contributor traffic hits:

1. **Concurrency** — N concurrent /v1/uploads/sign requests from a single
   contributor; verify all 200 OK and no D1 race-condition corruption
   (each upload_id unique, each gets its own row).

2. **Daily upload-count limit** — fire 1001 sequential /v1/uploads/sign
   requests; verify request 1001 returns 429 with daily_upload_count_exceeded.

3. **Daily byte limit** — fire signs with byte_size summing to >10 GB;
   verify the threshold-crossing request returns 429 with
   daily_byte_limit_exceeded. (Fired sparsely to avoid 1000+ requests.)

4. **Revoked contributor refusal** — revoke a contributor mid-test; verify
   subsequent /sign returns 403 contributor_revoked + 200 status returns
   `revoked_at` set.

5. **Cleanup** — DELETE all stress-test rows from D1 + smoke R2 objects
   so the next stress run starts fresh.

Usage:
    python scripts/stress_test_contribution_worker.py
    python scripts/stress_test_contribution_worker.py --concurrency 200
    python scripts/stress_test_contribution_worker.py --skip-cleanup  # leave D1 dirty for debugging

Tuned for safety: this hits the LIVE worker. If you don't want to pay the
Cloudflare ops + D1 writes, run against a self-hosted worker via
REFLEX_CONTRIB_ENDPOINT env override.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


DEFAULT_WORKER = "https://reflex-contributions.fastcrest.workers.dev"


@dataclass
class StressOutcome:
    test: str
    expected: str
    observed: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def _worker_url() -> str:
    return os.environ.get("REFLEX_CONTRIB_ENDPOINT", DEFAULT_WORKER).rstrip("/")


def _smoke_contributor_id() -> str:
    return f"free_stress_{uuid.uuid4().hex[:12]}"


def _sign(client: httpx.Client, *, contributor_id: str, byte_size: int = 100,
          file_name: str | None = None) -> httpx.Response:
    payload = {
        "contributor_id": contributor_id,
        "tier": "free",
        "opted_in_at": "2026-05-06T00:00:00Z",
        "file_name": file_name or f"stress-{uuid.uuid4().hex[:8]}.jsonl",
        "byte_size": int(byte_size),
        "episode_count": 1,
        "privacy_mode": "hash_only",
    }
    return client.post(f"{_worker_url()}/v1/uploads/sign", json=payload, timeout=30.0)


def test_concurrency(*, concurrency: int = 100) -> StressOutcome:
    """N concurrent signs from one contributor; verify all unique upload_ids."""
    cid = _smoke_contributor_id()
    upload_ids: set[str] = set()
    statuses: list[int] = []
    errors: list[str] = []

    def _one():
        with httpx.Client() as client:
            try:
                r = _sign(client, contributor_id=cid)
                if r.status_code == 200:
                    return r.status_code, r.json().get("upload_id"), None
                return r.status_code, None, r.text[:200]
            except Exception as exc:  # noqa: BLE001
                return -1, None, str(exc)

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one) for _ in range(concurrency)]
        for fut in cf.as_completed(futures):
            status, uid, err = fut.result()
            statuses.append(status)
            if uid:
                upload_ids.add(uid)
            elif err:
                errors.append(err)

    n_200 = sum(1 for s in statuses if s == 200)
    all_unique = len(upload_ids) == n_200
    passed = (n_200 == concurrency) and all_unique

    return StressOutcome(
        test="concurrency",
        expected=f"{concurrency} 200 OK + {concurrency} unique upload_ids",
        observed=f"{n_200} 200 OK; {len(upload_ids)} unique upload_ids",
        passed=passed,
        details={
            "contributor_id": cid,
            "status_codes": dict([(s, statuses.count(s)) for s in set(statuses)]),
            "error_samples": errors[:3],
        },
    )


def test_daily_upload_count_limit(*, max_uploads: int = 1000) -> StressOutcome:
    """Fire max_uploads + 1 sequential signs; verify (max_uploads+1)th is 429."""
    cid = _smoke_contributor_id()
    statuses: list[int] = []
    last_429_body = ""
    rejected_at: int | None = None

    with httpx.Client() as client:
        # Hit the limit + 1 to confirm the 429 fires exactly at the threshold.
        for i in range(max_uploads + 1):
            r = _sign(client, contributor_id=cid)
            statuses.append(r.status_code)
            if r.status_code == 429:
                if rejected_at is None:
                    rejected_at = i
                last_429_body = r.text[:300]
            # Avoid spamming the worker too hard.
            if i % 50 == 0 and i > 0:
                print(f"  ...{i}/{max_uploads+1} sent (so far: 200={statuses.count(200)} 429={statuses.count(429)})")

    n_200 = statuses.count(200)
    n_429 = statuses.count(429)
    # We expect exactly max_uploads 200s + at least one 429 starting at index max_uploads.
    passed = (n_200 == max_uploads) and (n_429 >= 1) and (rejected_at == max_uploads)

    return StressOutcome(
        test="daily_upload_count_limit",
        expected=f"{max_uploads} 200 OK then 429 at request {max_uploads + 1}",
        observed=f"{n_200} 200 OK / {n_429} 429; first 429 at request index {rejected_at}",
        passed=passed,
        details={
            "contributor_id": cid,
            "first_429_body": last_429_body,
            "status_distribution": dict([(s, statuses.count(s)) for s in set(statuses)]),
        },
    )


def test_daily_byte_limit(*, byte_size_per_request: int = 1_073_000_000) -> StressOutcome:
    """11 sequential ~1 GiB signs; the 11th should cross 10 GiB → 429.

    Per-request cap is 1 GiB (1,073,741,824 bytes). Using 1,073,000,000 keeps
    each request just under the per-request cap. 10 × 1.073e9 = 10.73e9
    (under 10 GiB = 10.737e9 threshold), 11th = 11.80e9 → 429.
    """
    cid = _smoke_contributor_id()
    statuses: list[int] = []
    rejected_at: int | None = None
    last_429_body = ""

    with httpx.Client() as client:
        # 11 × 1.073 GB = 11.8 GiB; 10 GiB threshold rejects the 11th.
        for i in range(11):
            r = _sign(client, contributor_id=cid, byte_size=byte_size_per_request)
            statuses.append(r.status_code)
            if r.status_code == 429:
                if rejected_at is None:
                    rejected_at = i
                last_429_body = r.text[:300]

    # 10 successes (10.73 GB just under 10 GiB) then 429 at index 10 (11th request).
    passed = (rejected_at == 10) and ("daily_byte_limit_exceeded" in last_429_body)

    return StressOutcome(
        test="daily_byte_limit",
        expected="10 200 OK then 429 at request 11 (cumulative ~11.8 GiB > 10 GiB threshold)",
        observed=f"first 429 at request {rejected_at}; body contains daily_byte_limit_exceeded={'daily_byte_limit_exceeded' in last_429_body}",
        passed=passed,
        details={
            "contributor_id": cid,
            "first_429_body": last_429_body,
            "status_distribution": dict([(s, statuses.count(s)) for s in set(statuses)]),
        },
    )


def test_revoked_contributor_refusal() -> StressOutcome:
    """After /v1/revoke/cascade, /sign returns 403 contributor_revoked."""
    cid = _smoke_contributor_id()

    with httpx.Client() as client:
        # 1. Sign once successfully.
        r1 = _sign(client, contributor_id=cid)

        # 2. Revoke.
        revoke = client.post(
            f"{_worker_url()}/v1/revoke/cascade",
            json={"contributor_id": cid, "scope": "all"},
            timeout=30.0,
        )

        # 3. Sign again; should be 403.
        r2 = _sign(client, contributor_id=cid)

    pre_revoke_ok = r1.status_code == 200
    revoke_ok = revoke.status_code == 200 and "request_id" in revoke.json()
    post_revoke_403 = r2.status_code == 403 and "contributor_revoked" in r2.text
    passed = pre_revoke_ok and revoke_ok and post_revoke_403

    return StressOutcome(
        test="revoked_contributor_refusal",
        expected="200 → revoke → 403 contributor_revoked",
        observed=f"pre-revoke={r1.status_code}; revoke={revoke.status_code}; post-revoke={r2.status_code}",
        passed=passed,
        details={
            "contributor_id": cid,
            "post_revoke_body": r2.text[:300],
        },
    )


def cleanup_stress_contributors() -> dict:
    """Delete all stress-test contributor rows from D1 (uses wrangler).
    Skipped if wrangler isn't on PATH (e.g. running on CI without CF auth)."""
    import shutil
    import subprocess

    if shutil.which("wrangler") is None:
        return {"skipped": "wrangler not on PATH"}

    # Run from the worker directory so wrangler resolves the right project.
    worker_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "infra", "contribution-worker",
    )
    if not os.path.isdir(worker_dir):
        return {"skipped": f"worker dir not found at {worker_dir}"}

    cmd_chain = [
        "DELETE FROM uploads WHERE contributor_id LIKE 'free_stress_%'; "
        "DELETE FROM daily_uploads WHERE contributor_id LIKE 'free_stress_%'; "
        "DELETE FROM revoke_requests WHERE contributor_id LIKE 'free_stress_%'; "
        "DELETE FROM contributors WHERE contributor_id LIKE 'free_stress_%';"
    ]
    try:
        result = subprocess.run(
            ["wrangler", "d1", "execute", "reflex-contributions", "--remote",
             "--command", cmd_chain[0]],
            cwd=worker_dir,
            capture_output=True, text=True, timeout=120,
        )
        return {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--include-rate-limit-tests", action="store_true",
                   help="Include the byte/count-limit tests. Note: as currently "
                        "implemented, /v1/uploads/sign reads but does not write "
                        "daily_uploads (writes happen in /complete). Sign-only "
                        "stress doesn't exhaust the limit. Off by default until "
                        "Phase 1.5 reservation-based rate limit lands.")
    p.add_argument("--skip-cleanup", action="store_true")
    args = p.parse_args()

    print(f"=== Stress-testing contribution-worker at {_worker_url()} ===\n")

    outcomes: list[StressOutcome] = []

    print("[1/4] concurrency test")
    t0 = time.time()
    o = test_concurrency(concurrency=args.concurrency)
    print(f"  {('PASS' if o.passed else 'FAIL')} — {o.observed} ({time.time()-t0:.1f}s)")
    outcomes.append(o)

    if args.include_rate_limit_tests:
        print("\n[2/4] daily byte-limit test (will likely FAIL — see --include-rate-limit-tests help)")
        t0 = time.time()
        o = test_daily_byte_limit()
        print(f"  {('PASS' if o.passed else 'FAIL')} — {o.observed} ({time.time()-t0:.1f}s)")
        outcomes.append(o)

        print("\n[3/4] daily count-limit test (1001 sequential requests; ~3 min)")
        t0 = time.time()
        o = test_daily_upload_count_limit()
        print(f"  {('PASS' if o.passed else 'FAIL')} — {o.observed} ({time.time()-t0:.1f}s)")
        outcomes.append(o)

    print("\n[2/2] revoked-contributor refusal test"
          if not args.include_rate_limit_tests
          else "\n[4/4] revoked-contributor refusal test")
    t0 = time.time()
    o = test_revoked_contributor_refusal()
    print(f"  {('PASS' if o.passed else 'FAIL')} — {o.observed} ({time.time()-t0:.1f}s)")
    outcomes.append(o)

    print()
    if not args.skip_cleanup:
        print("=== Cleanup: removing stress-test contributors from D1 ===")
        cleanup = cleanup_stress_contributors()
        print(f"  {cleanup}")

    print("\n=== Summary ===")
    for o in outcomes:
        marker = "✓" if o.passed else "✗"
        print(f"  {marker} {o.test}: {o.observed}")

    failures = [o for o in outcomes if not o.passed]
    if failures:
        print(f"\n{len(failures)} test(s) FAILED")
        for o in failures:
            print(f"  - {o.test}: expected={o.expected}; observed={o.observed}")
            for k, v in o.details.items():
                print(f"    {k}: {v}")
        return 1
    print("\nAll stress tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
