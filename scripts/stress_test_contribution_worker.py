"""Retired live contribution-worker stress harness.

The former script used the removed unauthenticated upload-signing protocol and
obsolete 1 GiB/1000-upload limits. Keeping it runnable could produce misleading
results against Contributor Authentication v1 or mutate a live deployment.

Use the deterministic Worker suite in ``infra/contribution-worker/test`` for
authentication, the 100 MiB object cap, the 60-reservation rolling-hour limit,
capability concurrency, and recovery races. A new live load harness must use
throwaway Ed25519 principals and require an explicit non-production endpoint.
"""
from __future__ import annotations

import sys


RETIREMENT_MESSAGE = """\
RETIRED: scripts/stress_test_contribution_worker.py used the removed
unauthenticated contribution protocol and obsolete limits; no requests were sent.

Run `cd infra/contribution-worker && npm test` for the current signed protocol.
Do not revive live stress traffic without a non-production endpoint and
throwaway Contributor Authentication v1 keys.
"""


def main() -> int:
    print(RETIREMENT_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
