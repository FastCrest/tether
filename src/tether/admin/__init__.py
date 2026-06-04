"""Tether admin CLI — license issuance + revocation against the license worker.

Operator-only commands; not exposed via the public `tether` CLI. Run as
Python modules:

    python -m tether.admin.issue_license --customer-id alice@bigco.com --tier pro --expires-in 30
    python -m tether.admin.revoke_license --license-id lic_xxx --reason refund
    python -m tether.admin.list_licenses

Auth: set TETHER_ADMIN_TOKEN to the bearer token configured on the worker
(via `wrangler secret put ADMIN_TOKEN`). Set TETHER_LICENSE_ENDPOINT to the
deployed worker URL (e.g., https://tether-licenses.fastcrest.workers.dev).
"""
from __future__ import annotations

__all__: list[str] = []
