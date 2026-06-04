"""Revoke a Tether Pro license.

Usage:
    python -m tether.admin.revoke_license --license-id lic_xxx --reason refund

The customer's running deployment will fail its next heartbeat (within 24h)
and the server will refuse to keep serving. The local license file remains
on the customer's disk but is treated as invalid by `tether pro status`.
"""
from __future__ import annotations

import argparse
import sys

from tether.admin._client import admin_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revoke a Tether Pro license.")
    parser.add_argument("--license-id", required=True, help="License ID to revoke (e.g., lic_xxx).")
    parser.add_argument("--reason", default="admin_revoke", help="Reason for revocation (admin audit).")
    args = parser.parse_args(argv)

    resp = admin_request("POST", "/admin/revoke", {
        "license_id": args.license_id,
        "reason": args.reason,
    })

    print()
    print(f"  License revoked: {resp.get('license_id')}")
    print(f"    revoked_at: {resp.get('revoked_at')}")
    print(f"    reason:     {resp.get('reason')}")
    print()
    print("  Customer's deployment will fail next heartbeat (within 24h).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
