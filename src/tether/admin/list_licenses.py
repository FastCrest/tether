"""List all Tether Pro licenses with status + heartbeat info.

Usage:
    python -m tether.admin.list_licenses
    python -m tether.admin.list_licenses --limit 50 --json
"""
from __future__ import annotations

import argparse
import json
import sys

from tether.admin._client import AdminError, admin_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List Tether Pro licenses.")
    parser.add_argument("--limit", type=int, default=100, help="Max licenses to show.")
    parser.add_argument("--json", action="store_true", help="Output raw JSON (for scripts).")
    args = parser.parse_args(argv)

    if args.limit < 1:
        sys.stderr.write("ERROR: --limit must be >= 1\n")
        return 2

    try:
        resp = admin_request("GET", f"/admin/list?limit={args.limit}")
    except AdminError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return exc.exit_code
    licenses = resp.get("licenses", [])

    if args.json:
        print(json.dumps({"licenses": licenses}, indent=2))
        return 0

    if not licenses:
        print("No licenses issued yet.")
        return 0

    print()
    print(f"  {len(licenses)} license{'s' if len(licenses) != 1 else ''} (newest first):")
    print()
    print(f"  {'License ID':<32} {'Customer':<32} {'Tier':<12} {'Expires':<22} {'Status':<10} {'Last HB':<22} {'FPs (7d)':<8}")
    print(f"  {'-' * 32} {'-' * 32} {'-' * 12} {'-' * 22} {'-' * 10} {'-' * 22} {'-' * 8}")
    for L in licenses:
        status = "REVOKED" if L.get("revoked_at") else "active"
        last_hb = L.get("last_heartbeat") or "(never)"
        fps = L.get("distinct_fingerprints_7d") or 0
        print(
            f"  {str(L.get('license_id', ''))[:32]:<32} "
            f"{str(L.get('customer_id', ''))[:32]:<32} "
            f"{str(L.get('tier', ''))[:12]:<12} "
            f"{str(L.get('expires_at', ''))[:22]:<22} "
            f"{status:<10} "
            f"{str(last_hb)[:22]:<22} "
            f"{fps:<8}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
