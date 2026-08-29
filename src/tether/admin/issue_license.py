"""Issue a new Tether Pro license.

Usage:
    python -m tether.admin.issue_license \\
        --customer-id alice@bigco.com \\
        --tier pro \\
        --expires-in 30 \\
        --max-seats 1 \\
        --notes "First customer"

Outputs the license_id + activation code. Send the activation code to the
customer; they redeem with `tether pro activate REFLEX-XXXX-XXXX-XXXX`.
"""
from __future__ import annotations

import argparse
import sys

from tether.admin._client import AdminError, admin_request

VALID_TIERS = ["trial", "pro", "team", "enterprise", "educational", "research", "oss"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Issue a Tether Pro license. Pairs with `tether pro activate <code>`.",
    )
    parser.add_argument("--customer-id", required=True, help="Customer identifier (e.g., email).")
    parser.add_argument("--tier", default="pro", choices=VALID_TIERS, help="License tier.")
    parser.add_argument("--expires-in", type=int, default=30, help="Days until expiry.")
    parser.add_argument("--max-seats", type=int, default=1, help="Concurrent activations allowed.")
    parser.add_argument("--notes", default="", help="Internal notes (admin-only, not sent to customer).")
    args = parser.parse_args(argv)

    if args.expires_in < 1:
        sys.stderr.write("ERROR: --expires-in must be >= 1\n")
        return 2
    if args.max_seats < 1:
        sys.stderr.write("ERROR: --max-seats must be >= 1\n")
        return 2

    body = {
        "customer_id": args.customer_id,
        "tier": args.tier,
        "expires_in_days": args.expires_in,
        "max_seats": args.max_seats,
        "notes": args.notes,
    }

    try:
        resp = admin_request("POST", "/admin/issue", body)
    except AdminError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return exc.exit_code

    license_id = resp.get("license_id", "?")
    code = resp.get("activation_code", "?")
    code_expires = resp.get("activation_expires_at", "?")
    lic = resp.get("license", {})
    expires = lic.get("expires_at", "?")

    print()
    print(f"  License issued for {args.customer_id}")
    print(f"    license_id:        {license_id}")
    print(f"    tier:              {args.tier}")
    print(f"    expires_at:        {expires}")
    print(f"    max_seats:         {args.max_seats}")
    print()
    print(f"  Activation code:    {code}")
    print(f"  Code expires:       {code_expires}")
    print()
    print(f"  Send the code to {args.customer_id}. They redeem with:")
    print(f"      tether pro activate {code}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
