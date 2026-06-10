"""Pull pypistats download counts for the tether package and print a Markdown digest.

Aggregates the pre-rename `reflex-vla` package (all releases <= 0.11.x shipped
under that name) with the post-rename `tether` package once it is published.

Run weekly: python scripts/install_digest.py [--days 7]

Outputs a small Markdown block suitable for pasting into a Slack/Discord/HN
comment. No auth required (pypistats hits the public API).
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

import httpx

API = "https://pypistats.org/api/packages"
# Pre-rename name first; "tether" 404s until its first PyPI publish and is
# skipped gracefully until then.
PACKAGES = ["reflex-vla", "tether"]


def _fetch_one(package: str, endpoint: str) -> dict:
    r = httpx.get(f"{API}/{package}/{endpoint}", timeout=10.0)
    if r.status_code == 404:
        # Package not yet indexed by pypistats (typical for the first 1-3 days
        # after a PyPI publish — and for "tether" until it ships at all).
        return {"data": {} if endpoint == "recent" else []}
    r.raise_for_status()
    return r.json()


def _fetch(endpoint: str) -> dict:
    """Fetch `endpoint` for every package name and merge the results."""
    if endpoint == "recent":
        merged: dict[str, int] = {}
        for pkg in PACKAGES:
            for k, v in _fetch_one(pkg, endpoint).get("data", {}).items():
                merged[k] = merged.get(k, 0) + (v or 0)
        return {"data": merged}
    rows: list[dict] = []
    for pkg in PACKAGES:
        rows.extend(_fetch_one(pkg, endpoint).get("data", []))
    return {"data": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="Window for the digest (default 7)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = ap.parse_args()

    overall = _fetch("recent")
    versions = _fetch("python_minor")  # daily by Python version, recent
    systems = _fetch("system")  # OS breakdown

    last_day = overall.get("data", {}).get("last_day", 0)
    last_week = overall.get("data", {}).get("last_week", 0)
    last_month = overall.get("data", {}).get("last_month", 0)

    if args.json:
        print(json.dumps({
            "last_day": last_day,
            "last_week": last_week,
            "last_month": last_month,
            "packages": PACKAGES,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }, indent=2))
        return 0

    print(f"# tether install digest — {date.today().isoformat()}")
    print(f"(aggregated across PyPI packages: {', '.join(PACKAGES)})")
    print()
    print(f"- **Last day:** {last_day:,} downloads")
    print(f"- **Last 7 days:** {last_week:,} downloads")
    print(f"- **Last 30 days:** {last_month:,} downloads")
    print()

    # Top Python versions in the recent window
    py_rows = versions.get("data", [])
    if py_rows:
        cutoff = (date.today() - timedelta(days=args.days)).isoformat()
        recent = [r for r in py_rows if r.get("date", "") >= cutoff]
        totals: dict[str, int] = {}
        for r in recent:
            cat = r.get("category") or "unknown"
            totals[cat] = totals.get(cat, 0) + r.get("downloads", 0)
        if totals:
            print(f"## Python versions (last {args.days}d)")
            for cat, n in sorted(totals.items(), key=lambda kv: -kv[1])[:6]:
                print(f"- `{cat}`: {n:,}")
            print()

    # Top OSes
    sys_rows = systems.get("data", [])
    if sys_rows:
        cutoff = (date.today() - timedelta(days=args.days)).isoformat()
        recent = [r for r in sys_rows if r.get("date", "") >= cutoff]
        totals = {}
        for r in recent:
            cat = r.get("category") or "unknown"
            totals[cat] = totals.get(cat, 0) + r.get("downloads", 0)
        if totals:
            print(f"## OS (last {args.days}d)")
            for cat, n in sorted(totals.items(), key=lambda kv: -kv[1])[:5]:
                print(f"- `{cat}`: {n:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
