#!/usr/bin/env python3
"""Publish realtime-serving certificates into a latency table (standalone CLI).

Thin wrapper over :func:`tether.realtime_cert_publish.publish` so the publish
flow is runnable standalone (e.g. in CI) without importing the full tether CLI.
Equivalent to the ``tether publish-latency`` subcommand.

Usage:
    python scripts/publish_jetson_latency.py /tmp/orin-smolvla-cert [more-certs...]
    python scripts/publish_jetson_latency.py certs/*.json --no-readme
    python scripts/publish_jetson_latency.py CERT_DIR --out path/to/results.md

Each positional arg may be a cert JSON file or a directory containing
``realtime-serving-cert.json``. Pure stdlib + ``tether.realtime_cert``; no GPU
needed — runs anywhere the package is importable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tether.realtime_cert_publish import (
    DEFAULT_RESULTS_DOC,
    README_TABLE_BEGIN,
    README_TABLE_END,
    CertificateLoadError,
    publish,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "certs",
        nargs="+",
        help="cert JSON files, or dirs containing realtime-serving-cert.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_DOC,
        help=f"results doc path (default: {DEFAULT_RESULTS_DOC})",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="README to inject the table into (default: README.md)",
    )
    parser.add_argument(
        "--no-readme", action="store_true", help="don't touch the README"
    )
    parser.add_argument(
        "--title", default="Realtime serving latency", help="table heading"
    )
    args = parser.parse_args(argv)

    try:
        result = publish(
            args.certs,
            out=args.out,
            readme=None if args.no_readme else args.readme,
            title=args.title,
        )
    except CertificateLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result["table"])
    print(f"wrote {result['out']}  ({result['count']} certificate(s))")
    if not args.no_readme:
        if result["readme_updated"]:
            print(f"injected table into {args.readme}")
        else:
            print(
                f"note: markers not found in {args.readme}; skipped injection "
                f"(add {README_TABLE_BEGIN} / {README_TABLE_END})",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
