#!/usr/bin/env python3
"""Recommend adaptive RTC action-chunk thresholds from Tether JSONL traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tether.runtime.adaptive_calibration import (  # noqa: E402
    iter_adaptive_records,
    recommend_adaptive_chunk_thresholds,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace",
        nargs="+",
        type=Path,
        help="Tether JSONL or JSONL.gz trace files containing request.rtc records.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of indented JSON.",
    )
    args = parser.parse_args(argv)

    records = list(iter_adaptive_records(args.trace))
    recommendation = recommend_adaptive_chunk_thresholds(records)
    print(
        json.dumps(
            recommendation,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
