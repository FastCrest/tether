from __future__ import annotations

import os
from pathlib import Path

import pytest


def receipt_path(filename: str) -> Path:
    configured = os.environ.get("TETHER_RECEIPT_DIR")
    if configured:
        return (Path(configured) / filename).resolve()
    return (Path(__file__).parent.parent / "reflex_context" / filename).resolve()


def require_receipt(path: Path, producer: str) -> None:
    if path.is_file():
        return
    message = f"No trusted receipt at {path}; run {producer}"
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    fail_loud = os.environ.get("TETHER_REQUIRE_RECEIPTS") == "1" or ref_name.startswith("release/")
    if fail_loud:
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
