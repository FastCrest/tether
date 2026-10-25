"""Validate the installed CPU Doctor contract without depending on table wrapping."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


def validate(body: dict, expected_system: str) -> None:
    assert body["schema_version"] == 1, body
    checks = body["system_probe"]["checks"]
    by_name = {check["name"]: check for check in checks}
    assert by_name["Platform"]["status"] == "pass", by_name
    assert by_name["Platform"]["detail"].startswith(expected_system + " "), by_name
    assert by_name["Python version"]["status"] == "pass", by_name
    # Without a detected CUDA family the library label intentionally omits the
    # speculative SONAME. Require the actual diagnostic and its warning result.
    for label in ("TensorRT runtime", "CUDA cuBLAS", "CUDA cuDNN"):
        matches = [check for check in checks if check["name"].startswith(label)]
        assert len(matches) == 1, (label, checks)
        assert matches[0]["status"] == "warn", matches
        assert matches[0]["detail"], matches
    torch = by_name["torch + CUDA"]
    assert "available=False" in torch["detail"], torch
    assert body["system_probe"]["summary"]["fail"] == 0, body


if __name__ == "__main__":
    validate(json.loads(Path(sys.argv[1]).read_text()), platform.system())
