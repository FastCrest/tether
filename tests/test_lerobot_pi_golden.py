"""pi0/pi0.5 against a golden captured on the pre-migration stack (see _lerobot_pi_golden.py).

The fixture was captured with LeRobot 0.5.1 + transformers 5.3.0 + torch 2.7.1
and Tether cdb94dc (the GPU-qualified stack before ADR 0003):

    PYTHONPATH=<tether@cdb94dc>/src <venv-0.5.1>/bin/python \\
        tests/_lerobot_pi_golden.py capture {pi0,pi05} tests/fixtures/lerobot_pi_golden/{pi0,pi05}.pt
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "lerobot_pi_golden"
HELPER = HERE / "_lerobot_pi_golden.py"
FAMILIES = ["pi0", "pi05"]


def _require_lerobot_061():
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires the LeRobot 0.6.1 native extra")
    if version != "0.6.1":
        pytest.skip("this golden check exercises LeRobot 0.6.1")


def _run(*args, timeout=600):
    result = subprocess.run(
        [sys.executable, str(HELPER), *args], capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_golden_fixture_provenance():
    torch = pytest.importorskip("torch")
    sums = dict(
        reversed(line.split()) for line in (FIXTURES / "SHA256SUMS").read_text().splitlines()
    )
    for family in FAMILIES:
        path = FIXTURES / f"{family}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sums[path.name]
        meta = torch.load(path, weights_only=True)["meta"]
        assert meta == {"lerobot": "0.5.1", "transformers": "5.3.0", "torch": "2.7.1"}


@pytest.mark.parametrize("family", FAMILIES)
def test_native_and_patched_match_pre_migration_golden(family):
    _require_lerobot_061()
    out = _run("compare", family, str(FIXTURES / f"{family}.pt"))
    assert f"{family} golden parity passed" in out


@pytest.mark.parametrize("family", FAMILIES)
def test_monolithic_onnx_export_matches_pre_migration_golden(family, tmp_path):
    _require_lerobot_061()
    torch = pytest.importorskip("torch")
    if tuple(int(p) for p in torch.__version__.split(".")[:2]) < (2, 8):
        pytest.skip("onnx-diagnostic export patches need torch>=2.8")
    for module in ("onnx_diagnostic", "onnxruntime", "onnxscript"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"requires the monolithic extra ({module})")
    out = _run("export", family, str(FIXTURES / f"{family}.pt"), str(tmp_path), timeout=900)
    assert f"{family} onnx export golden parity passed" in out
