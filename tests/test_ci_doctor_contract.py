"""CI Doctor assertions preserve diagnostics independent of Rich line wrapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / ".github/scripts/check_cpu_doctor.py"
SPEC = importlib.util.spec_from_file_location("ci_doctor_contract", PATH)
assert SPEC and SPEC.loader
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


def cpu_report(system: str = "Darwin") -> dict:
    return {
        "schema_version": 1,
        "system_probe": {
            "summary": {"fail": 0},
            "checks": [
                {"name": "Platform", "status": "pass", "detail": system + " arm64"},
                {"name": "Python version", "status": "pass", "detail": "3.12"},
                {
                    "name": "torch + CUDA",
                    "status": "warn",
                    "detail": "torch 2.8, CUDA None, available=False",
                },
                *[
                    {"name": label, "status": "warn", "detail": "CUDA family not detected"}
                    for label in ("TensorRT runtime", "CUDA cuBLAS", "CUDA cuDNN")
                ],
            ],
        },
    }


def test_cpu_contract_accepts_explicit_absent_gpu_diagnostics() -> None:
    for system in ("Darwin", "Linux"):
        CONTRACT.validate(cpu_report(system), system)


@pytest.mark.parametrize(
    "missing",
    ["Platform", "Python version", "TensorRT runtime", "CUDA cuBLAS", "CUDA cuDNN", "torch + CUDA"],
)
def test_cpu_contract_rejects_missing_diagnostics(missing: str) -> None:
    body = cpu_report()
    body["system_probe"]["checks"] = [
        check for check in body["system_probe"]["checks"] if check["name"] != missing
    ]
    with pytest.raises((AssertionError, KeyError)):
        CONTRACT.validate(body, "Darwin")


def test_cpu_contract_rejects_wrong_platform_or_claimed_gpu() -> None:
    with pytest.raises(AssertionError):
        CONTRACT.validate(cpu_report("Linux"), "Darwin")
    body = cpu_report()
    body["system_probe"]["checks"][2]["detail"] = "available=True"
    with pytest.raises(AssertionError):
        CONTRACT.validate(body, "Darwin")


def test_audit_inventory_keeps_cpu_version_and_every_third_party_dependency() -> None:
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "ci_audit_inventory", PATH.with_name("audit_requirements.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.inventory(
        [
            SimpleNamespace(metadata={"Name": "torch"}, version="2.8.0+cpu"),
            SimpleNamespace(metadata={"Name": "fastcrest-tether"}, version="0.12.0"),
            SimpleNamespace(metadata={"Name": "httpx"}, version="0.28.1"),
        ]
    ) == ["httpx==0.28.1", "torch==2.8.0"]
