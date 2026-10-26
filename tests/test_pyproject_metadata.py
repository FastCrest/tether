"""Packaging metadata regressions for user-facing install paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).parent.parent


def _toml_loads(text: str):
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        tomli = pytest.importorskip("tomli")
        return tomli.loads(text)
    return tomllib.loads(text)


def test_serve_extra_requires_explicit_runtime_choice():
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    assert not any(Requirement(dep).name.startswith("onnxruntime") for dep in extras["serve"])
    for runtime_extra, expected in (("onnx", "onnxruntime"), ("gpu", "onnxruntime-gpu")):
        runtimes = {
            Requirement(dep).name
            for dep in extras["serve"] + extras[runtime_extra]
            if Requirement(dep).name.startswith("onnxruntime")
        }
        assert runtimes == {expected}


@pytest.mark.parametrize("extra", ["gpu", "gpu-min"])
def test_cuda12_serving_cohort_cannot_float_to_cuda13(extra):
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    requirements = {
        Requirement(dep).name: Requirement(dep)
        for dep in pyproject["project"]["optional-dependencies"][extra]
    }
    assert Version("1.25.1") in requirements["onnxruntime-gpu"].specifier
    assert Version("1.27.0") not in requirements["onnxruntime-gpu"].specifier
    for name, known_version in (
        ("nvidia-cuda-runtime-cu12", "12.8.90"),
        ("nvidia-cublas-cu12", "12.8.4.1"),
        ("nvidia-cudnn-cu12", "9.10.2.21"),
    ):
        assert Version(known_version) in requirements[name].specifier
        assert Version("13.0.0") not in requirements[name].specifier
    assert Version("2.8.0") in requirements["torch"].specifier
    assert Version("2.10.0") not in requirements["torch"].specifier
    # Explicit python_version: the GPU cohort needs ORT 1.25.1's Python 3.11 floor.
    assert requirements["torch"].marker.evaluate(
        {"sys_platform": "linux", "platform_machine": "x86_64", "python_version": "3.12"}
    )
    assert not requirements["torch"].marker.evaluate(
        {"sys_platform": "linux", "platform_machine": "x86_64", "python_version": "3.10"}
    )
    assert not requirements["torch"].marker.evaluate(
        {"sys_platform": "darwin", "platform_machine": "arm64", "python_version": "3.12"}
    )
    if extra == "gpu":
        assert "tensorrt" not in requirements
        assert Version("10.14.1.48.post1") in requirements["tensorrt-cu12"].specifier
        assert Version("10.16.0") not in requirements["tensorrt-cu12"].specifier
        assert Version("12.8.1") in requirements["cuda-toolkit"].specifier


def test_safety_extra_requires_xxe_safe_lxml():
    """Keep the URDF parser above the CVE-2026-41066 security floor."""
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    safety_deps = pyproject["project"]["optional-dependencies"]["safety"]
    requirements = {Requirement(dep).name: Requirement(dep) for dep in safety_deps}

    assert "lxml" in requirements
    assert str(requirements["lxml"].specifier) == ">=6.1.0"


def test_lockfile_selects_xxe_safe_lxml():
    """Frozen safety installs must not restore a vulnerable lxml release."""
    lock = _toml_loads((REPO_ROOT / "uv.lock").read_text())
    lxml_packages = [package for package in lock["package"] if package["name"] == "lxml"]

    assert len(lxml_packages) == 1
    assert Version(lxml_packages[0]["version"]) >= Version("6.1.0")


def test_package_install_smoke_uploads_deploy_proof_packet():
    workflow = (REPO_ROOT / ".github/workflows/package-install-smoke.yml").read_text()

    assert "tether smoke --offline --json" in workflow
    assert "tether deploy-proof /tmp/tether-package-smoke-export" in workflow
    assert "--api-key ci-secret" in workflow
    assert "--record-dir /tmp/tether-deploy-proof-traces" in workflow
    assert "--markdown-output /tmp/tether-smoke.md" in workflow
    assert "--act-samples 3" in workflow
    assert "--samples 3" in workflow
    assert "MANIFEST.json" in workflow
    assert "deployment-proof" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "tether-package-install-smoke-proof" in workflow


@pytest.mark.parametrize("machine", ["aarch64", "x86_64", "arm64"])
@pytest.mark.parametrize("python_version", ["3.10", "3.12", "3.13"])
def test_lerobot_extras_respect_jetson_numpy_boundary(machine, python_version):
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    env = {
        "sys_platform": "linux",
        "platform_machine": machine,
        "python_version": python_version,
    }
    for extra in ("native", "monolithic", "lerobot", "rtc", "eval-local"):
        requirement = next(
            Requirement(dep)
            for dep in pyproject["project"]["optional-dependencies"][extra]
            if Requirement(dep).name == "lerobot"
        )
        enabled = requirement.marker.evaluate(env)
        expected = python_version >= "3.12" and (
            machine == "x86_64" if extra == "eval-local" else machine != "aarch64"
        )
        assert enabled == expected, (extra, env)
    numpy = [
        Requirement(dep)
        for dep in pyproject["project"]["dependencies"]
        if Requirement(dep).name == "numpy" and Requirement(dep).marker.evaluate(env)
    ]
    assert len(numpy) == 1
    assert (Version("2.0") in numpy[0].specifier) == (machine != "aarch64")


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "x86_64"])
@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_eval_local_dependencies_match_supported_host(machine, system):
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    env = {"sys_platform": system, "platform_machine": machine, "python_version": "3.12"}
    for dep in pyproject["project"]["optional-dependencies"]["eval-local"]:
        assert Requirement(dep).marker.evaluate(env) == (
            system == "linux" and machine == "x86_64"
        ), dep


@pytest.mark.parametrize("python_version", ["3.10", "3.11", "3.12"])
def test_generic_gpu_extras_respect_ort_python_floor(python_version):
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    env = {"sys_platform": "linux", "platform_machine": "x86_64", "python_version": python_version}
    for extra in ("gpu", "gpu-min"):
        for dep in pyproject["project"]["optional-dependencies"][extra]:
            assert Requirement(dep).marker.evaluate(env) == (python_version >= "3.11"), dep


def test_so100_extra_uses_published_scservo_distribution():
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    requirements = {
        Requirement(dep).name: Requirement(dep)
        for dep in pyproject["project"]["optional-dependencies"]["so100"]
    }
    assert "feetech-servo-sdk" in requirements
    assert "scservo-sdk" not in requirements
    assert Version("1.0.0") in requirements["feetech-servo-sdk"].specifier
