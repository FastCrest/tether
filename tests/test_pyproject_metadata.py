"""Packaging metadata regressions for user-facing install paths."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent


def _toml_loads(text: str):
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        tomli = pytest.importorskip("tomli")
        return tomli.loads(text)
    return tomllib.loads(text)


def test_serve_extra_installs_cpu_onnxruntime():
    pyproject = _toml_loads((REPO_ROOT / "pyproject.toml").read_text())
    serve_deps = pyproject["project"]["optional-dependencies"]["serve"]

    assert any(dep.startswith("onnxruntime") for dep in serve_deps), (
        "`pip install 'fastcrest-tether[serve]'` must install CPU ONNX Runtime "
        "so `tether serve` works on a fresh CPU machine."
    )


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
