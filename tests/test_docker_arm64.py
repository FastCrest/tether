"""Lightweight sanity tests for the arm64 Dockerfile + publish workflow.

These tests don't run Docker or push anything — they verify the files
exist with the expected shape so a CI misconfiguration (renamed file,
missing platform, dropped step) fails fast on PR instead of at release.

Goal: docker-arm64-jetson (weight 7). Closes the 'docker run and go'
quickstart for Jetson users.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


class TestDockerfileArm64:
    def test_exists(self):
        assert (REPO_ROOT / "Dockerfile.arm64").exists(), (
            "Dockerfile.arm64 missing — arm64 image build will fail"
        )

    def test_has_entrypoint(self):
        content = (REPO_ROOT / "Dockerfile.arm64").read_text()
        assert "ENTRYPOINT" in content
        assert "tether" in content.lower()

    def test_installs_reflex(self):
        content = (REPO_ROOT / "Dockerfile.arm64").read_text()
        # The pip install line should pull tether-vla with [serve] extras.
        assert "pip install" in content
        assert "serve" in content

    def test_does_not_install_gpu_extra(self):
        """Jetson's CUDA is ABI-locked to the host JetPack. Installing
        onnxruntime-gpu wheels pinned to NVIDIA's CUDA 12.x inside the
        container breaks on the Jetson runtime because the pinned
        wheel's CUDA deps conflict with host libraries injected via
        --runtime=nvidia. Users install ORT-GPU from NVIDIA's Jetson
        Zoo themselves when they need GPU inference.
        """
        content = (REPO_ROOT / "Dockerfile.arm64").read_text()
        # Spot-check: the [gpu] extra or onnxruntime-gpu shouldn't be
        # in the install command.
        install_line = [
            line for line in content.splitlines()
            if "pip install" in line and "tether" in line.lower()
        ]
        joined = " ".join(install_line)
        assert "[gpu]" not in joined
        assert "onnxruntime-gpu" not in joined


class TestPublishWorkflow:
    WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docker-publish-arm64.yml"

    def test_exists(self):
        assert self.WORKFLOW_PATH.exists()

    def test_targets_arm64(self):
        content = self.WORKFLOW_PATH.read_text()
        assert "linux/arm64" in content

    def test_uses_qemu_emulation(self):
        """Cross-building from x86 runner to arm64 needs QEMU. Without
        this step the build fails silently or produces x86 artifacts."""
        content = self.WORKFLOW_PATH.read_text()
        assert "setup-qemu-action" in content

    def test_uses_buildx(self):
        content = self.WORKFLOW_PATH.read_text()
        assert "setup-buildx-action" in content

    def test_tags_with_arm64_suffix(self):
        """Tags should be like v0.2.0-arm64 so users can pick arch.
        Multi-arch manifest merging is a separate future workflow."""
        content = self.WORKFLOW_PATH.read_text()
        assert "-arm64" in content

    def test_uses_dockerfile_arm64(self):
        """The workflow must point at Dockerfile.arm64, not the default
        Dockerfile (which is x86-centric)."""
        content = self.WORKFLOW_PATH.read_text()
        assert "Dockerfile.arm64" in content

    def test_pushes_to_ghcr(self):
        content = self.WORKFLOW_PATH.read_text()
        assert "ghcr.io" in content or "REGISTRY" in content

    def test_valid_yaml(self):
        """The workflow file must parse as YAML — GitHub silently skips
        malformed workflows, turning 'push to main' into a no-op."""
        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(self.WORKFLOW_PATH.read_text())
        assert parsed is not None
        assert "jobs" in parsed
        # PyYAML maps the 'on:' key to Python True (bool) because `on`
        # is a YAML 1.1 boolean. Accept either spelling.
        assert "on" in parsed or True in parsed
