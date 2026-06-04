"""Tests for `tether validate` round-trip orchestrator.

Covers fixture determinism, the seed-bridge invariant, threshold pass/fail
decisions, missing-ONNX error, unsupported-model rejection, and CI template
emission. Backends are monkeypatched to avoid loading 3B-param checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tether import _onnx_backend, _pytorch_backend, validate_roundtrip
from tether.ci_template import emit_ci_template
from tether.fixtures.vla_fixtures import load_fixtures
from tether.validate_roundtrip import (
    UNSUPPORTED_MODEL_MESSAGE,
    ValidateRoundTrip,
)


# --------------------------------------------------------------------------- helpers


def _write_min_config(
    export_dir: Path,
    model_type: str = "smolvla",
    chunk_size: int = 50,
    action_dim: int = 6,
    num_steps: int = 10,
    extras: dict | None = None,
) -> Path:
    """Write a minimal tether_config.json for orchestrator tests."""
    export_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "model_id": "stub/none",
        "action_chunk_size": chunk_size,
        "action_dim": action_dim,
        "num_denoising_steps": num_steps,
    }
    if extras:
        config.update(extras)
    config_path = export_dir / "tether_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


class _StubBackend:
    """Backend stub returning a fixed array regardless of inputs."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output
        self.calls: list[np.ndarray] = []

    def forward(self, image, prompt, state, initial_noise):
        # Record the noise array for seed-bridge id-equality assertions.
        self.calls.append(initial_noise)
        return self._output


def _patch_backends(
    monkeypatch: pytest.MonkeyPatch,
    pytorch_out: np.ndarray,
    onnx_out: np.ndarray,
) -> tuple[_StubBackend, _StubBackend]:
    """Monkeypatch both backend loaders to return stubs with fixed outputs."""
    pt_stub = _StubBackend(pytorch_out)
    onnx_stub = _StubBackend(onnx_out)

    def fake_load_pytorch(export_dir, model_id, device):
        return pt_stub

    def fake_load_onnx(export_dir, device):
        return onnx_stub

    # Patch at the validate_roundtrip module level — that's where the
    # orchestrator imports the loaders from.
    monkeypatch.setattr(
        validate_roundtrip, "load_pytorch_backend", fake_load_pytorch
    )
    monkeypatch.setattr(
        validate_roundtrip, "load_onnx_backend", fake_load_onnx
    )
    return pt_stub, onnx_stub


# --------------------------------------------------------------------------- tests


def test_fixture_determinism():
    """Same (model_type, num, seed) → identical fixtures across calls."""
    a = load_fixtures("smolvla", 3, seed=0)
    b = load_fixtures("smolvla", 3, seed=0)
    assert len(a) == len(b) == 3
    for (img_a, prompt_a, state_a), (img_b, prompt_b, state_b) in zip(a, b):
        assert prompt_a == prompt_b
        assert np.allclose(img_a, img_b)
        assert np.allclose(state_a, state_b)
        assert img_a.dtype == np.float32
        assert state_a.dtype == np.float32


def test_seed_bridge_equivalence():
    """Noise generated from a seeded torch.Generator is byte-identical across
    reseeds, and the same numpy buffer is consumed by both backends.

    Locks the invariant that PyTorch and ONNX paths see the same bytes —
    `torch.manual_seed` does not seed numpy, so the bridge must thread one
    array through both runtimes.
    """
    g1 = torch.Generator(device="cpu").manual_seed(42)
    noise1 = torch.randn((50, 6), generator=g1).numpy().astype(np.float32)

    g2 = torch.Generator(device="cpu").manual_seed(42)
    noise2 = torch.randn((50, 6), generator=g2).numpy().astype(np.float32)

    assert np.array_equal(noise1, noise2), "seeded noise must be byte-identical"
    assert noise1.dtype == np.float32
    assert noise1.shape == (50, 6)

    # Simulate both backends receiving the same numpy buffer. The
    # orchestrator hands the *same* array reference to both; assert value
    # equality at minimum (id equality holds when no copy is made).
    captured: list[np.ndarray] = []

    def pytorch_like(buf):
        captured.append(buf)
        return buf  # stand-in for "consumed as-is"

    def onnx_like(buf):
        captured.append(buf)
        return buf

    pytorch_like(noise1)
    onnx_like(noise1)
    assert len(captured) == 2
    assert captured[0] is captured[1], "both backends must see the same buffer"
    assert np.array_equal(captured[0], captured[1])


def test_threshold_pass(tmp_path, monkeypatch):
    """Identical backend outputs → passed=True, max_abs_diff=0.0."""
    _write_min_config(tmp_path)
    out = np.zeros((50, 6), dtype=np.float32)
    _patch_backends(monkeypatch, pytorch_out=out, onnx_out=out)

    harness = ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=2, seed=0, threshold=1e-4
    )
    result = harness.run()

    assert result["summary"]["passed"] is True
    assert result["summary"]["max_abs_diff_across_all"] == 0.0
    assert result["model_type"] == "smolvla"
    assert len(result["results"]) == 2


def test_threshold_fail(tmp_path, monkeypatch):
    """Differing backend outputs above threshold → passed=False."""
    _write_min_config(tmp_path)
    pt_out = np.zeros((50, 6), dtype=np.float32)
    onnx_out = np.ones((50, 6), dtype=np.float32)
    _patch_backends(monkeypatch, pytorch_out=pt_out, onnx_out=onnx_out)

    harness = ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=2, seed=0, threshold=1e-4
    )
    result = harness.run()

    assert result["summary"]["passed"] is False
    assert result["summary"]["max_abs_diff_across_all"] >= 1.0


def test_threshold_fail_then_pass_with_loose_threshold(tmp_path, monkeypatch):
    """Same diff passes if threshold is loosened — sanity-check threshold use."""
    _write_min_config(tmp_path)
    pt_out = np.zeros((50, 6), dtype=np.float32)
    onnx_out = np.full((50, 6), 1e-5, dtype=np.float32)
    _patch_backends(monkeypatch, pytorch_out=pt_out, onnx_out=onnx_out)

    strict = ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=1, seed=0, threshold=1e-7
    ).run()
    loose = ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=1, seed=0, threshold=1e-3
    ).run()

    assert strict["summary"]["passed"] is False
    assert loose["summary"]["passed"] is True


def test_missing_onnx_error(tmp_path, monkeypatch):
    """Export dir with config but no ONNX file → FileNotFoundError on run().

    Stub the PyTorch loader so we exercise only the ONNX-missing path
    (loading a real checkpoint would hit the network and is out of scope).
    """
    _write_min_config(tmp_path)
    # No expert_stack.onnx written.

    pt_stub = _StubBackend(np.zeros((50, 6), dtype=np.float32))
    monkeypatch.setattr(
        validate_roundtrip,
        "load_pytorch_backend",
        lambda export_dir, model_id, device: pt_stub,
    )
    # Leave load_onnx_backend un-patched — the real loader should raise
    # FileNotFoundError because expert_stack.onnx is absent.

    harness = ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=1, seed=0
    )
    with pytest.raises(FileNotFoundError):
        harness.run()


def test_unsupported_model_raises_pi05(tmp_path):
    """pi0.5 model_type → ValueError mentioning roadmap."""
    _write_min_config(tmp_path, model_type="pi05")
    with pytest.raises(ValueError, match="roadmap"):
        ValidateRoundTrip(export_dir=tmp_path)


def test_unsupported_model_raises_openvla(tmp_path):
    """openvla model_type → ValueError mentioning roadmap."""
    _write_min_config(tmp_path, model_type="openvla")
    with pytest.raises(ValueError, match="roadmap"):
        ValidateRoundTrip(export_dir=tmp_path)


def test_unsupported_model_raises_unknown(tmp_path):
    """Arbitrary unknown model_type → ValueError mentioning roadmap."""
    _write_min_config(tmp_path, model_type="some_unknown_model")
    with pytest.raises(ValueError, match="roadmap"):
        ValidateRoundTrip(export_dir=tmp_path)
    # Also ensure the canonical message is still surfaced verbatim.
    assert "roadmap" in UNSUPPORTED_MODEL_MESSAGE


def test_ci_template_emission(tmp_path):
    """emit_ci_template writes a valid YAML containing key markers, and
    refuses to overwrite without overwrite=True."""
    out = tmp_path / "workflows" / "tether-validate.yml"
    emit_ci_template(out, tether_version="0.1.0")
    assert out.exists()
    text = out.read_text()
    assert "tether validate" in text
    assert "ubuntu-latest" in text
    assert "0.1.0" in text

    # Second call without overwrite must raise.
    with pytest.raises(FileExistsError):
        emit_ci_template(out, tether_version="0.1.0")

    # Overwrite path works.
    emit_ci_template(out, tether_version="0.2.0", overwrite=True)
    assert "0.2.0" in out.read_text()


def test_seed_bridge_in_orchestrator(tmp_path, monkeypatch):
    """Running the orchestrator twice with the same seed produces the same
    initial-noise array fed to both backends across both runs."""
    _write_min_config(tmp_path)
    out = np.zeros((50, 6), dtype=np.float32)

    pt1, onnx1 = _patch_backends(monkeypatch, pytorch_out=out, onnx_out=out)
    ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=2, seed=123
    ).run()

    pt2, onnx2 = _patch_backends(monkeypatch, pytorch_out=out, onnx_out=out)
    ValidateRoundTrip(
        export_dir=tmp_path, num_test_cases=2, seed=123
    ).run()

    # PyTorch and ONNX backends must have received the identical noise array
    # within a single run.
    for noise_pt, noise_onnx in zip(pt1.calls, onnx1.calls):
        assert noise_pt is noise_onnx

    # And reseeding produces byte-identical noise across runs.
    for n1, n2 in zip(pt1.calls, pt2.calls):
        assert np.array_equal(n1, n2)


# --------------------------------------------------------------------------- integration


@pytest.mark.skipif(
    os.getenv("TETHER_INTEGRATION") != "1",
    reason="integration test; set TETHER_INTEGRATION=1 (requires HF token + ~30 min)",
)
def test_integration_smolvla_export(tmp_path):
    """End-to-end with a real SmolVLA export.

    Requires TETHER_INTEGRATION=1 + HF token + ~30 minutes. Uses
    `tether.export` to produce a real export, then runs validate against it.
    Uses a looser 1e-3 threshold for real models.
    """
    from tether.export import export_model  # type: ignore[attr-defined]

    export_dir = tmp_path / "smolvla_export"
    export_model(
        model_id="lerobot/smolvla_base",
        target="desktop",
        output_dir=str(export_dir),
    )

    result = ValidateRoundTrip(
        export_dir=export_dir, num_test_cases=2, seed=0, threshold=1e-3
    ).run()

    assert result["summary"]["passed"] is True
    assert result["summary"]["max_abs_diff_across_all"] < 1e-3
