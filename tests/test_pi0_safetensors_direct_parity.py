"""Parity test: Pi0VLA.flat_dict_from_safetensors() == Pi0VLA.prepare_inference_weights().

The Phase B safetensors-direct loader must produce bit-identical output to the
Phase A `from_lerobot_policy + prepare_inference_weights` pipeline. If they
diverge, the downstream consumers (InferenceWeightsRuntime, WeightBinder,
Lift #5 Triton kernels) will all be wrong.

The mapping logic (`_pi0_safetensors_mapping.py`) is also covered by a unit
test in `test_pi0_safetensors_mapping_unit.py` that uses cached key lists —
this test exercises the full pipeline against the real lerobot checkpoint
and verifies tensor values match.

**This test is heavy** (loads ~3.3 GB lerobot pi0_base checkpoint twice).
Skipped by default; opt-in via `REFLEX_RUN_PARITY=1`. Auto-run on Modal,
not in local CI.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("REFLEX_RUN_PARITY") != "1",
    reason="Heavy parity test — set REFLEX_RUN_PARITY=1 to run (loads ~3.3 GB checkpoint)",
)


def _find_pi0_safetensors() -> Path:
    """Locate the cached pi0_base safetensors blob."""
    base = Path.home() / ".cache/huggingface/hub/models--lerobot--pi0_base/snapshots"
    candidates = list(base.glob("*/model.safetensors"))
    if not candidates:
        pytest.skip(f"no pi0_base safetensors found in {base} — download with `huggingface-cli download lerobot/pi0_base`")
    return candidates[0]


def test_pi0_phase_b_key_set_matches_phase_a():
    """Phase B's flat dict has IDENTICAL keys to Phase A's."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    from reflex.models.vlas.pi0 import Pi0VLA

    safetensors_path = _find_pi0_safetensors()

    # Phase A: load PI0Policy → Pi0VLA → prepare_inference_weights
    policy = PI0Policy.from_pretrained("lerobot/pi0_base").to(dtype=torch.float32).cpu()
    vla = Pi0VLA.from_lerobot_policy(policy)
    phase_a = vla.prepare_inference_weights(prefix="")

    # Phase B: safetensors → flat dict directly
    phase_b = Pi0VLA.flat_dict_from_safetensors(
        str(safetensors_path),
        dtype=torch.float32,  # match Phase A's dtype for comparison
        device="cpu",
    )

    assert set(phase_b.keys()) == set(phase_a.keys()), (
        f"Phase A unique: {set(phase_a) - set(phase_b)} | "
        f"Phase B unique: {set(phase_b) - set(phase_a)}"
    )


def test_pi0_phase_b_tensor_values_match_phase_a():
    """Phase B's tensor values are bit-identical to Phase A's (modulo dtype)."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    from reflex.models.vlas.pi0 import Pi0VLA

    safetensors_path = _find_pi0_safetensors()

    policy = PI0Policy.from_pretrained("lerobot/pi0_base").to(dtype=torch.float32).cpu()
    vla = Pi0VLA.from_lerobot_policy(policy)
    phase_a = vla.prepare_inference_weights(prefix="")

    phase_b = Pi0VLA.flat_dict_from_safetensors(
        str(safetensors_path),
        dtype=torch.float32,
        device="cpu",
    )

    mismatches = []
    for key, a_tensor in phase_a.items():
        b_tensor = phase_b[key]
        if a_tensor.shape != b_tensor.shape:
            mismatches.append(f"{key}: shape A={a_tensor.shape} B={b_tensor.shape}")
            continue
        if not torch.equal(a_tensor, b_tensor):
            max_abs = (a_tensor - b_tensor).abs().max().item()
            mismatches.append(f"{key}: max_abs_diff={max_abs:.3e}")

    assert not mismatches, f"first 5 mismatches:\n" + "\n".join(mismatches[:5])
