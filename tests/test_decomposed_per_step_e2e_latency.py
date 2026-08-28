"""Receipt-checker for per-step expert E2E latency bench (gate 5).

Reads the receipt produced by ``scripts/modal_per_step_e2e_latency.py``
at ``reflex_context/per_step_e2e_latency_last_run.json`` and asserts
the production-runtime per-step path stays within the chunk-latency
gate vs baked-loop.

Acceptance (matches gate 4):
  - per-step E2E median ≤ 1.20 × baked median
  - per-step E2E p99 ≤ 1.30 × baked p99

Pattern mirrors ``tests/test_decomposed_per_step_parity.py``.
"""

from __future__ import annotations

import json

import pytest

from receipt_test_support import receipt_path, require_receipt

pytestmark = pytest.mark.receipt

RECEIPT = receipt_path("per_step_e2e_latency_last_run.json")

MEDIAN_OVERHEAD_MAX = 0.20
P99_RATIO_MAX = 1.30


def _load_receipt() -> dict:
    require_receipt(RECEIPT, "scripts/modal_per_step_e2e_latency.py")
    return json.loads(RECEIPT.read_text())


class TestPerStepE2ELatencyReceipt:
    """Gate 5 receipt checks against the Modal-produced JSON."""

    def test_receipt_exists(self):
        require_receipt(RECEIPT, "scripts/modal_per_step_e2e_latency.py")

    def test_e2e_passes_gate(self):
        receipt = _load_receipt()
        assert receipt["passes_overall"], (
            f"E2E overhead exceeds gate: "
            f"median={receipt['median_overhead_pct']:.1%} "
            f"(max {MEDIAN_OVERHEAD_MAX:.0%}), "
            f"p99={receipt['p99_ratio']:.2f}x "
            f"(max {P99_RATIO_MAX:.2f}x). See gate-5 experiment note."
        )

    def test_vlm_phases_match(self):
        """vlm_prefix.onnx is identical between baked + per-step builds
        (same export code path), so vlm phase wall time should match across
        the two conditions within ~5% noise. Gate 5 v3 vs v4 showed this
        diverges by 2x without cudnn_conv_algo_search=HEURISTIC pinning —
        if this assertion fails, the bench was likely run without the pin."""
        receipt = _load_receipt()
        baked_vlm = receipt["baked"]["vlm"]["median_ms"]
        per_step_vlm = receipt["per_step"]["vlm"]["median_ms"]
        ratio = max(baked_vlm, per_step_vlm) / max(min(baked_vlm, per_step_vlm), 1e-9)
        assert ratio < 1.10, (
            f"vlm_prefix wall time differs by {ratio:.2f}x between baked + "
            f"per-step (baked={baked_vlm:.2f}ms, per_step={per_step_vlm:.2f}ms). "
            f"Should be < 1.10x — same ONNX, same compute. Likely "
            f"cudnn_conv_algo_search wasn't pinned to HEURISTIC. See gate-5 "
            f"experiment note."
        )

    def test_both_conditions_use_cuda_for_prefix_and_expert(self):
        receipt = _load_receipt()
        for condition in ("baked", "per_step"):
            providers = receipt[condition]["providers"]
            assert providers["vlm_prefix"][0] == "CUDAExecutionProvider"
            assert providers["expert_denoise"][0] == "CUDAExecutionProvider"
