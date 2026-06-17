"""Receipt-checker for per-step expert ONNX parity (gate 3).

The actual parity measurement runs on Modal A100 via
``scripts/modal_per_step_parity.py`` and writes to
``reflex_context/per_step_parity_last_run.json``.

This test reads that receipt and asserts the gates fired green:

    cos     ≥ 0.99999  (research sidecar Lens 5 — tighter than spec's 0.999
                        because Tether's existing exports already hit
                        +0.99999994 stage-by-stage; anything looser hides
                        a real bug per CLAUDE.md no-good-enough-precision)
    max_abs ≤ 1e-5     (FP32 FMA-rounding floor across 10 Euler steps)

Pattern mirrors ``tests/test_cuda_runtime_parity.py:17``.

Skip if the receipt doesn't exist (CI runs this; locally you fire the
Modal job first then re-run pytest).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = (
    Path(__file__).parent.parent.parent
    / "reflex_context"
    / "per_step_parity_last_run.json"
).resolve()


def _load_receipt() -> dict | None:
    if not RECEIPT.exists():
        return None
    return json.loads(RECEIPT.read_text())


COS_MIN = 0.99999
MAX_ABS_MAX = 1e-5


class TestPerStepParityReceipt:
    """Per-cell + overall gate checks against the Modal-produced receipt."""

    def test_receipt_exists(self):
        if not RECEIPT.exists():
            pytest.skip(
                f"Run scripts/modal_per_step_parity.py to populate {RECEIPT}"
            )

    def test_all_cells_pass_overall_gate(self):
        receipt = _load_receipt()
        if receipt is None:
            pytest.skip("No receipt — run Modal job")
        failed = [
            label for label, r in receipt["cells"].items()
            if not r["passes_overall"]
        ]
        assert not failed, (
            f"Per-step parity failed for cells: {failed}. "
            f"See {RECEIPT} for cos / max_abs values."
        )

    def test_thresholds_are_correct(self):
        receipt = _load_receipt()
        if receipt is None:
            pytest.skip("No receipt — run Modal job")
        # The receipt must have used the tightened gates from research
        # sidecar Lens 5 — not the spec's looser cos≥0.999.
        assert receipt["thresholds"]["cos_min"] == COS_MIN, (
            f"Receipt used wrong cos threshold {receipt['thresholds']['cos_min']}; "
            f"research sidecar Lens 5 mandates {COS_MIN}"
        )
        assert receipt["thresholds"]["max_abs_max"] == MAX_ABS_MAX

    @pytest.mark.parametrize("expected_cell", ["pi05_teacher_n10", "pi05_teacher_n1"])
    def test_per_cell_cos(self, expected_cell):
        receipt = _load_receipt()
        if receipt is None:
            pytest.skip("No receipt — run Modal job")
        if expected_cell not in receipt["cells"]:
            pytest.skip(f"Cell {expected_cell} not in receipt")
        cell = receipt["cells"][expected_cell]
        assert cell["cos"] >= COS_MIN, (
            f"{expected_cell}: cos {cell['cos']} below threshold {COS_MIN}. "
            f"Indicates real bug (Euler ordering / dt sign / t schedule). "
            f"See research sidecar Lens 5."
        )

    @pytest.mark.parametrize("expected_cell", ["pi05_teacher_n10", "pi05_teacher_n1"])
    def test_per_cell_max_abs(self, expected_cell):
        receipt = _load_receipt()
        if receipt is None:
            pytest.skip("No receipt — run Modal job")
        if expected_cell not in receipt["cells"]:
            pytest.skip(f"Cell {expected_cell} not in receipt")
        cell = receipt["cells"][expected_cell]
        assert cell["max_abs"] <= MAX_ABS_MAX, (
            f"{expected_cell}: max_abs {cell['max_abs']:.3e} above threshold "
            f"{MAX_ABS_MAX:.0e}. Indicates float-ordering drift accumulating "
            f"across Euler steps. See research sidecar Lens 2 FM-2."
        )

    def test_cuda_provider_active(self):
        """Both sessions must have actually used CUDAExecutionProvider, not
        silently fallen back to CPU. Mirrors the existing parity-test
        provider check at tests/test_cuda_runtime_parity.py."""
        receipt = _load_receipt()
        if receipt is None:
            pytest.skip("No receipt — run Modal job")
        for label, cell in receipt["cells"].items():
            assert cell["used_provider_baked"] == "CUDAExecutionProvider", (
                f"{label}: baked session used {cell['used_provider_baked']}, "
                f"not CUDA — silent fallback voids the parity claim"
            )
            assert cell["used_provider_per_step"] == "CUDAExecutionProvider", (
                f"{label}: per-step session used {cell['used_provider_per_step']}, "
                f"not CUDA — silent fallback voids the parity claim"
            )
