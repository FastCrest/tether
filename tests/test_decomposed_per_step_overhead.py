"""Receipt-checker for per-step expert ORT-call overhead bench (gate 4).

The actual overhead measurement runs on Modal A100 via
``scripts/modal_per_step_overhead.py`` and writes to
``reflex_context/per_step_overhead_last_run.json``.

This test reads that receipt and asserts:
  - The IOBinding path passes the overhead gate (median ≤ +20%, p99 ≤ 1.30x)
  - The naive path was also measured (gives the apples-to-apples diff
    showing IOBinding is load-bearing)
  - CUDA EP was actually used (not silent CPU fallback)

Pattern mirrors ``tests/test_decomposed_per_step_parity.py``.

Skip if the receipt doesn't exist (CI runs this; locally you fire the
Modal job first then re-run pytest).
"""

from __future__ import annotations

import json

import pytest

from receipt_test_support import receipt_path, require_receipt

pytestmark = pytest.mark.receipt

RECEIPT = receipt_path("per_step_overhead_last_run.json")

MEDIAN_OVERHEAD_MAX = 0.20  # ≤ +20% median chunk overhead
P99_RATIO_MAX = 1.30  # ≤ 1.30x p99


def _load_receipt() -> dict:
    require_receipt(RECEIPT, "scripts/modal_per_step_overhead.py")
    return json.loads(RECEIPT.read_text())


class TestPerStepOverheadReceipt:
    """Gate 4 receipt checks against the Modal-produced JSON."""

    def test_receipt_exists(self):
        require_receipt(RECEIPT, "scripts/modal_per_step_overhead.py")

    def test_iobinding_path_passes_gate(self):
        receipt = _load_receipt()
        gate = receipt["iobinding_gate"]
        assert gate["passes_overall"], (
            f"IOBinding overhead exceeds gate: median={gate['median_overhead_pct']:.1%} "
            f"(max {MEDIAN_OVERHEAD_MAX:.0%}), p99={gate['p99_ratio']:.2f}x "
            f"(max {P99_RATIO_MAX:.2f}x). See gate-4 experiment note."
        )

    def test_iobinding_strictly_better_than_naive(self):
        """The whole point of IOBinding is to skip per-iter past_kv copies.
        If naive matches IOBinding, IOBinding refactor is dead code."""
        receipt = _load_receipt()
        naive_med = receipt["per_step_naive"]["median_ms"]
        iob_med = receipt["per_step_iobinding"]["median_ms"]
        assert iob_med < naive_med, (
            f"IOBinding ({iob_med:.2f}ms) should be faster than naive "
            f"({naive_med:.2f}ms). If equal, the IOBinding code path "
            f"isn't actually skipping host-device copies."
        )

    def test_cuda_provider_active(self):
        """Both sessions must have actually used CUDAExecutionProvider, not
        silently fallen back to CPU. Mirrors gate 3 receipt's check."""
        receipt = _load_receipt()
        providers = receipt["providers"]
        assert providers["baked"] == "CUDAExecutionProvider", (
            f"Baked session used {providers['baked']!r}, not CUDA — "
            f"silent fallback voids the bench numbers"
        )
        assert providers["per_step"] == "CUDAExecutionProvider", (
            f"Per-step session used {providers['per_step']!r}, not CUDA — "
            f"silent fallback voids the bench numbers"
        )
