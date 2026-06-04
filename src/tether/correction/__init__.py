"""A2C2 — Asynchronous Action Chunk Correction.

Plug-in residual correction head atop any chunk-predicting VLA. Frozen base policy;
small MLP (~100 KB) emits per-chunk-position residuals conditioned on observation +
estimated inference delay. Composes with RTC (boundary smoothing) — A2C2 corrects
*within* a chunk; RTC smooths *between* chunks.

Paper: arxiv 2509.23224 — Sendai, Alvarez, Matsushima, Matsuo, Iwasawa.

Phase B.4 (gate harness) and Phase B.5 (runtime hook) both use the same
A2C2Head class from `tether.kernels.a2c2_correction` (numpy, .npz). The
training functions live here in `tether.correction.a2c2_training` (numpy
backprop + Adam — no torch dependency).

Path A unification 2026-04-25: prior PyTorch A2C2Head was dropped; trainers
and runtime now share one architecture + one checkpoint format.
"""

from tether.correction.a2c2_training import (
    A2C2TrainResult,
    build_a2c2_input_batch,
    evaluate_mse,
    train_a2c2_head,
)
from tether.correction.transfer_gate import (
    GateDecision,
    GateReport,
    GateThresholds,
    compute_gate_report,
)
from tether.kernels.a2c2_correction import (
    A2C2Config,
    A2C2Head,
    positional_encoding,
)

__all__ = [
    "A2C2Config",
    "A2C2Head",
    "A2C2TrainResult",
    "build_a2c2_input_batch",
    "evaluate_mse",
    "positional_encoding",
    "train_a2c2_head",
    "GateDecision",
    "GateReport",
    "GateThresholds",
    "compute_gate_report",
]
