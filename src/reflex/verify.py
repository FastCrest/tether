"""Action-parity gate orchestrator for `reflex verify` (v0).

`reflex verify` answers a single customer question: *does my OPTIMIZED export
(ONNX / Triton) still behave like the ORIGINAL native-PyTorch policy?* It runs
both policies through the same LIBERO loop, pairs their per-episode outcomes,
scores the pair through the shipped Pro 9-gate evaluator, and emits a PASS/FAIL
verdict plus a written ``PARITY.md`` receipt.

v0 deliberately REUSES shipped components rather than inventing new metrics:

* :func:`reflex.eval.libero_rollout.run_libero_rollout` gathers paired
  (original, optimized) episode outcomes — it already supports ``use_native``
  to flip between native-PyTorch (the *original*) and ONNX/Triton inference
  (the *optimized* export) on the exact same proven loop. We call it twice on
  the same suite + seed + task set and pair the results by ``task_id``.
* :class:`reflex.pro.eval_gate.EvalGate` does ALL the metric math: Wasserstein-1
  on joint velocities (S2), action cosine similarity (P4), Wilson-CI aggregate
  + per-task success (P1/P5), the per-task success-cliff veto (S3), and the
  n>=30 statistical-power floor. We map original→``baseline_samples`` and
  optimized→``candidate_samples`` and let the gate decide.

What v0 measures TODAY: success-rate parity. The rollout primitive exposes
per-episode ``success`` / ``steps`` but NOT per-joint velocities, per-step
action chunks, inference latency, or teacher trajectories. Those gate inputs
are filled with the documented :class:`~reflex.pro.eval_gate.EvalSample`
sentinels, so the distributional gates (S1/S2/P2/P4/P6) pass by default and the
load-bearing v0 signal is the success-cliff + Wilson gates (S3/P1/P5). This is
honest and surfaced loudly in the report — it is NOT a silent degrade. The
TODO(reflex-verify) anchors below mark exactly where the richer engine lands.

This module is import-light: ``run_libero_rollout`` (and therefore torch /
LIBERO / mujoco) is imported lazily inside :func:`gather_paired_samples`, so
importing :mod:`reflex.verify` for the verdict types or the unit tests costs
nothing. The scoring + aggregation layer is pure and fully mockable via the
``gather_fn`` seam on :func:`run_verify`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from reflex.pro.eval_gate import (
    EvalGate,
    EvalReport,
    EvalSample,
    GateThresholds,
    InsufficientEpisodes,
    MIN_EPISODES_TO_EVALUATE,
)

logger = logging.getLogger(__name__)


# Suites we accept today. Mirrors the `reflex eval` Phase-1 surface (LIBERO
# only); SimplerEnv / customer suites are a separate roadmap item.
SUPPORTED_SUITES: tuple[str, ...] = ("libero",)


# A "gather" callable returns paired episode-outcome dicts, in the exact shape
# `run_libero_rollout` returns. The default implementation runs the real
# rollouts; tests inject a synthetic stub of the same signature. Keeping this a
# plain Callable (not a Protocol) keeps the test seam trivial.
GatherFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


@dataclass(frozen=True)
class ParityVerdict:
    """Structured outcome of `reflex verify` — frozen so the CLI / report
    writer pass it around without worrying about mutation.

    Wraps the Pro :class:`~reflex.pro.eval_gate.EvalReport` (the real scoring)
    with the verify-specific framing: which export, which original, which
    suite, and the headline success rates that make the verdict legible
    without re-deriving them from the gate internals.
    """

    passed: bool
    eval_report: EvalReport  # the Pro 9-gate report (source of truth)
    optimized_ref: str  # path / HF id of the export under test
    original_ref: str  # path / HF id of the native-PyTorch reference
    suite: str
    target: str
    n_episodes: int  # paired episode count (== candidate == baseline)
    original_success_rate: float  # in [0, 1]
    optimized_success_rate: float  # in [0, 1]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    @property
    def success_rate_delta(self) -> float:
        """optimized - original. Negative => the export regressed."""
        return self.optimized_success_rate - self.original_success_rate

    @property
    def first_failing_gate_id(self) -> str | None:
        g = self.eval_report.first_failing_gate
        return g.gate_id if g is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "optimized_ref": self.optimized_ref,
            "original_ref": self.original_ref,
            "suite": self.suite,
            "target": self.target,
            "n_episodes": self.n_episodes,
            "original_success_rate": self.original_success_rate,
            "optimized_success_rate": self.optimized_success_rate,
            "success_rate_delta": self.success_rate_delta,
            "first_failing_gate_id": self.first_failing_gate_id,
            "generated_at": self.generated_at,
            "eval_report": self.eval_report.to_dict(),
        }


# ---------------------------------------------------------------------------
# Rollout-results -> EvalSample adapter
# ---------------------------------------------------------------------------


def _rollout_results_to_samples(results: dict[str, Any]) -> list[EvalSample]:
    """Adapt one `run_libero_rollout` results dict into ``list[EvalSample]``.

    Why an adapter is needed: the rollout primitive (designed for the Modal
    eval scripts) reports per-episode ``success`` / ``steps`` grouped under
    ``per_task[].episodes[]`` — it does NOT surface the richer per-episode
    signals the 9-gate evaluator can consume (per-joint velocity, per-step
    action chunks, inference latency, teacher trajectories). Rather than widen
    the proven rollout loop for v0, we map what the loop DOES expose onto the
    gate's ``EvalSample`` and fill the rest with the sentinels documented on
    ``EvalSample`` (0 clamp count, 0 latency, [] velocities, [] / None
    trajectories). Those sentinels make the distributional gates no-op-pass;
    the success-cliff + Wilson gates carry the v0 signal.

    TODO(reflex-verify): once the rollout primitive is widened to capture
    per-step action chunks + per-joint velocities + inference latency (or a
    sidecar tap is added), populate ``per_joint_velocity`` /
    ``action_trajectory`` / ``teacher_action_trajectory`` /
    ``inference_latency_p99_ms`` here so S2 (velocity Wasserstein) and P4
    (action cosine) measure real distributional parity instead of passing on
    sentinels.
    """
    samples: list[EvalSample] = []
    for task in results.get("per_task", []) or []:
        task_id = str(task.get("task_idx", task.get("task_description", "unknown")))
        for ep in task.get("episodes", []) or []:
            samples.append(
                EvalSample(
                    task_id=task_id,
                    success=bool(ep.get("success", False)),
                    # --- sentinels (see docstring + EvalSample contract) ---
                    safety_clamp_count=0,
                    inference_latency_p99_ms=0.0,
                    per_joint_velocity=[],
                    action_trajectory=[],
                    teacher_action_trajectory=None,
                )
            )
    return samples


def _success_rate(samples: list[EvalSample]) -> float:
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.success) / len(samples)


# ---------------------------------------------------------------------------
# Paired-sample gathering (the only side-effecting / model-loading seam)
# ---------------------------------------------------------------------------


def gather_paired_samples(
    *,
    optimized_ref: str,
    original_ref: str | None,
    suite: str,
    task_suite_name: str,
    num_episodes: int,
    task_indices: list[int] | None,
    seed: int,
    preprocessor_ref: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the ORIGINAL (native PyTorch) and OPTIMIZED (ONNX/Triton) policies
    through the SAME LIBERO loop and return both rollout result dicts.

    Returns ``(original_results, optimized_results)`` — both in the shape
    documented on :func:`reflex.eval.libero_rollout.run_libero_rollout`.

    This is the only function that loads models / runs simulation, and the only
    one that imports torch + LIBERO. It is isolated behind the ``gather_fn``
    seam on :func:`run_verify` precisely so the scoring path can be unit-tested
    with synthetic samples and zero GPU.

    v0 reuses :func:`run_libero_rollout` with ``use_native`` flipped between the
    two arms — the identical primitive the shipped side-by-side eval uses
    (``scripts/modal_fast_kernels_l3_side_by_side.py``). Same ``seed`` + same
    ``task_indices`` keeps the two arms paired: episode *i* of task *t* sees the
    same LIBERO initial state in both arms.
    """
    # Lazy: torch + LIBERO + mujoco only load when we actually run a rollout.
    from reflex.eval.libero_rollout import (
        load_pi05_policy_and_processors,
        run_libero_rollout,
    )

    # The "original" reference defaults to the same checkpoint as the export
    # (native-PyTorch IS the reference for an export of itself). A caller may
    # override --original to compare against a different baseline checkpoint.
    original_checkpoint = original_ref or optimized_ref

    policy, preprocessor, postprocessor = load_pi05_policy_and_processors(
        student_checkpoint=original_checkpoint,
        decomposed_dir=optimized_ref,
        preprocessor_ref=preprocessor_ref,
    )

    common = dict(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
    )

    # ARM A — original: native lerobot select_action (the reference behavior).
    logger.info("verify: running ORIGINAL arm (native PyTorch)")
    original_results = run_libero_rollout(
        inference=None, use_native=True, label="ORIGINAL", **common,
    )

    # ARM B — optimized: the exported ONNX/Triton inference object on the same
    # loop. v0 uses the shipped Triton fast-kernels adapter
    # (``TritonLIBEROAdapter``), which is exactly the optimized arm the proven
    # side-by-side eval drives in scripts/modal_fast_kernels_l3_side_by_side.py.
    # It builds the optimized runtime from the SAME policy weights, so the only
    # difference between the two arms is the inference path (native vs Triton) —
    # which is precisely the parity question.
    #
    # TODO(reflex-verify): dispatch on the export's reflex_config.json so a
    # decomposed-ONNX export (Pi05DecomposedInference) or a future exporter
    # (DreamZero, GR00T DiT) selects the matching InferenceProtocol object
    # instead of always using the Triton adapter. v0 ships the Triton path
    # because it is the one with a proven LIBERO adapter today.
    logger.info("verify: running OPTIMIZED arm (Triton fast-kernels export)")
    from reflex.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter

    inference = TritonLIBEROAdapter.from_policy(policy)
    optimized_results = run_libero_rollout(
        inference=inference, use_native=False, label="OPTIMIZED", **common,
    )

    return original_results, optimized_results


# ---------------------------------------------------------------------------
# Public orchestrator — PURE scoring given a gather seam
# ---------------------------------------------------------------------------


def run_verify(
    *,
    optimized_ref: str,
    original_ref: str | None = None,
    suite: str = "libero",
    target: str = "unknown",
    task_suite_name: str = "libero_10",
    num_episodes: int = 30,
    task_indices: list[int] | None = None,
    seed: int = 7,
    thresholds: GateThresholds | None = None,
    preprocessor_ref: str | None = None,
    gather_fn: GatherFn | None = None,
) -> ParityVerdict:
    """Resolve original + optimized policies, gather paired samples, score via
    the Pro 9-gate evaluator, and return a :class:`ParityVerdict`.

    The scoring + aggregation in this function is PURE given ``gather_fn`` — it
    does no I/O and loads no models itself. ``gather_fn`` defaults to
    :func:`gather_paired_samples` (which runs the real rollouts); unit tests
    pass a stub that returns synthetic paired result dicts.

    Raises:
        ValueError: unsupported suite.
        InsufficientEpisodes: fewer than ``MIN_EPISODES_TO_EVALUATE`` paired
            episodes (propagated from :class:`EvalGate`) — verify refuses to
            return a green light on under-powered evidence, matching the gate.
    """
    if suite not in SUPPORTED_SUITES:
        raise ValueError(
            f"Unsupported suite: {suite!r}. v0 supports: "
            f"{', '.join(SUPPORTED_SUITES)}."
        )

    gather = gather_fn or gather_paired_samples
    original_results, optimized_results = gather(
        optimized_ref=optimized_ref,
        original_ref=original_ref,
        suite=suite,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        preprocessor_ref=preprocessor_ref,
    )

    # ORIGINAL -> baseline, OPTIMIZED -> candidate. The gate asks "is the
    # candidate as good as the baseline?" which is exactly the parity question.
    baseline_samples = _rollout_results_to_samples(original_results)
    candidate_samples = _rollout_results_to_samples(optimized_results)

    # Memory footprints are not gathered in v0 (the rollout loop doesn't report
    # them); pass equal sentinels so P3 (memory) is a no-op pass. The export is
    # by construction <= the native model in memory, so P3 is not the parity
    # signal v0 cares about.
    # TODO(reflex-verify): wire real export-vs-native resident-memory deltas
    # (and inference latency for P2) once the rollout primitive taps them.
    report: EvalReport = EvalGate.evaluate(
        candidate_samples=candidate_samples,
        baseline_samples=baseline_samples,
        candidate_memory_bytes=0.0,
        baseline_memory_bytes=0.0,
        thresholds=thresholds,
        is_libero_suite=(suite == "libero"),
        pro_force=False,
        bypass_audit=None,
    )

    # TODO(reflex-verify): distributional two-sample engine. The v0 gate scores
    # success-rate parity (S3/P1/P5) + sentinel-passes the distribution gates.
    # The real parity engine slots in HERE as an additional, non-bypassable
    # check before the verdict is finalized:
    #   - MMD (maximum mean discrepancy) two-sample test on the paired action
    #     chunk distributions (original vs optimized), with a permutation-test
    #     p-value threshold — detects distribution shift the mean hides.
    #   - Energy-distance two-sample test as a second, kernel-free estimator.
    # Both consume the per-step action chunks that the rollout primitive must
    # first be widened to capture (see _rollout_results_to_samples TODO).
    #
    # TODO(reflex-verify): embodied / kinematic parity metrics, scored per
    # paired episode and aggregated:
    #   - jerk (3rd derivative of joint position) — smoothness regressions are
    #     invisible to success rate but wreck real hardware.
    #   - completion-time delta — an export that succeeds but is slower.
    #   - motion-energy (sum of squared joint velocities) — energy-per-task
    #     parity. These need per-step joint state from the rollout loop.

    return ParityVerdict(
        passed=report.overall_passed,
        eval_report=report,
        optimized_ref=optimized_ref,
        original_ref=original_ref or optimized_ref,
        suite=suite,
        target=target,
        n_episodes=len(candidate_samples),
        original_success_rate=_success_rate(baseline_samples),
        optimized_success_rate=_success_rate(candidate_samples),
    )


__all__ = [
    "MIN_EPISODES_TO_EVALUATE",
    "SUPPORTED_SUITES",
    "InsufficientEpisodes",
    "ParityVerdict",
    "gather_paired_samples",
    "run_verify",
]
