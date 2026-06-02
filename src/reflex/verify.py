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

import numpy as np

from reflex.pro.eval_gate import (
    EvalGate,
    EvalReport,
    EvalSample,
    GateThresholds,
    InsufficientEpisodes,
    MIN_EPISODES_TO_EVALUATE,
)
from reflex.verify_metrics import (
    EmbodiedParity,
    TwoSampleResult,
    aggregate_embodied,
    two_sample_test,
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
    two_sample: TwoSampleResult | None = None  # distributional parity (None if no per-step data)
    two_sample_episodes: int = 0  # # episodes the distributional test compared (both arms succeeded)
    embodied: EmbodiedParity | None = None  # kinematic parity (None if no per-step data)

    @property
    def candidate_not_worse(self) -> bool:
        """Rich-verdict helper: candidate is >= baseline on success AND has no
        embodied regression. A 'different but not worse' export is True here even
        when ``two_sample.distributions_differ`` (the distributional flag alone
        doesn't make the candidate worse — it surfaces a shift for review)."""
        no_embodied_regress = self.embodied is None or not self.embodied.regressed()
        return self.success_rate_delta >= 0.0 and no_embodied_regress

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
            "two_sample": self.two_sample.to_dict() if self.two_sample else None,
            "two_sample_episodes": self.two_sample_episodes,
            "candidate_not_worse": self.candidate_not_worse,
            "embodied": self.embodied.to_dict() if self.embodied else None,
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


def _collect_step_actions(results: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Stack every per-step *applied* action into ``(N, D)`` + an ``(N,)`` array of
    episode ids (globally unique across tasks) so the two-sample test can permute
    whole episodes, not steps.

    The applied action — what the policy actually commanded each control step —
    has identical layout (7-dim) for BOTH the native and the optimized arm, so
    the two-sample test compares like with like. The model-internal *predicted
    chunk* does NOT: native ``select_action`` exposes one action per call while
    the decomposed path returns a full multi-step chunk, so their flattened
    widths differ (7 vs 350) and are not comparable — comparing those silently
    no-ops the distributional gate, which is the bug this collector fixes.

    The episode ids matter just as much: per-step actions are autocorrelated
    within an episode, so the two-sample test MUST permute at episode granularity
    (see ``verify_metrics.two_sample_test``). Without them the test over-rejects.

    Returns ``((0, 0), (0,))`` when the rollout didn't capture trajectories (tap
    off or older results) — the two-sample test then no-ops.
    """
    rows: list[np.ndarray] = []
    groups: list[int] = []
    ep_uid = 0
    for task in results.get("per_task", []) or []:
        for ep in task.get("episodes", []) or []:
            acts = ep.get("actions", []) or []
            for act in acts:
                rows.append(np.asarray(act, dtype=np.float64).reshape(-1))
                groups.append(ep_uid)
            if acts:
                ep_uid += 1
    if not rows:
        return np.empty((0, 0)), np.empty((0,))
    width = min(r.shape[0] for r in rows)
    if not width:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack([r[:width] for r in rows]), np.asarray(groups)


def _collect_paired_succeeded_step_actions(
    original_results: dict[str, Any], optimized_results: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-step applied actions for both arms, RESTRICTED to episodes BOTH arms
    succeeded (matched by ``(task_idx, ep)``).

    Conditioning on commonly-succeeded episodes isolates the *policy* shift
    (e.g. bf16 vs fp32 numerics) from the *outcome* shift. If one arm fails more
    episodes, those failures inject very different actions (a robot flailing for
    520 steps) that make the pooled action distributions differ for a reason
    unrelated to per-step policy fidelity — i.e. the distributional test would
    flag a difference that is really just "this arm succeeds less", which the
    success-rate gate already measures. Comparing actions only where both arms
    accomplished the same task answers the question the gate actually wants:
    *given the same successful outcome, does the optimized policy act the same?*

    Returns ``(base_actions, base_groups, cand_actions, cand_groups,
    n_episodes)``. Each arm carries its own per-episode group ids (the
    two-sample test offsets them so the arms' episodes stay distinct units);
    a shared id per commonly-succeeded episode keeps the bookkeeping simple.
    Returns empties + 0 when no episode succeeded in both arms (the two-sample
    test then no-ops and only success-rate parity applies).
    """
    def _index(results: dict[str, Any]) -> dict:
        idx = {}
        for task in results.get("per_task", []) or []:
            ti = task.get("task_idx")
            for ep in task.get("episodes", []) or []:
                idx[(ti, ep.get("ep"))] = ep
        return idx

    oi, ci = _index(original_results), _index(optimized_results)
    b_rows: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    b_g: list[int] = []
    c_g: list[int] = []
    uid = 0
    for key in sorted(oi.keys() & ci.keys()):
        oe, ce = oi[key], ci[key]
        if not (oe.get("success") and ce.get("success")):
            continue
        oa = oe.get("actions") or []
        ca = ce.get("actions") or []
        if not oa or not ca:
            continue
        for a in oa:
            b_rows.append(np.asarray(a, dtype=np.float64).reshape(-1))
            b_g.append(uid)
        for a in ca:
            c_rows.append(np.asarray(a, dtype=np.float64).reshape(-1))
            c_g.append(uid)
        uid += 1
    empty = (np.empty((0, 0)), np.empty((0,)), np.empty((0, 0)), np.empty((0,)), 0)
    if not b_rows or not c_rows:
        return empty
    width = min(min(r.shape[0] for r in b_rows), min(r.shape[0] for r in c_rows))
    if not width:
        return empty
    base = np.vstack([r[:width] for r in b_rows])
    cand = np.vstack([r[:width] for r in c_rows])
    return base, np.asarray(b_g), cand, np.asarray(c_g), uid


def _collect_eef_and_steps(results: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Per-episode end-effector position trajectories + completion-step counts."""
    positions: list[np.ndarray] = []
    steps: list[float] = []
    for task in results.get("per_task", []) or []:
        for ep in task.get("episodes", []) or []:
            eef = ep.get("eef_positions") or []
            if len(eef) > 1:
                positions.append(np.asarray(eef, dtype=np.float64))
            steps.append(float(ep.get("steps", 0) or 0))
    return positions, steps


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
        capture_trajectories=True,
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

    # Distributional + embodied parity — the v0 TODOs, now wired. Computed from
    # the per-step trajectories the widened rollout tap captures. When the tap
    # is off / older results lack them, these stay None and only success-rate
    # parity applies (no silent degrade — the verdict records which ran).
    base_actions, base_groups, cand_actions, cand_groups, n_cmp = (
        _collect_paired_succeeded_step_actions(original_results, optimized_results)
    )
    two_sample: TwoSampleResult | None = None
    if (
        base_actions.size
        and cand_actions.size
        and base_actions.shape[1] == cand_actions.shape[1]
    ):
        # Episode-aware + outcome-conditioned: permute whole episodes (per-step
        # actions are autocorrelated → step-level permutation over-rejects ~100%)
        # over ONLY episodes both arms succeeded (isolates the per-step policy
        # shift from the outcome shift — see the collector docstring).
        two_sample = two_sample_test(
            base_actions,
            cand_actions,
            baseline_groups=base_groups,
            candidate_groups=cand_groups,
        )

    base_pos, base_steps = _collect_eef_and_steps(original_results)
    cand_pos, cand_steps = _collect_eef_and_steps(optimized_results)
    embodied: EmbodiedParity | None = None
    if base_pos and cand_pos:
        embodied = aggregate_embodied(
            baseline_positions=base_pos,
            candidate_positions=cand_pos,
            baseline_velocities=[np.diff(p, axis=0) for p in base_pos],
            candidate_velocities=[np.diff(p, axis=0) for p in cand_pos],
            baseline_completion_steps=base_steps,
            candidate_completion_steps=cand_steps,
        )

    # Non-bypassable: a shifted action distribution or an embodied regression
    # fails the verdict even when success-rate parity passed.
    passed = report.overall_passed
    if two_sample is not None and two_sample.distributions_differ:
        passed = False
    if embodied is not None and embodied.regressed():
        passed = False

    return ParityVerdict(
        passed=passed,
        two_sample_episodes=n_cmp,
        eval_report=report,
        optimized_ref=optimized_ref,
        original_ref=original_ref or optimized_ref,
        suite=suite,
        target=target,
        n_episodes=len(candidate_samples),
        original_success_rate=_success_rate(baseline_samples),
        optimized_success_rate=_success_rate(candidate_samples),
        two_sample=two_sample,
        embodied=embodied,
    )


__all__ = [
    "MIN_EPISODES_TO_EVALUATE",
    "SUPPORTED_SUITES",
    "InsufficientEpisodes",
    "ParityVerdict",
    "gather_paired_samples",
    "run_verify",
]
