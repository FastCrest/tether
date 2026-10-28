# Ported from an Apache-2.0 upstream @ 34e1c60; source path and licence in tether/accel/THIRD_PARTY_NOTICES.
# Modified by FastCrest, 2026-10-28: renamed modules/classes for tether.accel; see
# docs/provenance/instinct-port.md for the per-file change list.
"""Certification: paired non-inferiority on task success.

Ported from an Apache-2.0 upstream (see ``tether/accel/THIRD_PARTY_NOTICES`` and
``docs/provenance/instinct-port.md``). Tether changes: module name, the shared
:func:`check_declaration`, and wiring into ``tether verify`` / ``parity.cert.json``.

WHY THIS IS NOT `probe_bitexact`

Every gate in this project so far asks `max|delta action| == 0`. That is the right question for a
runtime pass and a meaningless one for a model-level optimization: a distilled student produces
different actions by construction. The question becomes statistical -- *is the student's task
success acceptably close to the teacher's* -- and statistical questions are easy to answer
dishonestly.

Three rules, each of which exists because the obvious alternative is wrong:

1. **PAIRED.** The same episodes, same seeds, teacher and student. Two independent 2500-episode
   runs would let ordinary between-run variance masquerade as a real difference; pairing removes
   it. The test is McNemar on the discordant pairs, which is the correct test for paired binary
   outcomes and ignores episodes where both arms agree.

2. **THE MARGIN IS DECLARED BEFORE THE RUN.** `certify()` requires `margin` as an argument and
   records it in the certificate. A threshold chosen after seeing the delta is not a gate, it is a
   narrative. This is the same discipline that made the equivalence tiers useful.

3. **IT CAN FAIL.** A certificate that only ever says yes is decoration. The verdict is
   non-inferiority at the declared margin, and `FAIL` is a normal outcome that should block a
   release.

Incomplete or unpaired inputs are refused outright rather than partially analysed, matching
Tether's fail-closed evidence rule (missing evidence fails, it is never filled in).

TWO INTERVAL METHODS, EXPLICITLY NAMED — an upstream migration note (2026-08-27):

* ``interval="wald_central95"`` (the default, and the only method that ever existed here):
  ``ci95`` is the paired Wald central 95% interval, the decision rule is its lower bound, and
  the serialized certificate is byte-identical to every certificate this module has stamped
  before the port. Nothing about the default flow changed; re-stamping prior outcomes (verified
  on the VA 2V/4A pooled run) reproduces the old block exactly.
* ``interval="tango_one_sided95"`` (opt-in; the preregistered pi0.5 TF32 closed-loop gate):
  the decision is Tango's matched-pair score ONE-SIDED 95% lower confidence bound
  (z=1.6448536269514722), which stays non-degenerate at zero discordance where Wald collapses
  to [0, 0]. The two one-sided bounds land in their own explicitly named field
  ``tango_central90`` — named for what the pair IS (a central 90% interval), never serialized
  as ``ci95`` — alongside ``lower_confidence_bound``/``ci_method``/``p_value_kind``. ``ci95``
  keeps its Wald meaning in both modes. The exact McNemar p is descriptive in this mode and
  never decides non-inferiority.

The PR that brought Tango silently rebound the ``ci95`` field to the central-90 pair and made
z=1.6449 the default for every family's stamp flow; this port keeps the repo-wide certificate
stable and makes the new rule a deliberate, named choice per certificate.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

#: z for a one-sided 95% bound (nominal alpha 0.05). Used only by the tango_one_sided95 mode.
ONE_SIDED_95_Z = 1.6448536269514722

#: The interval methods certify() accepts. Adding one here is a release decision, not a default
#: change: the default stays wald_central95 so every existing stamp flow is byte-stable.
INTERVAL_METHODS = ("wald_central95", "tango_one_sided95")

#: Certificate fields that exist only for the opt-in modes/gates. None-valued entries are
#: dropped from to_json() so a default-mode certificate serializes exactly as it always has.
_OPTIONAL_FIELDS = (
    "ci_method",
    "p_value_kind",
    "tango_central90",
    "lower_confidence_bound",
    "lower_confidence_level",
    "alpha",
    "task_collapse_gate",
    "collapsed_tasks",
)


class NotCertifiable(RuntimeError):
    """The inputs cannot support a certificate. Never downgrade this to a warning."""


@dataclass(frozen=True)
class Outcome:
    """One episode's result on one arm. `success` is the scored binary outcome."""

    episode_id: str
    seed: int
    task: str
    success: bool


@dataclass
class Certificate:
    teacher_hash: str
    student_hash: str
    n_pairs: int
    teacher_success: float
    student_success: float
    delta: float
    ci95: tuple[float, float]
    margin_declared: float
    verdict: str
    p_value: float
    discordant: tuple[int, int]  # (teacher-only wins, student-only wins)
    harness: str
    recipe: str
    seeds: str
    #: task -> (n, teacher successes, student successes). A macro number can hide a task that went
    #: to zero while others improved, and for a policy that is the failure that matters.
    per_task: Mapping[str, tuple] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ---- opt-in fields (tango_one_sided95 mode / task-collapse gate). None means "this
    # certificate was produced by a flow that did not use the feature" and the field is then
    # omitted from to_json(), keeping default-mode certificates byte-identical to the pre-port
    # format. See the module docstring's migration note.
    ci_method: str | None = None
    p_value_kind: str | None = None
    #: The two Tango one-sided 95% bounds. Named for what the pair IS — a central 90% interval —
    #: so it can never be mistaken for (or silently replace) ci95.
    tango_central90: tuple[float, float] | None = None
    lower_confidence_bound: float | None = None
    lower_confidence_level: float | None = None
    alpha: float | None = None
    task_collapse_gate: bool | None = None
    collapsed_tasks: tuple[str, ...] | None = None

    @property
    def passed(self) -> bool:
        return self.verdict.startswith("PASS")

    def to_json(self, **kw) -> str:
        d = asdict(self)
        d["ci95"] = list(self.ci95)
        d["discordant"] = list(self.discordant)
        d["per_task"] = {t: list(v) for t, v in self.per_task.items()}
        if self.tango_central90 is not None:
            d["tango_central90"] = list(self.tango_central90)
        if self.collapsed_tasks is not None:
            d["collapsed_tasks"] = list(self.collapsed_tasks)
        for k in _OPTIONAL_FIELDS:
            if d.get(k) is None:
                del d[k]
        return json.dumps(d, indent=2, **kw)

    def per_task_table(self) -> str:
        if not self.per_task:
            return "(no per-task breakdown)"
        out = [f"  {'task':<26}{'n':>4}{'teacher':>9}{'student':>9}{'delta':>8}"]
        for t, (n, tw, sw) in self.per_task.items():
            dt = (sw - tw) / max(n, 1)
            flag = "  <-- collapsed" if tw > 0 and sw == 0 else ""
            out.append(f"  {t:<26}{n:>4}{tw / n:>9.2f}{sw / n:>9.2f}{dt:>+8.2f}{flag}")
        return "\n".join(out)

    def __str__(self) -> str:
        b, c = self.discordant
        if self.ci_method is not None:
            return (
                f"teacher {self.teacher_success:.4f}   student {self.student_success:.4f}\n"
                f"delta   {self.delta:+.4f}   one-sided 95% lower bound "
                f"{self.lower_confidence_bound:+.4f}   (Wald 95% CI "
                f"[{self.ci95[0]:+.4f}, {self.ci95[1]:+.4f}] reported, not deciding)\n"
                f"paired  n={self.n_pairs}  discordant: teacher-only {b}, student-only {c}\n"
                f"CI      {self.ci_method}; {self.p_value_kind} p={self.p_value:.4g} "
                f"(descriptive)\n"
                f"margin declared BEFORE the run: {self.margin_declared:+.4f}\n"
                f"VERDICT: {self.verdict}"
            )
        return (
            f"teacher {self.teacher_success:.4f}   student {self.student_success:.4f}\n"
            f"delta   {self.delta:+.4f}   95% CI [{self.ci95[0]:+.4f}, {self.ci95[1]:+.4f}]\n"
            f"paired  n={self.n_pairs}  discordant: teacher-only {b}, student-only {c}  "
            f"p={self.p_value:.4g}\n"
            f"margin declared BEFORE the run: {self.margin_declared:+.4f}\n"
            f"VERDICT: {self.verdict}"
        )


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar. b, c are the discordant counts.

    Exact rather than chi-square because discordant counts on 50-task suites are routinely small,
    and the chi-square approximation is bad exactly there.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _paired_delta_ci(pairs: Sequence[tuple[bool, bool]], z: float = 1.96) -> tuple[float, float]:
    """95% CI on (student - teacher) success rate for PAIRED binary data.

    Uses the variance of the paired differences, which is driven entirely by the discordant pairs;
    an unpaired two-proportion interval would be wider and wrong here.
    """
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    d = [(1 if s else 0) - (1 if t else 0) for t, s in pairs]
    mean = sum(d) / n
    if n < 2:
        return (mean, mean)
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    half = z * math.sqrt(var / n)
    return (mean - half, mean + half)


def _tango_paired_score_bounds(
    pairs: Sequence[tuple[bool, bool]], z: float = ONE_SIDED_95_Z
) -> tuple[float, float]:
    """Tango score bounds on the paired risk difference ``student - teacher``. Opt-in.

    Wald intervals have poor coverage for sparse discordance and collapse to ``[0, 0]`` when both
    arms agree everywhere. Tango's score interval inverts the efficient score statistic for
    matched binary proportions; it remains non-degenerate at that boundary and is the standard CI
    paired with the Nam/Tango risk-difference non-inferiority test.

    ``b`` is teacher-only success and ``c`` student-only success. This is the iterative form of
    Tango's interval (Statistics in Medicine 17 (1998), 891-908), verified against an independent
    brute-force constrained-MLE reference to <5e-6 including the 16-vs-17 discordant decision
    boundary at n=500. With the default ``z`` each endpoint is a ONE-SIDED 95% bound; the pair is
    a central 90% interval — which is why it is serialized as ``tango_central90``, never ``ci95``.
    """
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    b = sum(1 for teacher, student in pairs if teacher and not student)
    c = sum(1 for teacher, student in pairs if student and not teacher)
    estimate = (c - b) / n
    pa = 2.0 * n

    def score(candidate: float) -> float:
        pb = -b - c + (2 * n - c + b) * candidate
        pc = -b * candidate * (1.0 - candidate)
        discriminant = max(0.0, pb * pb - 4.0 * pa * pc)
        constrained_p10 = (math.sqrt(discriminant) - pb) / (2.0 * pa)
        variance = n * (2.0 * constrained_p10 + candidate * (1.0 - candidate))
        if variance <= 0.0:
            return math.copysign(math.inf, c - b - n * candidate)
        return (c - b - n * candidate) / math.sqrt(variance)

    def upper() -> float:
        if c == n:
            return 1.0
        root, step = estimate, 1.0 - estimate
        candidate = root
        for _ in range(80):
            step *= 0.5
            candidate = root + step
            value = score(candidate)
            if abs(value) < z:
                root = candidate
            if step < 1e-10 or abs(z - value) < 1e-8:
                break
        return min(1.0, candidate)

    def lower() -> float:
        if b == n:
            return -1.0
        root, step = estimate, 1.0 + estimate
        candidate = root
        for _ in range(80):
            step *= 0.5
            candidate = root - step
            value = score(candidate)
            if abs(value) < z:
                root = candidate
            if step < 1e-10 or abs(z - value) < 1e-8:
                break
        return max(-1.0, candidate)

    return (lower(), upper())


def required_pairs(pairs: Sequence[tuple[bool, bool]], margin: float, z: float = 1.96) -> int:
    """How many paired episodes would be needed to decide non-inferiority at this margin.

    Added after the first real run: 20 paired episodes produced a CI of [-0.12, +0.22] against a
    -0.05 margin. The verdict was FAIL, which was correct, but "FAIL" alone conflates *we measured
    a regression* with *we cannot tell yet* -- and those call for opposite responses. This turns
    the second case into an actionable number.
    """
    n = len(pairs)
    if n == 0:
        return 0
    d = [(1 if s else 0) - (1 if t else 0) for t, s in pairs]
    mean = sum(d) / n
    if n < 2:
        return 0
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    slack = mean - margin  # how much room between the estimate and the margin
    if slack <= 0 or var == 0:
        return 0  # a point estimate already at/below the margin
    return int(math.ceil(var * (z / slack) ** 2))


def _index(outcomes: Sequence[Outcome], arm: str) -> dict[tuple[str, int], Outcome]:
    seen: dict[tuple[str, int], Outcome] = {}
    for o in outcomes:
        key = (o.episode_id, o.seed)
        if key in seen:
            raise NotCertifiable(
                f"{arm}: duplicate episode {key}. Certification needs exactly one outcome per "
                f"(episode, seed); duplicates mean the run is not what it claims to be."
            )
        seen[key] = o
    return seen


def check_declaration(margin: float, interval: str = "wald_central95") -> None:
    """Refuse an invalid pre-run declaration. Callers run this BEFORE any rollout."""
    if interval not in INTERVAL_METHODS:
        raise NotCertifiable(
            f"interval={interval!r} is not a certificate method; one of {INTERVAL_METHODS}. "
            f"The decision rule is part of the certificate and cannot be guessed."
        )
    if (
        not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(margin)
    ):
        raise NotCertifiable(f"margin={margin!r} is not a finite number")
    if margin > 0:
        raise NotCertifiable(
            f"margin={margin} is positive. A non-inferiority margin is the acceptable LOSS and "
            f"must be <= 0; a positive value would certify a student that is worse than allowed."
        )


def certify(
    teacher: Sequence[Outcome],
    student: Sequence[Outcome],
    *,
    margin: float,
    teacher_hash: str = "?",
    student_hash: str = "?",
    harness: str = "?",
    recipe: str = "?",
    seeds: str = "?",
    min_pairs: int = 1,
    interval: str = "wald_central95",
    fail_on_task_collapse: bool = False,
) -> Certificate:
    """Paired non-inferiority certificate. `margin` MUST be supplied and is recorded.

    `margin` is the largest success-rate drop that is still acceptable, as a NEGATIVE number:
    -0.05 means "the student may be up to 5 points worse". Non-inferiority holds when the lower
    bound of the CI on (student - teacher) is above the margin.

    `interval` selects the decision rule — see the module docstring's migration note. The default
    is the historical Wald central 95% interval and is byte-stable; `tango_one_sided95` decides on
    Tango's one-sided 95% lower score bound and must be chosen deliberately (and preregistered).

    `fail_on_task_collapse` adds the secondary gate: any task where the teacher had at least one
    success and the student none fails the certificate regardless of the aggregate result.
    """
    check_declaration(margin, interval)

    ti, si = _index(teacher, "teacher"), _index(student, "student")
    common = sorted(set(ti) & set(si))
    notes: list[str] = []
    if not common:
        raise NotCertifiable(
            "no (episode, seed) pairs are common to both arms; the runs are not "
            "paired and cannot be compared this way."
        )
    only_t, only_s = set(ti) - set(si), set(si) - set(ti)
    if only_t or only_s:
        raise NotCertifiable(
            f"arms are not on the same episodes: {len(only_t)} teacher-only, {len(only_s)} "
            f"student-only. Certification refuses partial overlap rather than silently "
            f"comparing whatever happens to match."
        )
    if len(common) < min_pairs:
        raise NotCertifiable(f"{len(common)} pairs < min_pairs={min_pairs}")

    for k in common:
        if ti[k].task != si[k].task:
            raise NotCertifiable(
                f"episode {k} is task {ti[k].task!r} for the teacher and "
                f"{si[k].task!r} for the student"
            )

    pairs = [(ti[k].success, si[k].success) for k in common]
    n = len(pairs)
    per_task: dict[str, list] = {}
    for k in common:
        row = per_task.setdefault(ti[k].task, [0, 0, 0])
        row[0] += 1
        row[1] += 1 if ti[k].success else 0
        row[2] += 1 if si[k].success else 0
    t_rate = sum(1 for t, _ in pairs if t) / n
    s_rate = sum(1 for _, s in pairs if s) / n
    delta = s_rate - t_rate
    b = sum(1 for t, s in pairs if t and not s)  # teacher-only wins
    c = sum(1 for t, s in pairs if s and not t)  # student-only wins
    p = _mcnemar_exact(b, c)
    wald = _paired_delta_ci(pairs)
    tango: dict = {}

    if interval == "wald_central95":
        # THE HISTORICAL DEFAULT, byte-for-byte: decide on the Wald central 95% interval.
        lo, hi = wald
        if lo > margin:
            verdict = f"PASS non-inferiority at margin {margin:+.4f} (CI lower bound {lo:+.4f})"
        elif hi < margin:
            # the whole interval is below the margin: a regression, detected
            verdict = (
                f"FAIL (regression detected) at margin {margin:+.4f}: the entire 95% CI "
                f"[{lo:+.4f}, {hi:+.4f}] lies below the margin"
            )
        else:
            # the interval straddles the margin: we cannot tell, and saying so is honest
            need = required_pairs(pairs, margin)
            verdict = (
                f"FAIL (insufficient evidence) at margin {margin:+.4f}: CI [{lo:+.4f}, "
                f"{hi:+.4f}] straddles it. n={n} is too small to decide; "
                f"~{need} paired episodes are needed at the observed discordance rate"
            )
            notes.append(f"underpowered: {n} pairs, ~{need} needed for margin {margin:+.4f}")
    else:
        # tango_one_sided95, chosen deliberately: decide on the one-sided 95% lower score bound.
        lo, hi = _tango_paired_score_bounds(pairs, z=ONE_SIDED_95_Z)
        if lo > margin:
            verdict = (
                f"PASS non-inferiority at margin {margin:+.4f} "
                f"(one-sided 95% lower bound {lo:+.4f})"
            )
        elif hi < margin:
            verdict = (
                f"FAIL (regression detected) at margin {margin:+.4f}: the one-sided 95% "
                f"upper bound {hi:+.4f} lies below the margin"
            )
        else:
            need = required_pairs(pairs, margin, z=ONE_SIDED_95_Z)
            if need:
                tail = f"~{need} paired episodes are needed at the observed discordance rate"
                notes.append(f"underpowered: {n} pairs, ~{need} needed for margin {margin:+.4f}")
            else:
                tail = (
                    "the point estimate is at/below the margin, so more data at this effect "
                    "cannot establish NI"
                )
                notes.append("point estimate is at/below the non-inferiority margin")
            verdict = (
                f"FAIL (insufficient evidence) at margin {margin:+.4f}: one-sided 95% "
                f"bounds [{lo:+.4f}, {hi:+.4f}] straddle it. {tail}"
            )
        tango = dict(
            ci_method=(
                "Tango paired score one-sided 95% lower confidence bound "
                f"(nominal alpha=0.05; z={ONE_SIDED_95_Z})"
            ),
            # the ROLE ("descriptive, never deciding") is rendered by __str__; the field states
            # only what the number is, so downstream tooling can match on the test's name.
            p_value_kind="two-sided exact McNemar equality test",
            tango_central90=(lo, hi),
            lower_confidence_bound=lo,
            lower_confidence_level=0.95,
            alpha=0.05,
        )

    collapse: dict = {}
    if fail_on_task_collapse:
        collapsed = tuple(
            task
            for task, (_task_n, teacher_wins, student_wins) in sorted(per_task.items())
            if teacher_wins > 0 and student_wins == 0
        )
        collapse = dict(task_collapse_gate=True, collapsed_tasks=collapsed)
        if collapsed:
            notes.append(f"task collapse: {', '.join(collapsed)}")
            verdict = (
                "FAIL (secondary task-collapse gate): teacher had at least one success and "
                f"student had zero successes on {', '.join(collapsed)}"
            )
    if b + c == 0:
        notes.append(
            "zero discordant pairs: the arms agreed on every episode, so McNemar has no "
            "information and p=1 by construction"
        )
    return Certificate(
        teacher_hash=teacher_hash,
        student_hash=student_hash,
        n_pairs=n,
        teacher_success=t_rate,
        student_success=s_rate,
        delta=delta,
        ci95=wald,
        margin_declared=margin,
        verdict=verdict,
        p_value=p,
        discordant=(b, c),
        harness=harness,
        recipe=recipe,
        seeds=seeds,
        per_task={t: tuple(v) for t, v in sorted(per_task.items())},
        notes=tuple(notes),
        **tango,
        **collapse,
    )


def load_jsonl(path: str) -> list[Outcome]:
    """Per-episode JSONL: one object per episode with episode_id, seed, task, success."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                success = d["success"]
                if not isinstance(success, bool):
                    # bool("false") is True; a truthiness cast here silently scores a string
                    # column as all-success. Outcomes must be JSON true/false, nothing else.
                    raise TypeError("success must be JSON true/false")
                out.append(Outcome(str(d["episode_id"]), int(d["seed"]), str(d["task"]), success))
            except (KeyError, ValueError, TypeError) as e:
                raise NotCertifiable(
                    f"{path}:{i} is not a valid outcome record ({e}). "
                    f"Required: episode_id, seed, task, success"
                ) from e
    if not out:
        raise NotCertifiable(f"{path} contains no outcomes")
    return out
