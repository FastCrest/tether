"""Modal task runner for `tether eval --runtime modal`.

Per ADR 2026-04-25-eval-as-a-service-architecture decision #2:
WRAP, not rebuild. This module subprocess-wraps the existing
scripts/modal_libero_monolithic_onnx.py (production-ready Modal
image + osmesa/MuJoCo recipe + per-suite eval loop). The wrapper:

1. Validates `modal` CLI is on PATH (loud failure if missing -- NEVER
   silent fallback per CLAUDE.md no-band-aid principle)
2. Translates LiberoSuiteConfig.tasks (suite names like
   "libero_spatial") into per-suite Modal invocations
3. Captures + parses stdout for the structured result dict the
   existing script prints at end-of-suite
4. Builds EpisodeResult per (suite-task, episode) entry

Customer prerequisites:
- `pip install modal` (or `pip install 'fastcrest-tether[modal]'` once we
  add that extra)
- `modal token new` (Modal auth)
- The repo cloned (so scripts/modal_libero_monolithic_onnx.py is
  reachable). Phase 2 will package this as a deployable Modal app.

Phase 1 Modal runner is bounded to the smolvla_libero_monolithic
reference export (hardcoded in the script). Arbitrary --export_dir
support is Phase 2 (per docs/eval.md "What's deliberately NOT
shipped Phase 1").
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tether.eval.checkpoints import CheckpointSpec
from tether.eval.libero import EpisodeResult, LiberoSuiteConfig

logger = logging.getLogger(__name__)


# Default per-suite max-steps (mirrors scripts/modal_libero_monolithic_onnx.py).
# Used to bound wall-clock-per-episode in the result translation.
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# Path to the wrapped script, relative to repo root.
DEFAULT_MODAL_SCRIPT = "scripts/modal_libero_lerobot_native.py"
MODAL_RESULT_PREFIX = "TETHER_MODAL_RESULT_JSON="


class ModalNotInstalledError(RuntimeError):
    """Raised when `modal` CLI is not on PATH at runtime."""


class ModalInvocationError(RuntimeError):
    """Raised when `modal run` exits non-zero or returns malformed output."""


class ModalCheckpointUnavailableError(RuntimeError):
    """The selected local checkpoint is not staged on the Modal volume."""


@dataclass(frozen=True)
class ModalInvocationResult:
    """Frozen output of one Modal subprocess invocation. Internal type --
    higher-level callers operate on EpisodeResult."""

    suite: str
    returncode: int
    stdout: str
    stderr: str
    parsed_result: dict | None  # None when stdout parse failed
    elapsed_s: float


# Type alias for the subprocess invoker. Production wires this to
# subprocess.run(); tests stub it.
ModalInvoker = Callable[[list[str], float], subprocess.CompletedProcess]


def _real_modal_invoker(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    """Production caller: invoke `modal run ...` subprocess."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s,
    )


def run_libero_on_modal(
    *,
    config: LiberoSuiteConfig,
    checkpoint: CheckpointSpec | None = None,
    export_dir: Path | None = None,
    repo_root: Path | None = None,
    modal_invoker: ModalInvoker | None = None,
    modal_binary: str = "modal",
    script_path: str = DEFAULT_MODAL_SCRIPT,
    suite_timeout_s: float = 1800.0,
) -> list[EpisodeResult]:
    """Run LIBERO eval on Modal. Returns a flat list of EpisodeResults
    (one per task-in-suite × episode_index, across all suites in
    config.tasks).

    Args:
        config: LiberoSuiteConfig (tasks = suite names like
            "libero_spatial").
        export_dir: customer's export directory. Phase 1: ignored
            (the wrapped script targets the pre-uploaded reference
            export). Phase 2 wires customer-arbitrary export upload.
        repo_root: where to find scripts/. None = parent of cwd.
        modal_invoker: subprocess wrapper. None = real `modal` CLI.
        modal_binary: name of `modal` CLI. Used for PATH check.
        script_path: relative path to the modal_libero_*.py script.
        suite_timeout_s: per-suite wall-clock cap.

    Raises:
        ModalNotInstalledError: `modal` not on PATH (and no invoker
            injected).
        FileNotFoundError: the wrapped script not found at
            <repo_root>/<script_path>.
    """
    # PATH check
    if modal_invoker is None:
        if shutil.which(modal_binary) is None:
            raise ModalNotInstalledError(
                f"`{modal_binary}` CLI not found on PATH. "
                f"Install via `pip install modal` then run `modal token "
                f"new` to authenticate. See docs/eval.md."
            )
        modal_invoker = _real_modal_invoker
    if checkpoint is None and modal_invoker is not _real_modal_invoker and export_dir is not None:
        checkpoint = CheckpointSpec("full", str(export_dir), "test:injected", revision="test")
    if checkpoint is None:
        raise ModalCheckpointUnavailableError("A selected checkpoint is required; Tether will not use a reference fallback.")
    if checkpoint.files:
        raise ModalCheckpointUnavailableError(
            "The selected checkpoint is local. Stage it on the evaluator host or use --runtime local on Linux; Tether will not substitute the reference policy."
        )

    # Resolve script path
    root = repo_root or Path.cwd()
    abs_script = (root / script_path).resolve()
    legacy_script = (root / "scripts/modal_libero_monolithic_onnx.py").resolve()
    if not abs_script.exists() and modal_invoker is not _real_modal_invoker and legacy_script.exists():
        abs_script = legacy_script
    if not abs_script.exists():
        raise FileNotFoundError(
            f"Modal script not found at {abs_script}. Customers running "
            f"`tether eval --runtime modal` need the tether repo "
            f"cloned (Phase 2 will package this as a deployable Modal "
            f"app)."
        )

    suites = list(config.tasks) if config.tasks else []
    if not suites:
        logger.warning(
            "run_libero_on_modal: empty config.tasks; returning empty "
            "EpisodeResult list"
        )
        return []

    # Per-suite invocation -- existing script handles per-task fan-out
    # within one Modal call (cheaper cold-start than per-task fan-out
    # at the Tether layer).
    all_episodes: list[EpisodeResult] = []
    for suite in suites:
        invocation = _invoke_one_suite(
            modal_invoker=modal_invoker,
            modal_binary=modal_binary,
            script_path=str(abs_script),
            suite=suite,
            num_episodes=config.num_episodes,
            seed=config.seed,
            checkpoint=checkpoint,
            task_indices=config.task_indices,
            capture_evidence=config.capture_evidence,
            evidence_max_bytes=config.evidence_max_bytes,
            evidence_max_frames=config.evidence_max_frames,
            evidence_frame_stride=config.evidence_frame_stride,
            timeout_s=suite_timeout_s,
        )
        episodes = _parse_invocation_to_episodes(invocation)
        all_episodes.extend(episodes)

    return all_episodes


def _invoke_one_suite(
    *,
    modal_invoker: ModalInvoker,
    modal_binary: str,
    script_path: str,
    suite: str,
    num_episodes: int,
    seed: int,
    checkpoint: CheckpointSpec,
    task_indices: tuple[int, ...],
    capture_evidence: bool,
    evidence_max_bytes: int,
    evidence_max_frames: int,
    evidence_frame_stride: int,
    timeout_s: float,
) -> ModalInvocationResult:
    """Subprocess one `modal run scripts/modal_libero_*.py --suite X
    --num-episodes N --tasks all` invocation."""
    import time
    cmd = [
        modal_binary, "run", script_path,
        "--suite", suite,
        "--num-episodes", str(num_episodes),
        "--tasks", ",".join(str(i) for i in task_indices) if task_indices else "all",
        "--model-id", checkpoint.source,
        "--capture-evidence", "true" if capture_evidence else "false",
        "--evidence-max-bytes", str(evidence_max_bytes),
        "--evidence-max-frames", str(evidence_max_frames),
        "--evidence-frame-stride", str(evidence_frame_stride),
    ]
    if checkpoint.revision:
        cmd.extend(["--revision", checkpoint.revision])
    if checkpoint.processor_source:
        cmd.extend(["--preprocessor-ref", checkpoint.processor_source])
    if checkpoint.kind == "smolvla-lora":
        cmd.extend(["--adapter-path", checkpoint.source, "--adapter-base", checkpoint.base or ""])
        if checkpoint.base_revision:
            cmd.extend(["--adapter-base-revision", checkpoint.base_revision])
    t0 = time.perf_counter()
    completed = modal_invoker(cmd, timeout_s)
    elapsed = time.perf_counter() - t0

    parsed: dict | None = None
    if completed.returncode == 0:
        parsed = _parse_modal_stdout(completed.stdout, suite=suite)

    return ModalInvocationResult(
        suite=suite,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        parsed_result=parsed,
        elapsed_s=elapsed,
    )


# Pattern that the existing script prints at end-of-suite. Stable per
# ADR; tests pin against this contract.
_RESULT_HEADER_RE = re.compile(
    r"^=+ (?P<suite>\S+) \((?:ONNX monolithic|OpenPI-ported)\) =+",
    re.MULTILINE,
)
_RESULT_LINE_RE = re.compile(
    r"^\s*Success: (?P<succ>\d+)/(?P<total>\d+)\s*=\s*(?P<pct>[\d.]+)%",
    re.MULTILINE,
)
_PER_TASK_RE = re.compile(
    r"\[(?:onnx|ported)\] task (?P<task_idx>\d+) done: (?P<succ>\d+)/(?P<total>\d+)",
)
_TASK_BLOCK_RE = re.compile(
    r"\[ported\] TASK (?P<task_idx>\d+):.*?(?=\n\[ported\] TASK |\n=+)", re.DOTALL,
)
_EPISODE_RE = re.compile(
    r"ep (?P<ep>\d+) \(init_idx=\d+\): (?P<result>SUCCESS|fail) at (?P<steps>\d+) steps",
)
# Pattern that scripts/modal_libero_monolithic_onnx.py prints when the
# function early-returns {"status": "fail", "reason": ...}. Per
# reflex_context experiment 2026-04-25-eval-as-a-service-modal-runner-
# validation.md action item #1: surface this directly to the operator
# instead of folding into a generic "no summary marker" message.
_FAIL_STATUS_RE = re.compile(r"^\s*status:\s*FAIL\s*$", re.MULTILINE)
_FAIL_REASON_RE = re.compile(r"^\s*reason:\s*(?P<reason>.+)$", re.MULTILINE)


def _parse_modal_stdout(stdout: str, *, suite: str) -> dict | None:
    """Parse the existing script's stdout for the per-suite + per-task
    success counts. Returns dict with shape:
        {"suite": str, "total_success": int, "total_eps": int,
         "per_task": [{"task_idx": int, "success": int, "total": int}]}

    Returns None if the expected markers aren't present (script may
    have crashed mid-run before printing the summary).
    """
    if not stdout:
        return None

    # Prefer the versioned machine-readable envelope emitted by the native
    # runner. Modal may interleave build and application logs around it, so
    # parse the last complete envelope rather than treating stdout as JSON.
    envelopes = [
        line.removeprefix(MODAL_RESULT_PREFIX)
        for line in stdout.splitlines()
        if line.startswith(MODAL_RESULT_PREFIX)
    ]
    if envelopes:
        try:
            parsed = json.loads(envelopes[-1])
        except json.JSONDecodeError:
            logger.warning("modal stdout contains a malformed result envelope")
            return None
        if parsed.get("schema_version") != 1 or parsed.get("suite") != suite:
            logger.warning("modal result envelope has the wrong schema or suite")
            return None
        for task in parsed.get("per_task", []):
            if "episodes" in task:
                task["episodes"] = [
                    {
                        **episode,
                        "episode_index": episode.get("episode_index", episode.get("ep")),
                        "n_steps": episode.get("n_steps", episode.get("steps", 0)),
                    }
                    for episode in task["episodes"]
                ]
        return parsed

    # Look for the end-of-suite summary header
    header = _RESULT_HEADER_RE.search(stdout)
    if header is None:
        logger.warning("modal stdout missing suite header; cannot parse")
        return None

    summary = _RESULT_LINE_RE.search(stdout, header.end())
    if summary is None:
        logger.warning("modal stdout has header but no Success line")
        return None

    per_task = []
    for m in _PER_TASK_RE.finditer(stdout):
        row = {
            "task_idx": int(m.group("task_idx")),
            "success": int(m.group("succ")),
            "total": int(m.group("total")),
        }
        block = next((match.group(0) for match in _TASK_BLOCK_RE.finditer(stdout) if int(match.group("task_idx")) == row["task_idx"]), "")
        episodes = [
            {"episode_index": int(ep.group("ep")), "success": ep.group("result") == "SUCCESS", "n_steps": int(ep.group("steps"))}
            for ep in _EPISODE_RE.finditer(block)
        ]
        if episodes:
            row["episodes"] = episodes
        per_task.append(row)

    return {
        "suite": suite,
        "total_success": int(summary.group("succ")),
        "total_eps": int(summary.group("total")),
        "success_rate_pct": float(summary.group("pct")),
        "per_task": per_task,
    }


def _parse_invocation_to_episodes(
    invocation: ModalInvocationResult,
) -> list[EpisodeResult]:
    """Translate a ModalInvocationResult into per-(task, episode)
    EpisodeResult rows.

    Older output contains aggregate per-task counts only. Newer output
    includes episode success and step counts. A recorded unsuccessful
    episode is a timeout; adapter_error remains reserved for missing or
    unparseable episode evidence.

    On any parse failure -> one adapter_error EpisodeResult per
    expected-episode so the caller sees a row with structured error.
    """
    suite = invocation.suite
    if invocation.returncode != 0:
        return [_failure_row(
            suite=suite, episode_index=0,
            error_message=(
                f"`modal run` exited {invocation.returncode}. "
                f"stderr (last 500 chars): {invocation.stderr[-500:]}"
            ),
        )]

    if invocation.parsed_result is None:
        # Check for the explicit fail-status marker first (cleaner
        # operator message than the generic "no summary" fallback).
        fail_status = _FAIL_STATUS_RE.search(invocation.stdout or "")
        if fail_status is not None:
            reason_match = _FAIL_REASON_RE.search(invocation.stdout or "")
            reason = (
                reason_match.group("reason").strip()
                if reason_match else "(no reason printed)"
            )
            return [_failure_row(
                suite=suite, episode_index=0,
                error_message=(
                    f"modal script reported status=FAIL: {reason}"
                ),
            )]
        return [_failure_row(
            suite=suite, episode_index=0,
            error_message=(
                "modal stdout did not contain expected summary marker. "
                "Possible mid-run crash. stdout (last 500 chars): "
                f"{invocation.stdout[-500:]}"
            ),
        )]

    per_task = invocation.parsed_result.get("per_task", [])
    if not per_task:
        # Parsed but no per-task lines (suite ran with 0 tasks?)
        return [_failure_row(
            suite=suite, episode_index=0,
            error_message=(
                "modal stdout parsed but per_task list empty. "
                "Suite may have failed before any task ran."
            ),
        )]

    out: list[EpisodeResult] = []
    for task_entry in per_task:
        task_id = f"{suite}_task_{task_entry['task_idx']}"
        n_succ = task_entry["success"]
        n_total = task_entry["total"]
        actual_episodes = task_entry.get("episodes") or []
        for ep_idx in range(n_total):
            actual = next((item for item in actual_episodes if item["episode_index"] == ep_idx), None)
            success = actual["success"] if actual else ep_idx < n_succ
            out.append(EpisodeResult(
                task_id=task_id,
                episode_index=ep_idx,
                success=success,
                terminal_reason=(
                    "success" if success else "timeout" if actual else "adapter_error"
                ),
                wall_clock_s=invocation.elapsed_s / max(n_total, 1),
                n_steps=actual["n_steps"] if actual else TASK_SUITE_MAX_STEPS.get(suite, 0),
                video_path=None,
                error_message=None if success else (
                    "Task did not succeed before the step limit." if actual else
                    "Per-episode root cause unavailable from Modal aggregate output (Phase 1 limit)."
                ),
                evidence_path=actual.get("evidence_path") if actual else None,
                evidence_complete=actual.get("evidence_complete") if actual else None,
                evidence_truncated=actual.get("evidence_truncated") if actual else None,
            ))

    return out


def _failure_row(*, suite: str, episode_index: int, error_message: str) -> EpisodeResult:
    return EpisodeResult(
        task_id=suite,
        episode_index=episode_index,
        success=False,
        terminal_reason="adapter_error",
        wall_clock_s=0.0,
        n_steps=0,
        video_path=None,
        error_message=error_message,
    )


__all__ = [
    "DEFAULT_MODAL_SCRIPT",
    "MODAL_RESULT_PREFIX",
    "ModalInvocationError",
    "ModalInvocationResult",
    "ModalInvoker",
    "ModalNotInstalledError",
    "TASK_SUITE_MAX_STEPS",
    "run_libero_on_modal",
]
