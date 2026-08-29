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

Phase 1 Modal runner evaluates an export already present in the
pi0-onnx-outputs Modal volume. The local export_dir maps to an
ONNX subdirectory under /onnx_out; if that subdirectory has not been
uploaded/prepared, the Modal script fails loudly instead of silently
evaluating a reference export.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
DEFAULT_MODAL_SCRIPT = "scripts/modal_libero_monolithic_onnx.py"
MODAL_ONNX_OUTPUT_PATH = "/onnx_out"


class ModalNotInstalledError(RuntimeError):
    """Raised when `modal` CLI is not on PATH at runtime."""


class ModalInvocationError(RuntimeError):
    """Raised when `modal run` exits non-zero or returns malformed output."""


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
    export_dir: Path,
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
        export_dir: customer's export directory. The basename (or path
            relative to /onnx_out) is forwarded as the Modal volume subdir
            so `./my-export` evaluates `/onnx_out/my-export`, not the
            baked-in reference export.
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

    # Resolve script path
    root = repo_root or Path.cwd()
    abs_script = (root / script_path).resolve()
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

    onnx_subdir = _modal_onnx_subdir_for_export(export_dir)

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
            onnx_subdir=onnx_subdir,
            num_episodes=config.num_episodes,
            seed=config.seed,
            timeout_s=suite_timeout_s,
        )
        episodes = _parse_invocation_to_episodes(invocation)
        all_episodes.extend(episodes)

    return all_episodes


def _modal_onnx_subdir_for_export(export_dir: Path) -> str:
    """Map the user-facing export_dir to the Modal volume subdir.

    The Modal app mounts the shared ONNX volume at /onnx_out. A user may pass
    either a path already rooted there (`/onnx_out/foo`) or the local export dir
    they used for `tether export` (`./foo`). In both cases the wrapper must
    forward a specific subdir; otherwise the script falls back to its legacy
    smolvla_libero_monolithic reference export.
    """
    export_path = Path(export_dir).expanduser()
    modal_root = Path(MODAL_ONNX_OUTPUT_PATH)
    try:
        rel = export_path.resolve(strict=False).relative_to(modal_root)
    except ValueError:
        rel = Path(export_path.name)

    onnx_subdir = rel.as_posix()
    if not onnx_subdir or onnx_subdir == ".":
        raise ValueError(
            "export_dir must identify a concrete Modal ONNX subdirectory "
            f"under {MODAL_ONNX_OUTPUT_PATH}."
        )
    return onnx_subdir


def _invoke_one_suite(
    *,
    modal_invoker: ModalInvoker,
    modal_binary: str,
    script_path: str,
    suite: str,
    onnx_subdir: str,
    num_episodes: int,
    seed: int,
    timeout_s: float,
) -> ModalInvocationResult:
    """Subprocess one `modal run scripts/modal_libero_*.py --suite X
    --num-episodes N --tasks all` invocation."""
    import time
    cmd = [
        modal_binary, "run", script_path,
        "--suite", suite,
        "--num-episodes", str(num_episodes),
        "--tasks", "all",
        "--onnx-subdir", onnx_subdir,
    ]
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
    r"^=+ (?P<suite>\S+) \(ONNX monolithic\) =+",
    re.MULTILINE,
)
_RESULT_LINE_RE = re.compile(
    r"^\s*Success: (?P<succ>\d+)/(?P<total>\d+)\s*=\s*(?P<pct>[\d.]+)%",
    re.MULTILINE,
)
_PER_TASK_RE = re.compile(
    r"\[onnx\] task (?P<task_idx>\d+) done: (?P<succ>\d+)/(?P<total>\d+)",
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
        per_task.append({
            "task_idx": int(m.group("task_idx")),
            "success": int(m.group("succ")),
            "total": int(m.group("total")),
        })

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

    Modal's existing script returns aggregate per-task counts (not
    per-episode). We synthesize per-episode rows: first N successes
    are success=True, remaining failures are success=False with
    terminal_reason="adapter_error" (we don't have per-episode root
    cause from the aggregate).

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
        # Synthesize per-episode rows
        for ep_idx in range(n_total):
            success = ep_idx < n_succ
            out.append(EpisodeResult(
                task_id=task_id,
                episode_index=ep_idx,
                success=success,
                terminal_reason="success" if success else "adapter_error",
                wall_clock_s=invocation.elapsed_s / max(n_total, 1),
                n_steps=TASK_SUITE_MAX_STEPS.get(suite, 0),
                video_path=None,
                error_message=None if success else (
                    "Per-episode root cause unavailable from Modal "
                    "aggregate output (Phase 1 limit)."
                ),
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
    "ModalInvocationError",
    "ModalInvocationResult",
    "ModalInvoker",
    "ModalNotInstalledError",
    "TASK_SUITE_MAX_STEPS",
    "run_libero_on_modal",
]
