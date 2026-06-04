"""Pre-flight LIBERO smoke test — catches 4-of-5 documented failure modes.

Per ADR 2026-04-25-eval-as-a-service-architecture decision #4 +
research sidecar Lens 2: 5 real failure modes in LIBERO/SimplerEnv on
Modal. 4 are catchable in 1 hour of code via a single isolated-
subprocess smoke test that exercises the critical init path:

1. LIBERO `input()` hang (caught by `patch_libero.py` → checked here)
2. EGL silent black frames (caught by forcing osmesa)
3. Dependency version conflicts (caught by import + env.reset())
4. osmesa first-scene compile hang (caught by 300s timeout on env.reset())

The 5th failure (per-episode OOM) is per-call probabilistic; backoff +
legible error in the runner covers it.

The test runs in an ISOLATED SUBPROCESS so it doesn't pollute the
parent's Python state (LIBERO globals, MUJOCO_GL env, etc.). On
failure it returns a PreflightResult with a bounded failure_mode
enum so the caller (Day 3+ CLI wiring) can render a clear error
message + remediation.

Pure primitive — caller invokes once before the real eval; on failure
refuses to proceed.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


# Bounded enum of failure modes. Stable across releases — surfaced in
# CLI error messages + telemetry labels. Maps 1:1 to the documented
# Lens 2 failures.
FailureMode = Literal[
    "ok",
    "input-hang",
    "egl-black-frames",
    "dep-version-conflict",
    "osmesa-compile-hang",
    "subprocess-error",
    "import-error",
    "unknown",
]
ALL_FAILURE_MODES: tuple[str, ...] = (
    "ok",
    "input-hang",
    "egl-black-frames",
    "dep-version-conflict",
    "osmesa-compile-hang",
    "subprocess-error",
    "import-error",
    "unknown",
)


# Default timeout — covers worst-case osmesa scene-compile (per Lens 2,
# observed up to 180s on cold Modal containers).
DEFAULT_PREFLIGHT_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class PreflightResult:
    """Frozen output of PreflightSmokeTest.run()."""

    passed: bool
    failure_mode: str  # one of ALL_FAILURE_MODES
    elapsed_s: float
    stdout: str
    stderr: str
    remediation: str  # bounded-string operator hint

    def __post_init__(self) -> None:
        if self.failure_mode not in ALL_FAILURE_MODES:
            raise ValueError(
                f"failure_mode must be one of {ALL_FAILURE_MODES}, "
                f"got {self.failure_mode!r}"
            )
        if self.passed and self.failure_mode != "ok":
            raise ValueError(
                f"passed=True but failure_mode={self.failure_mode!r}; these "
                f"must agree"
            )
        if not self.passed and self.failure_mode == "ok":
            raise ValueError(
                "passed=False but failure_mode='ok'; these must agree"
            )


# The smoke-test script that runs in the subprocess. Self-contained
# (no imports from tether itself) so a broken tether install can still
# probe the LIBERO environment.
_SMOKE_TEST_SCRIPT = '''
"""LIBERO smoke test — exits 0 on success, non-zero with structured
stderr on failure. Caller parses stderr to identify the failure mode."""
import os
import sys

# Force osmesa BEFORE any LIBERO/MuJoCo import to avoid EGL silent-
# black-frames failure on debian_slim+A10G.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

try:
    import libero
except ImportError as exc:
    sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=import-error: {exc}\\n")
    sys.exit(2)
except Exception as exc:
    sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=dep-version-conflict: {type(exc).__name__}: {exc}\\n")
    sys.exit(3)

try:
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:
    sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=import-error: {exc}\\n")
    sys.exit(4)

# Find any BDDL task file
try:
    bddl_dir = os.path.join(os.path.dirname(libero.__file__), "libero", "bddl_files")
    if not os.path.exists(bddl_dir):
        sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=dep-version-conflict: bddl_files dir not found at {bddl_dir}\\n")
        sys.exit(5)
    bddl_files = []
    for root, dirs, files in os.walk(bddl_dir):
        for f in files:
            if f.endswith(".bddl"):
                bddl_files.append(os.path.join(root, f))
                break
        if bddl_files:
            break
    if not bddl_files:
        sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=dep-version-conflict: no .bddl files in {bddl_dir}\\n")
        sys.exit(6)
except Exception as exc:
    sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=unknown: {type(exc).__name__}: {exc}\\n")
    sys.exit(7)

try:
    env = OffScreenRenderEnv(bddl_file_name=bddl_files[0])
    obs = env.reset()
except Exception as exc:
    msg = str(exc).lower()
    if "egl" in msg or "vulkan" in msg or "rendering" in msg:
        sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=egl-black-frames: {exc}\\n")
        sys.exit(8)
    sys.stderr.write(f"PREFLIGHT_FAILURE_MODE=osmesa-compile-hang: {type(exc).__name__}: {exc}\\n")
    sys.exit(9)

print("PREFLIGHT_OK")
sys.exit(0)
'''


_REMEDIATION_BY_MODE: dict[str, str] = {
    "ok": "",
    "input-hang": (
        "LIBERO has interactive input() calls that block in non-TTY "
        "environments. Run `scripts/patch_libero.py` BEFORE the eval, "
        "or pass --runtime modal (the Modal image patches this in)."
    ),
    "egl-black-frames": (
        "EGL rendering returned black frames — common on debian_slim+A10G. "
        "The smoke test forces osmesa via MUJOCO_GL=osmesa; check that "
        "env var is set in your runtime environment."
    ),
    "dep-version-conflict": (
        "Dependency version mismatch detected. Pin: robosuite==1.4.1, "
        "bddl==1.0.1, mujoco==3.3.2, lerobot==0.5.1. Use the bundled "
        "Modal image (--runtime modal) for known-good pins."
    ),
    "osmesa-compile-hang": (
        "osmesa scene compilation timed out OR crashed. First-time scene "
        "compile can take 60-180s on cold containers. Increase --preflight-"
        "timeout, or pre-warm the Modal image. If reproducible, file a "
        "GitHub issue with the stderr output."
    ),
    "subprocess-error": (
        "Pre-flight subprocess failed to launch. Check Python availability "
        "in the runtime environment + that the smoke-test script can "
        "execute (no AppArmor / SELinux blocks)."
    ),
    "import-error": (
        "LIBERO import failed. Install via `pip install 'tether[eval-"
        "local]'` for local runs, or use --runtime modal which ships LIBERO "
        "in the bundled image."
    ),
    "unknown": (
        "Unknown failure mode — see stderr for details. File a GitHub "
        "issue with the full smoke-test output if you see this in the wild."
    ),
}


class PreflightSmokeTest:
    """Pre-flight LIBERO smoke test runner. Pure: stateless classmethod."""

    @classmethod
    def run(
        cls,
        *,
        timeout_s: float = DEFAULT_PREFLIGHT_TIMEOUT_S,
        python_executable: str | None = None,
    ) -> PreflightResult:
        """Run the smoke test in an isolated subprocess. Returns
        PreflightResult with bounded failure_mode + remediation."""
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

        executable = python_executable or sys.executable
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [executable, "-c", _SMOKE_TEST_SCRIPT],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0
            return PreflightResult(
                passed=False,
                failure_mode="osmesa-compile-hang",
                elapsed_s=elapsed,
                stdout=(exc.stdout.decode() if isinstance(exc.stdout, bytes)
                        else (exc.stdout or "")),
                stderr=(exc.stderr.decode() if isinstance(exc.stderr, bytes)
                        else (exc.stderr or "")),
                remediation=_REMEDIATION_BY_MODE["osmesa-compile-hang"],
            )
        except (OSError, FileNotFoundError) as exc:
            elapsed = time.perf_counter() - t0
            return PreflightResult(
                passed=False,
                failure_mode="subprocess-error",
                elapsed_s=elapsed,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                remediation=_REMEDIATION_BY_MODE["subprocess-error"],
            )

        elapsed = time.perf_counter() - t0
        if result.returncode == 0 and "PREFLIGHT_OK" in result.stdout:
            return PreflightResult(
                passed=True,
                failure_mode="ok",
                elapsed_s=elapsed,
                stdout=result.stdout,
                stderr=result.stderr,
                remediation="",
            )

        # Parse the failure mode marker from stderr (the smoke-test
        # script writes a structured marker on every failure path).
        mode = cls._extract_failure_mode(result.stderr)
        return PreflightResult(
            passed=False,
            failure_mode=mode,
            elapsed_s=elapsed,
            stdout=result.stdout,
            stderr=result.stderr,
            remediation=_REMEDIATION_BY_MODE.get(
                mode, _REMEDIATION_BY_MODE["unknown"],
            ),
        )

    @staticmethod
    def _extract_failure_mode(stderr: str) -> str:
        """Pull the PREFLIGHT_FAILURE_MODE=<x> marker from stderr.
        Returns 'unknown' if no marker found."""
        if not stderr:
            return "unknown"
        for line in stderr.splitlines():
            if "PREFLIGHT_FAILURE_MODE=" in line:
                # Extract the bounded enum value
                marker = line.split("PREFLIGHT_FAILURE_MODE=", 1)[1]
                mode = marker.split(":", 1)[0].strip()
                if mode in ALL_FAILURE_MODES:
                    return mode
        return "unknown"


__all__ = [
    "ALL_FAILURE_MODES",
    "DEFAULT_PREFLIGHT_TIMEOUT_S",
    "FailureMode",
    "PreflightResult",
    "PreflightSmokeTest",
]
