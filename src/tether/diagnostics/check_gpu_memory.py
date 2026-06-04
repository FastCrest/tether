"""Check 8 — GPU memory headroom (LeRobot #2137).

Probes available GPU memory via nvidia-smi, compares against the
estimated model footprint. Skips cleanly on machines without nvidia-smi
(macOS, CPU-only Linux, or servers where the binary isn't on PATH).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import Check, CheckResult, register

CHECK_ID = "check_gpu_memory"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2137"


def _probe_nvidia_smi() -> tuple[float, float] | None:
    """Returns (total_mb, free_mb) for cuda:0 or None if probe fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    try:
        total_mb = float(parts[0])
        free_mb = float(parts[1])
    except ValueError:
        return None
    return total_mb, free_mb


def _run(model_path: str, rtc: bool = False, **kwargs) -> CheckResult:
    probe = _probe_nvidia_smi()
    if probe is None:
        return CheckResult(
            check_id=CHECK_ID,
            name="GPU memory",
            status="skip",
            expected="nvidia-smi available + accessible",
            actual="nvidia-smi not on PATH (or probe failed)",
            remediation=(
                "Skipped on systems without nvidia-smi (macOS, CPU-only Linux). "
                "On Linux with NVIDIA hardware: ensure nvidia-smi is on PATH."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    total_mb, free_mb = probe
    free_gb = free_mb / 1024.0
    total_gb = total_mb / 1024.0

    # Estimate model GPU footprint
    p = Path(model_path)
    if not p.exists():
        # Can't estimate without a model — report GPU state but skip the comparison
        return CheckResult(
            check_id=CHECK_ID,
            name="GPU memory",
            status="skip",
            expected="export dir exists for footprint estimation",
            actual=f"GPU has {free_gb:.1f}/{total_gb:.1f} GB free; export dir missing",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    onnx_total = sum(f.stat().st_size for f in p.glob("*.onnx"))
    onnx_total += sum(f.stat().st_size for f in p.glob("*.bin"))
    onnx_total += sum(f.stat().st_size for f in p.glob("*.data"))
    estimated_gb = (onnx_total * 1.6) / (1024 ** 3)  # ×1.6 for KV cache + activations

    # Add RTC chunk buffer overhead estimate (~50MB) when --rtc set
    if rtc:
        estimated_gb += 0.05

    if estimated_gb > free_gb * 0.9:
        return CheckResult(
            check_id=CHECK_ID,
            name="GPU memory",
            status="fail",
            expected=f"model footprint ≤ 90% of free GPU mem ({free_gb:.1f}GB free)",
            actual=f"estimated {estimated_gb:.1f}GB needed (×1.6 file size)",
            remediation=(
                f"Model needs ~{estimated_gb:.1f} GB GPU but only {free_gb:.1f} GB free. "
                f"Either: (a) free other GPU processes, (b) re-export FP16 "
                f"(`tether export --fp16`, ~50% smaller), or (c) deploy to a host with "
                f"more GPU memory."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="GPU memory",
        status="pass",
        expected=f"model fits in available GPU mem with headroom",
        actual=f"~{estimated_gb:.1f}GB est, {free_gb:.1f}/{total_gb:.1f}GB free",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="GPU memory",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
