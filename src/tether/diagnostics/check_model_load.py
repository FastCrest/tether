"""Check 1 — Model load timeout/OOM (LeRobot #386, #414).

Verifies the export directory exists, contains an ONNX file, and the
ONNX file fits in available system RAM with 20% headroom.

Does NOT actually load the model (avoids the 30-90s ORT JIT cost on
every doctor run). Loading is deferred to actual `tether serve`.
"""
from __future__ import annotations

from pathlib import Path

from . import Check, CheckResult, register

CHECK_ID = "check_model_load"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/414"


def _run(model_path: str, **kwargs) -> CheckResult:
    p = Path(model_path)

    # Existence
    if not p.exists():
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="fail",
            expected=f"export dir exists at {model_path}",
            actual="path does not exist",
            remediation=(
                f"Create the export first: `tether export <hf_id> --output {model_path}`. "
                f"See README quickstart."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if not p.is_dir():
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="fail",
            expected="export path is a directory",
            actual=f"{model_path} is a file, not a directory",
            remediation=(
                "tether serve expects a directory containing model.onnx + tether_config.json, "
                "not a single .onnx file."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Find ONNX file(s)
    onnx_files = sorted(p.glob("*.onnx"))
    if not onnx_files:
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="fail",
            expected="at least one .onnx file in the export directory",
            actual=f"found 0 .onnx files in {model_path}",
            remediation=(
                "Re-export with `tether export <hf_id> --output <dir>`. The export "
                "should produce model.onnx (and optionally model.onnx.data for external weights)."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Estimate memory: file size × 1.4 (overhead for ORT session + activations)
    total_bytes = sum(f.stat().st_size for f in p.glob("*.onnx"))
    total_bytes += sum(f.stat().st_size for f in p.glob("*.bin"))  # external weights
    total_bytes += sum(f.stat().st_size for f in p.glob("*.data"))  # external weights variant
    estimated_mem_gb = (total_bytes * 1.4) / (1024 ** 3)

    # Available memory check (psutil optional — skip if absent)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="warn",
            expected=f"~{estimated_mem_gb:.1f}GB needed; psutil to verify available RAM",
            actual="psutil not installed — cannot verify available memory",
            remediation=(
                "pip install psutil to enable memory headroom checks. Without it, doctor "
                "reports model size but can't verify it fits."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if estimated_mem_gb > available_gb * 0.8:  # require 20% headroom
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="fail",
            expected=f"model footprint ≤ 80% of available RAM ({available_gb:.1f}GB available)",
            actual=f"estimated {estimated_mem_gb:.1f}GB (file ×1.4) > 80% headroom",
            remediation=(
                f"Model needs ~{estimated_mem_gb:.1f}GB but only {available_gb:.1f}GB free. "
                f"Either: (a) export FP16 (`tether export --fp16`, ~50% smaller), "
                f"(b) close other processes, or (c) deploy to a larger host."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Model load",
        status="pass",
        expected=f"model fits in available RAM with headroom",
        actual=f"~{estimated_mem_gb:.1f}GB est, {available_gb:.1f}GB available",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="Model load",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
