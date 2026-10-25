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

# Heuristic, not a measurement: the TensorRT execution provider parses the full weights and then
# builds an engine, so its peak is a multiple of the file size, well above ORT's 1.4x session
# footprint. One observed point: a 1.6 GB FP32 export was killed by the kernel out-of-memory killer
# during the engine build on an 8 GB Jetson Orin Nano (2026-10-21, tester bundle). Replace this with
# a measured factor when one exists.
TRT_BUILD_FACTOR = 4.0


def _available_bytes() -> int | None:
    """Available RAM: psutil when installed, else the kernel's MemAvailable. None if neither."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


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

    # Available memory check. Without psutil, Linux (including Jetson) still reports MemAvailable.
    available_bytes = _available_bytes()
    if available_bytes is None:
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="warn",
            expected=f"~{estimated_mem_gb:.1f}GB needed; available RAM readable",
            actual="cannot read available memory (no psutil and no /proc/meminfo)",
            remediation=(
                "pip install psutil to enable memory headroom checks. Without it, doctor "
                "reports model size but can't verify it fits."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )
    available_gb = available_bytes / (1024 ** 3)

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

    trt_build_gb = (total_bytes * TRT_BUILD_FACTOR) / (1024 ** 3)
    if _is_jetson() and trt_build_gb > available_gb * 0.8:
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="warn",
            expected=f"TensorRT engine build peak (heuristic ~{TRT_BUILD_FACTOR:.0f}x weights) fits in available RAM",
            actual=f"~{trt_build_gb:.1f}GB estimated build peak vs {available_gb:.1f}GB available",
            remediation=(
                "The model loads, but building a TensorRT engine needs several times the weight size "
                "and a Jetson shares one memory pool. A 1.6GB FP32 export was killed by the "
                "out-of-memory killer during the engine build on an 8GB Orin Nano. Export FP16 "
                "(about half the size), free memory, or build the engine on a larger machine."
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
