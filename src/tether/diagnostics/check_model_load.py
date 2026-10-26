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

# TensorRT engine-build peak host memory, fitted to native TensorRT 10.14 builds of SmolVLA on an
# x86 A10G (2026-10-25): 14.2 GB FP32 / 10 baked steps, 12.6 GB mixed FP16 / 10 steps, 9.6 GB FP32 /
# 1 step. That is ~2x the file for the ONNX parse plus ~23 bytes per parameter in the builder, plus
# the workspace and CUDA context. Same model as Studio's preflight. First Jetson points, Orin Nano
# Super 8 GB, JetPack 6 (L4T 36.4.3), ORT 1.24.0, tester runs:
# - SigLIP 372 MB encoder, TensorRT FP16 build: model 4.6 GB, measured process peak 4.2 GB
#   (board 5.4 of 7.6 GB), 152 s (2026-10-26). One tactic was skipped on a 367 MB CUDA allocation
#   with 5.9 GB available.
# - SmolVLA 1.6 GB FP32: model 14.0 GB, killed with 4.5 GB resident and all 12.3 GB of swap used,
#   so at least 16.8 GB (2026-10-21).
# On a Jetson the GPU shares RAM, and TensorRT's CUDA allocations cannot be swapped: swap only
# helps the host side of the build. Two points do not refit the coefficients.
PARSE_FACTOR = 2.0
BUILDER_BYTES_PER_PARAMETER = 23.0
WORKSPACE_AND_CONTEXT_BYTES = (1 << 30) + 600 * 1024 * 1024


def _swap_free_bytes() -> int:
    """Free swap, which a one-time TensorRT build can use; 0 when unreadable."""
    try:
        import psutil

        return int(psutil.swap_memory().free)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("SwapFree:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _trt_build_pending(export_dir: Path) -> bool:
    """Whether serving this export would build a TensorRT engine now.

    False when TensorRT is switched off (TETHER_TRT_EP=0), its execution provider is not
    installed, or an engine is already cached next to the export (no build happens then).
    """
    import os

    if os.environ.get("TETHER_TRT_EP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        import onnxruntime

        if "TensorrtExecutionProvider" not in onnxruntime.get_available_providers():
            return False
    except ImportError:
        return False
    return not any((export_dir / ".ort_trt" / "engines").glob("*.engine"))


def _is_reduced_precision(export_dir: Path) -> bool:
    import json

    try:
        precision = json.loads((export_dir / "tether_config.json").read_text()).get("precision")
    except (OSError, ValueError, AttributeError):
        return False
    # Not "fp16": decomposed exports carry that label over FP32 ONNX files.
    return precision in {"mixed_float16", "float16"}


def trt_build_peak_bytes(file_bytes: int, *, reduced_precision: bool) -> int:
    parameters = file_bytes / (2 if reduced_precision else 4)
    return int(file_bytes * PARSE_FACTOR + parameters * BUILDER_BYTES_PER_PARAMETER + WORKSPACE_AND_CONTEXT_BYTES)


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
    from tether.jetson import is_jetson

    return is_jetson()


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

    trt_build_gb = trt_build_peak_bytes(total_bytes, reduced_precision=_is_reduced_precision(p)) / (1024 ** 3)
    build_budget_gb = available_gb + _swap_free_bytes() / (1024 ** 3)
    if _is_jetson() and _trt_build_pending(p) and trt_build_gb > build_budget_gb * 0.8:
        return CheckResult(
            check_id=CHECK_ID,
            name="Model load",
            status="warn",
            expected="TensorRT engine build peak (measured model: 2x parse + ~23 B/parameter) fits in available RAM",
            actual=f"~{trt_build_gb:.1f}GB estimated build peak vs {build_budget_gb:.1f}GB available RAM + free swap",
            remediation=(
                "The model loads, but a TensorRT engine build needs far more memory than the "
                "weights, and a Jetson shares one memory pool. A 1.6GB FP32 export was killed during "
                "the engine build on an 8GB Orin Nano even with 12GB of swap; an FP16 copy still "
                "peaked at 12.6GB on a desktop GPU. Swap helps only the host side of the one-time "
                "build (the engine is cached afterwards): TensorRT's GPU allocations come from free "
                "RAM and cannot swap. Free memory, use a smaller export, or serve with CUDA instead "
                "of TensorRT."
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
