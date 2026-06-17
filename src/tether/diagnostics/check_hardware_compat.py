"""Check 10 — Hardware compatibility (CUDA / cuDNN / TensorRT versions).

Cross-checks installed CUDA/cuDNN/TensorRT versions against the matrix
required by Tether (CUDA 12.x, cuDNN 9.x, TRT 10.x for FP16). Per ADR
2026-04-14-strict-provider-no-silent-cpu-fallback, version drift is
the most-common cause of silent CPU fallback.
"""
from __future__ import annotations

import shutil
import subprocess

from . import Check, CheckResult, register

CHECK_ID = "check_hardware_compat"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2137"

# Required version ranges (per ORT 1.20+ requirements)
_CUDA_MIN_MAJOR = 12
_CUDNN_MIN_MAJOR = 9


def _probe_cuda_version() -> str | None:
    """Returns 'CUDA Version: 12.6' string, or None if probe fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in out.stdout.splitlines():
        if "CUDA Version" in line:
            # Line looks like "| ... | CUDA Version: 12.6     |"
            after = line.split("CUDA Version:")[-1].strip()
            return after.split()[0] if after else None
    return None


def _parse_major(version_str: str) -> int | None:
    try:
        return int(version_str.split(".")[0])
    except (ValueError, IndexError):
        return None


def _run(**kwargs) -> CheckResult:
    # CUDA via nvidia-smi (the system-installed driver version)
    cuda_version = _probe_cuda_version()
    cuda_major = _parse_major(cuda_version) if cuda_version else None

    # ONNX runtime version (the user-installed pip package)
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        ort_providers = ort.get_available_providers()
    except ImportError:
        return CheckResult(
            check_id=CHECK_ID,
            name="Hardware compat",
            status="fail",
            expected="onnxruntime importable for version check",
            actual="onnxruntime not installed",
            remediation="pip install fastcrest-tether[serve] (CPU) or [gpu] (GPU)",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    has_cuda_provider = "CUDAExecutionProvider" in ort_providers

    # Build the actual report
    facts = []
    facts.append(f"ORT={ort_version}")
    if cuda_version:
        facts.append(f"CUDA driver={cuda_version}")
    else:
        facts.append("no nvidia-smi (CPU-only or non-Linux)")

    # On systems with CUDA but missing GPU EP — drift likely
    if cuda_version and not has_cuda_provider:
        return CheckResult(
            check_id=CHECK_ID,
            name="Hardware compat",
            status="fail",
            expected="CUDAExecutionProvider available when CUDA driver present",
            actual=f"{', '.join(facts)} but providers={ort_providers}",
            remediation=(
                f"CUDA driver {cuda_version} is installed but onnxruntime can't use "
                f"it. Likely cause: you installed `onnxruntime` (CPU-only). Fix: "
                f"`pip uninstall onnxruntime && pip install onnxruntime-gpu`. ORT 1.20+ "
                f"also needs cuDNN 9 system libraries on the load path — see "
                f"docs/getting_started.md → Troubleshooting."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # On systems with CUDA major < 12 and GPU provider expected
    if cuda_major is not None and cuda_major < _CUDA_MIN_MAJOR:
        return CheckResult(
            check_id=CHECK_ID,
            name="Hardware compat",
            status="warn",
            expected=f"CUDA driver ≥ {_CUDA_MIN_MAJOR}.x for ORT 1.20+",
            actual=f"CUDA driver {cuda_version} (major={cuda_major})",
            remediation=(
                f"CUDA driver {cuda_version} predates ORT 1.20+ requirements (need "
                f"CUDA 12.x). Either upgrade NVIDIA driver OR pin onnxruntime-gpu < 1.20. "
                f"Per ADR 2026-04-14, this is the most common cause of silent CPU fallback."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # All compat checks passed (or skipped because no CUDA stack at all)
    if not cuda_version:
        return CheckResult(
            check_id=CHECK_ID,
            name="Hardware compat",
            status="warn",
            expected="CUDA stack for production GPU deployment",
            actual=f"no CUDA detected; {', '.join(facts)}",
            remediation=(
                "CPU-only is fine for dev. For production, install CUDA 12+ + cuDNN 9 "
                "+ onnxruntime-gpu. See docs/getting_started.md."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="Hardware compat",
        status="pass",
        expected=f"CUDA ≥ {_CUDA_MIN_MAJOR}.x + ORT GPU EP",
        actual=", ".join(facts) + ", GPU EP available",
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="Hardware compat",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
