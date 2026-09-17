"""Check 10 — Hardware compatibility (CUDA / cuDNN / TensorRT versions).

Cross-checks installed CUDA/cuDNN/TensorRT versions against the matrix
required by Tether (CUDA 12.x, cuDNN 9.x, TRT 10.x for FP16). Per ADR
2026-04-14-strict-provider-no-silent-cpu-fallback, version drift is
the most-common cause of silent CPU fallback.

Two probes live here: a discrete-GPU probe through `nvidia-smi`, and a Jetson
probe that does not. `nvidia-smi` is not installed on a Tegra board, so gating
everything on it made this check report "no CUDA detected" on an Orin running
CUDA 12.6 with both GPU execution providers available. Jetson facts come from
`tether.jetson`, which is also what `tether.agent.hardware` uses.
"""
from __future__ import annotations

import shutil
import subprocess

from tether.jetson import MIN_L4T_MAJOR_FOR_CUDA12, jetson_profile, l4t_release

from . import Check, CheckResult, register

CHECK_ID = "check_hardware_compat"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2137"

# Required version ranges (per ORT 1.20+ requirements)
_CUDA_MIN_MAJOR = 12
_CUDNN_MIN_MAJOR = 9


def _probe_cuda_version() -> str | None:
    """Returns 'CUDA Version: 12.6' string, or None if probe fails.

    Discrete-GPU path only. A Jetson has no `nvidia-smi`; see `_jetson_result`.
    """
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


def _result(status, expected, actual, remediation="") -> CheckResult:
    return CheckResult(
        check_id=CHECK_ID,
        name="Hardware compat",
        status=status,
        expected=expected,
        actual=actual,
        remediation=remediation,
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


def _jetson_result(ort_version: str, ort_providers: list[str]) -> CheckResult:
    """Verdict for a Tegra board. Never consults `nvidia-smi`."""
    profile = jetson_profile()
    release = l4t_release()
    facts = [f"ORT={ort_version}", f"L4T={profile['l4t_release'] or 'unreadable'}"]
    if profile["jetpack"]:
        facts.append(f"JetPack={profile['jetpack']}")
    if profile["cuda"]:
        facts.append(f"CUDA={profile['cuda']}")
    if profile["tensorrt"]:
        facts.append(f"TensorRT={profile['tensorrt']}")
    summary = ", ".join(facts)
    has_gpu_ep = any(
        provider in ort_providers
        for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
    )

    if release is not None and not release.ships_cuda_12:
        return _result(
            "fail",
            f"L4T R{MIN_L4T_MAJOR_FOR_CUDA12}+ (JetPack 6+) for ORT {_CUDA_MIN_MAJOR}.x CUDA EP",
            summary,
            f"JetPack {profile['jetpack'] or f'R{release.major}'} ships CUDA 11.4. ORT 1.20+ "
            f"requires CUDA 12.x, so CUDAExecutionProvider will silently fall to CPU. "
            f"Flash JetPack 6 (L4T R36+) on an Orin, or install "
            f"fastcrest-tether[serve,onnx] and accept CPU-only inference.",
        )

    if profile["cuda"] is None:
        return _result(
            "warn",
            "CUDA toolkit installed under JetPack",
            summary,
            "This is a Jetson, but neither /usr/local/cuda/version.json nor `nvcc` reports a "
            "CUDA version. Install the JetPack CUDA toolkit: "
            "`sudo apt-get install nvidia-jetpack` (or the cuda-toolkit-12-* metapackage).",
        )

    if not has_gpu_ep:
        return _result(
            "fail",
            "TensorrtExecutionProvider or CUDAExecutionProvider on a JetPack 6 board",
            f"{summary} but providers={ort_providers}",
            "CUDA is installed but onnxruntime cannot use it. On Jetson the PyPI "
            "`onnxruntime-gpu` wheel is the wrong build — install NVIDIA's JetPack wheel from "
            "https://pypi.jetson-ai-lab.dev/ (or the jp6/cu126 index) so the EPs link against "
            "JetPack's own CUDA/cuDNN/TensorRT. See docs/getting_started.md → Jetson.",
        )

    if profile["tensorrt"] is None:
        return _result(
            "warn",
            "TensorRT installed under JetPack",
            summary,
            "GPU execution providers are available but TensorRT's version could not be read "
            "from NvInferVersion.h or dpkg. FP16 engine builds will be unverifiable. Install "
            "`nvidia-tensorrt` from the JetPack apt repository.",
        )

    return _result("pass", f"JetPack 6+ CUDA {_CUDA_MIN_MAJOR}.x + ORT GPU EP", summary + ", GPU EP available")


def _run(**kwargs) -> CheckResult:
    # ONNX runtime version (the user-installed pip package)
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        ort_providers = ort.get_available_providers()
    except ImportError:
        return _result(
            "fail",
            "onnxruntime importable for version check",
            "onnxruntime not installed",
            "pip install fastcrest-tether[serve] (CPU) or [gpu] (GPU)",
        )

    # Jetson first: a Tegra board answers none of the questions below.
    if l4t_release() is not None:
        return _jetson_result(ort_version, list(ort_providers))

    # CUDA via nvidia-smi (the system-installed driver version)
    cuda_version = _probe_cuda_version()
    cuda_major = _parse_major(cuda_version) if cuda_version else None
    has_cuda_provider = "CUDAExecutionProvider" in ort_providers

    facts = [f"ORT={ort_version}"]
    facts.append(f"CUDA driver={cuda_version}" if cuda_version else "no nvidia-smi (CPU-only or non-Linux)")

    # On systems with CUDA but missing GPU EP — drift likely
    if cuda_version and not has_cuda_provider:
        return _result(
            "fail",
            "CUDAExecutionProvider available when CUDA driver present",
            f"{', '.join(facts)} but providers={ort_providers}",
            f"CUDA driver {cuda_version} is installed but onnxruntime can't use "
            f"it. Likely cause: you installed `onnxruntime` (CPU-only). Fix: "
            f"`pip uninstall onnxruntime && pip install onnxruntime-gpu`. ORT 1.20+ "
            f"also needs cuDNN 9 system libraries on the load path — see "
            f"docs/getting_started.md → Troubleshooting.",
        )

    # On systems with CUDA major < 12 and GPU provider expected
    if cuda_major is not None and cuda_major < _CUDA_MIN_MAJOR:
        return _result(
            "warn",
            f"CUDA driver ≥ {_CUDA_MIN_MAJOR}.x for ORT 1.20+",
            f"CUDA driver {cuda_version} (major={cuda_major})",
            f"CUDA driver {cuda_version} predates ORT 1.20+ requirements (need "
            f"CUDA 12.x). Either upgrade NVIDIA driver OR pin onnxruntime-gpu < 1.20. "
            f"Per ADR 2026-04-14, this is the most common cause of silent CPU fallback.",
        )

    # All compat checks passed (or skipped because no CUDA stack at all)
    if not cuda_version:
        return _result(
            "warn",
            "CUDA stack for production GPU deployment",
            f"no CUDA detected; {', '.join(facts)}",
            "CPU-only is fine for dev. For production, install CUDA 12+ + cuDNN 9 "
            "+ onnxruntime-gpu. See docs/getting_started.md.",
        )

    return _result(
        "pass",
        f"CUDA ≥ {_CUDA_MIN_MAJOR}.x + ORT GPU EP",
        ", ".join(facts) + ", GPU EP available",
    )


register(Check(
    check_id=CHECK_ID,
    name="Hardware compat",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
