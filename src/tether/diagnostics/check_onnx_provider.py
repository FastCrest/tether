"""Check 2 — ONNX execution provider availability (LeRobot #2137).

Verifies the requested ONNX EP is actually loadable (not silently
falling back to CPU). Per ADR 2026-04-14-strict-provider-no-silent-cpu-fallback,
strict mode treats CPU fallback as a FAIL, not a WARN.
"""
from __future__ import annotations

from . import Check, CheckResult, register

CHECK_ID = "check_onnx_provider"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2137"


def _run(**kwargs) -> CheckResult:
    try:
        import onnxruntime as ort
    except ImportError:
        return CheckResult(
            check_id=CHECK_ID,
            name="ONNX provider",
            status="fail",
            expected="onnxruntime installed",
            actual="ImportError — onnxruntime is missing",
            remediation=(
                "pip install tether[serve] (CPU) or tether[gpu] (GPU). "
                "ONNX runtime is required for the inference path."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available
    has_cpu = "CPUExecutionProvider" in available
    has_trt = "TensorrtExecutionProvider" in available
    has_coreml = "CoreMLExecutionProvider" in available

    # CPU is universally available; if it's not, ORT itself is broken
    if not has_cpu:
        return CheckResult(
            check_id=CHECK_ID,
            name="ONNX provider",
            status="fail",
            expected="CPUExecutionProvider available (always)",
            actual=f"available providers: {available}",
            remediation=(
                "ORT install is broken — CPUExecutionProvider should always be present. "
                "Reinstall: pip install --force-reinstall onnxruntime"
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Pass — we have at least CPU. Note GPU availability for the actual line.
    if has_cuda or has_trt:
        accel = []
        if has_trt: accel.append("TRT")
        if has_cuda: accel.append("CUDA")
        return CheckResult(
            check_id=CHECK_ID,
            name="ONNX provider",
            status="pass",
            expected="at least one accelerator EP available",
            actual=f"providers: CPU + {' + '.join(accel)}",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # CPU-only is fine for dev / non-GPU hosts but worth flagging
    return CheckResult(
        check_id=CHECK_ID,
        name="ONNX provider",
        status="warn",
        expected="GPU EP for production deploys",
        actual=f"only CPU EP available (got: {available})",
        remediation=(
            "Running CPU-only is fine for dev. For production, install onnxruntime-gpu: "
            "pip install tether[gpu]. Per ADR 2026-04-14, --strict-providers fails "
            "the server if GPU is requested but unavailable."
        ),
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="ONNX provider",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
