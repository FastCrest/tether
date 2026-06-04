"""Probe cuda-graph capture on the decomposed export's two ORT sessions.

Reports which sessions ORT can capture on the current hardware + software
stack. Surfaces A10G's vlm_prefix OOM limitation (from ADR
2026-04-24-cuda-graphs-architecture Day-0 spike findings) so customers
know what they'll get BEFORE they run `tether serve --cuda-graphs`.

Behavior:
- SKIP if export is not decomposed (cuda-graphs doesn't apply to monolithic)
- SKIP if onnxruntime-gpu is not installed (CPU-only env)
- SKIP if CUDAExecutionProvider is not available
- PASS if both sessions capture
- WARN if only expert_denoise captures (A10G-tier; expected on 24GB GPUs)
- FAIL if neither captures (misconfigured env, unsupported hardware)

Cost: loads both ORT sessions (~6.5 GB model weights) and runs one synthetic
forward each. Typically 15-30s on GPU; skipped without CUDA. Users can skip
explicitly via `tether doctor --skip cuda_graphs`.
"""
from __future__ import annotations

import json
from pathlib import Path

from tether.diagnostics import Check, CheckResult, register


_CHECK_ID = "cuda_graphs"
_CHECK_NAME = "cuda-graphs capture compat"


def _skip(reason: str, actual: str = "") -> CheckResult:
    return CheckResult(
        check_id=_CHECK_ID,
        name=_CHECK_NAME,
        status="skip",
        expected=reason,
        actual=actual,
        remediation="",
        duration_ms=0.0,
    )


def _run(*, model_path: str, embodiment_name: str, rtc: bool) -> CheckResult:
    export = Path(model_path)
    cfg_path = export / "tether_config.json"
    if not cfg_path.exists():
        return _skip(
            "decomposed export required (tether_config.json)",
            f"no tether_config.json at {export}",
        )

    cfg = json.loads(cfg_path.read_text())
    if cfg.get("export_kind") != "decomposed":
        return _skip(
            "cuda-graphs applies only to decomposed exports",
            f"export_kind={cfg.get('export_kind')!r}",
        )

    try:
        import onnxruntime as ort
    except ImportError as e:
        return _skip("onnxruntime-gpu installed", f"ImportError: {e}")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return _skip(
            "CUDAExecutionProvider available",
            f"providers={ort.get_available_providers()}",
        )

    try:
        from tether.runtime.cuda_graphs import (
            build_cuda_graph_providers,
            try_capture_or_fall_back,
        )
    except ImportError as e:
        return CheckResult(
            check_id=_CHECK_ID,
            name=_CHECK_NAME,
            status="fail",
            expected="tether.runtime.cuda_graphs importable",
            actual=f"ImportError: {e}",
            remediation="reinstall tether: `pip install -e .[gpu]`",
            duration_ms=0.0,
        )

    prefix_path = export / cfg["decomposed"]["vlm_prefix_onnx"]
    expert_path = export / cfg["decomposed"]["expert_denoise_onnx"]

    def _build_prefix(cg_enabled: bool) -> "ort.InferenceSession":
        return ort.InferenceSession(
            str(prefix_path),
            providers=build_cuda_graph_providers(enabled=cg_enabled),
        )

    def _build_expert(cg_enabled: bool) -> "ort.InferenceSession":
        return ort.InferenceSession(
            str(expert_path),
            providers=build_cuda_graph_providers(enabled=cg_enabled),
        )

    # Probe both sessions. try_capture_or_fall_back emits metrics on fallback
    # (tether_cuda_graph_capture_failed_at_init_total) so operators can see
    # the same signal in Prometheus + doctor report.
    prefix_wrapped = try_capture_or_fall_back(
        _build_prefix,
        session_name="vlm_prefix",
        embodiment=embodiment_name,
        model_id="doctor-probe",
    )
    expert_wrapped = try_capture_or_fall_back(
        _build_expert,
        session_name="expert_denoise",
        embodiment=embodiment_name,
        model_id="doctor-probe",
    )

    prefix_ok = prefix_wrapped.captured
    expert_ok = expert_wrapped.captured

    if prefix_ok and expert_ok:
        return CheckResult(
            check_id=_CHECK_ID,
            name=_CHECK_NAME,
            status="pass",
            expected="both sessions capture",
            actual=(
                "vlm_prefix + expert_denoise captured → full 2-graph cuda-graphs "
                "available. Expect ~16x compound per-chunk speedup (4.87x vlm_prefix "
                "* 12.23x expert_denoise, amortized by episode-cache hit rate)."
            ),
            remediation="",
            duration_ms=0.0,
        )

    if expert_ok and not prefix_ok:
        return CheckResult(
            check_id=_CHECK_ID,
            name=_CHECK_NAME,
            status="warn",
            expected="both sessions capture",
            actual=(
                "vlm_prefix capture failed (typical on A10G 24GB — insufficient "
                "memory for ORT's capture workspace); expert_denoise captured "
                "cleanly. cuda-graphs runs in tier-aware mode: expert-only "
                "capture delivers ~4x per-chunk speedup on cache-miss paths; "
                "vlm_prefix stays eager."
            ),
            remediation=(
                "For full 16x compound speedup: deploy on A100-40GB+ / AGX Orin "
                "64GB / Thor 128GB. A10G tier is documented as expert-only per "
                "ADR 2026-04-24-cuda-graphs-architecture. Phase 1.5 feature "
                "vlm-prefix-a10g-reexport may extend full capture to A10G later."
            ),
            duration_ms=0.0,
        )

    # Neither captured → misconfigured or unsupported hardware
    return CheckResult(
        check_id=_CHECK_ID,
        name=_CHECK_NAME,
        status="fail",
        expected="at least expert_denoise captures",
        actual=(
            f"vlm_prefix.captured={prefix_ok}, expert_denoise.captured={expert_ok}. "
            f"Check Prometheus metric tether_cuda_graph_capture_failed_at_init_total "
            f"for the failure reason on each session."
        ),
        remediation=(
            "cuda-graphs not supported on this hardware. Serve WITHOUT --cuda-graphs "
            "flag. Possible causes: (a) CUDA EP failed to load (cuBLAS/cuDNN missing), "
            "(b) insufficient GPU memory for even expert_denoise capture, "
            "(c) unsupported ONNX ops in this export. Inspect the doctor JSON output "
            "or Prometheus metrics for the specific exception."
        ),
        duration_ms=0.0,
    )


register(Check(
    check_id=_CHECK_ID,
    name=_CHECK_NAME,
    severity="warn",
    github_issue=None,
    run_fn=_run,
))
