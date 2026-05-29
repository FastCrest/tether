"""ONNX Runtime provider planning shared by Reflex runtimes.

The legacy decomposed server, monolithic pi0 server, monolithic SmolVLA
server, and CLI benchmark path should all make the same GPU/provider choice.
Keeping this logic in one place prevents a repeat of benchmark-only CUDA paths
that bypass TensorRT engine/timing caches.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_BLACKWELL_GPU_PATTERNS = (
    "rtx 50",          # GeForce RTX 5070/5080/5090
    "rtx pro 60",      # RTX PRO 6000 Blackwell
    "blackwell",
    "b200",            # B200 datacenter
    "gb200",           # GB200 datacenter
)

_LARGE_EXTERNAL_DATA_BYTES = 8 * 1024 * 1024 * 1024


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def provider_name(provider: Any) -> str:
    """Return provider name from ORT's string-or-(name, options) format."""
    return provider[0] if isinstance(provider, tuple) else str(provider)


def gpu_provider_requested(providers: list[Any]) -> bool:
    names = {provider_name(p) for p in providers}
    return bool(names & {"CUDAExecutionProvider", "TensorrtExecutionProvider"})


def gpu_provider_active(active_providers: list[str]) -> bool:
    return bool(set(active_providers) & {"CUDAExecutionProvider", "TensorrtExecutionProvider"})


def gpu_is_blackwell() -> bool:
    """Best-effort detection for GPUs where ORT-bundled TRT can segfault."""
    import subprocess as _sub

    try:
        proc = _sub.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, _sub.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    names_lower = (proc.stdout or "").lower()
    return any(pat in names_lower for pat in _BLACKWELL_GPU_PATTERNS)


def log_blackwell_trt_warning() -> None:
    bar = "=" * 72
    logger.warning(
        "\n%s\n"
        "Blackwell GPU detected (RTX 50-series / B200 / GB200, sm_100)\n"
        "%s\n"
        "TensorRT EP is disabled because older ORT-bundled TensorRT builds can "
        "segfault during session initialization on Blackwell. Reflex will use "
        "CUDAExecutionProvider instead. Inference remains correct, but slower.\n"
        "%s",
        bar,
        bar,
        bar,
    )


@dataclass(frozen=True)
class OrtProviderPlan:
    providers: list[Any]
    requested_device: str
    available_providers: list[str]
    used_trt: bool
    trt_disabled_reason: str
    trt_cache_dir: str
    trt_engine_cache_dir: str
    trt_timing_cache_dir: str
    trt_ep_context_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def _large_external_data_threshold() -> int:
    raw = os.environ.get("REFLEX_LARGE_EXTERNAL_DATA_BYTES")
    if raw is None:
        return _LARGE_EXTERNAL_DATA_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _LARGE_EXTERNAL_DATA_BYTES


def onnx_external_data_bytes(onnx_path: str | Path | None) -> int:
    """Return sibling external-data bytes for a model path.

    Reflex exports use ``model.onnx.data``. Some conversion paths use ``.bin``.
    This intentionally stays filesystem-based so it can run without loading a
    multi-GB ModelProto into memory.
    """
    if onnx_path is None:
        return 0
    path = Path(onnx_path)
    if not path.exists():
        return 0
    candidates = [
        path.with_suffix(path.suffix + ".data"),
        path.with_suffix(".bin"),
    ]
    total = 0
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            total += candidate.stat().st_size
            seen.add(candidate)
    for sibling in path.parent.glob(f"{path.stem}*.data"):
        if sibling.exists() and sibling not in seen:
            total += sibling.stat().st_size
            seen.add(sibling)
    return total


def onnx_has_large_external_data(onnx_path: str | Path | None) -> bool:
    return onnx_external_data_bytes(onnx_path) >= _large_external_data_threshold()


def make_ort_session_options(onnx_path: str | Path | None = None) -> Any:
    """Return shared ORT session options for Reflex runtime sessions.

    ORT's default warning logs can flood stderr for graphs with repeated
    ScatterND nodes. Keep runtime logs focused on errors unless a user asks for
    more detail with REFLEX_ORT_LOG_SEVERITY=0|1|2.

    Very large external-data graphs, such as pi0/pi0.5 monoliths, can trip ORT
    or TensorRT graph-optimizer paths that materialize an optimized ModelProto
    beyond protobuf's 2GB message limit. For those graphs, default to disabling
    online graph optimizations; users can override with
    REFLEX_ORT_GRAPH_OPT_LEVEL=disable|basic|extended|all.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    raw_level = os.environ.get("REFLEX_ORT_LOG_SEVERITY", "3")
    try:
        opts.log_severity_level = int(raw_level)
    except ValueError:
        opts.log_severity_level = 3

    graph_opt_raw = os.environ.get("REFLEX_ORT_GRAPH_OPT_LEVEL")
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "0": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "1": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "2": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "3": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if graph_opt_raw is not None:
        opts.graph_optimization_level = levels.get(
            graph_opt_raw.strip().lower(),
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
    elif onnx_has_large_external_data(onnx_path):
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        logger.info(
            "Disabled ORT graph optimizations for large external-data ONNX "
            "(external_data=%.1f GB)",
            onnx_external_data_bytes(onnx_path) / 1e9,
        )
    return opts


def onnx_has_scatternd_reduction(onnx_path: str | Path | None) -> bool:
    """Return True for ONNX graphs TensorRT 10.x cannot import efficiently.

    TensorRT 10.5 rejects ScatterND nodes carrying the ONNX ``reduction``
    attribute. ORT can still fall those nodes back to CUDA, but large VLA
    graphs may spend minutes attempting many unsupported TRT subgraphs before
    benching. Preflight lets us choose CUDA EP directly and report that
    honestly instead of timing out during provider initialization.
    """
    if onnx_path is None:
        return False
    path = Path(onnx_path)
    if not path.exists():
        return False
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        logger.debug("TRT ScatterND preflight skipped for %s: %s", path, exc)
        return False
    for node in model.graph.node:
        if node.op_type == "ScatterND" and any(attr.name == "reduction" for attr in node.attribute):
            return True
    return False


def build_ort_provider_plan(
    export_dir: str | Path,
    *,
    device: str = "cuda",
    requested_providers: list[Any] | None = None,
    available_providers: list[str] | None = None,
    max_batch: int = 1,
    prefer_trt: bool = True,
    onnx_path: str | Path | None = None,
    trt_workspace_bytes: int = 4 * 1024 * 1024 * 1024,
    trt_cache_dirname: str = ".ort_trt",
) -> OrtProviderPlan:
    """Build a production ORT provider list.

    Defaults:
      - explicit caller providers always win
      - CUDA requests prefer TensorRT EP when available and batch==1
      - TensorRT EP gets engine + timing caches under the export directory
      - CUDA EP remains second priority for unsupported TRT subgraphs
      - CPU EP remains final fallback

    Env toggles:
      - REFLEX_TRT_EP=0 disables TensorRT preference
      - REFLEX_TRT_TIMING_CACHE=0 disables TRT timing cache
      - REFLEX_TRT_DUMP_EP_CONTEXT=1 asks ORT to dump an EPContext model
    """
    export_path = Path(export_dir)
    available = list(available_providers or get_available_providers())
    available_set = set(available)
    device_norm = (device or "cpu").lower()

    if requested_providers is not None:
        return OrtProviderPlan(
            providers=list(requested_providers),
            requested_device=device_norm,
            available_providers=available,
            used_trt=any(provider_name(p) == "TensorrtExecutionProvider" for p in requested_providers),
            trt_disabled_reason="explicit providers supplied",
            trt_cache_dir="",
            trt_engine_cache_dir="",
            trt_timing_cache_dir="",
            trt_ep_context_path="",
        )

    if device_norm != "cuda":
        return OrtProviderPlan(
            providers=["CPUExecutionProvider"],
            requested_device=device_norm,
            available_providers=available,
            used_trt=False,
            trt_disabled_reason="device is not cuda",
            trt_cache_dir="",
            trt_engine_cache_dir="",
            trt_timing_cache_dir="",
            trt_ep_context_path="",
        )

    trt_cache_dir = export_path / trt_cache_dirname
    engine_cache_dir = trt_cache_dir / "engines"
    timing_cache_dir = trt_cache_dir / "timing"
    ep_context_path = trt_cache_dir / "model_ctx.onnx"

    env_trt_enabled = _as_bool(os.environ.get("REFLEX_TRT_EP"), default=True)
    timing_enabled = _as_bool(os.environ.get("REFLEX_TRT_TIMING_CACHE"), default=True)
    dump_ep_context = _as_bool(os.environ.get("REFLEX_TRT_DUMP_EP_CONTEXT"), default=False)
    blackwell = gpu_is_blackwell()
    scatternd_reduction = onnx_has_scatternd_reduction(onnx_path)
    large_external_data = onnx_has_large_external_data(onnx_path)

    providers: list[Any] = []
    used_trt = False
    disabled_reason = ""
    if not prefer_trt or not env_trt_enabled:
        disabled_reason = "TensorRT disabled by configuration"
    elif "TensorrtExecutionProvider" not in available_set:
        disabled_reason = "TensorrtExecutionProvider unavailable"
    elif max_batch > 1:
        disabled_reason = f"max_batch={max_batch} > 1"
    elif blackwell:
        disabled_reason = "Blackwell GPU detected"
        log_blackwell_trt_warning()
    elif scatternd_reduction:
        disabled_reason = "ScatterND reduction unsupported by TensorRT EP"
    elif large_external_data:
        disabled_reason = (
            "large external-data ONNX; TensorRT EP/ORT graph optimizer can "
            "materialize a >2GB protobuf"
        )
    else:
        engine_cache_dir.mkdir(parents=True, exist_ok=True)
        timing_cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options: dict[str, Any] = {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engine_cache_dir),
            "trt_max_workspace_size": trt_workspace_bytes,
        }
        if timing_enabled:
            trt_options.update(
                {
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(timing_cache_dir),
                }
            )
        if dump_ep_context:
            trt_options.update(
                {
                    "trt_dump_ep_context_model": True,
                    "trt_ep_context_file_path": str(ep_context_path),
                }
            )
        providers.append(("TensorrtExecutionProvider", trt_options))
        used_trt = True

    providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")

    return OrtProviderPlan(
        providers=providers,
        requested_device=device_norm,
        available_providers=available,
        used_trt=used_trt,
        trt_disabled_reason=disabled_reason,
        trt_cache_dir=str(trt_cache_dir) if used_trt else "",
        trt_engine_cache_dir=str(engine_cache_dir) if used_trt else "",
        trt_timing_cache_dir=str(timing_cache_dir) if used_trt and timing_enabled else "",
        trt_ep_context_path=str(ep_context_path) if used_trt and dump_ep_context else "",
    )
