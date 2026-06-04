"""Helper that composes the policy-versioning substrate for 2-policy serve.

Per ADR 2026-04-25-policy-versioning-architecture: this is the load-bearing
glue between the substrate (PolicyRouter + Policy + PolicyCrashTracker +
TwoPolicyDispatcher + memory checks) and the FastAPI server runtime.

Single entry point: `setup_two_policy_serving(export_a, export_b, split,
no_rtc, ...)` returns a `TwoPolicyServingState` with two loaded
TetherServer instances + their PolicyRuntimes + the dispatcher. The
caller (server.create_app in 2-policy mode) wires this into the FastAPI
lifespan + /act handler.

Memory-safety check fires BEFORE either TetherServer is loaded -- per
ADR refuse-to-load: 2 × model_size > 0.7 × total_gpu_bytes -> abort.
This prevents the OOM-at-first-inference failure mode.

Pure factory -- the returned state object holds references; lifecycle
(start/stop) is the caller's responsibility (server.lifespan).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tether.runtime.policy import (
    Policy,
    validate_memory_for_two_policies,
    validate_split_and_no_rtc,
)
from tether.runtime.two_policy_dispatcher import TwoPolicyDispatcher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwoPolicyServingState:
    """Frozen output of setup_two_policy_serving().

    Caller (server.create_app) holds references for the lifespan +
    handler dispatch path. Both servers are LOADED + READY.
    """

    server_a: Any  # TetherServer instance
    server_b: Any  # TetherServer instance
    policy_a: Policy  # bundled metadata (for headers + record-replay)
    policy_b: Policy
    runtime_a: Any | None  # PolicyRuntime for slot a; None when backend lacks run_batch
    runtime_b: Any | None  # PolicyRuntime for slot b
    dispatcher: TwoPolicyDispatcher  # /act dispatch target
    split_a_percent: int
    no_rtc_enforced: bool


async def setup_two_policy_serving(
    *,
    export_a: str | Path,
    export_b: str | Path,
    split_a_percent: int = 50,
    no_rtc: bool = True,
    crash_threshold: int = 5,
    server_factory: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    skip_memory_check: bool = False,
    memory_safety_factor: float = 0.7,
    **server_kwargs: Any,
) -> TwoPolicyServingState:
    """Compose the 2-policy serving stack: load 2 TetherServers, build
    PolicyRuntimes per server, build TwoPolicyDispatcher.

    Args:
        export_a / export_b: paths to the two model exports.
        split_a_percent: % of episodes routed to slot A in [0, 100].
            Default 50 for clean A/B.
        no_rtc: must be True in 2-policy mode (per ADR; cross-policy
            RTC carry-over produces OOD actions). Validated up-front.
        crash_threshold: per-slot consecutive-crash counter threshold.
            Default 5 matches single-policy convention.
        server_factory: callable that builds a TetherServer from an
            export_dir + the **server_kwargs. None -> imports
            tether.runtime.server.TetherServer. Tests stub.
        runtime_factory: callable that builds + starts a PolicyRuntime
            wrapping a server. None -> imports + uses default.
            Tests stub.
        skip_memory_check: when True, bypass the
            validate_memory_for_two_policies check. Use only when
            you've verified VRAM independently (e.g., on a CPU-only
            host where the GPU probe returns 0).
        memory_safety_factor: passed through to
            validate_memory_for_two_policies. Default 0.7 leaves 30%
            for cuDNN workspace, IO buffers, OS.
        **server_kwargs: forwarded to both server_factory calls
            (device, providers, safety_config, etc.).

    Returns:
        TwoPolicyServingState with both servers loaded + dispatcher
        wired. Caller is responsible for lifecycle (lifespan
        start/stop of the runtimes; TetherServer cleanup).

    Raises:
        ValueError: invalid args (split out of bounds, no_rtc=False).
        FileNotFoundError: either export_dir doesn't exist.
        Whatever the underlying TetherServer.load() raises.
    """
    # ---- Validate flag combo (mirrors the CLI's Day 5 check) ----
    validate_split_and_no_rtc(
        split_a_percent=split_a_percent, no_rtc=no_rtc,
    )

    # ---- Validate paths exist BEFORE any compute ----
    export_a_path = Path(export_a)
    export_b_path = Path(export_b)
    if not export_a_path.exists():
        raise FileNotFoundError(f"--policy-a export not found: {export_a}")
    if not export_b_path.exists():
        raise FileNotFoundError(f"--policy-b export not found: {export_b}")

    # ---- Memory refuse-to-load check (probe + validate) ----
    if not skip_memory_check:
        try:
            model_size = _estimate_export_size_bytes(export_a_path)
            total_gpu = _probe_total_gpu_bytes()
            if model_size > 0 and total_gpu > 0:
                validate_memory_for_two_policies(
                    model_size_bytes=model_size,
                    total_gpu_bytes=total_gpu,
                    safety_factor=memory_safety_factor,
                )
        except ValueError:
            # Memory check FAILED -- propagate loudly. NOT silently degrade.
            raise
        except Exception as exc:  # noqa: BLE001
            # Probe couldn't run -- log + proceed (we can't validate
            # but we shouldn't block the user; their hardware may not
            # respond to the probe but still have the VRAM).
            logger.warning(
                "two_policy.memory_check_skipped reason=%s -- "
                "proceeding without 2x VRAM validation",
                exc,
            )

    # ---- Build servers via factory (tests stub; production uses TetherServer) ----
    if server_factory is None:
        from tether.runtime.server import TetherServer
        def _default_factory(export_dir: str, **kwargs):
            srv = TetherServer(export_dir=export_dir, **kwargs)
            srv.load()
            return srv
        server_factory = _default_factory

    logger.info("two_policy.loading_server_a export=%s", export_a)
    server_a = server_factory(export_dir=str(export_a_path), **server_kwargs)
    logger.info("two_policy.loading_server_b export=%s", export_b)
    server_b = server_factory(export_dir=str(export_b_path), **server_kwargs)

    # ---- Build per-policy runtimes (one queue/scheduler per policy) ----
    # runtime_factory may be sync OR async. Production wires PolicyRuntime
    # which needs `await runtime.start()`; tests use sync stubs. Detect
    # the return value type and await if it's a coroutine.
    import inspect as _inspect
    runtime_a = None
    runtime_b = None
    if runtime_factory is not None:
        _ra = runtime_factory(server=server_a, slot="a")
        runtime_a = await _ra if _inspect.iscoroutine(_ra) else _ra
        _rb = runtime_factory(server=server_b, slot="b")
        runtime_b = await _rb if _inspect.iscoroutine(_rb) else _rb

    # ---- Compose Policy bundles for headers + record-replay ----
    policy_a = Policy(
        slot="a",
        model_id=getattr(server_a, "model_id", None) or export_a_path.name,
        model_hash=_safe_hash(server_a, export_a_path),
        export_dir=str(export_a_path),
        runtime=runtime_a,
        action_guard=getattr(server_a, "_action_guard", None),
        rtc_adapter=None,  # forced None in 2-policy mode (no_rtc=True)
    )
    policy_b = Policy(
        slot="b",
        model_id=getattr(server_b, "model_id", None) or export_b_path.name,
        model_hash=_safe_hash(server_b, export_b_path),
        export_dir=str(export_b_path),
        runtime=runtime_b,
        action_guard=getattr(server_b, "_action_guard", None),
        rtc_adapter=None,
    )

    # ---- Build the dispatcher ----
    # When a per-slot PolicyRuntime is provided, route /act through its
    # bounded queue + cost-budget scheduler (chunk-budget-batching
    # benefit per ADR 2026-04-24-chunk-budget-batching-architecture
    # decision: per-policy queues land in the same refactor as
    # policy-versioning). When runtime is None, fall back to the direct
    # async predict path (correctness preserved; no batching benefit).
    async def _predict_a(request):
        if runtime_a is not None:
            return await runtime_a.submit(request)
        return await server_a.predict_from_base64_async(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )

    async def _predict_b(request):
        if runtime_b is not None:
            return await runtime_b.submit(request)
        return await server_b.predict_from_base64_async(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )

    dispatcher = TwoPolicyDispatcher(
        policy_a=policy_a, policy_b=policy_b,
        predict_a=_predict_a, predict_b=_predict_b,
        split_a_percent=split_a_percent,
        crash_threshold=crash_threshold,
    )

    logger.info(
        "two_policy.ready slot_a=%s slot_b=%s split_a_percent=%d "
        "crash_threshold=%d",
        policy_a.model_version, policy_b.model_version,
        split_a_percent, crash_threshold,
    )

    return TwoPolicyServingState(
        server_a=server_a, server_b=server_b,
        policy_a=policy_a, policy_b=policy_b,
        runtime_a=runtime_a, runtime_b=runtime_b,
        dispatcher=dispatcher,
        split_a_percent=split_a_percent,
        no_rtc_enforced=True,
    )


def _safe_hash(server: Any, export_path: Path) -> str:
    """Pull the model hash from the server (preferred) OR derive a
    16-hex placeholder from the export dir name. Used for the
    X-Tether-Model-Version header + record-replay correlation."""
    h = getattr(server, "_model_hash", None)
    if isinstance(h, str) and h:
        return h
    # Stable per-export placeholder (sha256 of the dir name, first 16
    # hex chars). Not the actual weights hash; just unique per export.
    import hashlib
    return hashlib.sha256(str(export_path).encode()).hexdigest()[:16]


def _estimate_export_size_bytes(export_dir: Path) -> int:
    """Estimate VRAM footprint from the on-disk export size.

    Sums the size of all *.onnx + *.safetensors + *.bin files in the
    export. This OVER-estimates a bit (FP32 weights load to FP16 in
    ORT) but the over-estimate is the safe direction for the
    refuse-to-load check.

    Returns 0 when the directory has no recognized weight files.
    """
    if not export_dir.exists():
        return 0
    total = 0
    for pattern in ("*.onnx", "*.onnx.data", "*.safetensors", "*.bin"):
        for p in export_dir.glob(pattern):
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _probe_total_gpu_bytes() -> int:
    """Probe total VRAM in bytes. Returns 0 when no probe path
    succeeds (CPU-only host, missing nvidia-smi, etc.).

    Tries 2 sources: torch.cuda.get_device_properties (most reliable
    when torch is installed), then nvidia-smi --query-gpu=memory.total.
    """
    # 1. torch path (preferred -- doesn't shell out)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory)
    except Exception:  # noqa: BLE001
        pass

    # 2. nvidia-smi fallback
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0])
            return mb * 1024 * 1024
    except Exception:  # noqa: BLE001
        pass

    return 0


__all__ = [
    "TwoPolicyServingState",
    "setup_two_policy_serving",
]
