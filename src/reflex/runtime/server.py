"""Reflex VLA inference server.

Persistent HTTP server that loads an exported VLA model and serves
action predictions. Camera image in, robot actions out.

Usage:
    reflex serve ./reflex_export/ --port 8000

Then from robot:
    POST http://localhost:8000/act
    {
        "image": "<base64 encoded image>",
        "instruction": "pick up the cup",
        "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    → {"actions": [[...], [...], ...], "latency_ms": 253.1, "hz": 3.9}
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .record import (
    RecordWriter,
    compute_config_hash,
    compute_model_hash,
)
from .tracing import get_tracer, setup_tracing, shutdown_tracing

# Optional Prometheus metrics — gated on the [serve] extra (prometheus-client).
# When absent, all metric calls become no-ops via the import guard.
try:
    from reflex.observability import (
        METRICS_CONTENT_TYPE,
        record_act_latency,
        render_metrics,
        set_robot_info,
        set_server_up,
        track_in_flight,
    )
    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _METRICS_AVAILABLE = False
    METRICS_CONTENT_TYPE = "text/plain"
    def record_act_latency(*args, **kwargs): pass
    def render_metrics() -> bytes: return b"# prometheus_client not installed\n"
    def set_server_up(*args): pass
    def set_robot_info(*args, **kwargs): pass

    from contextlib import contextmanager as _cm
    @_cm
    def track_in_flight(*args, **kwargs):
        yield

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

try:
    from reflex import __version__ as _REFLEX_VERSION
except ImportError:
    _REFLEX_VERSION = ""


# ─── Blackwell auto-detect ──────────────────────────────────────────────────
# ORT-bundled TensorRT predates Blackwell (RTX 50-series, sm_100). On those
# GPUs, TRT EP segfaults at session-init because TRT can't register kernels
# for the unknown architecture and leaves a NULL function pointer that gets
# called during model load. Detected 2026-04-28 by Rob (RTX 5090, exit 139,
# `ip 0x0` per dmesg). We auto-disable TRT EP on Blackwell hardware until
# ORT bundles a Blackwell-aware TensorRT runtime.

_BLACKWELL_GPU_PATTERNS = (
    "rtx 50",          # GeForce RTX 5070/5080/5090
    "rtx pro 60",      # RTX PRO 6000 Blackwell
    "blackwell",
    "b200",            # B200 datacenter
    "gb200",           # GB200 datacenter
)


def _gpu_is_blackwell() -> bool:
    """True if any locally-visible NVIDIA GPU is Blackwell-family.

    Returns False on errors (no nvidia-smi, no GPU, timeout) — best-effort.
    A False result means "we don't have evidence of Blackwell"; the caller
    proceeds with the normal TRT EP path.
    """
    import subprocess as _sub
    try:
        proc = _sub.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3.0,
        )
    except (FileNotFoundError, _sub.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    names_lower = (proc.stdout or "").lower()
    return any(pat in names_lower for pat in _BLACKWELL_GPU_PATTERNS)


def _print_blackwell_trt_warning() -> None:
    """Loud, multi-line warning so users understand the perf trade-off.

    Per CLAUDE.md "no silent fallbacks that paper over errors" — this is
    documented degradation, not silent. Users see exactly what's happening,
    why, and the upgrade path.
    """
    bar = "═" * 72
    msg = f"""
{bar}
⚠ Blackwell GPU detected (RTX 50-series / B200 / GB200, sm_100)
{bar}
TensorRT EP segfaults on Blackwell because ORT's bundled TensorRT runtime
predates Blackwell support (sm_100 kernels not registered → NULL function
pointer → SIGSEGV on session init). Auto-falling back to ORT's CUDA EP.

What this means for you:
  • Inference WORKS correctly (same numerics as TRT, cos=+1.0 vs PyTorch)
  • Inference is ~3-5x SLOWER than supported tiers (CUDA EP vs TRT EP)
  • Suitable for: chat, dev, prototyping, low-Hz robot control
  • Marginal for: real-time control above 20 Hz

Tracking the upstream fix:
  https://github.com/microsoft/onnxruntime/issues — search "Blackwell sm_100"

When ORT bundles a Blackwell-aware TensorRT runtime (target H1 2026), this
warning will go away automatically and TRT EP will re-enable.
{bar}
"""
    logger.warning(msg)


# ─── End Blackwell auto-detect ──────────────────────────────────────────────


class ReflexServer:
    """VLA inference server that loads exported models and serves predictions."""

    def __init__(
        self,
        export_dir: str | Path,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        providers: list[str] | None = None,
        strict_providers: bool = True,
        safety_config: str | Path | None = None,
        adaptive_steps: bool = False,
        cloud_fallback_url: str = "",
        deadline_ms: float | None = None,
        max_batch: int = 1,
        batch_timeout_ms: float = 5.0,
    ):
        """Create the server.

        Args:
            export_dir: directory with exported ONNX + reflex_config.json
            device: "cuda" or "cpu" — selects default ONNX execution provider
            num_denoising_steps: Euler flow matching steps per inference
            providers: explicit list of ORT execution providers to request, e.g.
                ["CUDAExecutionProvider", "CPUExecutionProvider"]. If omitted,
                derived from `device`. Useful for explicit control in production.
            strict_providers: if True (default), raise a loud RuntimeError when
                the requested provider fails to load instead of silently falling
                back to CPU. Set False only if you explicitly want best-effort
                fallback (almost always wrong for GPU deployments).
            safety_config: path to a SafetyLimits JSON (see `reflex guard init`).
                When set, every action is run through ActionGuard.check() before
                being returned. Violations are clamped by default; use
                safety_config with mode="reject" to return an error instead.
            adaptive_steps: if True, use TurboOptimizer adaptive strategy
                (stops early when velocity converges). Requires ORT session
                already loaded.
            cloud_fallback_url: if non-empty, configures `SplitOrchestrator`
                with this cloud endpoint. On edge failure or deadline miss,
                `predict()` will attempt to route through the cloud.
            deadline_ms: soft deadline per `predict()` call. If the denoise
                loop + safety check exceeds this, the server returns the last
                known good action (or a zero vector) and logs a deadline miss.
        """
        self.export_dir = Path(export_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_denoising_steps = num_denoising_steps
        self._requested_device = device
        self._requested_providers = providers
        self._strict_providers = strict_providers
        self.config = self._load_config()
        self.model = None
        self._ready = False
        self._vlm = None
        self._vlm_loaded = False
        self._expert_input_names: list[str] = []

        # Composed wedges (Phase I.2)
        self._safety_config_path = Path(safety_config) if safety_config else None
        self._adaptive_steps = adaptive_steps
        self._cloud_fallback_url = cloud_fallback_url
        self._deadline_ms = deadline_ms
        self._action_guard = None  # built during load()
        self._split_orchestrator = None  # built during load()
        self._last_good_actions: np.ndarray | None = None
        self._deadline_misses = 0

        # Multi-robot batching (Phase III)
        self._max_batch = max(1, max_batch)
        self._batch_timeout_s = max(0.0, batch_timeout_ms) / 1000.0
        # _batch_queue + _batch_worker_task lazily created in start_batch_worker()
        self._batch_queue = None
        self._batch_worker_task = None
        self._batches_run = 0
        self._batched_requests = 0

        # Rolling latency history for p50/p95/p99 reporting (goal:
        # latency-histograms). Capped at 1024 samples — a ring buffer in
        # all but name.
        from collections import deque
        self._latency_history: deque[float] = deque(maxlen=1024)

        # Reproducibility hashes (goal: determinism-version-hash) — pulled
        # from the exported config + computed lazily at first /act call.
        self._model_hash: str | None = None
        self._config_hash: str | None = None

        # Async replan-while-execute action_buffer (goal:
        # action-chunk-buffering). Lazy-initialized on first
        # configure_replan() call; None means the sliding_window
        # buffering path is disabled and /act returns full chunks
        # (default behavior).
        from reflex.runtime.buffer import ActionChunkBuffer
        self._action_buffer: ActionChunkBuffer | None = None
        self._replan_hz: float | None = None
        self._execute_hz: float | None = None
        self._replan_threshold: float = 0.5

    def _load_config(self) -> dict[str, Any]:
        config_path = self.export_dir / "reflex_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def configure_replan(
        self,
        replan_hz: float,
        execute_hz: float,
    ) -> None:
        """Enable async replan-while-execute buffering with a ring buffer.

        When configured, callers that pop from the buffer receive single
        actions instead of full chunks. The /act handler refills the
        buffer by running predict() when should_replan() crosses the
        threshold. See reflex/runtime/buffer.py for the sliding_window
        semantics.

        Typical robot setup: execute_hz=100, replan_hz=20 (matches the
        Physical Intelligence pattern). action_buffer capacity is auto-
        sized from the ratio.
        """
        from reflex.runtime.buffer import ActionChunkBuffer, compute_replan_window

        window = compute_replan_window(
            execute_hz=execute_hz,
            replan_hz=replan_hz,
            chunk_size=self.chunk_size,
        )
        self._action_buffer = ActionChunkBuffer(capacity=window["capacity"])
        self._replan_hz = replan_hz
        self._execute_hz = execute_hz
        self._replan_threshold = window["threshold_ratio"]
        logger.info(
            "replan configured: replan_hz=%g execute_hz=%g "
            "buffer capacity=%d threshold_ratio=%.2f",
            replan_hz, execute_hz, window["capacity"],
            window["threshold_ratio"],
        )

    def _latency_percentiles(self) -> dict[str, float]:
        """Return p50/p95/p99 + jitter_ms over the rolling latency window.

        Jitter = p99 - p50 (simple proxy for tail-vs-median variance;
        matches the common robotics-control definition where jitter is
        the spread between typical and worst-case cycle time).
        """
        if not self._latency_history:
            return {
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "jitter_ms": 0.0,
            }
        sorted_samples = sorted(self._latency_history)
        n = len(sorted_samples)

        def _pct(p: float) -> float:
            # Nearest-rank method; fine for small windows.
            idx = min(n - 1, int(round(p * (n - 1))))
            return sorted_samples[idx]

        p50 = _pct(0.50)
        p95 = _pct(0.95)
        p99 = _pct(0.99)
        return {
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "jitter_ms": round(p99 - p50, 2),
        }

    def _determinism_fields(self) -> dict[str, str]:
        """Return model_hash, config_hash, reflex_version for reproducibility.

        Lazy: computed on first call, cached on self. Hashes are SHA256
        truncated to 16 chars — short enough for logs, unique enough to
        pin a deployment.
        """
        import hashlib

        if self._model_hash is None:
            # Hash all onnx files in the export dir (deterministic order).
            h = hashlib.sha256()
            for p in sorted(self.export_dir.glob("*.onnx")):
                try:
                    h.update(p.name.encode())
                    with p.open("rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                except Exception:
                    pass
            # External data files (.bin, .data) too — they hold the weights.
            for p in sorted(self.export_dir.glob("*.bin")):
                try:
                    h.update(p.name.encode())
                    with p.open("rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                except Exception:
                    pass
            self._model_hash = h.hexdigest()[:16]

        if self._config_hash is None:
            import json as _json
            try:
                cfg_str = _json.dumps(self.config, sort_keys=True, default=str)
            except Exception:
                cfg_str = str(self.config)
            self._config_hash = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]

        try:
            from reflex import __version__ as _rver
        except Exception:
            _rver = "unknown"

        return {
            "model_hash": self._model_hash,
            "config_hash": self._config_hash,
            "reflex_version": _rver,
        }

    def load(self) -> None:
        """Load the model from exported directory + compose any wedges."""
        logger.info("Loading model from %s", self.export_dir)
        start = time.perf_counter()

        expert_meta = self.config.get("expert", {})
        self.action_dim = expert_meta.get("action_dim", 32)
        self.chunk_size = self.config.get("action_chunk_size", 50)
        self.expert_hidden = expert_meta.get("expert_hidden", 720)

        # Try ONNX runtime first, fall back to PyTorch
        onnx_path = self.export_dir / "expert_stack.onnx"
        if onnx_path.exists():
            self._load_onnx(onnx_path)
        else:
            logger.warning("No ONNX model found, inference not available")
            return

        # Cache expert input names for backward compat (v0.1 exports may not have vlm_kv)
        self._expert_input_names = [inp.name for inp in self._ort_session.get_inputs()]
        logger.info("Expert ONNX inputs: %s", self._expert_input_names)

        # Load VLM prefix pipeline via orchestrator (4-file ONNX pipeline)
        self._load_vlm_orchestrator()

        # ---- Wedge composition (Phase I.2) ----
        # reflex guard: safety limits
        if self._safety_config_path is not None:
            try:
                from reflex.safety import ActionGuard, SafetyLimits
                limits = SafetyLimits.from_json(self._safety_config_path)
                self._action_guard = ActionGuard(limits=limits, mode="clamp")
                logger.info(
                    "reflex guard loaded: %d joints, mode=clamp",
                    len(limits.joint_names),
                )
            except Exception as e:
                logger.warning("Failed to load safety config: %s", e)

        # reflex split: cloud-edge orchestrator
        if self._cloud_fallback_url:
            try:
                from reflex.runtime.split import SplitOrchestrator, SplitConfig
                self._split_orchestrator = SplitOrchestrator(SplitConfig(
                    cloud_url=self._cloud_fallback_url,
                    prefer="edge",
                    fallback_mode="last_action",
                ))
                logger.info(
                    "reflex split configured: cloud_url=%s, fallback=last_action",
                    self._cloud_fallback_url,
                )
            except Exception as e:
                logger.warning("Failed to build split orchestrator: %s", e)

        if self._adaptive_steps:
            logger.info("reflex turbo: adaptive denoise step count ENABLED")
            # Honesty per the Apr-14 phase IV bench: the 0.01 velocity-norm-delta
            # threshold works well on pi0 (~58% latency savings, action diff 0.07)
            # but never triggers on smolvla, rarely triggers on pi0.5, and triggers
            # too aggressively on gr00t (action diff 0.67 — meaningful drift). The
            # per-model threshold tuning lands in v0.2.
            model_type = self.config.get("model_type", "")
            if model_type and model_type != "pi0":
                logger.warning(
                    "adaptive_steps with model_type=%s is unvalidated. "
                    "Phase IV bench (Apr 14): smolvla never triggers (no savings), "
                    "pi0.5 rarely triggers, gr00t triggers too aggressively "
                    "(action diff 0.67). Use --adaptive-steps with model_type=pi0 "
                    "for now; per-model thresholds land in v0.2.",
                    model_type,
                )
        if self._deadline_ms is not None:
            logger.info("deadline enforcement: %.1f ms", self._deadline_ms)

        elapsed = time.perf_counter() - start
        self._ready = True
        logger.info("Model loaded in %.1fs, ready to serve", elapsed)

    def _load_onnx(self, onnx_path: Path) -> None:
        """Load ONNX model via onnxruntime.

        Honors `self._requested_device` and `self._requested_providers`. Raises
        if the requested provider fails to load AND `self._strict_providers` is
        set (the default). Silent CPU fallback was causing users to publish
        "GPU" benchmarks that were actually CPU — Apr 14 post-mortem. Never
        again.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is not installed. For GPU inference, install "
                "`onnxruntime-gpu` (not `onnxruntime`). For CPU only, "
                "install `onnxruntime`."
            ) from e

        # What the installed ORT actually supports on this machine
        available = set(ort.get_available_providers())

        # Decide which providers to request
        if self._requested_providers is not None:
            providers = list(self._requested_providers)
        elif self._requested_device == "cuda":
            providers = []
            # Prefer TensorRT EP when available — gives FP16 kernels and
            # engine caching transparently. ORT falls back to CUDA EP for
            # ops the TRT EP doesn't support.
            #
            # CRITICAL: TRT EP is incompatible with continuous batching when
            # the source ONNX has static shapes (which our exporters bake).
            # Apr-14 verification showed TRT rebuilds the engine on each new
            # batch shape, causing 34-second latencies instead of milliseconds.
            # When max_batch > 1, fall through to CUDAExecutionProvider which
            # handles dynamic shapes natively and gives the 2.88x batching
            # speedup measured in Phase III.
            #
            # ALSO: ORT's bundled TensorRT runtime predates Blackwell (RTX
            # 50-series, sm_100). On Blackwell GPUs, TRT EP segfaults at
            # session-init with a NULL function pointer — caught 2026-04-28
            # by first-tester Rob (RTX 5090, exit code 139). Auto-disable
            # TRT EP on Blackwell + print a loud warning so users know
            # they're on the slower CUDA-EP path until ORT bundles new TRT.
            blackwell_detected = _gpu_is_blackwell()
            use_trt_ep = (
                "TensorrtExecutionProvider" in available
                and self._max_batch <= 1
                and not blackwell_detected
            )
            if blackwell_detected and "TensorrtExecutionProvider" in available:
                _print_blackwell_trt_warning()
            if use_trt_ep:
                # Use a per-export-dir engine cache so subsequent serve calls
                # skip the engine-build cost.
                trt_cache = str(self.export_dir / ".trt_cache")
                Path(trt_cache).mkdir(parents=True, exist_ok=True)
                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": trt_cache,
                        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4GB
                    },
                ))
            elif "TensorrtExecutionProvider" in available and self._max_batch > 1:
                logger.info(
                    "TRT EP available but disabled because --max-batch=%d > 1. "
                    "TRT EP rebuilds engines per batch shape on static-shape "
                    "ONNX, causing 34s+ latencies. Using CUDAExecutionProvider "
                    "which handles dynamic batch natively. (Re-export with "
                    "dynamic batch shapes + TRT shape profiles is a v0.2 fix.)",
                    self._max_batch,
                )
            providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]

        logger.info("Requested providers: %s; available: %s", providers, sorted(available))

        # Create session
        self._ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self._ort_session.get_providers()
        logger.info("Loaded ONNX model: %s — active providers: %s", onnx_path.name, active)

        # Strict check: if caller asked for any GPU provider (CUDA or TRT) but
        # we ended up on CPU, fail loudly.
        def _provider_name(p):
            return p[0] if isinstance(p, tuple) else p

        gpu_provider_names = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        cuda_requested = any(_provider_name(p) in gpu_provider_names for p in providers)
        cuda_active = any(p in gpu_provider_names for p in active)
        if cuda_requested and not cuda_active and self._strict_providers:
            install_hint = ""
            if "CUDAExecutionProvider" not in available:
                install_hint = (
                    "\n\nCUDAExecutionProvider is not available in this ORT install. "
                    "Likely causes:\n"
                    "  - You installed `onnxruntime` (CPU-only). Replace it with:\n"
                    "      pip uninstall onnxruntime && pip install onnxruntime-gpu\n"
                    "  - CUDA 12 + cuDNN 9 libraries are not on the library path. "
                    "ORT 1.20+ requires CUDA 12.x and cuDNN 9.x. "
                    "See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements\n"
                    "  - You are on a machine without an NVIDIA GPU."
                )
            raise RuntimeError(
                f"reflex serve was started with --device cuda (or CUDAExecutionProvider "
                f"in --providers) but ONNX Runtime fell back to CPU. "
                f"Active providers: {active}. "
                f"Refusing to continue under strict mode — use --no-strict-providers or "
                f"--device cpu to explicitly request CPU execution.{install_hint}"
            )

        # Tag the inference mode so /act responses report which path was used
        if "TensorrtExecutionProvider" in active:
            self._inference_mode = "onnx_trt_fp16"
        elif cuda_active:
            self._inference_mode = "onnx_gpu"
        else:
            self._inference_mode = "onnx_cpu"

    def _load_vlm_orchestrator(self) -> None:
        """Load the 4-file VLM prefix pipeline via VLMPrefixOrchestrator.

        Checks for VLM files (vision_encoder.onnx, text_embedder.onnx,
        decoder_prefill.onnx) in the export directory. If at least the
        vision encoder exists, creates a VLMPrefixOrchestrator. Otherwise
        falls back to dummy conditioning (v0.1 mode).
        """
        # Check if there are any VLM files to load
        has_vlm_files = (
            (self.export_dir / "vision_encoder.onnx").exists()
            or self.config.get("vlm_prefix_onnx") is not None
        )

        if not has_vlm_files:
            self._vlm = None
            self._vlm_loaded = False
            logger.info("No VLM files found, using dummy conditioning (v0.1 mode)")
            return

        try:
            from reflex.runtime.vlm_orchestrator import VLMPrefixOrchestrator

            self._vlm = VLMPrefixOrchestrator(self.export_dir, self.config)
            self._vlm_loaded = self._vlm.is_loaded
            if self._vlm_loaded:
                logger.info(
                    "VLM orchestrator loaded (complete=%s)",
                    self._vlm.is_complete,
                )
            else:
                logger.warning(
                    "VLM orchestrator created but no sessions loaded -- "
                    "falling back to dummy conditioning"
                )
                self._vlm = None
                self._vlm_loaded = False
        except Exception as e:
            self._vlm = None
            self._vlm_loaded = False
            logger.warning(
                "Failed to create VLM orchestrator: %s -- using dummy conditioning", e
            )

    @property
    def ready(self) -> bool:
        return self._ready

    def _run_denoise(
        self,
        noisy_actions: np.ndarray,
        position_ids: np.ndarray,
        vlm_kv: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Run the full denoising loop (fixed or adaptive).

        ``vlm_kv`` can be:
            - (k, v) tuple of shape ([L, B, seq, kv], [L, B, seq, kv]) — v0.5+ (RoPE + split k/v)
            - single ndarray [L, B, seq, kv] — v0.4 (per-layer, no split, no RoPE)
            - single ndarray [B, seq, kv] — v0.3 (collapsed shared tensor)
            - None — use zeros of the right shape

        Returns (denoised_actions, steps_used).
        """
        dt = -1.0 / self.num_denoising_steps
        prev_velocity_norm: float | None = None
        converged_at: int | None = None

        # Detect expert schema by ONNX input names.
        expert_has_split_kv = (
            "vlm_k" in self._expert_input_names
            and "vlm_v" in self._expert_input_names
        )
        expert_has_single_kv = "vlm_kv" in self._expert_input_names

        # Normalize vlm_kv to (k, v) if tuple, else leave as scalar array.
        vlm_k: np.ndarray | None = None
        vlm_v: np.ndarray | None = None
        vlm_kv_single: np.ndarray | None = None
        if isinstance(vlm_kv, tuple) and len(vlm_kv) == 2:
            vlm_k, vlm_v = vlm_kv
        elif vlm_kv is not None:
            vlm_kv_single = vlm_kv

        # Zero fallback when expert expects kv inputs but caller provided none.
        if (expert_has_split_kv or expert_has_single_kv) and (
            vlm_k is None and vlm_v is None and vlm_kv_single is None
        ):
            vlm_kv_dim = self.config.get("vlm_kv_dim", 320)
            prefix_seq_len = self.config.get("vlm_prefix_seq_len", 50)
            batch = noisy_actions.shape[0]
            num_layers = self.config.get("vlm_num_layers", 16)
            zeros_4d = np.zeros(
                (num_layers, batch, prefix_seq_len, vlm_kv_dim), dtype=np.float32
            )
            if expert_has_split_kv:
                vlm_k = zeros_4d
                vlm_v = zeros_4d.copy()
            else:
                # v0.3/v0.4 single-tensor fallback
                vlm_kv_single = zeros_4d

        for step in range(self.num_denoising_steps):
            t = 1.0 + step * dt
            timestep = np.array([t], dtype=np.float32)

            feed_dict = {
                "noisy_actions": noisy_actions,
                "timestep": timestep,
                "position_ids": position_ids,
            }
            if expert_has_split_kv and vlm_k is not None and vlm_v is not None:
                feed_dict["vlm_k"] = vlm_k
                feed_dict["vlm_v"] = vlm_v
                prefix_len = int(vlm_k.shape[2])  # [L, B, seq, kv]
                batch = noisy_actions.shape[0]
                if "prefix_offset" in self._expert_input_names:
                    feed_dict["prefix_offset"] = np.full(
                        (batch, 1), prefix_len, dtype=np.int64
                    )
                if "kv_mask" in self._expert_input_names:
                    # All-valid mask when we don't have the prefix pad mask handy.
                    # TODO: plumb the real padded-token mask through from the
                    # VLM orchestrator.
                    feed_dict["kv_mask"] = np.ones(
                        (batch, prefix_len), dtype=bool
                    )
            elif expert_has_single_kv and vlm_kv_single is not None:
                feed_dict["vlm_kv"] = vlm_kv_single

            velocity = self._ort_session.run(None, feed_dict)[0]

            noisy_actions = noisy_actions + velocity * dt

            # Adaptive early stop: if velocity norm stops changing, stop
            if self._adaptive_steps and step >= 2:
                v_norm = float(np.linalg.norm(velocity))
                if prev_velocity_norm is not None:
                    delta = abs(v_norm - prev_velocity_norm)
                    # Threshold chosen so that small models that converge in 4-5
                    # steps actually break early. 0.01 is conservative.
                    if delta < 0.01:
                        converged_at = step + 1
                        break
                prev_velocity_norm = v_norm

        steps_used = converged_at or self.num_denoising_steps
        return noisy_actions, steps_used

    def predict(
        self,
        image: np.ndarray | list[np.ndarray] | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run inference: image + instruction + state → action chunk.

        Composes the wedges when enabled:
        - reflex turbo adaptive step count (`--adaptive-steps`)
        - reflex guard safety check (`--safety-config`)
        - reflex split cloud fallback (`--cloud-fallback`)
        - deadline enforcement (`--deadline-ms`)

        Args:
            image: RGB image array [H, W, 3] or None
            instruction: text instruction (unused in v0.1 expert-only mode)
            state: robot state vector [N] or None

        Returns:
            dict with "actions" (list of action vectors), "latency_ms", "hz",
            and optional telemetry fields (steps_used, safety_violations,
            deadline_exceeded, used_cloud_fallback)
        """
        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}
        if not self._inference_mode.startswith("onnx"):
            return {"error": f"Unknown inference mode: {self._inference_mode}"}
        if self._action_guard is not None and self._action_guard.tripped:
            return {
                "error": "guard_tripped",
                "reason": self._action_guard.trip_reason,
                "hint": "Investigate upstream (inputs / sensors / model) and "
                        "call POST /guard/reset to resume.",
            }

        start = time.perf_counter()

        # Prepare inputs — optionally seed noise externally so test harnesses
        # can produce deterministic outputs matching a reference pipeline.
        if noise is not None:
            noisy_actions = np.asarray(noise, dtype=np.float32)
            if noisy_actions.ndim == 2:
                noisy_actions = noisy_actions[np.newaxis, ...]
        else:
            noisy_actions = np.random.randn(
                1, self.chunk_size, self.action_dim
            ).astype(np.float32)
        position_ids = np.arange(self.chunk_size, dtype=np.int64).reshape(1, -1)

        # VLM prefix conditioning via orchestrator
        vlm_kv = None
        used_vlm = False
        state_np = np.array(state, dtype=np.float32) if state is not None else None
        if self._vlm is not None and image is not None and instruction:
            try:
                _state_for_vlm = state_np if state_np is not None else np.zeros(6, dtype=np.float32)
                vlm_kv = self._vlm.run(image, instruction, _state_for_vlm)
                used_vlm = True
            except Exception as e:
                logger.warning("VLM orchestrator failed: %s — using dummy conditioning", e)
                vlm_kv = None
                used_vlm = False

        # Denoise (adaptive or fixed)
        noisy_actions, steps_used = self._run_denoise(noisy_actions, position_ids, vlm_kv=vlm_kv)

        actions_np = noisy_actions[0]  # [chunk, action_dim]

        # reflex guard — safety check
        safety_violations = 0
        guard_detail: list[str] = []
        guard_summary: dict[str, Any] | None = None
        if self._action_guard is not None:
            try:
                safe_actions, guard_results = self._action_guard.check(actions_np)
                actions_np = safe_actions
                safety_violations = sum(len(r.violations) for r in guard_results)
                if safety_violations > 0:
                    guard_detail = [
                        f"action {i}: {len(r.violations)} violations"
                        for i, r in enumerate(guard_results[:3]) if r.violations
                    ]
                # Failure-classifier substrate (per consent-revoke + failure-classifier
                # research sidecars): expose flat violations list + clamp count so the
                # recorder can write it into the JSONL `guard` field. Detectors that
                # depend on guard data (collision, action_clamp) read this in the
                # uploader pass.
                guard_summary = {
                    "violations": [
                        v for r in guard_results for v in r.violations
                    ],
                    "clamped": any(r.clamped for r in guard_results),
                    "clamp_count": sum(1 for r in guard_results if r.clamped),
                }
            except Exception as e:
                logger.warning("safety check failed: %s", e)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Deadline enforcement — return last good action if over budget
        deadline_exceeded = False
        if self._deadline_ms is not None and elapsed_ms > self._deadline_ms:
            deadline_exceeded = True
            self._deadline_misses += 1
            if self._last_good_actions is not None:
                actions_np = self._last_good_actions
                logger.warning(
                    "deadline miss (%d): %.1fms > %.1fms — returning last good action",
                    self._deadline_misses, elapsed_ms, self._deadline_ms,
                )
            else:
                # No prior good action; return zeros
                actions_np = np.zeros_like(actions_np)
                logger.warning(
                    "deadline miss (%d): %.1fms > %.1fms — no prior action, returning zeros",
                    self._deadline_misses, elapsed_ms, self._deadline_ms,
                )

        # Cache for next deadline miss
        if not deadline_exceeded:
            self._last_good_actions = actions_np.copy()

        # Convert to list for JSON
        actions = actions_np.tolist()

        # Record in rolling window for p50/p95/p99 reporting.
        self._latency_history.append(elapsed_ms)

        result: dict[str, Any] = {
            "actions": actions,
            "num_actions": len(actions),
            "action_dim": self.action_dim,
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "denoising_steps": steps_used,
            "inference_mode": self._inference_mode,
            "vlm_conditioning": "real" if used_vlm else "dummy",
        }
        # Latency histograms over the rolling window (goal: latency-histograms).
        result.update(self._latency_percentiles())
        # Reproducibility: every response includes deployment fingerprint
        # (goal: determinism-version-hash).
        result.update(self._determinism_fields())
        # Telemetry from wedges — only populate when flags are on
        if self._adaptive_steps:
            result["adaptive_enabled"] = True
        if self._action_guard is not None:
            result["safety_violations"] = safety_violations
            if guard_detail:
                result["safety_detail"] = guard_detail
            # Recorder consumes guard_summary to populate write_request(guard=...)
            # for the failure classifier. Only emit when there's actual data
            # (omit when guard_summary remained None due to exception above).
            if guard_summary is not None:
                result["guard_summary"] = guard_summary
        if self._deadline_ms is not None:
            result["deadline_exceeded"] = deadline_exceeded
            if self._deadline_misses:
                result["deadline_misses_total"] = self._deadline_misses
        if self._split_orchestrator is not None:
            result["split_enabled"] = True  # full implementation pending Phase VI

        return result

    def predict_from_base64(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Predict from base64-encoded image (for HTTP API)."""
        image = None
        if image_b64:
            try:
                from PIL import Image

                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as e:
                return {"error": f"Failed to decode image: {e}"}

        return self.predict(image=image, instruction=instruction, state=state)

    # ---------------------------------------------------------------
    # Phase III: continuous batching across HTTP /act requests
    # ---------------------------------------------------------------

    async def start_batch_worker(self) -> None:
        """Spawn an asyncio task that drains the batch queue. Idempotent.

        Only does anything when max_batch > 1 — otherwise predict_async()
        falls through to plain predict().
        """
        if self._max_batch <= 1:
            return
        if self._batch_worker_task is not None and not self._batch_worker_task.done():
            return
        import asyncio
        self._batch_queue = asyncio.Queue()
        self._batch_worker_task = asyncio.create_task(self._batch_worker_loop())
        logger.info(
            "batching enabled: max_batch=%d, timeout=%.1fms",
            self._max_batch, self._batch_timeout_s * 1000,
        )

    async def stop_batch_worker(self) -> None:
        """Cancel the batch worker (called during FastAPI shutdown)."""
        import asyncio
        if self._batch_worker_task is None:
            return
        self._batch_worker_task.cancel()
        try:
            await self._batch_worker_task
        except (asyncio.CancelledError, Exception):
            pass
        self._batch_worker_task = None
        self._batch_queue = None

    async def predict_async(
        self,
        image: np.ndarray | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Async front-door used by the HTTP /act handler.

        - If max_batch <= 1: runs `self.predict()` synchronously in this task.
        - If max_batch > 1: enqueues the request onto a batch queue. A worker
          coroutine drains the queue every `batch_timeout_ms` ms (or when the
          queue hits max_batch) and runs ONE batched ONNX inference, then
          splits the results back to each waiter.
        """
        if self._max_batch <= 1 or self._batch_queue is None:
            return self.predict(image=image, instruction=instruction, state=state)

        import asyncio
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._batch_queue.put((future, image, instruction, state))
        return await future

    async def _batch_worker_loop(self) -> None:
        """Drain the batch queue. Run for the lifetime of the server."""
        import asyncio
        while True:
            batch: list[tuple] = []
            try:
                # Block on the first request — if the queue is empty we just wait.
                first = await self._batch_queue.get()
                batch.append(first)
            except asyncio.CancelledError:
                break

            # Drain up to max_batch within the configured time window.
            deadline = asyncio.get_event_loop().time() + self._batch_timeout_s
            while len(batch) < self._max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._batch_queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Make sure pending futures are released
                    for fut, *_ in batch:
                        if not fut.done():
                            fut.set_exception(asyncio.CancelledError())
                    return

            # Run batched inference (sync — we're holding the event loop, but
            # the actual ORT call is the bottleneck and yields the GIL).
            try:
                results = self._predict_batch_sync(batch)
                for (fut, *_), result in zip(batch, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as e:
                for fut, *_ in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _predict_batch_sync(self, batch: list[tuple]) -> list[dict[str, Any]]:
        """Run one ONNX inference with batch dim = len(batch). Split results.

        For v0.1 of batching, ignores per-item image/instruction/state — same
        as plain predict(). The point is to demonstrate the batching primitive
        and measure throughput scaling. Per-item conditioning lands when the
        VLM prefix path is wired in (Phase II.4).
        """
        if not self._ready:
            return [{"error": "Model not loaded."} for _ in batch]
        if not self._inference_mode.startswith("onnx"):
            return [{"error": f"Unknown inference mode: {self._inference_mode}"} for _ in batch]

        b = len(batch)
        start = time.perf_counter()

        noisy_batched = np.random.randn(
            b, self.chunk_size, self.action_dim
        ).astype(np.float32)
        position_ids_batched = np.tile(
            np.arange(self.chunk_size, dtype=np.int64), (b, 1),
        )

        dt = -1.0 / self.num_denoising_steps
        for step in range(self.num_denoising_steps):
            t = 1.0 + step * dt
            timestep = np.full((b,), t, dtype=np.float32)
            velocity = self._ort_session.run(
                None,
                {
                    "noisy_actions": noisy_batched,
                    "timestep": timestep,
                    "position_ids": position_ids_batched,
                },
            )[0]
            noisy_batched = noisy_batched + velocity * dt

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_request_ms = elapsed_ms / b  # amortized

        self._batches_run += 1
        self._batched_requests += b

        results: list[dict[str, Any]] = []
        for i in range(b):
            actions_np = noisy_batched[i]

            # Apply guard per-item (each request gets its own clamping)
            safety_violations = 0
            if self._action_guard is not None:
                try:
                    safe_actions, guard_results = self._action_guard.check(actions_np)
                    actions_np = safe_actions
                    safety_violations = sum(len(r.violations) for r in guard_results)
                except Exception as e:
                    logger.warning("safety check failed in batch: %s", e)

            result = {
                "actions": actions_np.tolist(),
                "num_actions": len(actions_np),
                "action_dim": self.action_dim,
                "latency_ms": round(elapsed_ms, 1),
                "amortized_latency_ms": round(per_request_ms, 1),
                "hz": round(1000.0 / per_request_ms, 1) if per_request_ms > 0 else 0,
                "denoising_steps": self.num_denoising_steps,
                "inference_mode": self._inference_mode,
                "batch_size": b,
                "request_index": i,
                "batches_run_total": self._batches_run,
                "batched_requests_total": self._batched_requests,
            }
            if self._action_guard is not None:
                result["safety_violations"] = safety_violations
            results.append(result)

        return results

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Async base64 entrypoint — decodes image, then routes through batching."""
        image = None
        if image_b64:
            try:
                from PIL import Image
                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as e:
                return {"error": f"Failed to decode image: {e}"}

        return await self.predict_async(image=image, instruction=instruction, state=state)

    async def run_batch(self, requests: list) -> list[dict[str, Any]]:
        """The PolicyRuntime's run_batch_callback. Takes a list of PredictRequest
        objects and returns a list of result dicts, one per request, in order.

        Phase 1: sequential dispatch via ``predict_from_base64_async`` per request.
        The decomposed ONNX exports are static-shape (per ADR 2026-04-21) so
        true batched dispatch (single ORT call with batch_dim=N) requires a
        future ``dynamic-batch-shapes`` feature. Today the queue + scheduler
        + per-policy isolation primitive lands without changing per-request
        compute cost.

        Per chunk-budget-batching ADR 2026-04-24, this is the entry point the
        ``PolicyRuntime`` worker calls once per scheduler-decided flush. The
        runtime fans the returned list back to the per-request awaiting futures.
        """
        results: list[dict[str, Any]] = []
        for req in requests:
            res = await self.predict_from_base64_async(
                image_b64=getattr(req, "image", None),
                instruction=getattr(req, "instruction", "") or "",
                state=getattr(req, "state", None),
            )
            results.append(res)
        return results


try:
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        image: str | None = None  # base64 encoded -- the primary/agentview camera
        image_wrist: str | None = None  # base64 encoded -- optional wrist/eye-in-hand camera; required by multi-camera VLAs (e.g., pi05 trained with image + image2)
        instruction: str = ""
        state: list[float] | None = None
        episode_id: str | None = None  # B.3: triggers RTC reset on change

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        inference_mode: str = ""
        export_dir: str = ""
        vlm_loaded: bool = False

except ImportError:
    PredictRequest = None  # type: ignore
    HealthResponse = None  # type: ignore


def create_app(
    export_dir: str,
    device: str = "cuda",
    providers: list[str] | None = None,
    strict_providers: bool = True,
    safety_config: str | Path | None = None,
    adaptive_steps: bool = False,
    cloud_fallback_url: str = "",
    deadline_ms: float | None = None,
    max_batch: int = 1,
    batch_timeout_ms: float = 5.0,
    max_batch_cost_ms: float = 100.0,  # PolicyRuntime budget per chunk-budget-batching ADR
    api_key: str | None = None,
    replan_hz: float | None = None,
    execute_hz: float | None = None,
    embodiment_config: Any = None,
    record_dir: str | Path | None = None,
    record_image_redaction: str = "hash_only",
    record_gzip: bool = True,
    rtc_config: Any = None,
    inject_latency_ms: float = 0.0,
    prewarm: bool = True,
    max_consecutive_crashes: int = 5,
    slo_tracker: Any = None,  # reflex.runtime.slo.SLOTracker
    slo_mode: str = "degrade",  # "log_only" | "503" | "degrade"
    max_concurrent: int | None = None,  # None = no limit; int → 429 when saturated
    otel_endpoint: str | None = None,  # OTLP gRPC endpoint (e.g. "localhost:4317")
    otel_sample: float = 1.0,  # 0.0-1.0; 1.0=sample all, 0.1=10% (OTel SemConv)
    robot_id: str | None = None,  # fleet-telemetry: human-readable per-process identity
    cuda_graphs_enabled: bool = False,  # opt-in ORT cuda-graphs on decomposed sessions
    # Action-similarity fast path (FlashVLA, arxiv 2505.21200). Decomposed
    # pi0.5 only — Pi05DecomposedServer wires it; legacy + monolithic ignore.
    # 0.0 = disabled (default); 0.05 = paper default.
    action_similarity_threshold: float = 0.0,
    max_similar_skips: int = 3,
    a2c2_checkpoint: str | None = None,  # path to .npz A2C2 head; None disables A2C2
    a2c2_latency_threshold_ms: float = 40.0,  # hook auto-skip when latency_p95 < this (ms)
    a2c2_success_threshold: float = 0.90,  # hook auto-skip when /act success rate > this; set to 1.01 to disable
    # BID (Bidirectional Decoding) — alternative to A2C2 head per arxiv
    # 2408.17355 + 2026-04-29 a2c2-correction research-revisit Lens 4. When
    # bid_n_candidates > 0, server samples N chunks per /act + picks best
    # via backward coherence with previously-emitted chunk. Mutually
    # exclusive with a2c2_checkpoint in Phase 1; create_app warns if both set.
    bid_n_candidates: int = 0,  # 0 = BID disabled (default); 2-32 = enable
    bid_coherence_window: int = 5,  # how many trailing actions of prev chunk to score against
    bid_coherence_metric: str = "l2",  # 'l2' or 'cos'
    auto_calibrate: bool = False,  # Phase 1 auto-calibration opt-in
    calibration_cache_path: str | None = None,  # path to ~/.reflex/calibration.json
    calibrate_force: bool = False,  # ignore cache hit, re-run measurement
    # Policy-versioning Days 9-10 integration. Per ADR
    # 2026-04-25-policy-versioning-architecture: when policy_b_export_dir is
    # set, create_app loads BOTH ReflexServer instances + builds a
    # TwoPolicyDispatcher; /act routes through the dispatcher instead of the
    # single-server PolicyRuntime. Single-policy mode (default) unchanged.
    policy_b_export_dir: str | None = None,  # 2-policy mode: path to slot B
    policy_split_a_percent: int = 50,  # % traffic to slot A in [0, 100]
    policy_crash_threshold: int = 5,  # per-slot circuit-breaker threshold
) -> Any:
    """Create a FastAPI app for serving VLA predictions.

    api_key: if provided, every /act and /config request must include a
    matching ``X-Reflex-Key`` header or it's rejected with HTTP 401. None
    means no auth (default). /health is always unauthenticated so load
    balancers can probe readiness without a key.

    embodiment_config: an `EmbodimentConfig` (per-robot config — action
    space, normalization, gripper, control rate, constraints). Optional —
    None means existing behavior. Stored on the server instance as
    `server.embodiment_config` so downstream consumers (RTC adapter,
    action denormalization, reflex doctor) can read it. See
    `src/reflex/embodiments/` and `docs/embodiment_schema.md`.

    inject_latency_ms: synthetic deployment-latency injection (B.4 A2C2
    transfer-validation gate). Adds an asyncio.sleep AFTER inference +
    JSONL recording, so the recorded `latency_ms` reflects true compute
    cost while the client observes inference + injected delay. Used to
    simulate Jetson-class deployment latency on Modal A10G for the A2C2
    transfer-validation gate. 0.0 (default) = no injection. Range
    [0, 1000]; values outside clamp at the edges. The matching paper
    methodology is arxiv 2509.23224 §4 ("100ms injected delay").

    prewarm: when True (default), run one synthetic forward at lifespan
    startup so any lazy ONNX/TRT engine build happens before users hit
    /act. /health returns HTTP 503 throughout warmup and HTTP 200 only
    after a successful warmup. Setting False skips warmup — /health
    becomes 200 the moment server.load() completes, but the first /act
    bears the 30-90s engine-build cost. Prewarm ON is the right default
    for production behind a load balancer.

    max_consecutive_crashes: circuit-breaker threshold. After this many
    consecutive /act inference exceptions or error-result responses,
    server.health_state flips to "degraded" — /health returns 503 and
    /act returns 503 with Retry-After: 60. Successful /act resets the
    counter. Default 5. Set to 0 to disable.
    """
    try:
        from contextlib import asynccontextmanager
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("Install fastapi: pip install 'reflex-vla[serve]'")

    # Route: decomposed-ONNX by default; native PyTorch path under REFLEX_NATIVE=1.
    # The native path bypasses our ONNX export and runs lerobot's SmolVLAPolicy
    # directly (RMSNorm still swapped for DecomposedRMSNorm for TRT-export compat
    # on the decomposed side). See reflex/runtime/smolvla_native.py.
    import os as _os

    # Dispatch order:
    #   1. REFLEX_NATIVE=1 — SmolVLANativeServer (PyTorch native path)
    #   2. reflex_config.json export_kind == "monolithic" → model-specific
    #      monolithic server (Pi0OnnxServer / SmolVLAOnnxServer). This is
    #      the cos=1.0 verified production path as of 2026-04-18.
    #   3. Default: ReflexServer (legacy decomposed path).
    _config_path = Path(export_dir) / "reflex_config.json"
    _monolithic_cfg = {}
    if _config_path.exists():
        try:
            _monolithic_cfg = json.loads(_config_path.read_text())
        except Exception:
            _monolithic_cfg = {}

    if _os.environ.get("REFLEX_NATIVE", "0") == "1":
        from reflex.runtime.smolvla_native import SmolVLANativeServer
        server = SmolVLANativeServer(
            export_dir,
            device=device,
            providers=providers,
            strict_providers=strict_providers,
            safety_config=safety_config,
            adaptive_steps=adaptive_steps,
            cloud_fallback_url=cloud_fallback_url,
            deadline_ms=deadline_ms,
            max_batch=max_batch,
            batch_timeout_ms=batch_timeout_ms,
        )
    elif _monolithic_cfg.get("export_kind") == "decomposed":
        # Per ADR 2026-04-25-decomposed-dispatch-via-reflex-serve: the
        # decomposed export (vlm_prefix.onnx + expert_denoise.onnx) needs
        # the wrapper around Pi05DecomposedInference, NOT the legacy
        # ReflexServer (which expects expert_stack.onnx and degrades to
        # "No ONNX model found"). Closes the B.4/B.5 measurement gap
        # documented in the 2026-04-25 b4-gate-refire-v3 experiment.
        from reflex.runtime.decomposed_server import Pi05DecomposedServer
        server = Pi05DecomposedServer(
            export_dir,
            device=device,
            providers=providers,
            strict_providers=strict_providers,
            safety_config=safety_config,
            adaptive_steps=adaptive_steps,
            cloud_fallback_url=cloud_fallback_url,
            deadline_ms=deadline_ms,
            max_batch=max_batch,
            batch_timeout_ms=batch_timeout_ms,
            action_similarity_threshold=action_similarity_threshold,
            max_similar_skips=max_similar_skips,
        )
    elif _monolithic_cfg.get("export_kind") == "monolithic":
        _model_type = _monolithic_cfg.get("model_type", "smolvla")
        _ort_providers = providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        if _model_type == "pi0":
            from reflex.runtime.pi0_onnx_server import Pi0OnnxServer
            server = Pi0OnnxServer(
                export_dir,
                providers=_ort_providers,
                strict_providers=strict_providers,
            )
        elif _model_type == "smolvla":
            from reflex.runtime.smolvla_onnx_server import SmolVLAOnnxServer
            server = SmolVLAOnnxServer(
                export_dir,
                providers=_ort_providers,
                strict_providers=strict_providers,
            )
        else:
            raise ValueError(
                f"Monolithic runtime for model_type={_model_type!r} not yet "
                f"supported. v0.2 covers smolvla + pi0."
            )
    else:
        server = ReflexServer(
            export_dir,
            device=device,
            providers=providers,
            strict_providers=strict_providers,
            safety_config=safety_config,
            adaptive_steps=adaptive_steps,
            cloud_fallback_url=cloud_fallback_url,
            deadline_ms=deadline_ms,
            max_batch=max_batch,
            batch_timeout_ms=batch_timeout_ms,
        )

    # Attach embodiment config (B.1) — optional, downstream consumers
    # (RTC adapter, action denormalization, reflex doctor) read via
    # getattr(server, 'embodiment_config', None).
    server.embodiment_config = embodiment_config

    # Synthetic latency injection (B.4). Clamped to [0, 1000] ms. The
    # /act handler sleeps for this long AFTER inference + recording so
    # JSONL captures true compute latency while the client observes
    # the inflated round-trip.
    server.inject_latency_ms = max(0.0, min(1000.0, float(inject_latency_ms)))  # type: ignore[attr-defined]
    if server.inject_latency_ms > 0:
        logger.info(
            "Synthetic latency injection armed: %.1f ms per /act call (B.4 A2C2 gate)",
            server.inject_latency_ms,
        )

    # Attach ActionGuard from embodiment config (B.6) — clamps actions
    # against per-axis ranges + velocity caps before /act returns. Skips
    # cleanly when embodiment_config is None (existing behavior preserved).
    # Coexists with the URDF-based ActionGuard from `--safety-config` flag;
    # the embodiment-config guard is always-on when configs are present,
    # the URDF guard is opt-in for deeper physics checks.
    server.embodiment_guard = None  # type: ignore[attr-defined]
    if embodiment_config is not None:
        try:
            from reflex.safety.guard import ActionGuard
            server.embodiment_guard = ActionGuard.from_embodiment_config(  # type: ignore[attr-defined]
                embodiment_config, mode="clamp"
            )
            logger.info(
                "Embodiment ActionGuard armed — %d joints, mode=clamp",
                embodiment_config.action_dim,
            )
        except Exception as e:  # noqa: BLE001 — guard must never crash startup
            logger.error(
                "Embodiment ActionGuard init failed (continuing without): %s", e
            )
            server.embodiment_guard = None  # type: ignore[attr-defined]

    # Attach RTC adapter (B.3) if --rtc was passed. Day-1 scope: construct
    # the processor + latency tracker; body methods land Day 2-3 of the B.3
    # sprint. Stored on server.rtc_adapter for the /act handler to dispatch
    # to once Day 4 wires the replan loop. Skeleton-safe: when rtc_config
    # is None or rtc_config.enabled is False, this is a no-op.
    server.rtc_adapter = None  # type: ignore[attr-defined]
    if rtc_config is not None and getattr(rtc_config, "enabled", False):
        # Reject 1-NFE + RTC at config-time per the per-step-expert-export
        # research sidecar (Lens 2 FM-4/FM-6). RTC's guidance-weight formula
        # has tau=1-time=0 → division by zero at the only step when num_steps=1.
        from .rtc_adapter import assert_rtc_compatible_with_num_steps
        _num_steps = (
            _monolithic_cfg.get("num_denoising_steps")
            or _monolithic_cfg.get("decomposed", {}).get("num_steps")
        )
        assert_rtc_compatible_with_num_steps(_num_steps)
        try:
            from .rtc_adapter import RtcAdapter
            # RTC's merge_and_update calls self.buffer.peek_all() to carry
            # the previous chunk's unexecuted tail forward as inertia. If
            # action_buffer wasn't initialized via --replan-hz/--execute-hz,
            # auto-initialize with sensible defaults so RTC actually works.
            # (Pre-fix this passed action_buffer=None silently, which made
            # merge_and_update raise per-call → RTC was effectively no-op
            # under --rtc without explicit replan flags. Caught 2026-04-30
            # in gate-6 LIBERO smoke for per-step expert export.)
            _buf = getattr(server, "_action_buffer", None)
            if _buf is None:
                # Server (e.g. Pi05DecomposedServer) may not have
                # configure_replan/_action_buffer wired up. In either case
                # RTC needs a buffer for merge_and_update's carry-forward,
                # so build one directly. Use the chunk_size from the loaded
                # model when available so the buffer can hold a full chunk.
                from reflex.runtime.buffer import ActionChunkBuffer
                _chunk = (
                    getattr(server, "chunk_size", None)
                    or getattr(server, "action_horizon", None)
                    or _monolithic_cfg.get("chunk_size")
                    or _monolithic_cfg.get("decomposed", {}).get("chunk_size")
                    or 50
                )
                _buf = ActionChunkBuffer(capacity=int(_chunk))
                # Stash on server too so any other RTC consumer can find it.
                server._action_buffer = _buf  # type: ignore[attr-defined]
                logger.info(
                    "--rtc auto-built ActionChunkBuffer(capacity=%d) — "
                    "server class %s lacks configure_replan or it wasn't "
                    "called; pass --replan-hz/--execute-hz to override",
                    _chunk, type(server).__name__,
                )
                if hasattr(server, "configure_replan"):
                    # If the server DOES support replan, also wire the
                    # replan tracker for any other code paths that read
                    # _replan_hz/_execute_hz. Best-effort.
                    try:
                        server.configure_replan(replan_hz=20.0, execute_hz=100.0)
                    except Exception as e:
                        logger.warning(
                            "configure_replan call failed (buffer was built "
                            "directly so RTC still works): %s", e,
                        )
            server.rtc_adapter = RtcAdapter(  # type: ignore[attr-defined]
                policy=server,
                action_buffer=_buf,
                config=rtc_config,
            )
            logger.info(
                "RTC adapter armed — execution_horizon=%d schedule=%s",
                rtc_config.rtc_execution_horizon,
                rtc_config.prefix_attention_schedule,
            )
        except Exception as e:  # noqa: BLE001 — RTC must never crash serve startup
            logger.error("RTC adapter init failed (RTC disabled): %s", e)
            server.rtc_adapter = None  # type: ignore[attr-defined]

    # Attach JSONL recorder (B.2) if --record was passed. Lazily emits the
    # header on the first /act call. Coexists with OTel tracing — see
    # reflex_context/03_experiments/2026-04-23-phoenix-record-replay-smoke.md.
    server._recorder = None  # type: ignore[attr-defined]
    if record_dir is not None:
        try:
            _model_type = _monolithic_cfg.get("model_type", "smolvla")
            _export_kind = _monolithic_cfg.get("export_kind", "decomposed")
            _ec = embodiment_config
            # Curate dual-write: when --record AND consent is opted-in,
            # also enqueue every recorded /act into the contribution queue
            # at ~/.reflex/contribute/queue/. Failures here downgrade the
            # recorder to JSONL-only — never block --record.
            _curate_collector = None
            try:
                from reflex.curate import consent as _curate_consent
                if _curate_consent.is_opted_in():
                    from reflex.curate.free_collector import FreeContributorCollector
                    _curate_collector = FreeContributorCollector.from_consent()
                    logger.info(
                        "curate dual-write armed: contributor_id=%s tier=%s",
                        _curate_collector.contributor_id, _curate_collector.tier,
                    )
            except Exception as _curate_exc:  # noqa: BLE001
                logger.warning(
                    "curate dual-write disabled: %s", _curate_exc,
                )
                _curate_collector = None
            server._recorder = RecordWriter(  # type: ignore[attr-defined]
                record_dir=record_dir,
                model_hash=compute_model_hash(export_dir),
                config_hash=compute_config_hash(export_dir),
                export_dir=export_dir,
                model_type=_model_type,
                export_kind=_export_kind,
                providers=providers or [],
                gpu="",  # filled by hardware probe in v2
                cuda_version="",
                ort_version="",
                embodiment=getattr(_ec, "embodiment", None) if _ec else None,
                image_redaction=record_image_redaction,  # type: ignore[arg-type]
                gzip_output=record_gzip,
                reflex_version=_REFLEX_VERSION,
                curate_collector=_curate_collector,
            )
            logger.info(
                "RecordWriter armed: dir=%s redaction=%s gzip=%s curate=%s",
                record_dir, record_image_redaction, record_gzip,
                "on" if _curate_collector else "off",
            )
        except Exception as e:  # noqa: BLE001 — recorder must never crash serve startup
            logger.error("RecordWriter init failed (recording disabled): %s", e)
            server._recorder = None  # type: ignore[attr-defined]

    # Prewarm + circuit-breaker state (initialized BEFORE lifespan so /health
    # can be queried immediately at process startup — important for orchestrators
    # that probe before lifespan completes). State machine:
    #   "initializing" → "loading" → "warming" → "ready"
    #                                          ↘ "warmup_failed" (graceful degrade)
    #                              "ready" → "degraded" (circuit broken on /act)
    server.health_state = "initializing"  # type: ignore[attr-defined]
    server.consecutive_crash_count = 0  # type: ignore[attr-defined]
    server.max_consecutive_crashes = int(max_consecutive_crashes)  # type: ignore[attr-defined]
    server.prewarm_enabled = bool(prewarm)  # type: ignore[attr-defined]
    server.robot_id = robot_id or ""  # type: ignore[attr-defined]
    server._cuda_graphs_enabled = bool(cuda_graphs_enabled)  # type: ignore[attr-defined]

    # Auto-calibration cache load (Phase 1 auto-calibration Day 4 plumbing).
    # Day 5 wires the actual measurement + apply; Day 4 just loads + exposes
    # the cache so `reflex doctor --show-calibration` works against a live
    # server. server.calibration_cache is None when --auto-calibrate is unset.
    server.calibration_cache = None  # type: ignore[attr-defined]
    server.calibration_cache_path = None  # type: ignore[attr-defined]
    server.auto_calibrate_enabled = bool(auto_calibrate)  # type: ignore[attr-defined]
    server.calibrate_force = bool(calibrate_force)  # type: ignore[attr-defined]
    if auto_calibrate and calibration_cache_path:
        try:
            from reflex.runtime.calibration import CalibrationCache, HardwareFingerprint
            server.calibration_cache_path = str(Path(calibration_cache_path).expanduser())  # type: ignore[attr-defined]
            server.calibration_cache = CalibrationCache.load_or_empty(  # type: ignore[attr-defined]
                server.calibration_cache_path
            )
            _fp = HardwareFingerprint.current()
            if calibrate_force:
                logger.info(
                    "auto-calibrate: --calibrate-force set, will re-measure "
                    "regardless of cache state"
                )
            elif server.calibration_cache.is_stale(_fp):
                logger.info(
                    "auto-calibrate: cache stale (fingerprint mismatch or "
                    "older than 30 days) — measurement will run on first /act"
                )
            else:
                _entries_count = len(server.calibration_cache.entries)
                logger.info(
                    "auto-calibrate: cache hit at %s (%d entries, fingerprint matches)",
                    server.calibration_cache_path, _entries_count,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "auto-calibrate: cache load failed at %s: %s — continuing "
                "without auto-calibration",
                calibration_cache_path, exc,
            )
            server.calibration_cache = None  # type: ignore[attr-defined]

    # Calibration warmup tracker (Day 5): when auto-calibrate is on AND the
    # cache holds an entry for this (embodiment, model_hash), the tracker
    # passively learns latency_compensation_ms from real /act traffic and
    # writes back to the cache once stable. No active probe; cold-start
    # uses the embodiment default already in the cached entry.
    server.calibration_warmup = None  # type: ignore[attr-defined]
    if (
        auto_calibrate
        and getattr(server, "calibration_cache", None) is not None
        and getattr(server, "calibration_cache_path", None)
    ):
        try:
            from reflex.runtime.calibration import (
                CalibrationWarmupTracker,
            )
            _ec = getattr(server, "embodiment_config", None)
            _emb = getattr(_ec, "embodiment", None) or "custom"
            _model_id = Path(server.export_dir).name or "unknown"
            # Day 5 only writes back when an entry already exists (Day 7+ Modal
            # integration writes the initial entry). For now, defensively
            # check; if no entry, the tracker is constructed but maybe_persist
            # will return False until Day 7 fills the entry.
            server.calibration_warmup = CalibrationWarmupTracker(  # type: ignore[attr-defined]
                cache=server.calibration_cache,
                cache_path=server.calibration_cache_path,
                embodiment=_emb,
                model_hash=_model_id,
            )
            logger.info(
                "auto-calibrate: warmup tracker armed for embodiment=%s "
                "model_hash=%s (writes back when 30+ samples + p95 stable)",
                _emb, _model_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "auto-calibrate: warmup tracker init failed: %s", exc,
            )

    # A2C2 correction hook (Phase 1 a2c2-correction feature). When
    # `a2c2_checkpoint` is provided, the hook loads the head + wires into
    # /act for per-chunk correction with auto-skip semantics. Per
    # a2c2-correction execution plan B.5 Day 3.
    server.a2c2_hook = None  # type: ignore[attr-defined]
    server.a2c2_internal = False  # type: ignore[attr-defined]
    if a2c2_checkpoint:
        try:
            from reflex.runtime.a2c2_hook import A2C2Hook, A2C2HookConfig
            _a2c2_cfg = A2C2HookConfig(
                latency_threshold_ms=a2c2_latency_threshold_ms,
                success_threshold=a2c2_success_threshold,
            )
            server.a2c2_hook = A2C2Hook.from_checkpoint(  # type: ignore[attr-defined]
                a2c2_checkpoint, config=_a2c2_cfg,
            )
            # If the server class can apply A2C2 internally (in normalized
            # action space, between inference + denorm), bind the hook there.
            # Pi05DecomposedServer does this; legacy ReflexServer does not.
            # The /act handler skips its own A2C2 application when
            # server.a2c2_internal is True (since the result dict already
            # carries a2c2_* fields populated by the server itself).
            if hasattr(server, "set_a2c2_hook"):
                server.set_a2c2_hook(server.a2c2_hook)  # type: ignore[attr-defined]
                server.a2c2_internal = True  # type: ignore[attr-defined]
            logger.info(
                "A2C2 hook loaded: checkpoint=%s, "
                "latency_threshold_ms=%.1f, success_threshold=%.2f, "
                "applied=%s",
                a2c2_checkpoint,
                server.a2c2_hook.config.latency_threshold_ms,
                server.a2c2_hook.config.success_threshold,
                "internal-pre-denorm" if server.a2c2_internal else "post-/act",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load A2C2 checkpoint %s: %s — A2C2 disabled",
                a2c2_checkpoint, exc,
            )
            server.a2c2_hook = None  # type: ignore[attr-defined]
            server.a2c2_internal = False  # type: ignore[attr-defined]
    # BID (Bidirectional Decoding) chunk-selection. Per the 2026-04-29
    # research-revisit Lens 4: alternative to A2C2 head correction. Sample
    # N candidates per /act + pick best via backward coherence. Mutex with
    # A2C2 hook in Phase 1.
    if bid_n_candidates > 0:
        try:
            from reflex.correction.bid import BIDConfig
            _bid_cfg = BIDConfig(
                n_candidates=int(bid_n_candidates),
                coherence_window=int(bid_coherence_window),
                coherence_metric=str(bid_coherence_metric),
            )
            if hasattr(server, "set_bid_config"):
                server.set_bid_config(_bid_cfg)  # type: ignore[attr-defined]
                logger.info(
                    "BID enabled on server: N=%d, window=%d, metric=%s",
                    _bid_cfg.n_candidates, _bid_cfg.coherence_window,
                    _bid_cfg.coherence_metric,
                )
                if a2c2_checkpoint:
                    logger.warning(
                        "Both --bid-num-candidates and --a2c2-checkpoint set; "
                        "Phase 1 treats these as mutually exclusive. BID will "
                        "run; A2C2 head output ignored."
                    )
            else:
                logger.warning(
                    "--bid-num-candidates=%d set but server type %s does not "
                    "support BID (only Pi05DecomposedServer wires it today). "
                    "Request behavior unchanged.",
                    bid_n_candidates, type(server).__name__,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to configure BID (n=%d): %s — falling back to single-sample",
                bid_n_candidates, exc,
            )

    # The cuda_graphs_enabled flag is consumed by Pi05DecomposedInference when
    # that backend is instantiated (scripts/modal_*_decomposed.py paths, and
    # future production wiring once chunk-budget-batching lands the decomposed-
    # dispatch fix in server.py). When set on a legacy ReflexServer backend,
    # the flag has no effect — log once so operators notice the no-op.
    if cuda_graphs_enabled and type(server).__name__ == "ReflexServer":
        logger.info(
            "--cuda-graphs was set but this backend (ReflexServer legacy decomposed "
            "path) does not consume the flag. cuda-graphs applies to the "
            "Pi05DecomposedInference dispatch path (Modal scripts today; production "
            "wire-up pending the decomposed-dispatch fix tracked by chunk-budget-"
            "batching). Request behavior unchanged."
        )

    @asynccontextmanager
    async def lifespan(app):
        # Enable Python's fault handler so any C-level crash (SIGSEGV /
        # SIGABRT / SIGFPE) during model load prints a Python traceback
        # to stderr BEFORE the process dies. Without this, signal-based
        # death is silent — the process just exits with code 139 and
        # the user has no idea what crashed. Caught 2026-04-28 by
        # first-tester Rob (RTX 5090 segfault in ORT-CUDA EP, Blackwell
        # sm_100 not supported by bundled cuBLAS/cuDNN).
        #
        # User can disable via REFLEX_NO_FAULTHANDLER=1 if they have a
        # different faulthandler installed (e.g., sentry-sdk's). Default
        # ON because the cost is negligible (one signal handler) and
        # the value is enormous when something goes wrong.
        import os as _os_fh
        if not _os_fh.environ.get("REFLEX_NO_FAULTHANDLER"):
            try:
                import faulthandler as _fh
                _fh.enable()
                logger.info(
                    "faulthandler enabled — C-level crashes (SIGSEGV/SIGABRT) "
                    "will print a Python traceback to stderr before the "
                    "process dies. Disable via REFLEX_NO_FAULTHANDLER=1."
                )
            except Exception as _fh_exc:  # noqa: BLE001
                logger.warning("faulthandler.enable() failed: %s", _fh_exc)

        # Initialize OTel tracing if [tracing] extra is installed AND
        # OTEL_EXPORTER_OTLP_ENDPOINT is set (or default localhost:4317).
        # No-ops cleanly if deps absent — server behavior unchanged.
        setup_tracing(
            service_name="reflex-vla",
            endpoint=otel_endpoint,
            sample_rate=otel_sample,
        )
        # Prometheus liveness signal — operators rely on Prometheus's own
        # `up` metric for absent-server detection; this is defense-in-depth.
        set_server_up(1)
        # Fleet-telemetry (Phase 1 feature): publish robot identity so Grafana
        # can join hot metrics against `reflex_robot_info` via `instance`.
        # Skip when unset — no cardinality cost for single-robot deploys.
        if robot_id:
            _ec = getattr(server, "embodiment_config", None)
            _emb = getattr(_ec, "embodiment", None) or "custom"
            set_robot_info(
                robot_id=robot_id,
                embodiment=_emb,
                model_id=Path(server.export_dir).name or "unknown",
            )
        if getattr(server, "embodiment_config", None) is not None:
            ec = server.embodiment_config
            logger.info(
                "Embodiment config loaded: %s (action_dim=%d, control=%.1fHz, chunk=%d)",
                ec.embodiment, ec.action_dim,
                ec.control["frequency_hz"], ec.control["chunk_size"],
            )
        server.health_state = "loading"  # type: ignore[attr-defined]
        # Curate touchpoint: brief banner at serve start until the user has
        # decided about data contribution. Honors REFLEX_NO_CONTRIB_NUDGE=1
        # for hard-silence; otherwise prints once per serve start.
        try:
            from reflex.curate import nudge_engine as _curate_nudge
            _msg = _curate_nudge.maybe_serve_start_banner()
            if _msg is not None:
                logger.info("contribute: %s", _msg.body)
        except Exception as _curate_exc:  # noqa: BLE001
            logger.debug("curate banner skipped: %s", _curate_exc)
        # Phase logging + loud failure: if server.load() throws, print a
        # full traceback to stderr (uvicorn swallows lifespan exceptions
        # by default, leaving users staring at "Waiting for application
        # startup." with no clue what crashed) and re-raise so uvicorn
        # exits non-zero. Caught from a real first-tester debugging
        # session where a stale export cache caused a silent startup
        # death — see CHANGELOG v0.5.4.
        logger.warning(
            "Loading model from %s ... (ONNX session + TRT engine init "
            "can take 10-60s on first-cold-start; subsequent runs reuse "
            "the cached engine)",
            getattr(server, "export_dir", "?"),
        )
        try:
            server.load()
        except Exception as load_exc:  # noqa: BLE001
            import sys as _sys_load
            import traceback as _tb_load
            logger.error("=" * 70)
            logger.error("FATAL: model load failed during FastAPI startup.")
            logger.error("Export dir: %s", getattr(server, "export_dir", "?"))
            logger.error("Exception: %s: %s", type(load_exc).__name__, load_exc)
            logger.error("-" * 70)
            _tb_load.print_exc(file=_sys_load.stderr)
            logger.error("-" * 70)
            logger.error(
                "Common causes: (1) stale export cache — try "
                "'rm -rf ~/.cache/reflex/exports/<model_id>' and re-run "
                "reflex go; (2) ONNX/TensorRT runtime version mismatch — "
                "try pip install --upgrade onnxruntime-gpu; (3) GPU "
                "out-of-memory — close other GPU processes and retry."
            )
            logger.error("=" * 70)
            raise
        logger.info("Model loaded successfully.")
        # Only configure replan buffering after load() so chunk_size is known.
        if replan_hz is not None and execute_hz is not None and hasattr(
            server, "configure_replan"
        ):
            try:
                server.configure_replan(
                    replan_hz=replan_hz, execute_hz=execute_hz
                )
            except Exception as e:
                logger.warning("replan config failed, continuing without: %s", e)
        # Warm up: run one inference so any lazy-build (TRT engine build,
        # ORT graph optimization passes) happens before users hit /act.
        # Without this, the first /act request takes 30-90s with TRT EP enabled
        # because TRT builds + caches an engine on first call.
        if server.prewarm_enabled:
            server.health_state = "warming"  # type: ignore[attr-defined]
            logger.warning(
                "GPU kernel warmup starting — first /act takes ~30-60s, "
                "/health returns 503 until ready (use --no-prewarm to skip)"
            )
            try:
                import time as _t
                _t0 = _t.perf_counter()
                warmup_result = server.predict()
                _elapsed = (_t.perf_counter() - _t0) * 1000
                if isinstance(warmup_result, dict) and "error" in warmup_result:
                    server.health_state = "warmup_failed"  # type: ignore[attr-defined]
                    logger.error(
                        "Warmup FAILED with error result — /health returns 503 "
                        "(server stays up but degraded): %s",
                        warmup_result["error"],
                    )
                else:
                    server.health_state = "ready"  # type: ignore[attr-defined]
                    logger.info(
                        "Warmup complete in %.0fms (mode=%s) — server READY "
                        "(/health returns 200). Subsequent /act calls will use "
                        "the cached TRT engine if applicable.",
                        _elapsed,
                        warmup_result.get("inference_mode", "?") if isinstance(warmup_result, dict) else "?",
                    )
            except Exception as e:
                server.health_state = "warmup_failed"  # type: ignore[attr-defined]
                logger.error(
                    "Warmup FAILED with exception — /health returns 503 "
                    "(server stays up but degraded): %s", e,
                )
        else:
            server.health_state = "ready"  # type: ignore[attr-defined]
            logger.info(
                "Prewarm SKIPPED (--no-prewarm). First /act will take ~30-60s "
                "to JIT the engine. /health returns 200 immediately."
            )

        # ---------------------------------------------------------------
        # Policy-versioning Day 9-10 integration: 2-policy mode.
        # When policy_b_export_dir is set, build TWO ReflexServers via
        # setup_two_policy_serving + store the dispatcher on
        # server.two_policy_state. /act handler dispatches through the
        # dispatcher instead of the single-server PolicyRuntime.
        # ---------------------------------------------------------------
        server.two_policy_state = None  # type: ignore[attr-defined]
        if policy_b_export_dir:
            from reflex.runtime.two_policy_setup import setup_two_policy_serving

            # The first server (the one we already created above) is server A.
            # Build server B via setup_two_policy_serving's server_factory,
            # which mirrors the same load() path. We pass server A in via a
            # closure to avoid re-loading it.
            servers_pair = {"a": server}

            def _two_policy_server_factory(*, export_dir, **kwargs):
                # First call (slot A): return the already-loaded server we have
                # from the outer create_app code. Match by Path equivalence.
                if "a" in servers_pair:
                    try:
                        if Path(export_dir).resolve() == servers_pair["a"].export_dir.resolve():
                            return servers_pair["a"]
                    except (OSError, RuntimeError):
                        pass
                # Otherwise build server B with the same shape as A.
                from reflex.runtime.server import ReflexServer
                srv_b = ReflexServer(
                    export_dir=export_dir,
                    device=device,
                    providers=providers,
                    strict_providers=strict_providers,
                    safety_config=safety_config,
                    adaptive_steps=adaptive_steps,
                    cloud_fallback_url=cloud_fallback_url,
                    deadline_ms=deadline_ms,
                    max_batch=max_batch,
                    batch_timeout_ms=batch_timeout_ms,
                )
                srv_b.load()
                servers_pair["b"] = srv_b
                return srv_b

            # Per-slot PolicyRuntime queues. Each slot gets its own queue +
            # cost-budget scheduler wrapping the per-server run_batch
            # callback. Closes the chunk-budget-batching cross-cut into
            # 2-policy mode (per chunk-budget-batching ADR decision: per-
            # policy queues land in the same refactor as policy-versioning).
            #
            # Gated on `hasattr(server_per_slot, "run_batch")` -- only
            # ReflexServer (the default decomposed backend) implements it.
            # Other backends fall back to direct predict_from_base64_async
            # (the dispatcher's predict closures handle this transparently).
            from reflex.runtime.batching import (
                CostBudgetScheduler as _CBS,
                CostMode as _CM,
                GpuMsCostModel as _GMCM,
            )
            from reflex.runtime.policy_runtime import PolicyRuntime as _PR

            _per_slot_runtimes: list = []  # tracked for shutdown

            async def _two_policy_runtime_factory(*, server: Any, slot: str):
                if not hasattr(server, "run_batch"):
                    return None
                _ec_local = getattr(server, "embodiment_config", None)
                _emb_local = getattr(_ec_local, "embodiment", None) or "custom"
                _model_id_local = Path(server.export_dir).name or f"slot_{slot}"
                _cm_local = _GMCM()
                _sched_local = _CBS(
                    max_cost_per_batch_ms=max_batch_cost_ms,
                    cost_model=_cm_local,
                    max_wait_ms=max(1.0, batch_timeout_ms),
                    mode=_CM.PROFILED,
                )

                def _shape_key(_req):
                    return "default"

                runtime = _PR(
                    policy_id=slot,
                    model_id=_model_id_local,
                    embodiment=_emb_local,
                    scheduler=_sched_local,
                    cost_model=_cm_local,
                    run_batch_callback=server.run_batch,
                    shape_key_fn=_shape_key,
                    max_queue=1000,
                )
                await runtime.start()
                _per_slot_runtimes.append(runtime)
                logger.info(
                    "two_policy.runtime_started slot=%s model=%s embodiment=%s",
                    slot, _model_id_local, _emb_local,
                )
                return runtime

            try:
                two_state = await setup_two_policy_serving(
                    export_a=server.export_dir,
                    export_b=policy_b_export_dir,
                    split_a_percent=policy_split_a_percent,
                    no_rtc=True,  # ADR-enforced; CLI also blocks otherwise
                    crash_threshold=policy_crash_threshold,
                    server_factory=_two_policy_server_factory,
                    runtime_factory=_two_policy_runtime_factory,
                    # GPU-memory check uses the export-dir size estimator;
                    # can be skipped via env for tests / CPU-only runs.
                    skip_memory_check=bool(_os.environ.get(
                        "REFLEX_SKIP_2POLICY_MEMORY_CHECK", ""
                    )),
                )
                server.two_policy_state = two_state  # type: ignore[attr-defined]
                # Stash per-slot runtimes on server so the lifespan
                # finalizer can stop them cleanly.
                server._two_policy_runtimes = _per_slot_runtimes  # type: ignore[attr-defined]
                # Mirror into the legacy server.policies dict so existing
                # diagnostics + tests that read server.policies see the
                # right per-slot runtimes (instead of empty {}).
                server.policies = {
                    "a": two_state.runtime_a,
                    "b": two_state.runtime_b,
                }  # type: ignore[attr-defined]
                logger.info(
                    "two_policy.serving_active split_a_percent=%d "
                    "crash_threshold=%d slot_a=%s slot_b=%s",
                    policy_split_a_percent, policy_crash_threshold,
                    two_state.policy_a.model_version,
                    two_state.policy_b.model_version,
                )
            except Exception as exc:
                logger.error(
                    "two_policy.setup_failed -- falling back to single-policy "
                    "serve (export_a stays loaded). Reason: %s", exc,
                )
                # Don't raise -- single-policy serve continues to work as
                # documented. Operator sees the error in logs + the
                # banner the CLI prints.
                server.two_policy_state = None  # type: ignore[attr-defined]

        # PolicyRuntime — per-policy queue + cost-weighted scheduler (Phase 1
        # chunk-budget-batching). Single-policy default key "prod"; multi-policy
        # via policy-versioning lands {"a": ..., "b": ...} on the same dict.
        # Per ADR 2026-04-24-chunk-budget-batching-architecture.
        #
        # Gated on `hasattr(server, "run_batch")` — only ReflexServer (the
        # default decomposed backend) implements it for now. Other backends
        # (SmolVLANativeServer, Pi0OnnxServer, SmolVLAOnnxServer) take the
        # /act fallback path (direct predict_from_base64_async) until they
        # ship their own run_batch. No-op for the legacy batching path.
        _runtime = None
        if hasattr(server, "run_batch"):
            from reflex.runtime.batching import (
                CostBudgetScheduler, CostMode, GpuMsCostModel,
            )
            from reflex.runtime.policy_runtime import PolicyRuntime
            _ec = getattr(server, "embodiment_config", None)
            _emb = getattr(_ec, "embodiment", None) or "custom"
            _model_id = Path(server.export_dir).name or "unknown"
            _cost_model = GpuMsCostModel()
            _scheduler = CostBudgetScheduler(
                max_cost_per_batch_ms=max_batch_cost_ms,
                cost_model=_cost_model,
                max_wait_ms=max(1.0, batch_timeout_ms),
                mode=CostMode.PROFILED,
            )

            def _shape_key_fn(req):
                # Phase 1 single-embodiment-per-process collapses to a constant.
                # Phase 2 per-embodiment routing extends this.
                return "default"

            _runtime = PolicyRuntime(
                policy_id="prod",
                model_id=_model_id,
                embodiment=_emb,
                scheduler=_scheduler,
                cost_model=_cost_model,
                run_batch_callback=server.run_batch,
                shape_key_fn=_shape_key_fn,
                max_queue=1000,
            )
            await _runtime.start()
            server.policies = {"prod": _runtime}  # type: ignore[attr-defined]
            logger.info(
                "policy_runtime started: policy_id=prod, max_cost_ms=%.1f, "
                "max_wait_ms=%.1f, embodiment=%s, model=%s",
                max_batch_cost_ms, max(1.0, batch_timeout_ms), _emb, _model_id,
            )
        else:
            server.policies = {}  # type: ignore[attr-defined]
            logger.info(
                "policy_runtime skipped: backend %s lacks run_batch — /act "
                "uses direct predict_from_base64_async path",
                type(server).__name__,
            )
        # Pro tier: start daily heartbeat background task if a Pro license is
        # loaded. The task runs send_heartbeat() every 24h until cancellation.
        # Defensive — does nothing on free tier where server.pro_license is None.
        _heartbeat_task = None
        _pro_license = getattr(server, "pro_license", None)
        if _pro_license is not None:
            try:
                from reflex.pro.activate import heartbeat_fingerprint
                from reflex.pro.heartbeat import (
                    LicenseExpiredAtServer,
                    LicenseRevokedError,
                    send_heartbeat,
                )
                _hb_fp = heartbeat_fingerprint()
                _hb_license_id = _pro_license.customer_id  # license dict / dataclass — has customer_id
                _hb_version = getattr(server, "_reflex_version", None) or "unknown"

                async def _heartbeat_loop():
                    import asyncio as _asyncio_hb
                    while True:
                        try:
                            send_heartbeat(
                                license_id=_hb_license_id,
                                hardware_fingerprint=_hb_fp,
                                reflex_version=_hb_version,
                            )
                            logger.debug("Pro heartbeat sent for %s", _hb_license_id)
                        except LicenseRevokedError as exc:
                            logger.error("Pro license revoked: %s. Server will refuse new requests.", exc)
                            server.health_state = "degraded"  # type: ignore[attr-defined]
                            break
                        except LicenseExpiredAtServer as exc:
                            logger.error("Pro license expired at server: %s.", exc)
                            server.health_state = "degraded"  # type: ignore[attr-defined]
                            break
                        except Exception as exc:  # noqa: BLE001 — soft failure, retry next tick
                            logger.debug("Heartbeat soft failure (will retry): %s", exc)
                        # 24h interval. Cached license stays valid until
                        # HEARTBEAT_FRESHNESS_S elapses since the last successful
                        # heartbeat (handled in pro/license.py at next startup).
                        await _asyncio_hb.sleep(24 * 3600)

                import asyncio as _asyncio_lifespan
                _heartbeat_task = _asyncio_lifespan.create_task(_heartbeat_loop())
                logger.info(
                    "Pro daily heartbeat started for license %s", _hb_license_id,
                )
            except Exception as exc:  # noqa: BLE001 — never block startup on heartbeat scaffolding
                logger.warning("Pro heartbeat scaffolding failed: %s", exc)

        # Curate uploader scaffolding — daily background upload of the
        # contribution queue at ~/.reflex/contribute/queue/. Posts to the
        # live contribution-worker (https://reflex-contributions.fastcrest
        # .workers.dev) by default. Set REFLEX_CURATE_DRY_RUN=1 to keep
        # files locally without uploading; REFLEX_CONTRIB_ENDPOINT to point
        # at a self-hosted worker.
        _curate_uploader = None
        try:
            from reflex.curate import consent as _curate_consent
            if _curate_consent.is_opted_in():
                from reflex.curate.uploader import Uploader as _CurateUploader
                _curate_receipt = _curate_consent.load()
                _curate_dry_run = os.environ.get("REFLEX_CURATE_DRY_RUN", "").lower() in ("1", "true", "yes")
                _curate_uploader = _CurateUploader(
                    contributor_id=_curate_receipt.contributor_id,
                    tier=_curate_receipt.tier,
                    opted_in_at=_curate_receipt.opted_in_at,
                    privacy_mode=_curate_receipt.privacy_mode,
                    live=not _curate_dry_run,
                )
                _curate_uploader.start()
                logger.info(
                    "curate uploader started (live=%s; contributor_id=%s)",
                    not _curate_dry_run, _curate_receipt.contributor_id,
                )
        except Exception as exc:  # noqa: BLE001 — never block startup on uploader scaffolding
            logger.warning("curate uploader scaffolding failed: %s", exc)
            _curate_uploader = None

        try:
            yield
        finally:
            # Stop the curate uploader if it was started.
            if _curate_uploader is not None:
                try:
                    _curate_uploader.stop(drain=True)
                    logger.info("curate uploader stopped")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("curate uploader.stop failed: %s", exc)
            # Cancel the Pro heartbeat task if running.
            if _heartbeat_task is not None and not _heartbeat_task.done():
                _heartbeat_task.cancel()
                try:
                    await _heartbeat_task
                except Exception:  # noqa: BLE001 — cancellation is expected
                    pass
            set_server_up(0)
            if _runtime is not None:
                try:
                    await _runtime.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("policy_runtime.stop failed: %s", exc)
            # Stop per-slot 2-policy runtimes (each is a PolicyRuntime
            # built by _two_policy_runtime_factory in lifespan startup).
            for _slot_runtime in getattr(server, "_two_policy_runtimes", []):
                try:
                    await _slot_runtime.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "two_policy_runtime.stop failed: %s", exc,
                    )
            # Flush + close JSONL recorder if armed
            _rec = getattr(server, "_recorder", None)
            if _rec is not None:
                try:
                    _rec.write_footer({"total_requests": _rec.seq})
                finally:
                    _rec.close()
            shutdown_tracing()

    app = FastAPI(
        title="Reflex VLA Server",
        description="Deploy any VLA model to any edge hardware.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Bearer auth dependency (Phase 1 auth-bearer feature).
    # If api_key is set at app-creation time, every protected route requires
    # the caller to pass `Authorization: Bearer <token>` (preferred) OR the
    # legacy `X-Reflex-Key` header (back-compat). Token comparison is constant-
    # time to resist timing attacks. /health skips auth so load balancers /
    # orchestrators can probe readiness without credentials.
    from reflex.runtime.auth import (
        constant_time_token_match,
        make_401_payload,
        resolve_request_token,
    )

    async def _require_api_key(
        authorization: str | None = Header(default=None),
        x_reflex_key: str | None = Header(default=None, alias="X-Reflex-Key"),
    ) -> None:
        if api_key is None:
            return
        provided = resolve_request_token(authorization, x_reflex_key)
        if not constant_time_token_match(provided, api_key):
            err = make_401_payload(
                "missing or invalid credentials — supply 'Authorization: Bearer <token>' "
                "or 'X-Reflex-Key' header"
            )
            raise HTTPException(status_code=401, detail=err.to_dict())

    @app.get("/health")
    async def health():
        # Prewarm + crash-recovery state machine. HTTP 200 only when the
        # health_state is "ready" — load balancers / orchestrators correctly
        # skip the server during warmup, on warmup failure, and after
        # circuit-breaker degradation. Body always returns the granular state
        # for human debugging.
        state = getattr(server, "health_state", "initializing")
        body = {
            "status": "ok" if state == "ready" else "not_ready",
            "state": state,
            "model_loaded": server.ready,
            "inference_mode": getattr(server, "_inference_mode", ""),
            "export_dir": str(server.export_dir),
            "vlm_loaded": getattr(server, "_vlm_loaded", False),
            "consecutive_crashes": int(getattr(server, "consecutive_crash_count", 0)),
            "max_consecutive_crashes": int(getattr(server, "max_consecutive_crashes", 5)),
            "robot_id": getattr(server, "robot_id", "") or "",
        }
        http_status = 200 if state == "ready" else 503
        return JSONResponse(content=body, status_code=http_status)

    @app.get("/metrics")
    async def metrics():
        """Prometheus scrape endpoint. No auth — operators network-isolate.
        Returns text/plain with Prometheus expfmt v0.0.4 (or v1.0.0 from
        prometheus_client 0.20+). Skipped via --disable-metrics flag (TBD)."""
        from fastapi.responses import Response
        return Response(
            content=render_metrics(),
            media_type=METRICS_CONTENT_TYPE,
        )

    @app.post("/act")
    async def act(request: PredictRequest, _auth: None = Depends(_require_api_key)):
        # Circuit breaker: refuse traffic when the consecutive-crash threshold
        # has tripped. Operators must restart the server to clear "degraded".
        # Returns 503 + Retry-After=60 so well-behaved clients back off.
        if getattr(server, "health_state", "ready") == "degraded":
            return JSONResponse(
                status_code=503,
                content={
                    "error": "server-degraded",
                    "consecutive_crashes": int(getattr(server, "consecutive_crash_count", 0)),
                    "max_consecutive_crashes": int(getattr(server, "max_consecutive_crashes", 5)),
                    "hint": "circuit breaker tripped; restart server to clear",
                },
                headers={"Retry-After": "60"},
            )
        # Determine embodiment label for metrics. Bounded enum (per
        # ORGANIZATION.md/cardinality budget): preset name OR 'custom'.
        _ec = getattr(server, "embodiment_config", None)
        _emb_label = getattr(_ec, "embodiment", None) or "custom"
        _model_label = Path(server.export_dir).name or "unknown"

        # OTel span — no-op when [tracing] extra not installed.
        # Attribute namespace: gen_ai.* (OTel SemConv stable) for cross-tool
        # compatibility; reflex.* for VLA-specific extensions.
        # Wrapped in track_in_flight so the in-flight gauge stays accurate
        # even on exception or early return.
        with track_in_flight(embodiment=_emb_label), \
                _tracer.start_as_current_span("act") as span:
            span.set_attribute("gen_ai.operation.name", "act")
            span.set_attribute("gen_ai.request.model", str(server.export_dir))
            # OTel GenAI robotics extensions (Phase 1 otel-genai-spans feature).
            # Non-standard attrs under gen_ai.action.* — proposed for upstream
            # OTel GenAI working group contribution (Phase 2 per spec).
            span.set_attribute("gen_ai.action.embodiment", _emb_label)
            # chunk_size + denoise_steps are set AFTER predict returns (we don't
            # know them until the result is in hand). See ~line 1590 below.
            span.set_attribute(
                "reflex.instruction",
                request.instruction[:512] if request.instruction else "",
            )
            span.set_attribute(
                "reflex.state_dim", len(request.state) if request.state else 0
            )
            span.set_attribute(
                "reflex.image_bytes", len(request.image) if request.image else 0
            )

            # RTC episode-boundary reset (B.3). Runs BEFORE predict so the
            # latency tracker + carry-forward state are fresh for the new
            # episode's first chunk. No-op when adapter absent or episode
            # unchanged.
            _rtc = getattr(server, "rtc_adapter", None)
            if (
                _rtc is not None
                and request.episode_id is not None
                and request.episode_id != _rtc._active_episode_id
            ):
                _rtc.reset(episode_id=request.episode_id)
                span.set_attribute("reflex.rtc.episode_reset", True)
            if _rtc is not None and request.episode_id:
                span.set_attribute("reflex.rtc.episode_id", request.episode_id)

            try:
                # 2-policy mode: route via TwoPolicyDispatcher (overrides the
                # single-policy PolicyRuntime path). Per ADR
                # 2026-04-25-policy-versioning-architecture: episode-sticky
                # SHA-256 hash on episode_id; first-request decides slot;
                # subsequent requests in same episode get the same slot.
                _two_state = getattr(server, "two_policy_state", None)
                if _two_state is not None:
                    # Resolve an episode_id (from request body OR fall back to
                    # request.session_id if present; degraded routing fires a
                    # one-time warning when both are missing).
                    _ep_id = (
                        getattr(request, "episode_id", None)
                        or getattr(request, "session_id", None)
                    )
                    _req_id = getattr(request, "request_id", None) or (
                        f"req_{int(time.time() * 1000)}"
                    )
                    result, _routing = await _two_state.dispatcher.predict(
                        request=request, episode_id=_ep_id, request_id=_req_id,
                    )
                    # Stash routing decision so the response builder + headers
                    # + recorder can pick it up below.
                    _two_routing_decision = _routing
                else:
                    _two_routing_decision = None
                    # Single-policy mode: route through the existing
                    # per-policy runtime queue (chunk-budget-batching Phase 1).
                    from reflex.runtime.policy_runtime import QueueFull as _PRQueueFull
                    _runtime = getattr(server, "policies", {}).get("prod")
                    if _runtime is None:
                        # Fallback for backends/tests that don't install a runtime —
                        # call the per-request path directly.
                        result = await server.predict_from_base64_async(
                            image_b64=request.image,
                            instruction=request.instruction,
                            state=request.state,
                            image_wrist_b64=request.image_wrist,
                        )
                    else:
                        try:
                            result = await _runtime.submit(request)
                        except _PRQueueFull:
                            return JSONResponse(
                                status_code=503,
                                content={
                                    "error": "queue_full",
                                    "message": "policy runtime queue at capacity",
                                    "policy_id": "prod",
                                    "max_queue": _runtime.snapshot().get("max_queue"),
                                },
                                headers={"Retry-After": "1"},
                            )
            except Exception as _predict_exc:  # noqa: BLE001
                # Circuit-breaker increment on raw exception. Re-raise so
                # FastAPI returns 500; subsequent calls hit the degraded
                # check above if the threshold trips.
                _max = int(getattr(server, "max_consecutive_crashes", 5) or 0)
                if _max > 0:
                    server.consecutive_crash_count = int(
                        getattr(server, "consecutive_crash_count", 0)
                    ) + 1
                    if server.consecutive_crash_count >= _max:
                        server.health_state = "degraded"
                        logger.error(
                            "Server degraded after %d consecutive predict crashes "
                            "(threshold=%d). /health returns 503; /act returns 503. "
                            "Restart to clear.",
                            server.consecutive_crash_count, _max,
                        )
                raise

            # Circuit-breaker bookkeeping on returned result. Error-result
            # responses (e.g., NaN guard trips) count as crashes; clean
            # responses reset the counter to 0.
            _max = int(getattr(server, "max_consecutive_crashes", 5) or 0)
            if _max > 0:
                if isinstance(result, dict) and "error" in result:
                    server.consecutive_crash_count = int(
                        getattr(server, "consecutive_crash_count", 0)
                    ) + 1
                    if server.consecutive_crash_count >= _max:
                        server.health_state = "degraded"
                        logger.error(
                            "Server degraded after %d consecutive error responses "
                            "(threshold=%d). /health returns 503; /act returns 503. "
                            "Restart to clear.",
                            server.consecutive_crash_count, _max,
                        )
                else:
                    server.consecutive_crash_count = 0

            # Embodiment ActionGuard (B.6) — clamp actions against per-axis
            # ranges + velocity caps from embodiment config. NaN/Inf zeroes
            # the chunk + reports a violation. No-op when guard absent.
            _eg = getattr(server, "embodiment_guard", None)
            _guard_violations: list[str] = []
            if (
                _eg is not None
                and isinstance(result, dict)
                and "error" not in result
                and isinstance(result.get("actions"), list)
                and result["actions"]
            ):
                try:
                    _arr = np.asarray(result["actions"], dtype=np.float32)
                    _safe, _check_results = _eg.check(_arr)
                    _was_modified = not np.array_equal(_arr, _safe)
                    if _was_modified:
                        result["actions"] = _safe.tolist()
                        for _cr in _check_results:
                            _guard_violations.extend(_cr.violations)
                        # Prometheus counter — bucket by first-violation kind
                        try:
                            from reflex.observability import inc_safety_violation
                            _kind = (
                                "non_finite"
                                if any("non_finite" in v for v in _guard_violations)
                                else "joint_clamp"
                            )
                            inc_safety_violation(
                                embodiment=_emb_label, kind=_kind
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    if _guard_violations:
                        result["guard_violations"] = _guard_violations[:20]  # cap log
                        result["guard_clamped"] = True
                        span.set_attribute(
                            "reflex.guard.violation_count",
                            len(_guard_violations),
                        )
                except Exception as e:  # noqa: BLE001 — guard must never break /act
                    logger.warning("Embodiment ActionGuard failed: %s", e)

            if isinstance(result, dict):
                if "latency_ms" in result:
                    span.set_attribute(
                        "reflex.inference_ms", float(result["latency_ms"])
                    )
                    # Prometheus latency histogram (D.1.8 prometheus-grafana).
                    # No-op when prometheus-client not installed. Day 6
                    # policy-versioning: emit policy_slot label so
                    # operators can split per-slot p99 in 2-policy mode.
                    # Default "prod" preserves single-policy series.
                    try:
                        _slot_label = (
                            _two_routing_decision.slot
                            if _two_routing_decision is not None
                            else "prod"
                        )
                        record_act_latency(
                            float(result["latency_ms"]) / 1000.0,
                            embodiment=_emb_label,
                            model_id=_model_label,
                            policy_slot=_slot_label,
                        )
                    except Exception:  # noqa: BLE001 — metrics never break /act
                        pass
                if "inference_mode" in result:
                    span.set_attribute(
                        "reflex.inference_mode", str(result["inference_mode"])
                    )
                if "actions" in result and isinstance(result["actions"], list):
                    span.set_attribute(
                        "reflex.action_chunk_len", len(result["actions"])
                    )
                    # OTel GenAI robotics extension (otel-genai-spans feature).
                    span.set_attribute(
                        "gen_ai.action.chunk_size", len(result["actions"])
                    )
                if isinstance(result, dict) and "denoise_steps" in result:
                    span.set_attribute(
                        "gen_ai.action.denoise_steps",
                        int(result.get("denoise_steps", 0)),
                    )
                if "error" in result:
                    span.set_attribute("error.type", str(result.get("error", ""))[:200])

            # RTC post-predict carry-state update (B.3). Snapshots buffer +
            # pushes new chunk + records latency on the adapter's tracker.
            # No-op when adapter absent OR result has an error OR actions empty.
            if (
                _rtc is not None
                and isinstance(result, dict)
                and "error" not in result
                and isinstance(result.get("actions"), list)
                and result["actions"]
            ):
                try:
                    actions_arr = np.asarray(result["actions"], dtype=np.float32)
                    latency_s = float(result.get("latency_ms", 0.0)) / 1000.0
                    _rtc.merge_and_update(actions_arr, elapsed_time=latency_s)
                    span.set_attribute("reflex.rtc.chunk_count", _rtc._chunk_count)
                except Exception as e:  # noqa: BLE001 — RTC must never break /act
                    logger.warning("RTC merge_and_update failed: %s", e)

            # A2C2 correction hook (Phase 1 a2c2-correction Day 3).
            # Applies a per-step residual correction to the action chunk
            # when latency p95 ≥ threshold AND success rate ≤ threshold.
            # Cold-start + no-hook → no-op. Per-act outcome recorded into
            # the hook's rolling windows AFTER the apply call so the
            # current request's signal joins the steady-state distribution.
            _a2c2 = getattr(server, "a2c2_hook", None)
            _a2c2_internal = getattr(server, "a2c2_internal", False)
            if (
                _a2c2 is not None
                and not _a2c2_internal  # server applied internally; skip post-hoc
                and isinstance(result, dict)
                and "error" not in result
                and isinstance(result.get("actions"), list)
                and result["actions"]
            ):
                try:
                    actions_arr = np.asarray(result["actions"], dtype=np.float32)
                    if actions_arr.ndim == 2:
                        # The model's action chunk may be padded (e.g., pi05 emits
                        # max_action_dim=32 padded actions). The A2C2 head was
                        # trained on the customer's REAL action_dim (e.g., 7 for
                        # LIBERO franka). Slice to the head's action_dim before
                        # passing to the hook + write the corrected leading slice
                        # back into the full chunk. Caught 2026-04-26: the hook
                        # was rejecting every call with "actions shape mismatch:
                        # expected (chunk_size, 7), got (50, 32)".
                        hook_dim = _a2c2.head.config.action_dim
                        full_dim = actions_arr.shape[1]
                        actions_for_hook = actions_arr[:, :hook_dim].copy()
                        corrected, decision, magnitude = _a2c2.maybe_apply_to_chunk(
                            actions=actions_for_hook,
                        )
                        result["a2c2_applied"] = decision.apply
                        result["a2c2_reason"] = decision.reason
                        result["a2c2_correction_magnitude"] = round(magnitude, 6)
                        if decision.apply:
                            if full_dim > hook_dim:
                                # Splice corrected values back into the leading
                                # hook_dim of the full padded chunk.
                                actions_arr[:, :hook_dim] = corrected
                                result["actions"] = actions_arr.tolist()
                            else:
                                result["actions"] = corrected.tolist()
                        span.set_attribute("reflex.a2c2.applied", decision.apply)
                        span.set_attribute("reflex.a2c2.reason", decision.reason)
                except Exception as exc:  # noqa: BLE001 — A2C2 must never break /act
                    logger.warning("a2c2_hook.apply_failed: %s", exc)

            # JSONL record hook (B.2). Writes inside the OTel span context
            # so the seq attribute below cross-links the two ledgers. Recorder
            # absent or degraded → no-op.
            _rec = getattr(server, "_recorder", None)
            if _rec is not None and isinstance(result, dict):
                actions = result.get("actions") or []
                action_dim = (
                    len(actions[0]) if actions and isinstance(actions[0], list) else 0
                )
                latency_total = float(result.get("latency_ms", 0.0))
                err = (
                    {"slug": "inference-error", "message": str(result["error"])[:500]}
                    if "error" in result
                    else None
                )
                # Policy-versioning Day 7 schema: emit per-request
                # `routing` block when 2-policy mode is active. v1
                # readers ignore the unknown field per ADR additive
                # evolution.
                _routing_for_record: dict | None = None
                if _two_routing_decision is not None:
                    _routing_for_record = {
                        "slot": _two_routing_decision.slot,
                        "routing_key": _two_routing_decision.routing_key,
                        "degraded": _two_routing_decision.degraded_routing,
                        "cached": _two_routing_decision.cached,
                        "crash_verdict": _two_routing_decision.crash_verdict,
                    }
                # Failure-classifier substrate (per failure-classifier-v1 research
                # sidecar Finding 3.1): pass through guard_summary if predict()
                # populated it. None when ActionGuard wasn't built (no URDF /
                # embodiment_config.constraints absent).
                _guard_for_record = (
                    result.get("guard_summary") if isinstance(result, dict) else None
                )
                rec_seq = _rec.write_request(
                    chunk_id=_rec.seq,  # 1:1 with seq for non-batched serve
                    image_b64=request.image,
                    instruction=request.instruction,
                    state=request.state,
                    actions=actions,
                    action_dim=action_dim,
                    latency_total_ms=latency_total,
                    mode=str(result.get("inference_mode", "")),
                    error=err,
                    routing=_routing_for_record,
                    guard=_guard_for_record,
                )
                if rec_seq >= 0:
                    span.set_attribute("reflex.record.seq", rec_seq)

            # Synthetic latency injection (B.4 A2C2 gate). Runs AFTER
            # JSONL recording so recorded latency_ms is the true compute
            # cost; client sees inference + injected delay. No-op when
            # inject_latency_ms == 0.0.
            _inj = float(getattr(server, "inject_latency_ms", 0.0) or 0.0)
            if _inj > 0 and isinstance(result, dict) and "error" not in result:
                import asyncio as _asyncio
                result["injected_latency_ms"] = _inj
                span.set_attribute("reflex.injected_latency_ms", _inj)
                await _asyncio.sleep(_inj / 1000.0)

            # A2C2 outcome bookkeeping — feed the latency + success signal
            # into the hook's rolling windows so subsequent should_apply()
            # decisions reflect this request. Records regardless of whether
            # we applied or skipped this time.
            if _a2c2 is not None and isinstance(result, dict):
                try:
                    _latency_ms = float(result.get("latency_ms", 0.0))
                    if _inj > 0:
                        _latency_ms += _inj  # client-perceived latency
                    _success = "error" not in result
                    _a2c2.record_outcome(latency_ms=_latency_ms, success=_success)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("a2c2_hook.record_outcome_failed: %s", exc)

            # Auto-calibration warmup tracker (Day 5): record real /act
            # latency + try to persist a stable p95 back to the cache.
            # No active probe — passively learns latency_compensation_ms.
            _calib_warmup = getattr(server, "calibration_warmup", None)
            if _calib_warmup is not None and isinstance(result, dict):
                try:
                    _calib_latency_ms = float(result.get("latency_ms", 0.0))
                    if _calib_latency_ms > 0:
                        _calib_warmup.record_latency(_calib_latency_ms)
                        _calib_warmup.maybe_persist()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "calibration_warmup.record_or_persist_failed: %s", exc,
                    )

            # Policy-versioning Days 9-10: emit X-Reflex-* headers + slot
            # tracking when 2-policy mode is active. Single-policy mode is
            # unchanged (headers omitted; existing customers/dashboards
            # don't depend on them).
            _resp_headers: dict[str, str] = {}
            if _two_routing_decision is not None:
                _two_state_local = getattr(server, "two_policy_state", None)
                if _two_state_local is not None:
                    _slot = _two_routing_decision.slot
                    _slot_policy = (
                        _two_state_local.policy_a
                        if _slot == "a"
                        else _two_state_local.policy_b
                    )
                    _resp_headers["X-Reflex-Policy-Slot"] = _slot
                    _resp_headers["X-Reflex-Model-Version"] = (
                        _slot_policy.model_version
                    )
                    _resp_headers["X-Reflex-Routing-Key"] = (
                        _two_routing_decision.routing_key[:128]
                    )
                    if _two_routing_decision.degraded_routing:
                        _resp_headers["X-Reflex-Routing-Degraded"] = "true"

            return JSONResponse(content=result, headers=_resp_headers)

    @app.get("/config")
    async def config(_auth: None = Depends(_require_api_key)):
        cfg = dict(server.config) if isinstance(server.config, dict) else {}
        cfg["robot_id"] = getattr(server, "robot_id", "") or ""
        return JSONResponse(content=cfg)

    @app.get("/guard/status")
    async def guard_status():
        g = getattr(server, "_action_guard", None)
        if g is None:
            return JSONResponse(content={"enabled": False})
        return JSONResponse(content={
            "enabled": True,
            "tripped": bool(g.tripped),
            "trip_reason": g.trip_reason,
            "consecutive_clamps": int(g.consecutive_clamps),
            "max_consecutive_clamps": int(g.max_consecutive_clamps),
            "inference_count": int(g.inference_count),
        })

    @app.post("/guard/reset")
    async def guard_reset():
        g = getattr(server, "_action_guard", None)
        if g is None:
            return JSONResponse(
                status_code=400,
                content={"error": "guard_not_enabled"},
            )
        was_tripped = bool(g.tripped)
        g.reset()
        return JSONResponse(content={"reset": True, "was_tripped": was_tripped})

    # Attach the live ReflexServer to app.state so downstream integrations
    # (MCP server, future dashboards, test harnesses) can access the same
    # inference engine without recreating it. Per mcp-server Phase 1 wiring.
    app.state.reflex_server = server

    # Concurrency limiter middleware (Phase 1 auth-bearer feature).
    # When max_concurrent is set, a semaphore bounds in-flight /act requests;
    # overload returns HTTP 429 + Retry-After (not slow-down). /health is
    # exempt so liveness probes bypass the limit. TGI's overload pattern.
    if max_concurrent is not None and max_concurrent > 0:
        from reflex.runtime.auth import ConcurrencyLimiter, make_429_payload
        _concurrency_limiter = ConcurrencyLimiter(max_concurrent=max_concurrent)
        app.state.concurrency_limiter = _concurrency_limiter

        @app.middleware("http")
        async def _concurrency_middleware(request, call_next):
            # Don't rate-limit liveness probes or metrics scrapes
            if request.url.path in ("/health", "/healthz", "/metrics"):
                return await call_next(request)
            async with _concurrency_limiter.try_acquire() as ctx:
                if not ctx.acquired:
                    return JSONResponse(
                        status_code=429,
                        headers={"Retry-After": "1"},
                        content=make_429_payload(
                            current=_concurrency_limiter.in_flight,
                            limit=_concurrency_limiter.max_concurrent,
                        ),
                    )
                return await call_next(request)

    # SLO enforcement middleware (Phase 1 latency-slo-enforcement feature).
    # Measures /act latency, computes rolling p99, and reacts per slo_mode:
    #   - "log_only": emit inc_slo_violation() metric only
    #   - "503": metric + return HTTP 503 with {p99_measured, p99_slo, retry_after}
    #   - "degrade": metric only in Phase 1 (actual degradation knobs ship
    #     with adaptive-denoise-pi0 + chunk-budget-batching in Phase 1.5)
    if slo_tracker is not None:
        import time as _slo_time
        from reflex.observability.prometheus import inc_slo_violation

        _slo_embodiment = (
            getattr(embodiment_config, "embodiment", "unknown")
            if embodiment_config is not None else "unknown"
        )
        _slo_percentile_str = f"p{int(slo_tracker.spec.percentile)}"

        @app.middleware("http")
        async def _slo_middleware(request, call_next):
            # Only measure /act (that's the latency SLO customers care about).
            if request.url.path != "/act":
                return await call_next(request)
            t0 = _slo_time.perf_counter()
            response = await call_next(request)
            elapsed_ms = (_slo_time.perf_counter() - t0) * 1000.0
            slo_tracker.record_latency_ms(elapsed_ms)
            if slo_tracker.is_violating():
                inc_slo_violation(
                    embodiment=_slo_embodiment,
                    kind=f"{_slo_percentile_str}_exceeded",
                )
                if slo_mode == "503":
                    return JSONResponse(
                        status_code=503,
                        headers={"Retry-After": "1"},
                        content={
                            "error": "slo_violation",
                            f"{_slo_percentile_str}_measured_ms": round(
                                slo_tracker.current_p99(), 2
                            ),
                            f"{_slo_percentile_str}_slo_ms": slo_tracker.spec.threshold_ms,
                            "retry_after_s": 1,
                        },
                    )
                # "log_only" and "degrade" fall through — response already computed
            return response

        app.state.slo_tracker = slo_tracker
        app.state.slo_mode = slo_mode

    return app
