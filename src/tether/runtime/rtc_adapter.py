"""RTC adapter — wraps ``lerobot.policies.rtc.RTCProcessor`` for Tether serve.

Implements Real-Time Chunking (arxiv 2506.07339) so the robot keeps executing
the tail of one chunk while the next chunk is being computed. Net: 2-3x
effective throughput on high-latency deployments (Jetson-class).

Status: Day 1+2 of B.3 sprint shipped (construction + predict body).
Day 3 wires prev_chunk_left_over carry-forward; Day 4 integrates with
TetherServer's /act handler. Body methods do real work but assume the
underlying policy supports the lerobot RTC kwargs (inference_delay,
prev_chunk_left_over) — true for decomposed exports with Python denoise
loops; monolithic ONNX (loop baked in) ignores the kwargs silently.

Design: ``reflex_context/reference/deep_dive_lerobot_rtc.md``
Plan ref: ``reflex_context/features/01_serve/subfeatures/_rtc_a2c2/rtc-adapter_plan.md``
Goal: ``serve-rtc-wrapper`` (GOALS.yaml, weight 10)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from tether.runtime.adaptive_chunking import (
    AdaptiveChunkConfig,
    AdaptiveChunkController,
    AdaptiveChunkDecision,
    AdaptiveChunkSignal,
    overlapping_action_delta,
)
from tether.runtime.buffer import ActionChunkBuffer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Soft import of lerobot.policies.rtc
# ──────────────────────────────────────────────────────────────────
try:
    from lerobot.configs.types import RTCAttentionSchedule  # type: ignore
    from lerobot.policies.rtc import RTCProcessor  # type: ignore
    from lerobot.policies.rtc.configuration_rtc import RTCConfig  # type: ignore
    _RTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    RTCProcessor = None  # type: ignore
    RTCConfig = None  # type: ignore
    RTCAttentionSchedule = None  # type: ignore
    _RTC_AVAILABLE = False


# Schedule names supported by lerobot's RTCAttentionSchedule enum. Hard-coded
# (not derived from the import) so config validation works even when lerobot
# isn't installed — fail with a clear error at construction time, not at
# config-parse time.
_VALID_SCHEDULES = ("ZEROS", "ONES", "LINEAR", "EXP")


def require_rtc() -> None:
    """Raise if lerobot's RTC module isn't installed in this env."""
    if not _RTC_AVAILABLE:
        raise ImportError(
            "lerobot.policies.rtc not available. Install fastcrest-tether[rtc] "
            "or pip install lerobot>=0.5.1."
        )


def assert_rtc_compatible_with_num_steps(num_steps: int | None) -> None:
    """Reject 1-NFE + RTC at config-time per CLAUDE.md no-silent-fallbacks.

    RTC's prefix_weight schedule (lerobot RTCProcessor.denoise_step) computes
    guidance weights via ``tau = 1 - time`` and ``c = (1 - tau) / tau``. When
    ``num_steps == 1`` the only step runs at ``time == 1.0``, giving
    ``tau == 0`` and a division-by-zero in ``c``. This is a math-level
    singularity, not graceful degradation — fail loud at config-time
    rather than crash mid-inference.

    Surfaced in the per-step expert ONNX export research sidecar:
    ``reflex_context/features/03_export/per-step-expert-export_research.md``
    Lens 2 FM-4/FM-6.
    """
    if num_steps == 1:
        raise ValueError(
            "RTC guidance is not supported with num_steps=1. "
            "RTC's guidance-weight formula divides by tau=1-time, which "
            "is zero at the only step (time=1.0). Use num_steps >= 2 or "
            "disable RTC for 1-step inference. See "
            "features/03_export/per-step-expert-export_research.md "
            "Lens 2 FM-4 for the math."
        )


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────
@dataclass
class RtcAdapterConfig:
    """Config knobs for RTC behavior. Populated from per-embodiment YAML
    (see features/serve/per-embodiment-configs.md) or CLI flags.

    Maps onto lerobot's `RTCConfig` (configuration_rtc.py) plus Tether-side
    extras for latency tracking and gripper handling. The mapping is built
    in `_build_lerobot_rtc_config()`.
    """

    enabled: bool = False
    replan_hz: float = 20.0
    execute_hz: float = 100.0
    rtc_execution_horizon: int = 10  # actions locked to old chunk during replan
    prefix_attention_schedule: str = "LINEAR"  # ZEROS | ONES | LINEAR | EXP
    max_guidance_weight: float = 10.0
    debug: bool = False
    debug_maxlen: int = 100
    latency_percentile: int = 95      # p95 default; p99 if maintainer recommends
    cold_start_discard: int = 10      # first N chunks NOT recorded in latency tracker
    guidance_space: str = "normalized"  # 'normalized' | 'processed'
    gripper_dim_indices: list[int] = field(default_factory=list)
    skip_gripper_smoothing: bool = True
    adaptive_chunking_enabled: bool = False
    adaptive_min_horizon: int = 1
    adaptive_high_latency_ms: float = 120.0

    def __post_init__(self) -> None:
        """Validate the Tether-side extras. lerobot's RTCConfig validates
        its own fields when constructed via _build_lerobot_rtc_config()."""
        if self.prefix_attention_schedule not in _VALID_SCHEDULES:
            raise ValueError(
                f"prefix_attention_schedule must be one of {_VALID_SCHEDULES}, "
                f"got {self.prefix_attention_schedule!r}"
            )
        if self.max_guidance_weight <= 0:
            raise ValueError(
                f"max_guidance_weight must be positive, got {self.max_guidance_weight}"
            )
        if self.rtc_execution_horizon < 1:
            raise ValueError(
                f"rtc_execution_horizon must be >= 1, got {self.rtc_execution_horizon}"
            )
        if not 1 <= self.latency_percentile <= 99:
            raise ValueError(
                f"latency_percentile must be in [1, 99], got {self.latency_percentile}"
            )
        if self.adaptive_min_horizon < 1:
            raise ValueError(
                f"adaptive_min_horizon must be >= 1, got {self.adaptive_min_horizon}"
            )
        if self.adaptive_high_latency_ms <= 0:
            raise ValueError(
                "adaptive_high_latency_ms must be positive, "
                f"got {self.adaptive_high_latency_ms}"
            )


def _build_lerobot_rtc_config(cfg: RtcAdapterConfig) -> Any:
    """Build a lerobot RTCConfig from the Tether-side RtcAdapterConfig.

    Called only when cfg.enabled is True (require_rtc() guards the import).
    Raises if lerobot isn't installed.
    """
    require_rtc()
    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule(cfg.prefix_attention_schedule),
        max_guidance_weight=cfg.max_guidance_weight,
        execution_horizon=cfg.rtc_execution_horizon,
        debug=cfg.debug,
        debug_maxlen=cfg.debug_maxlen,
    )


# ──────────────────────────────────────────────────────────────────
# Latency tracking
# ──────────────────────────────────────────────────────────────────
class LatencyTracker:
    """Rolling-window percentile estimator for inference latency.

    Feeds RTC's internal ``update_latency`` path. Conservative percentile
    (p95 default) guards against under-sizing the replan budget.
    """

    def __init__(self, window_size: int = 50, percentile: int = 95, discard_first: int = 10):
        self._samples: list[float] = []
        self.window_size = window_size
        self.percentile = percentile
        self.discard_first = discard_first
        self._seen = 0

    def record(self, latency_s: float) -> None:
        """Record one inference wall-clock. Discards the first N (cold start)."""
        self._seen += 1
        if self._seen <= self.discard_first:
            return
        self._samples.append(latency_s)
        if len(self._samples) > self.window_size:
            self._samples.pop(0)

    def estimate(self) -> float:
        """Return conservative latency estimate for the scheduler."""
        if not self._samples:
            return 0.1  # 100ms fallback before any warm samples
        return float(np.percentile(self._samples, self.percentile))

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
        arr = np.array(self._samples)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "n": len(self._samples),
        }


# ──────────────────────────────────────────────────────────────────
# Policy protocol
# ──────────────────────────────────────────────────────────────────
class RtcCompatiblePolicy(Protocol):
    """Minimum interface an RtcAdapter expects from a policy.

    Pi05DecomposedInference, Pi0OnnxServer, and SmolVLANativeServer all
    implement ``predict_action_chunk(**kwargs) -> np.ndarray`` returning
    shape ``(B, chunk_size, action_dim)``. RTC wraps this call.
    """

    def predict_action_chunk(self, **kwargs: Any) -> np.ndarray: ...


# ──────────────────────────────────────────────────────────────────
# RtcAdapter
# ──────────────────────────────────────────────────────────────────
class RtcAdapter:
    """Tether-side wrapper around ``lerobot.policies.rtc.RTCProcessor``.

    Owns:
    - config parsing from YAML / CLI / per-embodiment overrides
    - episode-id reset hooks (SDK ``client.reset()`` calls)
    - interop with existing ``ActionChunkBuffer`` (buffer holds processed /
      denormalized actions; RTC internally holds original / normalized)
    - latency tracker feeding RTC's scheduler
    - gripper-dim bypass for binary components

    Does NOT own:
    - the RTC math itself (delegated to lerobot's processor)
    - denormalization (handled by the policy's postprocessor downstream)
    - robot-side safety clamping (Guard wedge)

    Usage::

        adapter = RtcAdapter(
            policy=decomposed_server,
            action_buffer=buf,
            config=RtcAdapterConfig(enabled=True, replan_hz=20, execute_hz=100),
        )
        actions = adapter.predict_chunk_with_rtc(batch)
        adapter.merge_and_update(actions, elapsed_time=latency_s)
    """

    def __init__(
        self,
        policy: RtcCompatiblePolicy,
        action_buffer: ActionChunkBuffer,
        config: RtcAdapterConfig | None = None,
    ):
        if config is None:
            config = RtcAdapterConfig()
        if config.enabled:
            require_rtc()

        self.policy = policy
        self.buffer = action_buffer
        self.config = config

        self.latency = LatencyTracker(
            percentile=config.latency_percentile,
            discard_first=config.cold_start_discard,
        )
        # Lerobot RTCProcessor — only constructed when enabled + dep available.
        # Verified against lerobot 0.5.1 source: RTCProcessor(rtc_config: RTCConfig).
        self._processor: Any = None
        if config.enabled:
            lerobot_cfg = _build_lerobot_rtc_config(config)
            self._processor = RTCProcessor(lerobot_cfg)
            logger.info(
                "RTCProcessor initialized — execution_horizon=%d schedule=%s "
                "max_guidance_weight=%.1f debug=%s",
                config.rtc_execution_horizon,
                config.prefix_attention_schedule,
                config.max_guidance_weight,
                config.debug,
            )

        self._active_episode_id: str | None = None
        self._chunk_count: int = 0
        self._prev_chunk_left_over: Any = None  # set in merge_and_update (Day 3)
        self._last_action_delta: float | None = None
        self._last_adaptive_signal = AdaptiveChunkSignal()
        self._last_adaptive_decision: AdaptiveChunkDecision | None = None
        self._adaptive_chunker: AdaptiveChunkController | None = None
        if config.adaptive_chunking_enabled:
            self._adaptive_chunker = AdaptiveChunkController(
                AdaptiveChunkConfig(
                    enabled=True,
                    min_horizon=config.adaptive_min_horizon,
                    base_horizon=config.rtc_execution_horizon,
                    max_horizon=action_buffer.capacity,
                    high_latency_ms=config.adaptive_high_latency_ms,
                )
            )

    # ---- Public API ---------------------------------------------

    def predict_chunk_with_rtc(self, batch: dict[str, Any]) -> np.ndarray:
        """Run one inference with RTC guidance applied.

        Steps:
        1. Estimate p95 latency from the tracker
        2. Compute actions_consumed = int(latency_s * config.execute_hz)
           — how many actions of the previous chunk the robot has already
           executed by the time this inference completes
        3. Call policy.predict_action_chunk(**batch, inference_delay=N,
           prev_chunk_left_over=...) — the policy is responsible for
           applying RTC guidance via lerobot.policies.rtc.RTCProcessor
           inside its denoising loop (decomposed exports with Python loops
           support this; monolithic ONNX ignores the kwargs silently)
        4. Record elapsed wall-clock on the latency tracker so the next
           call has a fresh estimate
        5. Return the chunk unchanged (denormalization happens inside the
           policy's postprocessor)

        Day 2 scope: prev_chunk_left_over is always None (first-chunk
        behavior). Day 3 wires the carry-forward from
        ``self._prev_chunk_left_over`` (set by ``merge_and_update``).
        """
        # Step 1+2: latency → actions_consumed
        latency_s = self.latency.estimate()
        actions_consumed = int(latency_s * self.config.execute_hz)
        adaptive_decision = self._decide_adaptive_horizon(latency_s)
        logger.debug(
            "[rtc] predict — latency p%d=%.3fs → %d actions consumed",
            self.config.latency_percentile, latency_s, actions_consumed,
        )

        # Step 3: call policy with RTC kwargs
        # `inference_delay` and `prev_chunk_left_over` are the lerobot
        # RTC contract (see RTCProcessor.denoise_step). Day 3: carry
        # forward the previous chunk's leftover so guidance has prefix.
        rtc_kwargs: dict[str, Any] = {
            "inference_delay": actions_consumed,
            "prev_chunk_left_over": self._prev_chunk_left_over,
        }
        if self.config.enabled:
            rtc_kwargs["execution_horizon"] = (
                adaptive_decision.horizon
                if adaptive_decision is not None
                else self.config.rtc_execution_horizon
            )

        t0 = time.monotonic()
        try:
            actions = self.policy.predict_action_chunk(**batch, **rtc_kwargs)
        except TypeError:
            # Policy doesn't accept RTC kwargs — fall back to plain call.
            # Useful for monolithic-ONNX policies whose forward doesn't
            # take inference_delay. The chunk we get back is the same
            # shape; RTC simply has no effect on this call.
            logger.debug(
                "[rtc] policy rejected RTC kwargs — falling back to plain call"
            )
            actions = self.policy.predict_action_chunk(**batch)
        elapsed_s = time.monotonic() - t0

        # Step 4: feed the latency tracker
        self.latency.record(elapsed_s)

        return actions

    def record_adaptive_signal(
        self,
        *,
        uncertainty: float | None = None,
        guard_margin: float | None = None,
        correction_magnitude: float | None = None,
    ) -> None:
        """Record external runtime signals for the next AAC decision.

        The /act handler calls this after guard/A2C2 post-processing. Keeping
        it separate from predict preserves the current response while allowing
        the next chunk horizon to react to the latest safety/correction facts.
        """
        self._last_adaptive_signal = AdaptiveChunkSignal(
            uncertainty=_clean_signal_value(uncertainty),
            guard_margin=_clean_signal_value(guard_margin),
            correction_magnitude=_clean_signal_value(correction_magnitude),
        )

    def _decide_adaptive_horizon(
        self,
        latency_s: float,
    ) -> AdaptiveChunkDecision | None:
        """Return current AAC decision, or None when disabled."""
        if self._adaptive_chunker is None:
            self._last_adaptive_decision = None
            return None
        decision = self._adaptive_chunker.decide(
            AdaptiveChunkSignal(
                uncertainty=self._last_adaptive_signal.uncertainty,
                guard_margin=self._last_adaptive_signal.guard_margin,
                correction_magnitude=(
                    self._last_adaptive_signal.correction_magnitude
                ),
                latency_ms=latency_s * 1000.0,
                action_delta=self._last_action_delta,
            ),
            capacity=self.buffer.capacity,
        )
        self._last_adaptive_decision = decision
        return decision

    def merge_and_update(
        self,
        actions: np.ndarray,
        elapsed_time: float,
    ) -> None:
        """Push the new chunk into the action buffer; record inference latency.

        Snapshots the buffer's current contents BEFORE push_chunk overwrites
        them, stashes as ``self._prev_chunk_left_over`` for the NEXT
        ``predict_chunk_with_rtc`` call. This is the carry-forward that
        gives RTC its prefix guidance — the new chunk's first N actions
        get pulled toward the previous chunk's unexecuted tail.

        On the first chunk, the buffer is empty so the snapshot is None;
        Day 2's "first-chunk-no-guidance" behavior is preserved.

        actions: shape ``(T, action_dim)`` — already denormalized by the policy.
        """
        # 1. Snapshot the previous chunk's leftover BEFORE we overwrite it.
        #    push_chunk(overwrite_stale=True) wipes the buffer; capture first.
        prev_leftover = self.buffer.peek_all()

        # 2. Push the new chunk (replaces stale)
        # buffer.push_chunk expects 2D; flatten batch dim if present
        if actions.ndim == 3 and actions.shape[0] == 1:
            chunk_2d = actions[0]
        else:
            chunk_2d = actions
        self._last_action_delta = overlapping_action_delta(prev_leftover, chunk_2d)
        self.buffer.push_chunk(chunk_2d)

        # 3. Stash the snapshot for the NEXT predict call's guidance
        self._prev_chunk_left_over = prev_leftover

        # 4. Record latency + bump chunk count
        self.latency.record(elapsed_time)
        self._chunk_count += 1

    def reset(self, episode_id: str | None = None) -> None:
        """Reset RTC state at episode boundary.

        Called by SDK ``client.reset()`` hooks or when the server detects a
        new ``episode_id`` via ``/act`` params.
        """
        self._active_episode_id = episode_id
        self._chunk_count = 0
        self._prev_chunk_left_over = None
        self._last_action_delta = None
        self._last_adaptive_signal = AdaptiveChunkSignal()
        self._last_adaptive_decision = None
        # Clear the latency window — old samples are stale on a fresh episode
        self.latency = LatencyTracker(
            percentile=self.config.latency_percentile,
            discard_first=self.config.cold_start_discard,
        )
        if self._processor is not None:
            self._processor.reset_tracker()
        logger.info("[rtc] reset — new episode_id=%s", episode_id)

    # ---- Introspection ------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Stats for logging / metrics. Consumed by Prometheus exporter
        (Phase 0.5.8) + ``tether monitor``."""
        stats = {
            "enabled": self.config.enabled,
            "chunk_count": self._chunk_count,
            "active_episode_id": self._active_episode_id,
            "latency": self.latency.summary(),
            "rtc_available": _RTC_AVAILABLE,
        }
        if self._last_action_delta is not None:
            stats["last_action_delta"] = self._last_action_delta
        if self._last_adaptive_signal.has_values():
            stats["adaptive_signal"] = self._last_adaptive_signal.as_dict()
        if self._last_adaptive_decision is not None:
            stats["adaptive_chunking"] = self._last_adaptive_decision.as_dict()
        return stats


def _clean_signal_value(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        cleaned = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cleaned):
        return None
    return cleaned


__all__ = [
    "RtcAdapter",
    "RtcAdapterConfig",
    "LatencyTracker",
    "RtcCompatiblePolicy",
    "require_rtc",
    "_VALID_SCHEDULES",
    "_build_lerobot_rtc_config",
]
