"""RTC adapter — Real-Time Chunking state for Tether serve.

Implements Real-Time Chunking (arxiv 2506.07339) so the robot keeps executing
the tail of one chunk while the next chunk is being computed.

The adapter owns the per-session carry-forward: the last returned chunk, its
``chunk_id``, the latency estimate and the leftover the next chunk is guided
toward. It does not own the guidance math. Torch runtimes hand the guidance
inputs to LeRobot's ``RTCProcessor``; the pi0.5 per-step ONNX loop uses the
numpy port in ``tether.runtime.rtc_guidance``. Both apply LeRobot 0.6.1's
identity-Jacobian correction, not the paper's VJP (see that module).

Leftover contract (HTTP ``/act``):

- The client reports ``actions_executed`` (index of the next unexecuted action
  of chunk ``chunk_id``). The leftover is ``chunk[actions_executed:]``.
- Without a report the count is estimated from wall-clock time since the chunk
  was returned and labelled ``leftover_source="estimated"``.
- The leftover is converted into the policy's normalized action space with
  ``policy.normalize_actions`` before guidance; the stored chunk is the one the
  client receives (after guards), in processed units.

Design: ``docs/research/rtc-guidance-adr-20261027.md``
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
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
    adaptive_chunking_canary: bool = False
    adaptive_min_horizon: int = 1
    adaptive_low_uncertainty: float = 0.20
    adaptive_high_uncertainty: float = 0.65
    adaptive_low_guard_margin: float = 0.05
    adaptive_high_correction_magnitude: float = 0.20
    adaptive_high_action_delta: float = 0.25
    adaptive_high_latency_ms: float = 120.0
    # HTTP /act keeps one carry-forward per episode_id. At the cap a new
    # episode evicts the least recently active stream that has no request in
    # flight and has been idle at least stream_min_idle_s; if none qualifies
    # the new episode is refused (503) rather than evicting a live robot.
    max_streams: int = 64
    # None: derived at serve time as max(2 s, 4 x chunk duration). The carry-
    # forward only guides a request that arrives within about one chunk
    # period, so a stream idle for a few chunk periods holds nothing useful.
    stream_min_idle_s: float | None = None
    # Deprecated (one release): the adapter is always strict now.
    strict_policy_kwargs: bool | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate the Tether-side extras. lerobot's RTCConfig validates
        its own fields when constructed via _build_lerobot_rtc_config()."""
        if self.strict_policy_kwargs is True:
            import warnings

            warnings.warn(
                "RtcAdapterConfig.strict_policy_kwargs is deprecated: RTC is always "
                "strict now; drop the keyword (removed in the next minor release)",
                DeprecationWarning,
                stacklevel=3,
            )
        elif self.strict_policy_kwargs is False:
            raise ValueError(
                "strict_policy_kwargs=False is no longer supported: the plain-inference "
                "fallback was removed, so a policy that rejects the RTC kwargs is "
                "refused instead of served unguided; drop the keyword"
            )
        if self.max_streams < 1:
            raise ValueError(f"max_streams must be >= 1, got {self.max_streams}")
        if self.stream_min_idle_s is not None and self.stream_min_idle_s < 0:
            raise ValueError(
                f"stream_min_idle_s must be >= 0, got {self.stream_min_idle_s}"
            )
        if self.adaptive_chunking_canary:
            self.adaptive_chunking_enabled = True
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
        if self.adaptive_low_uncertainty < 0:
            raise ValueError(
                "adaptive_low_uncertainty must be >= 0, "
                f"got {self.adaptive_low_uncertainty}"
            )
        if self.adaptive_high_uncertainty <= self.adaptive_low_uncertainty:
            raise ValueError(
                "adaptive_high_uncertainty must be greater than "
                "adaptive_low_uncertainty"
            )
        if self.adaptive_low_guard_margin < 0:
            raise ValueError(
                "adaptive_low_guard_margin must be >= 0, "
                f"got {self.adaptive_low_guard_margin}"
            )
        if self.adaptive_high_correction_magnitude < 0:
            raise ValueError(
                "adaptive_high_correction_magnitude must be >= 0, "
                f"got {self.adaptive_high_correction_magnitude}"
            )
        if self.adaptive_high_action_delta < 0:
            raise ValueError(
                "adaptive_high_action_delta must be >= 0, "
                f"got {self.adaptive_high_action_delta}"
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
    """Minimum interface an enabled RtcAdapter expects from a policy.

    ``normalize_actions`` maps processed (client-facing) actions back into the
    normalized space the denoiser works in; the leftover is converted with it.
    ``predict_action_chunk`` is only used by the in-process
    ``predict_chunk_with_rtc`` path; HTTP ``/act`` calls the server's predict
    with ``rtc=RtcGuidance``.
    """

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray: ...

    def predict_action_chunk(self, **kwargs: Any) -> np.ndarray: ...


class RtcRequestError(ValueError):
    """A client's ``chunk_id`` / ``actions_executed`` report was refused."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after

    def headers(self) -> dict[str, str] | None:
        return None if self.retry_after is None else {"Retry-After": str(self.retry_after)}

    def as_response(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self)}


@dataclass
class RtcPlan:
    """Guidance inputs for one chunk plus the telemetry that describes them."""

    guidance: Any  # tether.runtime.rtc_guidance.RtcGuidance
    leftover_source: str  # "none" | "client" | "estimated"
    actions_executed: int | None
    leftover_processed: np.ndarray | None

    def telemetry(self) -> dict[str, Any]:
        g = self.guidance
        left = g.prev_chunk_left_over
        return {
            "applied": left is not None,
            "guidance": "lerobot_identity_jacobian",
            "leftover_source": self.leftover_source,
            "actions_executed": self.actions_executed,
            "leftover_len": 0 if left is None else int(left.shape[0]),
            "inference_delay": int(g.inference_delay),
            "execution_horizon": int(g.execution_horizon),
            "schedule": g.schedule,
            "max_guidance_weight": float(g.max_guidance_weight),
        }


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

        if config.enabled and not callable(getattr(policy, "normalize_actions", None)):
            raise ValueError(
                f"RTC needs {type(policy).__name__}.normalize_actions to convert "
                "the leftover into the model's normalized action space"
            )
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
        self._prev_chunk_left_over: Any = None  # normalized leftover of the last plan
        self._last_chunk: np.ndarray | None = None  # processed, as returned
        self._last_chunk_id: str | None = None
        self._last_chunk_at: float = 0.0
        self._last_plan: RtcPlan | None = None
        self._last_action_delta: float | None = None
        self._last_adaptive_signal = AdaptiveChunkSignal()
        self._last_adaptive_decision: AdaptiveChunkDecision | None = None
        self._last_execution_horizon: int | None = None
        self._last_actions_consumed: int | None = None
        self._adaptive_chunker: AdaptiveChunkController | None = None
        if config.adaptive_chunking_enabled:
            self._adaptive_chunker = AdaptiveChunkController(
                AdaptiveChunkConfig(
                    enabled=True,
                    min_horizon=config.adaptive_min_horizon,
                    base_horizon=config.rtc_execution_horizon,
                    max_horizon=action_buffer.capacity,
                    low_uncertainty=config.adaptive_low_uncertainty,
                    high_uncertainty=config.adaptive_high_uncertainty,
                    low_guard_margin=config.adaptive_low_guard_margin,
                    high_correction_magnitude=(
                        config.adaptive_high_correction_magnitude
                    ),
                    high_action_delta=config.adaptive_high_action_delta,
                    high_latency_ms=config.adaptive_high_latency_ms,
                )
            )

    # ---- Public API ---------------------------------------------

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Processed actions → the policy's normalized action space."""
        fn = getattr(self.policy, "normalize_actions", None)
        if fn is None:
            if self.config.enabled:  # construction refuses this; defensive
                raise RuntimeError("RTC policy has no normalize_actions")
            fn = np.asarray  # disabled adapter: no guidance runs on it
        # Always a fresh array: an identity normalizer must not hand the
        # caller a view of the stored chunk.
        return np.array(fn(actions), dtype=np.float32, copy=True)

    def plan_guidance(
        self,
        *,
        actions_executed: int | None = None,
        chunk_id: str | None = None,
        now: float | None = None,
    ) -> RtcPlan:
        """Build the guidance inputs for the next chunk.

        Validates the client's report against the last chunk this adapter
        returned. Raises :class:`RtcRequestError` (HTTP status attached) when
        the report is incomplete, refers to a different chunk, or is out of
        range. Never guesses past a refused report.
        """
        from tether.runtime.rtc_guidance import RtcGuidance

        if (actions_executed is None) != (chunk_id is None):
            raise RtcRequestError(
                422, "rtc_report_incomplete",
                "actions_executed and chunk_id must be sent together",
            )
        if actions_executed is not None and actions_executed < 0:
            raise RtcRequestError(422, "rtc_actions_executed_invalid",
                                  "actions_executed must be >= 0")
        prev = self._last_chunk
        executed: int | None = None
        source = "none"
        if chunk_id is not None:
            if prev is None or chunk_id != self._last_chunk_id:
                raise RtcRequestError(
                    409, "rtc_chunk_mismatch",
                    f"chunk_id {chunk_id!r} is not the last chunk this session "
                    f"returned ({self._last_chunk_id!r}); send a new episode_id "
                    "to start over",
                )
            assert actions_executed is not None  # paired with chunk_id above
            if actions_executed > prev.shape[0]:
                raise RtcRequestError(
                    422, "rtc_actions_executed_invalid",
                    f"actions_executed={actions_executed} exceeds the chunk "
                    f"length {prev.shape[0]}",
                )
            executed, source = int(actions_executed), "client"
        elif prev is not None:
            now = time.monotonic() if now is None else now
            elapsed = max(0.0, now - self._last_chunk_at)
            executed = min(prev.shape[0], _ceil_actions(elapsed, self.config.execute_hz))
            source = "estimated"

        tail = None if prev is None else prev[executed:]
        if tail is not None and tail.shape[0] == 0:
            tail = None
        latency_s = self.latency.estimate()
        delay = _ceil_actions(latency_s, self.config.execute_hz)
        self._last_actions_consumed = delay
        horizon = self.config.rtc_execution_horizon
        decision = self._decide_adaptive_horizon(latency_s)
        if decision is not None and not self.config.adaptive_chunking_canary:
            horizon = decision.horizon
        self._last_execution_horizon = horizon
        plan = RtcPlan(
            guidance=RtcGuidance(
                inference_delay=delay,
                execution_horizon=horizon,
                prev_chunk_left_over=(
                    None if tail is None else self.normalize_actions(tail)
                ),
                schedule=self.config.prefix_attention_schedule,
                max_guidance_weight=self.config.max_guidance_weight,
            ),
            leftover_source=source,
            actions_executed=executed,
            leftover_processed=tail,
        )
        self._prev_chunk_left_over = plan.guidance.prev_chunk_left_over
        self._last_plan = plan
        return plan

    def predict_chunk_with_rtc(
        self,
        batch: dict[str, Any],
        *,
        actions_executed: int | None = None,
        chunk_id: str | None = None,
    ) -> np.ndarray:
        """In-process RTC call: plan guidance, then call the policy with
        LeRobot's kwargs (``inference_delay``, ``prev_chunk_left_over`` in
        normalized space, ``execution_horizon``). Follow with
        ``merge_and_update`` to record the returned chunk.

        A policy that does not accept the kwargs raises: plain inference would
        not apply RTC, so there is no fallback.
        """
        plan = self.plan_guidance(actions_executed=actions_executed, chunk_id=chunk_id)
        g = plan.guidance
        try:
            return self.policy.predict_action_chunk(
                **batch,
                inference_delay=g.inference_delay,
                prev_chunk_left_over=g.prev_chunk_left_over,
                execution_horizon=g.execution_horizon,
            )
        except TypeError as exc:
            raise RuntimeError(
                "This policy does not implement the inference_delay, "
                "prev_chunk_left_over and execution_horizon contract; plain "
                "inference would not apply RTC."
            ) from exc

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
        *,
        t_obs: float | None = None,
        now: float | None = None,
    ) -> str:
        """Record the chunk returned to the client; return its ``chunk_id``.

        ``actions``: ``(T, A)`` or ``(1, T, A)`` in processed units — exactly
        what the client will execute. The next ``plan_guidance`` slices its
        leftover from this chunk. Also pushes it into the action buffer for
        in-process integrations and records the inference latency.
        """
        chunk_2d = actions[0] if actions.ndim == 3 and actions.shape[0] == 1 else actions
        chunk_2d = np.asarray(chunk_2d, dtype=np.float32)
        plan = self._last_plan
        self._last_action_delta = overlapping_action_delta(
            None if plan is None else plan.leftover_processed, chunk_2d,
        )
        self.buffer.push_chunk(chunk_2d, t_obs=t_obs)
        self._chunk_count += 1
        self._last_chunk = chunk_2d.copy()
        self._last_chunk_at = time.monotonic() if now is None else now
        self._last_chunk_id = self._chunk_id_for(chunk_2d)
        self.latency.record(elapsed_time)
        if self.config.enabled and self._last_execution_horizon is None:
            self._last_execution_horizon = self.config.rtc_execution_horizon
        return self._last_chunk_id

    def _chunk_id_for(self, chunk: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(str(self._active_episode_id).encode())
        h.update(self._chunk_count.to_bytes(8, "little"))
        h.update(np.ascontiguousarray(chunk).tobytes())
        return h.hexdigest()[:16]

    def reset(self, episode_id: str | None = None) -> None:
        """Reset RTC state at episode boundary.

        Called by SDK ``client.reset()`` hooks or when the server detects a
        new ``episode_id`` via ``/act`` params.
        """
        self._active_episode_id = episode_id
        self._chunk_count = 0
        self._prev_chunk_left_over = None
        self._last_chunk = None
        self._last_chunk_id = None
        self._last_chunk_at = 0.0
        self._last_plan = None
        self._last_action_delta = None
        self._last_adaptive_signal = AdaptiveChunkSignal()
        self._last_adaptive_decision = None
        self._last_execution_horizon = None
        self._last_actions_consumed = None
        # Clear the latency window — old samples are stale on a fresh episode
        self.latency = LatencyTracker(
            percentile=self.config.latency_percentile,
            discard_first=self.config.cold_start_discard,
        )
        if self._processor is not None:
            self._processor.reset_tracker()
        logger.info("[rtc] reset — new episode %s", _episode_tag(episode_id))

    # ---- Introspection ------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Stats for logging / metrics. Consumed by Prometheus exporter
        (Phase 0.5.8) + ``tether monitor``."""
        stats: dict[str, Any] = {
            "enabled": self.config.enabled,
            "chunk_count": self._chunk_count,
            "active_episode_id": self._active_episode_id,
            "latency": self.latency.summary(),
            "rtc_available": _RTC_AVAILABLE,
            "configured_execution_horizon": self.config.rtc_execution_horizon,
            "action_buffer": asdict(self.buffer.stats()),
        }
        if self._last_execution_horizon is not None:
            stats["execution_horizon"] = self._last_execution_horizon
        if self._last_actions_consumed is not None:
            stats["actions_consumed"] = self._last_actions_consumed
        if self._last_action_delta is not None:
            stats["last_action_delta"] = self._last_action_delta
        if self._last_adaptive_signal.has_values():
            stats["adaptive_signal"] = self._last_adaptive_signal.as_dict()
        if self._last_adaptive_decision is not None:
            adaptive_chunking = self._last_adaptive_decision.as_dict()
            adaptive_chunking["canary"] = self.config.adaptive_chunking_canary
            if self._last_execution_horizon is not None:
                adaptive_chunking["applied_horizon"] = self._last_execution_horizon
            stats["adaptive_chunking"] = adaptive_chunking
        return stats


_EPISODE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")  # used with fullmatch


def _episode_tag(episode_id: str | None) -> str:
    """Loggable stand-in for a client-chosen id: length + short hash."""
    if episode_id is None:
        return "none"
    digest = hashlib.sha256(episode_id.encode("utf-8", "replace")).hexdigest()[:8]
    return f"len{len(episode_id)}:{digest}"


class RtcStream:
    """One client's RTC carry-forward: adapter, lock, users and replay memo.

    ``/act`` holds ``lock`` from guidance planning to chunk commit, so two
    in-flight requests on one stream cannot plan against the same leftover.
    ``users`` counts requests that hold or wait for the lock; eviction only
    takes streams with none (``lock.locked()`` is briefly False between a
    release and the woken waiter, so it cannot be the test). ``replay``
    answers an exact retry of the report that produced the last chunk.
    """

    def __init__(self, adapter: RtcAdapter, now: float, clock: Any, episode_id: str = ""):
        import asyncio

        self.episode_id = episode_id
        self.adapter = adapter
        self.lock = asyncio.Lock()
        self.users = 0
        self.last_activity = now
        self._clock = clock
        self._last_report: tuple[str, int] | None = None
        self._last_body: bytes | None = None
        self._last_headers: dict[str, str] = {}

    def release(self) -> None:
        self.users -= 1
        self.last_activity = self._clock()

    def start_episode(self, episode_id: str) -> None:
        """Explicit episode start under a reused episode_id: drop the carry."""
        self.adapter.reset(episode_id=episode_id)
        self.forget_replay()

    def invalidate(self) -> None:
        """Safety-state change (guard trip/reset, degrade, license change):
        drop the carry-forward and the replay memo. The next request plans
        from a fresh observation with no leftover."""
        self.start_episode(self.episode_id)

    def forget_replay(self) -> None:
        self._last_report = None
        self._last_body = None
        self._last_headers = {}

    def replay(self, chunk_id: str | None, actions_executed: int | None) -> tuple[bytes, dict[str, str]] | None:
        if chunk_id is None or self._last_body is None or self._last_report != (chunk_id, actions_executed):
            return None
        return self._last_body, dict(self._last_headers)

    def remember(
        self,
        chunk_id: str | None,
        actions_executed: int | None,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._last_report = (
            None if chunk_id is None or actions_executed is None else (chunk_id, int(actions_executed))
        )
        self._last_body = body
        self._last_headers = dict(headers or {})


class RtcStreams:
    """RTC state keyed by ``episode_id``: one stream per robot/episode.

    Never shared across keys. At ``max_streams`` a new key evicts the least
    recently active stream with no user that has been idle at least
    ``min_idle_s``; when none qualifies the new key is refused with 503, so
    a burst of unknown ids cannot evict a live robot.
    """

    def __init__(
        self,
        factory: Any,
        max_streams: int = 64,
        min_idle_s: float = 300.0,
        clock: Any = time.monotonic,
    ):
        self._factory = factory
        self._streams: dict[str, RtcStream] = {}
        self.max_streams = max_streams
        self.min_idle_s = min_idle_s
        self._clock = clock
        self.evictions = 0
        self.invalidations = 0
        self._safety_epoch: Any = None

    def invalidate_all(self, reason: str) -> None:
        """Drop every stream's carry-forward and replay memo."""
        for stream in self._streams.values():
            stream.invalidate()
        self.invalidations += 1
        logger.warning("[rtc] invalidated %d stream(s): %s", len(self._streams), reason)

    def sync_safety_epoch(self, epoch: Any) -> None:
        """Invalidate every stream when the serving safety state changed since
        the last request (guard tripped or reset, health, license)."""
        if self._safety_epoch is not None and epoch != self._safety_epoch:
            self.invalidate_all(f"safety state changed {self._safety_epoch!r} -> {epoch!r}")
        self._safety_epoch = epoch

    def acquire(self, episode_id: str | None) -> RtcStream:
        """The stream for ``episode_id`` with one more user; pair with
        ``stream.release()``. Raises :class:`RtcRequestError`."""
        if not episode_id:
            raise RtcRequestError(
                422, "rtc_episode_required",
                "--rtc keeps one carry-forward per stream; send episode_id "
                "(one per robot and episode) so streams never mix",
            )
        if not isinstance(episode_id, str) or not _EPISODE_ID.fullmatch(episode_id):
            raise RtcRequestError(
                422, "rtc_episode_invalid",
                "under --rtc episode_id must be 1-128 characters of "
                "A-Z a-z 0-9 . _ : -",
            )
        now = self._clock()
        stream = self._streams.get(episode_id)
        if stream is None:
            if len(self._streams) >= self.max_streams:
                self._evict_one(now)
            adapter = self._factory()
            adapter.reset(episode_id=episode_id)
            stream = RtcStream(adapter, now, self._clock, episode_id)
            self._streams[episode_id] = stream
        stream.users += 1
        stream.last_activity = now
        return stream

    def _evict_one(self, now: float) -> None:
        idle = [
            (stream.last_activity, key)
            for key, stream in self._streams.items()
            if stream.users == 0 and now - stream.last_activity >= self.min_idle_s
        ]
        if not idle:
            free_at = [
                self.min_idle_s - (now - stream.last_activity)
                for stream in self._streams.values() if stream.users == 0
            ]
            wait = min(free_at) if free_at else self.min_idle_s
            raise RtcRequestError(
                503, "rtc_streams_full",
                f"{len(self._streams)} RTC streams are active (cap {self.max_streams}, "
                f"none idle >= {self.min_idle_s:g} s); retry later or raise --rtc-max-streams",
                retry_after=max(1, math.ceil(wait)),
            )
        _, key = min(idle)
        del self._streams[key]
        self.evictions += 1
        logger.info("[rtc] evicted idle stream %s", _episode_tag(key))

    def __len__(self) -> int:
        return len(self._streams)

    def __contains__(self, episode_id: object) -> bool:
        return episode_id in self._streams


def _ceil_actions(seconds: float, hz: float) -> int:
    """Actions executed in ``seconds`` at ``hz``, rounded up as LeRobot does
    (``math.ceil(latency / time_per_action)``); 1e-6 rounding first keeps
    0.08 s x 100 Hz at 8, not 9."""
    return math.ceil(round(seconds * hz, 6))


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
    "RtcPlan",
    "RtcRequestError",
    "RtcStream",
    "RtcStreams",
    "RtcAdapterConfig",
    "LatencyTracker",
    "RtcCompatiblePolicy",
    "require_rtc",
    "_VALID_SCHEDULES",
    "_build_lerobot_rtc_config",
]
