"""ORT CUDA graphs wrapper for Tether serve (Phase 1 cuda-graphs feature).

Per ADR 2026-04-24-cuda-graphs-architecture:
  - ORT-native CUDA graphs (NOT torch.cuda.graph)
  - Two separate captured graphs per model (vlm_prefix + expert_denoise)
  - One shape per (model × embodiment) pair — ONNX is shape-specialized at export
  - Opt-in customer flag for Phase 1; flip to always-on in Phase 2 after telemetry

Research sidecar:
  features/01_serve/subfeatures/_perf_compound/cuda-graphs/cuda-graphs_research.md

Usage:

    from tether.runtime.cuda_graphs import build_cuda_graph_providers, CudaGraphWrapper

    providers = build_cuda_graph_providers(enabled=cuda_graphs_enabled)
    raw = ort.InferenceSession(path, providers=providers)
    wrapped = CudaGraphWrapper(raw, session_name="vlm_prefix",
                               embodiment=embodiment, model_id=model_id)
    result = wrapped.run(output_names, feed)  # first call captures, subsequent replay
    wrapped.invalidate()  # on model swap — caller must build a new session

The wrapper does NOT own session construction. Caller is responsible for
passing a session built with build_cuda_graph_providers(enabled=True). On
capture/replay failure, the wrapper increments the eager-fallback counter
and re-raises — callers are expected to catch the exception and route the
request to an eager fallback path (or surface the error, per their policy).
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, Union

import numpy as np

from tether.observability.prometheus import (
    inc_cuda_graph_capture_failed_at_init,
    inc_cuda_graph_captured,
    inc_cuda_graph_eager_fallback,
    inc_cuda_graph_replayed,
    observe_cuda_graph_capture_seconds,
    observe_cuda_graph_replay_seconds,
)

if TYPE_CHECKING:
    import onnxruntime as ort  # pragma: no cover

logger = logging.getLogger(__name__)


_ORT_TO_NUMPY_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


# Bounded enum of session names — matches the Prometheus label vocabulary.
VALID_SESSION_NAMES = frozenset({"vlm_prefix", "expert_denoise"})


def build_cuda_graph_providers(enabled: bool) -> list:
    """Return an ONNX Runtime providers list with CUDA graphs configured.

    When enabled=True: CUDAExecutionProvider captures on first .run() and
    replays thereafter (session-level graph capture).
    When enabled=False: eager CUDA execution with no graph capture.

    The list includes CPUExecutionProvider as a final fallback per the
    existing Tether convention (src/tether/runtime/pi05_decomposed_server.py
    session constructors).
    """
    cuda_opts: dict[str, str] = {}
    if enabled:
        cuda_opts["enable_cuda_graph"] = "1"
    return [
        ("CUDAExecutionProvider", cuda_opts),
        "CPUExecutionProvider",
    ]


class CudaGraphWrapper:
    """Wraps an ORT session with capture/replay metric emission + fallback tracking.

    First call to .run() triggers capture (ORT handles this internally because
    the session was built with enable_cuda_graph=1); wrapper records the
    capture time + increments captured_total counter. Subsequent calls replay;
    wrapper records replay time + increments replayed_total.

    On any exception during .run(), the wrapper increments the eager-fallback
    counter with a reason label and re-raises. Callers are expected to catch
    and route to an eager-path session (or surface the error).

    .invalidate() is a no-op on the session itself — ORT's graph is bound to
    the session lifecycle. Callers must re-create the session (with a fresh
    CudaGraphWrapper) to get a new capture.
    """

    __slots__ = (
        "_session",
        "_session_name",
        "_embodiment",
        "_model_id",
        "_captured",
    )

    def __init__(
        self,
        session: "ort.InferenceSession",
        session_name: str,
        embodiment: str,
        model_id: str,
        already_captured: bool = False,
    ):
        """Wrap an ORT session with capture/replay metrics.

        Args:
            already_captured: set True if the session was already capture-probed
                by the caller (e.g., by try_capture_or_fall_back). The wrapper
                skips the first-run capture path on next .run() and jumps straight
                to replay semantics. Caller is responsible for having emitted the
                captured-counter + capture-duration metrics in that case.
        """
        if session_name not in VALID_SESSION_NAMES:
            raise ValueError(
                f"session_name must be one of {sorted(VALID_SESSION_NAMES)}, got {session_name!r}"
            )
        self._session = session
        self._session_name = session_name
        self._embodiment = embodiment
        self._model_id = model_id
        self._captured = already_captured  # flipped true after first successful .run()

    @property
    def session(self) -> "ort.InferenceSession":
        """Underlying ORT session. Exposed for callers that need to introspect
        inputs/outputs (e.g., building a feed dict)."""
        return self._session

    @property
    def captured(self) -> bool:
        """True if at least one successful .run() has occurred (graph is captured)."""
        return self._captured

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> list:
        """Forward to the ORT session with metric emission + fallback tracking.

        - First call (captured == False): records capture wall-clock into the
          capture histogram, increments captured_total on success.
        - Subsequent calls: records replay wall-clock into the replay histogram,
          increments replayed_total.
        - Any exception increments eager_fallback_total with a reason label
          and is re-raised.
        """
        if not self._captured:
            t0 = time.perf_counter()
            try:
                result = self._session.run(output_names, input_feed)
            except Exception as exc:
                inc_cuda_graph_eager_fallback(
                    embodiment=self._embodiment,
                    model_id=self._model_id,
                    reason="capture_failed",
                )
                logger.error(
                    "cuda_graph.capture_failed session=%s model=%s embodiment=%s exc=%s: %s",
                    self._session_name,
                    self._model_id,
                    self._embodiment,
                    type(exc).__name__,
                    exc,
                )
                raise
            elapsed = time.perf_counter() - t0
            observe_cuda_graph_capture_seconds(
                embodiment=self._embodiment,
                session=self._session_name,
                seconds=elapsed,
            )
            self._captured = True
            inc_cuda_graph_captured(
                embodiment=self._embodiment,
                model_id=self._model_id,
                session=self._session_name,
            )
            inc_cuda_graph_replayed(
                embodiment=self._embodiment,
                model_id=self._model_id,
                session=self._session_name,
            )
            logger.info(
                "cuda_graph.captured session=%s model=%s embodiment=%s elapsed_ms=%.1f",
                self._session_name,
                self._model_id,
                self._embodiment,
                elapsed * 1000,
            )
            return result

        # Replay path
        t0 = time.perf_counter()
        try:
            result = self._session.run(output_names, input_feed)
        except Exception as exc:
            inc_cuda_graph_eager_fallback(
                embodiment=self._embodiment,
                model_id=self._model_id,
                reason="replay_failed",
            )
            logger.error(
                "cuda_graph.replay_failed session=%s model=%s embodiment=%s exc=%s: %s",
                self._session_name,
                self._model_id,
                self._embodiment,
                type(exc).__name__,
                exc,
            )
            raise
        elapsed = time.perf_counter() - t0
        observe_cuda_graph_replay_seconds(
            embodiment=self._embodiment,
            session=self._session_name,
            seconds=elapsed,
        )
        inc_cuda_graph_replayed(
            embodiment=self._embodiment,
            model_id=self._model_id,
            session=self._session_name,
        )
        return result

    def invalidate(self) -> None:
        """Reset the captured flag so the next .run() is treated as a capture.

        Note: this does NOT rebuild the underlying ORT session. Callers are
        expected to construct a fresh session (e.g., after a model swap) and
        pass it to a new CudaGraphWrapper. This method exists mainly for test
        fixtures that want to simulate an invalidation event without rebuilding.
        """
        self._captured = False
        logger.info(
            "cuda_graph.invalidated session=%s model=%s embodiment=%s",
            self._session_name,
            self._model_id,
            self._embodiment,
        )


class EagerSessionWrapper:
    """Transparent wrapper around an eager ORT session that exposes the same
    `.run()` / `.session` / `.captured` / `.invalidate()` API as CudaGraphWrapper.

    Returned by `try_capture_or_fall_back()` when cuda-graph capture fails at
    init time (e.g., OOM on A10G's 24 GB envelope for vlm_prefix). Callers
    don't have to branch on wrapper type — both respond to the same surface.

    `.captured` always returns False; `.invalidate()` is a no-op (there's no
    captured graph to invalidate).
    """

    __slots__ = ("_session", "_session_name", "_embodiment", "_model_id")

    def __init__(
        self,
        session: "ort.InferenceSession",
        session_name: str,
        embodiment: str,
        model_id: str,
    ):
        if session_name not in VALID_SESSION_NAMES:
            raise ValueError(
                f"session_name must be one of {sorted(VALID_SESSION_NAMES)}, got {session_name!r}"
            )
        self._session = session
        self._session_name = session_name
        self._embodiment = embodiment
        self._model_id = model_id

    @property
    def session(self) -> "ort.InferenceSession":
        return self._session

    @property
    def captured(self) -> bool:
        return False

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> list:
        return self._session.run(output_names, input_feed)

    def invalidate(self) -> None:
        pass  # no captured graph to invalidate


def _make_probe_feed(session: "ort.InferenceSession", seed: int = 0) -> dict[str, Any]:
    """Generate a synthetic feed dict matching `session`'s declared input
    names + shapes + dtypes. Used by `try_capture_or_fall_back()` to probe
    whether the session can capture without waiting for a real request.

    Any dynamic dim (None or symbolic) defaults to 1. Tether's decomposed
    exports are static-shape per ADR 2026-04-21, so dynamic dims should not
    appear in practice.
    """
    rng = np.random.default_rng(seed)
    feed: dict[str, Any] = {}
    for inp in session.get_inputs():
        shape = [1 if (isinstance(d, str) or d is None) else int(d) for d in inp.shape]
        dtype = _ORT_TO_NUMPY_DTYPE.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            feed[inp.name] = (rng.integers(0, 2, size=shape) > 0)
        else:
            feed[inp.name] = rng.integers(0, 100, size=shape, dtype=dtype)
    return feed


def try_capture_or_fall_back(
    session_factory: Callable[[bool], "ort.InferenceSession"],
    session_name: str,
    embodiment: str,
    model_id: str,
    probe_feed: Mapping[str, Any] | None = None,
) -> Union[CudaGraphWrapper, EagerSessionWrapper]:
    """Build a session with cuda_graph=True and probe-capture it via a synthetic
    forward. On success return a CudaGraphWrapper wrapping the captured session.
    On capture failure (OOM, unsupported op, etc.), emit a
    `tether_cuda_graph_capture_failed_at_init_total` metric, then rebuild the
    session with cuda_graph=False and return an EagerSessionWrapper.

    Used for graceful-degrade on hardware tiers where some sessions cannot
    capture due to memory constraints (e.g., A10G 24 GB for `vlm_prefix`).
    Distinct from in-request eager fallback (which uses
    `tether_cuda_graph_eager_fallback_total` for replay-time failures).

    Per ADR 2026-04-24-cuda-graphs-architecture decision #5 (post-spike
    refinement): Phase 1 ships tier-aware semantics — A100+ gets both
    sessions captured; A10G gets expert_denoise captured + vlm_prefix
    gracefully degraded to eager (surfaced via metric, not silent).

    Args:
        session_factory: Callable that takes `cuda_graphs_enabled: bool` and
            returns an `ort.InferenceSession`. Same factory is invoked twice
            on the capture-failure path (once with True, once with False),
            so it should be idempotent.
        session_name: "vlm_prefix" or "expert_denoise".
        embodiment: Prometheus label (e.g., "franka" / "so100" / "ur5").
        model_id: Prometheus label (e.g., "pi05-decomposed-libero").
        probe_feed: Optional explicit synthetic feed. If None, one is generated
            from the session's `get_inputs()` metadata.

    Returns:
        CudaGraphWrapper on capture success, EagerSessionWrapper on failure.
        Both expose the same `.run()` API so callers don't branch on type.
    """
    try:
        cg_session = session_factory(True)
        feed = probe_feed if probe_feed is not None else _make_probe_feed(cg_session)
        t0 = time.perf_counter()
        _ = cg_session.run(None, feed)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        logger.warning(
            "cuda_graph.capture_failed_at_init session=%s model=%s embodiment=%s "
            "reason=%s: %s — falling back to eager session for this process lifetime",
            session_name, model_id, embodiment, type(exc).__name__, exc,
        )
        inc_cuda_graph_capture_failed_at_init(
            embodiment=embodiment, model_id=model_id,
            session=session_name, reason=type(exc).__name__,
        )
        eager_session = session_factory(False)
        return EagerSessionWrapper(
            eager_session,
            session_name=session_name,
            embodiment=embodiment,
            model_id=model_id,
        )

    # Capture succeeded. Wrap with already_captured=True so subsequent .run()s
    # go through replay semantics (the caller does the capture path once here).
    observe_cuda_graph_capture_seconds(
        embodiment=embodiment, session=session_name, seconds=elapsed,
    )
    inc_cuda_graph_captured(
        embodiment=embodiment, model_id=model_id, session=session_name,
    )
    logger.info(
        "cuda_graph.captured_at_init session=%s model=%s embodiment=%s elapsed_ms=%.1f",
        session_name, model_id, embodiment, elapsed * 1000,
    )
    return CudaGraphWrapper(
        cg_session,
        session_name=session_name,
        embodiment=embodiment,
        model_id=model_id,
        already_captured=True,
    )
