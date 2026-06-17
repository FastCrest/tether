"""OTel tracing for the serve runtime.

Optional. If `opentelemetry-sdk` + `opentelemetry-exporter-otlp` aren't
installed (i.e. the `[tracing]` extra isn't in the env), `setup_tracing()`
no-ops and `get_tracer()` returns a no-op tracer. Server behavior is
unchanged in either case.

Wire-up:
    from tether.runtime.tracing import setup_tracing, get_tracer
    setup_tracing(service_name="tether", endpoint="localhost:4317")
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("act") as span:
        span.set_attribute("gen_ai.operation.name", "act")
        ...

Phoenix as the local backend:
    pip install arize-phoenix
    phoenix serve            # UI on :6006, OTLP gRPC on :4317
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRACING_AVAILABLE: bool | None = None
_TRACER_PROVIDER = None


def _check_otel_available() -> bool:
    global _TRACING_AVAILABLE
    if _TRACING_AVAILABLE is not None:
        return _TRACING_AVAILABLE
    try:
        import opentelemetry  # noqa: F401
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: F401
            OTLPSpanExporter,
        )
        _TRACING_AVAILABLE = True
    except ImportError:
        _TRACING_AVAILABLE = False
    return _TRACING_AVAILABLE


def setup_tracing(
    service_name: str = "tether",
    endpoint: str | None = None,
    sample_rate: float = 1.0,
) -> bool:
    """Initialize an OTLP-gRPC tracer provider. Idempotent.

    Returns True if tracing was set up, False if the optional deps aren't
    installed (logged at INFO level — not an error).

    Args:
        service_name: `service.name` resource attribute.
        endpoint: OTLP gRPC endpoint. Defaults to `OTEL_EXPORTER_OTLP_ENDPOINT`
            env var or `localhost:4317` (the Phoenix dev default).
        sample_rate: fraction of spans to sample (0.0-1.0). Default 1.0
            (sample everything). Use < 1.0 for high-traffic production
            deploys; the OTel SemConv doc recommends 0.1 as a starting
            point (per otel-genai-spans.md spec).
    """
    global _TRACER_PROVIDER

    if not _check_otel_available():
        logger.info(
            "OTel tracing skipped — `pip install fastcrest-tether[tracing]` to enable."
        )
        return False

    if _TRACER_PROVIDER is not None:
        return True  # already initialized

    if not (0.0 <= sample_rate <= 1.0):
        raise ValueError(
            f"sample_rate must be in [0.0, 1.0], got {sample_rate}"
        )

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

    endpoint = endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"
    )
    resource = Resource.create({"service.name": service_name})
    # Sampler: ratio-based under a parent-based shell so child spans inherit
    # the root's sampling decision (avoids partial traces).
    sampler = ParentBased(root=TraceIdRatioBased(sample_rate))
    provider = TracerProvider(resource=resource, sampler=sampler)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    logger.info(
        "OTel tracing initialized — service=%s endpoint=%s sample_rate=%.3f",
        service_name, endpoint, sample_rate,
    )
    return True


def get_tracer(name: str):
    """Return an OTel tracer (real if setup_tracing succeeded, no-op otherwise)."""
    if not _check_otel_available():
        return _NoopTracer()
    from opentelemetry import trace
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush + shut down the tracer provider. Call from server lifespan exit."""
    global _TRACER_PROVIDER
    if _TRACER_PROVIDER is None:
        return
    try:
        _TRACER_PROVIDER.shutdown()
    except Exception as e:
        logger.warning("Tracing shutdown failed: %s", e)
    _TRACER_PROVIDER = None


class _NoopSpan:
    def set_attribute(self, *a, **kw): pass
    def set_attributes(self, *a, **kw): pass
    def add_event(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
    def end(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _NoopTracer:
    def start_as_current_span(self, *a, **kw): return _NoopSpan()
    def start_span(self, *a, **kw): return _NoopSpan()
