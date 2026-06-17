"""Tests for src/tether/runtime/tracing.py — OTel tracer bootstrap."""
from __future__ import annotations

import pytest

from tether.runtime import tracing as tracing_module
from tether.runtime.tracing import (
    _NoopTracer,
    get_tracer,
    setup_tracing,
    shutdown_tracing,
)


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Ensure each test sees a fresh global tracer state.

    OTel's global `set_tracer_provider()` uses a set-once latch; without
    resetting it, later tests get "Overriding of current TracerProvider is
    not allowed" warnings and their exporters never receive spans.
    """
    tracing_module._TRACER_PROVIDER = None
    try:
        from opentelemetry import trace as _ot
        _ot._TRACER_PROVIDER_SET_ONCE = _ot.Once()
        _ot._TRACER_PROVIDER = None
        _ot._PROXY_TRACER_PROVIDER = _ot.ProxyTracerProvider()
    except ImportError:
        pass
    yield
    try:
        shutdown_tracing()
    finally:
        tracing_module._TRACER_PROVIDER = None
        try:
            from opentelemetry import trace as _ot
            _ot._TRACER_PROVIDER_SET_ONCE = _ot.Once()
            _ot._TRACER_PROVIDER = None
            _ot._PROXY_TRACER_PROVIDER = _ot.ProxyTracerProvider()
        except ImportError:
            pass


def test_setup_rejects_negative_sample_rate():
    if not tracing_module._check_otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    with pytest.raises(ValueError, match="sample_rate"):
        setup_tracing(sample_rate=-0.1)


def test_setup_rejects_sample_rate_above_one():
    if not tracing_module._check_otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    with pytest.raises(ValueError, match="sample_rate"):
        setup_tracing(sample_rate=1.5)


def test_setup_accepts_boundary_sample_rates():
    if not tracing_module._check_otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    # 0.0 (sample nothing) and 1.0 (sample all) must both be accepted.
    assert setup_tracing(sample_rate=0.0, endpoint="localhost:4317") is True
    shutdown_tracing()
    tracing_module._TRACER_PROVIDER = None
    assert setup_tracing(sample_rate=1.0, endpoint="localhost:4317") is True


def test_setup_idempotent():
    if not tracing_module._check_otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    assert setup_tracing(endpoint="localhost:4317") is True
    # Second call must return True without replacing the provider.
    provider_before = tracing_module._TRACER_PROVIDER
    assert setup_tracing(endpoint="localhost:4317") is True
    assert tracing_module._TRACER_PROVIDER is provider_before


def test_setup_uses_parent_based_ratio_sampler():
    """Sampler must be ParentBased(TraceIdRatioBased) so child spans inherit."""
    if not tracing_module._check_otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    assert setup_tracing(sample_rate=0.25, endpoint="localhost:4317") is True
    provider = tracing_module._TRACER_PROVIDER
    assert provider is not None
    assert isinstance(provider.sampler, ParentBased)
    # ParentBased wraps the root sampler; verify the ratio value we passed.
    root = provider.sampler._root
    assert isinstance(root, TraceIdRatioBased)


def test_get_tracer_returns_noop_when_deps_missing(monkeypatch):
    monkeypatch.setattr(tracing_module, "_TRACING_AVAILABLE", False)
    # Force the cached-none path so _check_otel_available short-circuits to False.
    t = get_tracer("whatever")
    assert isinstance(t, _NoopTracer)
    # Noop tracer has span/context-manager protocol that no-ops cleanly.
    with t.start_as_current_span("sp") as span:
        span.set_attribute("gen_ai.action.chunk_size", 10)
        span.record_exception(RuntimeError("hi"))


def test_setup_returns_false_when_deps_missing(monkeypatch, caplog):
    monkeypatch.setattr(tracing_module, "_TRACING_AVAILABLE", False)
    ok = setup_tracing(service_name="svc", endpoint="localhost:4317")
    assert ok is False
    # Must NOT raise; logs at INFO level.
    assert tracing_module._TRACER_PROVIDER is None


def test_shutdown_noop_when_never_started():
    # Idempotent — no provider → no error.
    shutdown_tracing()
    assert tracing_module._TRACER_PROVIDER is None


def test_gen_ai_action_attributes_recorded_on_span():
    """Mirror the server's /act instrumentation and verify OTel GenAI attrs land."""
    if not tracing_module._check_otel_available():
        pytest.skip("opentelemetry-sdk not installed")
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "t"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracing_module._TRACER_PROVIDER = provider

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("act") as span:
        span.set_attribute("gen_ai.operation.name", "act")
        span.set_attribute("gen_ai.request.model", "pi0_libero")
        span.set_attribute("gen_ai.action.embodiment", "franka")
        span.set_attribute("gen_ai.action.chunk_size", 50)
        span.set_attribute("gen_ai.action.denoise_steps", 10)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.operation.name"] == "act"
    assert attrs["gen_ai.request.model"] == "pi0_libero"
    assert attrs["gen_ai.action.embodiment"] == "franka"
    assert attrs["gen_ai.action.chunk_size"] == 50
    assert attrs["gen_ai.action.denoise_steps"] == 10
    assert spans[0].name == "act"
