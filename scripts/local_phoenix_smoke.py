"""Smoke test: emit one OTel span to a local Phoenix instance.

Doesn't load any VLA — just verifies the tracing module wires up correctly,
the exporter connects, and a span lands in Phoenix.

Run:
    /Users/romirjain/tools/phoenix-env/bin/phoenix serve   # in another shell
    .venv/bin/python scripts/local_phoenix_smoke.py
    open http://localhost:6006   # confirm "tether" project + an "act" span
"""
from __future__ import annotations

import sys
import time

from tether.runtime.tracing import get_tracer, setup_tracing, shutdown_tracing


def main() -> int:
    ok = setup_tracing(service_name="tether", endpoint="localhost:4317")
    if not ok:
        print("FAIL: setup_tracing returned False (extras not installed)")
        return 1
    print("OK: tracing initialized")

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("act") as span:
        span.set_attribute("gen_ai.operation.name", "act")
        span.set_attribute("gen_ai.request.model", "smoke-test/synthetic")
        span.set_attribute("tether.instruction", "pick up the red cup")
        span.set_attribute("tether.state_dim", 6)
        span.set_attribute("tether.image_bytes", 12345)
        time.sleep(0.05)  # simulate inference
        span.set_attribute("tether.inference_ms", 50.0)
        span.set_attribute("tether.inference_mode", "smoke-test")
        span.set_attribute("tether.action_chunk_len", 50)
        print("OK: span emitted")

    shutdown_tracing()
    print("OK: tracing flushed")
    print()
    print("Open http://localhost:6006 — look for project 'tether' with one 'act' span.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
