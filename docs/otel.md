# OpenTelemetry tracing

`tether serve` emits OpenTelemetry GenAI-semantic-convention spans for every `/act` call. Point Phoenix, Datadog, Honeycomb, New Relic, or any OTLP-compatible backend at it — no integration code on your side.

## Install

```bash
pip install 'fastcrest-tether[tracing]'
```

Without the `[tracing]` extra, tracing no-ops silently; the server emits nothing and costs nothing. Your serve behavior is unchanged.

## Quick start — Phoenix (local dev)

Phoenix is a free OSS trace UI; the easiest way to see what Tether emits.

```bash
# Terminal 1 — Phoenix UI on :6006, OTLP gRPC collector on :4317
pip install arize-phoenix
phoenix serve

# Terminal 2 — Tether pointing at it
tether serve ./my-export/ --otel-endpoint localhost:4317 --otel-sample 1.0
```

Hit `/act` and open `http://localhost:6006` — every request shows up as an `act` span with all the robotics attributes below.

## Quick start — Datadog

```bash
# Datadog Agent exposes OTLP on :4317 when you enable receivers.otlp in datadog.yaml
tether serve ./my-export/ \
    --otel-endpoint localhost:4317 \
    --otel-sample 0.1
```

Spans land under `service:tether` in APM.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--otel-endpoint` | `$OTEL_EXPORTER_OTLP_ENDPOINT` or `localhost:4317` | OTLP gRPC endpoint (collector, Phoenix, Datadog Agent, etc.) |
| `--otel-sample` | `1.0` | Fraction of traces to sample. Parent-based — child spans inherit the root's decision, so you never see partial traces. |

### Sampling guidance

- **Dev / staging:** `--otel-sample 1.0` (sample everything, see every request)
- **Production, low traffic (<100 RPS):** `--otel-sample 1.0` is fine — OTel's `BatchSpanProcessor` decouples export from the hot path
- **Production, high traffic (>1000 RPS):** `--otel-sample 0.1` per OTel GenAI SemConv recommendation. Attribute serialization still happens on every request, but export batching overhead scales down linearly.

You can also set `OTEL_EXPORTER_OTLP_ENDPOINT` as an env var if you prefer env-based config over CLI flags.

## Attribute reference

Every `/act` call produces one root span named `act`.

### Standard OTel GenAI attributes

| Attribute | Type | Value |
|---|---|---|
| `gen_ai.operation.name` | string | `"act"` |
| `gen_ai.request.model` | string | Export directory path (post policy-versioning, this becomes the routing slot) |

### Robotics extensions (Tether-specific, proposed upstream Phase 2)

| Attribute | Type | Value |
|---|---|---|
| `gen_ai.action.embodiment` | string | Embodiment preset name (`franka`, `so100`, …) or `"custom"` |
| `gen_ai.action.chunk_size` | int | Number of actions returned in this chunk |
| `gen_ai.action.denoise_steps` | int | Diffusion denoise iterations (when adaptive-denoise turbo is active) |

### Tether-specific (prefix intentional; not for upstream)

| Attribute | Type | Value |
|---|---|---|
| `tether.instruction` | string | First 512 chars of the instruction text (truncated — never raw bytes) |
| `tether.state_dim` | int | Length of the proprio-state vector |
| `tether.image_bytes` | int | Size of the posted image in bytes (**size only** — never the image data) |
| `tether.rtc.episode_id` | string | Episode identifier when RTC adapter is active |
| `tether.rtc.episode_reset` | bool | True on the first span of a new episode |
| `tether.inference_mode` | string | `"decomposed"`, `"monolithic"`, `"native"`, etc. |
| `tether.action_chunk_len` | int | Same as `gen_ai.action.chunk_size`; kept for backward compat. |
| `error.type` | string | Populated only when the server returned an error result |

## PII and security

Tether explicitly **does not** attribute:

- Raw image bytes (could contain faces, text, identifiers on screens)
- API keys or auth headers
- Raw state vectors when they exceed 512 dims (use `tether.state_dim` for size only)

Instructions are truncated to 512 chars. If your instructions carry secrets, filter them client-side before calling `/act`.

## What's not emitted yet (Phase 1.5)

- Per-sub-op child spans (`vlm_prefix`, `expert_denoise`, `rtc_eval`) linked to the `act` root. Useful for root-causing which stage dominated latency.
- `gen_ai.usage.input_tokens` analog computed from `image_bytes` + instruction token count (needs a per-tokenizer calibration table).

File a GitHub issue if either is blocking an integration.

## Troubleshooting

**"Tracing skipped — pip install fastcrest-tether[tracing] to enable"**
The `[tracing]` extra isn't installed. Run `pip install 'fastcrest-tether[tracing]'` and restart serve.

**Spans appear in Phoenix but not in Datadog**
Your Datadog Agent probably has `receivers.otlp` disabled. Enable OTLP gRPC on :4317 in `datadog.yaml`, or point `--otel-endpoint` at an OTel Collector that forwards to Datadog.

**`--otel-sample 0.1` but I see every trace**
Parent-based sampling inherits from incoming parent context. If an upstream client is propagating a sampled trace context (via W3C traceparent), Tether honors it. Either disable propagation upstream or rely on the root sampler by ensuring `/act` spans are the root.

**Export slowing down `/act`**
`BatchSpanProcessor` is async — export failures shouldn't bubble to request latency. If you see correlation, drop `--otel-sample` or check collector health.
