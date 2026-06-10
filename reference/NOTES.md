# Reflex-VLA Runtime: Reference Implementation Audit

**Date**: April 22, 2026  
**Goal**: Compare what we have in `src/reflex/runtime/` against best-in-class implementations from reference repos (vllm, triton, tgi, ray-serve, trtllm, lerobot, openpi). Produce actionable "what to grep, copy, build" guidance for each category.

---

## 1. KV Cache / Prefix Caching

### Why it matters
Episode-aware caching is our #1 serve differentiation. VLAs operate on fixed-size action chunks; identical image+instruction pairs should reuse the VLM prefix cache to avoid re-computing vision encodings. This is key to competitive throughput on repeated observations.

### What we have now
- **`src/reflex/runtime/server.py:451-493`** — `VLMPrefixOrchestrator` loads 4-file ONNX pipeline (vision_encoder, text_embedder, decoder_prefill, state_encoder). Computes prefix_kv once per request in `predict()` at line 651-663.
- **Limitation**: No caching layer. Each inference re-runs the prefix pipeline even if image+instruction are identical. No hash-based lookup or block-level cache tracking.
- **`src/reflex/runtime/vlm_orchestrator.py:1-200+`** — Full VLM orchestrator implementation. Does pipeline management, not caching.

### Best reference impl
- **vLLM v1**: `/reference/vllm/vllm/v1/core/kv_cache_manager.py` (likely ~2000 LOC) + `/reference/vllm/vllm/v1/core/scheduler.py:35-92` shows the cache config and metrics collection.
  - Uses block-level KV cache with reuse detection via `BlockHash` (mentioned at `/reference/vllm/vllm/v1/request.py:29`).
  - Tracks `PrefixCacheStats` in `/reference/vllm/vllm/v1/metrics/stats.py:18-100` with hit rate over rolling window.
  - Test at `/reference/vllm/tests/v1/core/test_prefix_caching.py` and reset logic in `/reference/vllm/tests/v1/core/test_reset_prefix_cache_e2e.py`.
- **LeRobot**: No KV cache (single-image policy eval), but `/reference/lerobot/src/lerobot/policies/xvla/modeling_xvla.py:64` shows `self.chunk_size` tracking for stateful action generation.

### Why it's relevant
Caching identical prefixes avoids re-computing vision embedding (typically 50-200ms) when the robot sees the same scene twice. For a robot looping at 10Hz on a static scene, every 3-4 frames are identical. Prefix cache would cut inference latency 30-50% in those cases.

### Effort to adapt
- **Medium** (port the pattern)  
  Design: add a `PrefixCacheManager` class that:
  - Hashes (image, instruction) → 64-bit signature
  - Stores recent (hash → prefix_kv) entries in an LRU of size ~100
  - In `predict()`, check cache before calling `self._vlm.run()`
  - Track cache hit rate in response telemetry
  - Vllm's `/reference/vllm/vllm/v1/request.py:71,95` shows how they pass `cache_salt` and `block_hasher` — a good pattern to steal.

### Priority for our roadmap
**HIGH** — This is one of our explicit wedge goals (Phase I.2). Implement in v0.2 as a single-layer episode-aware prefix cache before moving to vllm-style block-level caching.

---

## 2. Continuous Batching

### Why it matters
We currently batch multiple HTTP /act requests into a single ONNX inference (Phase III). vLLM and TGI both use dynamic queue-draining with timeout to batch arbitrary-sized requests without waiting for the queue to fill.

### What we have now
- **`src/reflex/runtime/server.py:770-943`** — Full async batching implementation:
  - `start_batch_worker()` spawns an asyncio task that drains a queue every `batch_timeout_ms`.
  - `predict_async()` routes requests into the queue.
  - `_batch_worker_loop()` collects up to `max_batch` items within the timeout, then calls `_predict_batch_sync()`.
  - Tracks `_batches_run`, `_batched_requests` for telemetry.
  - **Limitation**: Same VLM conditioning for all batch items (line 874 comment). Per-item image/instruction lands in Phase II.4.
  - No continuous batching of *prepare* phase; all denoising happens in parallel in a single ONNX call.

### Best reference impl
- **vLLM v1 async scheduler**: `/reference/vllm/vllm/v1/core/sched/async_scheduler.py` (read first few hundred lines) — async queue draining with dynamic scheduling.
- **vLLM v1 batching**: `/reference/vllm/vllm/v1/worker/ubatching.py` — "micro-batching" for heterogeneous request sizes (prompt lengths).
- **TGI Neuron backend**: `/reference/tgi/backends/neuron/tests/server/test_continuous_batching.py` shows the continuous batching test pattern.
- **Ray Serve**: `/reference/ray/doc/source/serve/doc_code/tutorial_batch.py` shows the batching decorator pattern (much simpler than our async loop).

### Why it's relevant
Our 2.88x throughput gain (Phase III bench, Apr 14) comes from batching 4 requests in one ONNX call. vLLM generalizes this to arbitrary batch sizes with per-token scheduling (more complex). We're already doing the right thing; no urgent changes needed.

### Effort to adapt
- **Small** (already implemented)  
  Current implementation is sound. Improvements would be:
  - Support per-item image/instruction in batch (requires 4-file VLM pipeline to run per-item; likely 2-3x latency hit for now).
  - Support dynamic batch sizing (e.g., skip inference if batch is empty for >10ms). vLLM's async scheduler does this via priorities.

### Priority for our roadmap
**MEDIUM** — What we have works. Defer per-item VLM conditioning to Phase II.4 as currently planned.

---

## 3. Auth + Middleware

### Why it matters
Production serving needs authentication (API keys), request logging, rate limiting. TGI and Ray Serve both have patterns worth stealing.

### What we have now
- **`src/reflex/runtime/server.py:1134-1148`** — Simple API key auth via header:
  - `_require_api_key()` dependency checks `X-Reflex-Key` header.
  - Missing or wrong key → 401.
  - `/health` skips auth (good for load balancers).
  - No rate limiting, no request ID tracking, no structured logging middleware.

### Best reference impl
- **TGI**: `/reference/tgi/backends/neuron/server/text_generation_server/server.py:52-80` uses gRPC with `ExceptionInterceptor` (line 68). For REST, check `/reference/tgi` for Python FastAPI middleware (likely in a separate file).
- **Ray Serve**: `/reference/ray/python/ray/_private/authentication/grpc_authentication_server_interceptor.py` shows token validation pattern for gRPC. For HTTP, the docs show middleware via FastAPI.
- **vLLM**: Uses OpenAI-compatible API with simple bearer token auth in the frontend server (not shown in this ref repo snapshot).

### Why it's relevant
Current header-based auth is fine for MVP. Production needs:
- Request tracing (per-request ID → logs).
- Rate limiting (per-API-key, global).
- Structured logging (JSON for log aggregation).

### Effort to adapt
- **Small** (copy verbatim with attribution)  
  We already have the pattern. To add request tracing:
  ```python
  # In FastAPI app, add middleware before other routes:
  @app.middleware("http")
  async def add_request_id(request, call_next):
      request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
      logger.debug(f"req_id={request_id} method={request.method} path={request.url.path}")
      response = await call_next(request)
      response.headers["X-Request-ID"] = request_id
      return response
  ```

### Priority for our roadmap
**MEDIUM** — Implement request tracing in v0.2. Rate limiting can wait until v0.3 if we have per-key usage tracking in the backend.

---

## 4. Hot-Reload / Model Swap

### Why it matters
Swap a loaded model for a new one (e.g., upgrade pi0 → pi0.5) without restarting the server. Ray Serve and Kubernetes do this via versioning and canary deployments.

### What we have now
- **Nothing yet**. Current design requires restarting the server to load a new model.
- Workaround: use a load balancer to kill old replicas and spin up new ones, but that's infrastructure-level, not server-level.

### Best reference impl
- **Ray Serve**: `/reference/ray/python/ray/serve/deployment.py:68-100` shows the `Deployment` and `Application` model, which supports zero-downtime rollouts via versioning.
  - `/reference/ray/doc/source/serve/doc_code/deployment_version.py` (not in snap, but referenced in tests) shows version management.
  - Key pattern: versioned deployments can be updated while old versions drain requests.
- **Kubernetes native**: Not in refs, but the pattern is StatefulSet with rolling updates. Triton's deployment examples in `/reference/triton/deploy/` likely show this.

### Why it's relevant
For a prod robot fleet, being able to push a new model without downtime is crucial. Current restart path means 30s-2m of no inference while the new model JITs/warms up.

### Effort to adapt
- **Large** (need to redesign for VLA)  
  Requires:
  - Wrap the ReflexServer instance in a versioned handle.
  - During a version swap, drain in-flight requests to the old version, then unload model, load new model, mark as ready.
  - Surface this via a new HTTP endpoint `/models/load` or `/models/swap?version=v0.2.0`.
  - Integrate with the batching queue to handle requests in flight during swap.
  - Test chaos: what happens if a request arrives during the swap? (Should queue or reject cleanly.)

### Priority for our roadmap
**MEDIUM** — Defer to v0.3 (Phase IV). Get single-model serving + batching + episode cache solid first.

---

## 5. Multi-Model Serving

### Why it matters
Triton's core value prop: serve many models on the same hardware via intelligent scheduling. We only serve one model per server instance.

### What we have now
- **Nothing**. Each server loads exactly one model (one export_dir → one ReflexServer instance).

### Best reference impl
- **Triton Inference Server**: The whole architecture (C++ backend, model repository structure, scheduler) is built around this. See `/reference/triton/` directory.
  - Key: `/reference/triton/CMakeLists.txt` and `/reference/triton/build.py` show the build system, but the real multi-model logic is in C++ (not included in this snapshot).
  - Triton's scheduler decides which GPU to route each model's request to, when to batch across models, etc.
- **vLLM**: Single-model only (like us). Uses LoRA adapters (`/reference/vllm/vllm/v1/request.py:70`) instead of full model swaps.

### Why it's relevant
A robot warehouse might run 10 different VLAs (pick, place, inspect, etc.). Multi-model serving lets one server handle all 10, auto-scaling based on load per model. This is a nice-to-have, not essential for a single-robot demo.

### Effort to adapt
- **Large** (need to redesign for VLA)  
  Requires:
  - Change server to load a model registry: `{model_name → ReflexServer}`.
  - Route /act requests to the right server by model name in the payload.
  - Manage memory: loading 5 VLAs simultaneously might OOM. Need LRU eviction.
  - Batch requests within each model, or across models if they have the same export format (unlikely for VLAs).

### Priority for our roadmap
**LOW** — Not on the critical path. Single-model serving is the MVP. Multi-model is Phase V at earliest.

---

## 6. Action-Chunk Handling + RTC (Real-Time Control)

### Why it matters
VLAs generate action chunks (e.g., 50-frame trajectory). Robots execute at 100-1000 Hz. We need to:
- Buffer the chunk and pop one action per tick (done in `/src/reflex/runtime/buffer.py`).
- Handle RTCs (soft constraints on action bounds, like "gripper < 1.0"). Robots may also want to interrupt a chunk mid-execution if new observations arrive.

### What we have now
- **`src/reflex/runtime/buffer.py`** — Full action chunk buffer implementation (thread-safe ring buffer, replan on demand). This is solid.
- **No RTC support yet**. We have safety clamping (`src/reflex/safety/...`) but not soft action constraints or mid-chunk interruption.

### Best reference impl
- **LeRobot**: `/reference/lerobot/src/lerobot/policies/xvla/action_hub.py` — Action space definitions (continuous, discrete, hybrid, normalized).
  - `/reference/lerobot/src/lerobot/policies/xvla/modeling_xvla.py:51-82` shows how the policy handles `chunk_size` and action space building.
  - `/reference/lerobot/src/lerobot/policies/xvla/processor_xvla.py` (not read yet) likely shows normalization and denormalization of actions.
  - No explicit RTC logic in the code; RTCs are typically baked into the training data or applied by the simulation/hardware abstraction layer.
- **OpenPI**: `/reference/openpi/` (not explored deeply). May have action space definitions.

### Why it's relevant
Our buffer is great, but we lack:
1. **Action denormalization**: VLA outputs normalized [-1, 1] actions. We need to denorm to robot joint space (depends on robot).
2. **RTC enforcement**: Some robots want actions soft-constrained (e.g., "gripper < 1.0 is ideal but clamp hard at 0.99"). This is different from our safety guard (hard clamps).
3. **Episode state tracking**: If the robot hits a hard limit mid-chunk (e.g., gripper hits closed), we should know to request a fresh plan. Currently the server doesn't get that feedback.

### Effort to adapt
- **Medium** (port the pattern)  
  Steps:
  1. Read `/reference/lerobot/src/lerobot/policies/xvla/action_hub.py` fully.
  2. Extract action space definitions (min/max per joint, continuous vs. discrete).
  3. Add action denormalization to `predict()` post-inference. Needs a config field in `tether_config.json` (action_space definition).
  4. For RTC: add soft constraint checking in a new `ActionRTC` class (similar to `ActionGuard`). Return violations as telemetry, let the robot decide to replan.

### Priority for our roadmap
**HIGH** — Action denormalization is essential for any real robot. Needed in v0.2. RTC/soft constraints can wait until v0.3 (Phase III.5) once we have real robot feedback loops.

---

## 7. Telemetry / Per-Request Metrics

### Why it matters
Production systems need structured telemetry: latency distribution, cache hit rate, safety violations, deadline misses. Triton exports Prometheus metrics. We return inline telemetry in the response.

### What we have now
- **`src/reflex/runtime/server.py:176-256`** — Per-request telemetry:
  - Latency percentiles (p50/p95/p99) over a 1024-sample rolling window (`_latency_history`, line 117).
  - Determinism fields (model_hash, config_hash, reflex_version, line 208-256).
  - Per-wedge metrics (safety_violations, deadline_exceeded, adaptive_steps_enabled, line 732-744).
  - Telemetry in batch responses (line 925-941).

### Best reference impl
- **vLLM v1**: `/reference/vllm/vllm/v1/metrics/stats.py` — Prefix cache hit rate tracking (line 35-100), SQL-like aggregation with rolling windows. Also `/reference/vllm/vllm/v1/metrics/perf.py` (not read, but named `PerfStats` and used in scheduler).
  - Integrates with observability via `KVCacheMetricsCollector` (line 88-91 of scheduler.py).
- **Triton**: Exports Prometheus-format metrics at `/metrics` endpoint (not shown in refs, but well-known).
- **LeRobot**: Uses `draccus` for structured config logging, but not deep observability.

### Why it's relevant
Current approach (inline in response) is fine for MVP. For prod:
- Ops teams want Prometheus metrics scraped at /metrics endpoint (one number per metric, not per-request).
- We need counters (total requests, cache hits, deadline misses) + histograms (latency, inference time, queue wait time).

### Effort to adapt
- **Small** (copy verbatim with attribution)  
  Add a `/metrics` endpoint that exports Prometheus format:
  ```python
  @app.get("/metrics")
  async def metrics():
      return PlainTextResponse(
          f"reflex_requests_total {server._request_count}\n"
          f"reflex_cache_hits_total {server._cache_hits}\n"
          f"reflex_latency_p50_ms {server._latency_history.percentile(50)}\n"
      )
  ```
  Or use `prometheus_client` library for automatic scraping.

### Priority for our roadmap
**MEDIUM** — Implement in v0.2 alongside request ID tracing. This is low-effort and high-value for prod diagnostics.

---

## 8. Episode / Session State Tracking

### Why it matters
A robot executing actions from an action chunk may hit a physical limit (gripper fully closed, arm joint limit). When that happens, the next VLA inference should know "this is a new episode, not a continuation of the prior one." Otherwise the VLM sees stale observations and the plan becomes invalid.

### What we have now
- **Nothing yet**. No session tracking.
- Workaround: robots can send a new instruction on each HTTP /act call, which resets the VLM context.

### Best reference impl
- **LeRobot**: `/reference/lerobot/src/lerobot/policies/xvla/modeling_xvla.py:24` uses `from collections import deque` and line 64 tracks `chunk_size`. Inside the policy, there's likely a `.reset()` method to clear history when an episode ends (not visible in the excerpt, but standard).
- **vLLM**: `/reference/vllm/vllm/v1/request.py:33-56` defines `StreamingUpdate` and `resumable: bool` (line 75). Used for session continuation in long-running tasks. The scheduler checks `request.resumable` to decide if a request can continue or needs fresh KV.

### Why it's relevant
For a real robot:
1. The robot executes the chunk until: (a) it runs out of actions, or (b) a limit is hit (gripper limit, joint limit, timeout, operator interrupt).
2. If (b), the robot should send a new /act call with a flag like `reset_session=True` or `episode_boundary=True`.
3. The server should then clear the action buffer and reset any per-request state (e.g., previous KV cache context).

### Effort to adapt
- **Medium** (port the pattern)  
  Steps:
  1. Add a `session_id` field to the `PredictRequest` pydantic model (line 968-971).
  2. Add an optional `reset_session` bool flag.
  3. In the server, track `_session_id` and `_session_start_time`.
  4. When `reset_session=True` or `session_id` changes, clear the action buffer and log the boundary.
  5. Return `session_id` in the response for the robot to echo back.

### Priority for our roadmap
**MEDIUM** — Needed for real robot deployments. Implement in v0.2 as part of the buffer replan work.

---

## 9. Real-Time Streaming / Chunked Response

### Why it matters
For action-chunk serving, we currently return the full chunk in one /act response. Some robots might want to stream actions as they're computed (streaming output for long-running inference) or stream intermediate state (e.g., progress of denoising loop).

### What we have now
- **Single response per /act call**. No streaming.
- Clients get the full action chunk once inference finishes.

### Best reference impl
- **TGI (Hugging Face Text Generation Inference)**: Designed for text streaming. `/reference/tgi/backends/neuron/server/text_generation_server/server.py` is gRPC-based (not HTTP streaming), but the logic is there.
  - HTTP streaming support is in the outer FastAPI layer (not in this snapshot).
- **Ray Serve**: `/reference/ray/python/ray/serve/tests/test_streaming_response.py` and `/reference/ray/doc/source/serve/doc_code/streaming_tutorial.py` show how to use `StreamingResponse` in FastAPI.
  - Pattern: return a generator and FastAPI chunks it.

### Why it's relevant
For a robot operating at 100 Hz, waiting for full chunk inference (50-500ms) before the first action is acceptable. Streaming is a nice-to-have for latency dashboards ("show denoising progress") but not critical.

### Effort to adapt
- **Small** (copy verbatim with attribution)  
  Add an optional query param `?stream=true` to /act:
  ```python
  @app.post("/act")
  async def act(request: PredictRequest, stream: bool = False, ...):
      if stream:
          async def generate():
              for i, action in enumerate(actions):
                  yield f'data: {{"action": {action}, "index": {i}}}\n'
              yield "data: [DONE]\n"
          return StreamingResponse(generate(), media_type="text/event-stream")
      else:
          # Current behavior: return all at once
  ```

### Priority for our roadmap
**LOW** — Not essential for v0.1. Defer to v0.3 if robots ask for it. Current batch/chunk latency is acceptable.

---

## Summary: Action Items by Priority

| Category | Status | Effort | Priority | Recommended Release |
|----------|--------|--------|----------|---------------------|
| KV cache / Prefix caching | Partial (orchestrator, no caching) | Medium | **HIGH** | v0.2 |
| Continuous batching | **Complete** | – | Medium | v0.1 ✓ |
| Auth + middleware | Partial (API key only) | Small | Medium | v0.2 |
| Hot-reload / model swap | Not started | Large | Medium | v0.3 |
| Multi-model serving | Not started | Large | **LOW** | v0.4+ |
| Action-chunk + RTC | Partial (buffer only, no denorm/RTC) | Medium | **HIGH** | v0.2 |
| Telemetry / metrics | Partial (inline, no /metrics) | Small | Medium | v0.2 |
| Episode / session state | Not started | Medium | Medium | v0.2 |
| Real-time streaming | Not started | Small | **LOW** | v0.3 |

---

## Implementation Roadmap

### v0.2 (Next, 4-6 weeks)
1. **Prefix cache layer** (`PrefixCacheManager`) — grep vllm `/reference/vllm/vllm/v1/core/kv_cache_manager.py` for block-level patterns; start with simple LRU.
2. **Action denormalization** — grep lerobot `/reference/lerobot/src/lerobot/policies/xvla/action_hub.py` + `processor_xvla.py`.
3. **Request tracing** (X-Request-ID header) + `/metrics` endpoint.
4. **Episode / session state** — add `session_id` to request/response.
5. **Safety RTC** (soft constraints) — separate from hard guard clamping.

### v0.3 (6-10 weeks later)
1. **Hot-reload / model swap** — versioned deployments (grep ray `/reference/ray/python/ray/serve/deployment.py`).
2. **Streaming /act responses** (optional).
3. **Per-item VLM conditioning in batches** (Phase II.4).

### v0.4+ (Future)
1. **Multi-model serving** (low priority unless demand).

---

## Key Files to Read / Grep

| Reference Repo | File Path | Why Read It |
|----------------|-----------|------------|
| vLLM | `vllm/v1/core/kv_cache_manager.py` | Prefix cache implementation |
| vLLM | `vllm/v1/core/sched/scheduler.py` | Async scheduling + KV metrics |
| vLLM | `vllm/v1/metrics/stats.py` | Caching metrics + telemetry |
| vLLM | `vllm/v1/request.py` | Session/streaming state + `StreamingUpdate` |
| LeRobot | `src/lerobot/policies/xvla/action_hub.py` | Action space definitions |
| LeRobot | `src/lerobot/policies/xvla/processor_xvla.py` | Action denormalization |
| LeRobot | `src/lerobot/policies/xvla/modeling_xvla.py` | Policy session / chunk size tracking |
| Ray Serve | `python/ray/serve/deployment.py` | Versioned deployments (hot-reload) |
| Ray Serve | `doc/source/serve/doc_code/streaming_tutorial.py` | HTTP streaming (low priority) |
| Ray Serve | `doc/source/serve/doc_code/tutorial_batch.py` | Batching patterns |
| TGI Neuron | `backends/neuron/server/text_generation_server/server.py` | gRPC server + interceptors |
| Triton | `/` (whole directory) | Multi-model serving architecture (large, low priority) |

---

## Copy-Paste: Strategic Wedges from Server.py

Keep these architectural wins from our current implementation:

- **Batching loop** (`server.py:770-943`): Clean async queue draining. Just add per-item VLM support later.
- **Latency tracking** (`server.py:176-206`): Rolling window + percentiles is production-ready.
- **Action buffer** (`buffer.py:1-200+`): Thread-safe ring buffer with replan heuristics is solid.
- **Wedge composition** (`server.py:283-331`): Safety + split orchestrator + adaptive steps plugged in cleanly. Keep this pattern for new RTC wedge.

---

## What NOT to Copy from References

- **Triton's C++ backend system**: Over-engineered for single-model VLA serving.
- **vLLM's full scheduler**: We don't need token-level scheduling; action chunks are fixed-size.
- **Ray Serve's full actor deployment system**: Too much infra for a single CLI. Use only if we go to multi-tenant serving (v0.4+).

---

## Sibling project: EasyInference (NOT a competitor — own project; pattern source for Phase 1)

**Path:** `/Users/romirjain/Desktop/building projects/EasyInference-main/`

EasyInference is a sibling open-source project (2 products: ISB-1 benchmark standard + InferScope CLI/MCP). It targets LLM inference (H100/H200/B200/GB200/MI300X/MI355X), not VLA inference. Different unit-of-work (tokens vs action chunks), different latency model (ms-per-token vs ms-per-chunk × control rate), different customer (chatbot ops vs robotics engineers). NOT cloned into `reference/` because the convention there is competitor-OSS-grep-first. This is a SIBLING — pattern source, not benchmark target.

**Phase 1 features that should lift InferScope patterns:**

| Reflex feature (Phase 1) | InferScope pattern to lift | VLA-specific adaptation |
|---|---|---|
| `reflex bench` revamp (D.7) | ISB-1 measurement methodology — warmup discard, p50/p95/p99 + tail latency, jitter calc, Markdown + JSON report | Per-chunk vs per-token; account for diffusion-loop denoise steps; flow-matching reproducibility seeds |
| `mcp-server` (planned) | InferScope MCP scaffolding — tools registration, transport wiring, prompt patterns | Expose `/act` as the tool (not `/chat/completions`); state + image inputs; episode_id in tool params |
| `latency-slo-enforcement` (--slo p99=Xms) | InferScope SLO violation rolling-window tracker + degradation strategy | Latency unit = chunk; threshold check pairs with `--adaptive-steps` + `--deadline-ms`; 503 + measured p99 in body |
| `model_resolver.py` cloud half (shipped) | ISB-1 hardware DB (H100/H200/B200/GB200/MI300X/MI355X cost + throughput) | Consume for cloud-side training-substrate decisions; keep our Jetson-side DB for edge decisions |

**Workflow when building any Phase 1 feature with an InferScope analog:**
1. Open the matching code under `EasyInference-main/products/{isb1,inferscope}/`
2. Internalize the design (notebook to scratch file)
3. Write Reflex's analogous version with VLA-aware semantics
4. **Do NOT vendor the code** — copy patterns, write fresh. Avoids two-codebases-out-of-sync drift.

**End of Audit. Next step: prioritize v0.2 features above and start implementation.**
