# The Wedge: Server Architecture

The Reflex inference server uses a **Wedge composition pattern** to safely, adaptively, and reliably run vision-language-action (VLA) models in production. Four orthogonal "wedges" (Safety, Split, Adaptive, Deadline) layer safety constraints, cloud failover, latency optimization, and hard time limits around the core denoise-and-predict loop.

> **Model layer (v0.10.0+):** the *model* the server runs on is composed via the **BaseVLA spine** — a 6-slot taxonomy (`vision_backbone`, `llm_backbone`, `vlm_backbone`, `projector`, `vla_head`, `text_encoder`) where every supported VLA is a thin (~100 LOC) `BaseVLA` subclass declaring which slots it uses. Pi0VLA / Pi05VLA / SmolVLA use the 2-tower split (vision_backbone + llm_backbone separate); GR00TVLA uses the fused `vlm_backbone` slot for Eagle. OpenVLA stays as a non-spine shim per decision S-4. See `src/reflex/models/vlas/` for the worked composition examples; the per-VLA `NAME_MAPPING` ClassVar handles HF-checkpoint key renames per decision S-1.

---

## Overview

When a client posts to `/act`, the server runs inference through this pipeline:

```
POST /act request
    ↓
[Denoise loop with Adaptive early-stopping]
    ↓
[Safety check: ActionGuard joint clamping + velocity caps]
    ↓
[Deadline enforcement: return last-good or zeros if over budget]
    ↓
[Optional: Split fallback to cloud if enabled]
    ↓
200 OK with actions + telemetry
```

Each wedge is **independently toggleable** via startup flags. This allows:

- **Development**: test a model without safety constraints
- **Production**: enable all wedges for bulletproof safety + resilience
- **Edge deployment**: skip Split (no cloud), enable Safety + Deadline
- **Cloud deployment**: prefer Split to edge, enable Adaptive for latency

---

## The Four Wedges

### 1. Safety Wedge (ActionGuard)

**Purpose**: Enforce per-axis joint position and velocity limits.

**What it does**:

- Receives the predicted action chunk from the model
- Clamps each axis to embodiment-specific joint ranges (`qmin`, `qmax`)
- Clamps velocities to per-axis limits (`v_max`)
- Detects and zeros NaN/Inf values
- Records violation counts for telemetry

**When violations occur**:

- **Mode `clamp` (default)**: Silently clamp and log. Client receives clamped actions and `safety_violations` count in the response.
- **Mode `reject`**: Return a zeroed action vector instead of the rejected action. The `/act` request still completes successfully rather than returning HTTP 400.
- **Trip mode**: After N consecutive clamps, trip the guard. While tripped, `/act` returns 200 with an error until explicitly reset via `POST /guard/reset`.

**Configuration**:

```bash
# Generate safety limits from robot URDF:
reflex guard init --urdf robot.urdf --output safety.json

# Start server with safety enforcement:
reflex serve ./my-export/ --safety-config safety.json
```

**Safety config schema** (`safety.json`):

```json
{
  "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", ...],
  "position_min": [-3.14, -2.0, 0.0, ...],
  "position_max": [3.14, 2.0, 3.14, ...],
  "velocity_max": [2.0, 1.5, 2.5, ...],
  "effort_max": [50.0, 50.0, 50.0, ...],
  "workspace_min": [-1.0, -1.0, 0.0],
  "workspace_max": [1.0, 1.0, 1.5]
}
```

**Telemetry**:

```json
{
  "safety_violations": 3,
  "safety_detail": ["action 5: 1 violations", "action 10: 2 violations"],
  "guard_summary": {
    "violations": [
      "joint_clamp:shoulder_pan",
      "joint_clamp:shoulder_pan",
      "non_finite:elbow_flex"
    ],
    "clamped": true,
    "clamp_count": 2
  }
}
```

---

### 2. Split Wedge (SplitOrchestrator)

**Purpose**: Configure cloud fallback integration for future split orchestration.

**What it does**:

- Initializes split orchestration when `--cloud-fallback` is set
- Exposes `split_enabled: true` in `/act` responses when configured
- Stores cloud fallback URL for future dispatch/routing support

**Current status**:

- Full edge/cloud request dispatch is not active in the current server path yet.
- Routing preferences and fallback strategies are planned for a future phase.

**Configuration**:

```bash
# Edge-first (cloud is optional):
reflex serve ./my-export/ --cloud-fallback http://cloud-vla:8000
```

**Split config** (internal, created by the server):

```python
SplitConfig(
    cloud_url="http://cloud-vla:8000",
)
```

**Telemetry**:

```json
{
  "split_enabled": true
}
```

---

### 3. Adaptive Wedge (TurboOptimizer)

**Purpose**: Reduce latency by early-stopping the denoise loop when velocity converges.

**What it does**:

- Tracks velocity norm across denoising steps
- Stops denoising early if velocity norm delta < 0.01 (configurable threshold)
- Typical savings: **20–50% latency reduction** on models that converge fast (e.g., pi0)
- No action quality loss when threshold is tuned per-model

**How it works**:

```
Step 0: noisy_actions ~ N(0, I)
Step 1: denoise(noisy_actions)
  → velocity = gradient of noisy_actions
  → v_norm = ||velocity||
Step 2: denoise(noisy_actions)
  → v_norm_new = ||gradient||
  → delta = |v_norm_new - v_norm|
  → if delta < 0.01: STOP, use Step 2 result
  → else: continue
...
Step N: final denoise step (if not converged)
```

**Model-specific performance** (Phase 4 benchmarks, 2026-04-14):

- **pi0** ✅ **58% latency savings**, action diff 0.07 (imperceptible)
- **smolvla** ❌ Never triggers (already fast)
- **pi0.5** ⚠️ Rarely triggers (inconsistent benefit)
- **gr00t** ❌ Triggers too aggressively (action diff 0.67, meaningful drift)

**Recommendation**: Use `--adaptive-steps` only with `pi0` or models validated in your benchmarks. Per-model threshold tuning lands in v0.2.

**Configuration**:

```bash
# Enable adaptive denoising:
reflex serve ./my-export/ --adaptive-steps

# Monitor convergence at runtime:
# (telemetry field denoising_steps will vary per request)
```

**Telemetry**:

```json
{
  "denoising_steps": 7, // < full 10-step default
  "adaptive_enabled": true,
  "latency_ms": 42.5 // faster than fixed-step baseline
}
```

---

### 4. Deadline Wedge

**Purpose**: Guarantee a response within a strict time budget (e.g., "return by 100ms or die trying").

**What it does**:

- Measures elapsed time from request start to just before returning
- If elapsed > deadline, returns either:
  - **Last good action** when a prior successful inference result is cached
  - **Zero vector** when no prior successful action has been cached yet
- Logs the deadline miss for alerting
- Records miss count for SLO tracking

**Why this matters**:

- Robot controllers often have hard real-time constraints (e.g., 10 Hz control loop = 100 ms per cycle)
- Missing a deadline is better than hanging; client can retry or use fallback action
- Helps isolate stuck inference from crashing the whole robot

**Configuration**:

```bash
# Hard 100ms deadline:
reflex serve ./my-export/ --deadline-ms 100

# Deadline fallback still enforces the budget (returns last-good/zeros on miss):
reflex serve ./my-export/ --deadline-ms 200  # with alerting
```

**Telemetry**:

```json
{
  "latency_ms": 152.4,
  "deadline_exceeded": true,
  "deadline_misses_total": 3,
  "reason": "returned last-good action"
}
```

---

## Full /act Request Flow

Here's a detailed trace of a request through all wedges:

```
1. Client POST /act
   { "image": "base64...", "instruction": "pick up cup", "state": [0.1, 0.2, ...] }

2. Server starts timer and OTel span

3. API authentication & validation (header/key checks)

4. Policy routing (single-policy or 2-policy dispatcher)

5. **DENOISE LOOP** (with optional Adaptive early-stop)
   for step in 0..num_denoising_steps:
       noisy_actions = euler_step(noisy_actions, ...)
       if adaptive_steps and step >= 2:
           v_norm = ||gradient(noisy_actions)||
           if |v_norm - prev_v_norm| < 0.01:
               BREAK  # converged

6. **SAFETY CHECK** (ActionGuard)
   if safety_config is set:
       safe_actions, violations = guard.check(actions)
       if violations > 0:
           log "safety_violations: {count}"
           if mode == "reject": actions = zeros_like(actions)  # still 200 OK
           if max_consecutive_clamps exceeded: return guard_tripped error
       actions = safe_actions

7. **DEADLINE CHECK**
   elapsed = time.perf_counter() - start
   if deadline_ms and elapsed > deadline_ms:
       if last_good_actions exists:
           actions = last_good_actions
       else:
           actions = zeros(shape)
       deadline_exceeded = true

8. **SPLIT FALLBACK** (if cloud_fallback_url set)
   if actions is empty or error occurred:
       if split_orchestrator.cloud_available():
           actions = await split_orchestrator.infer_cloud(...)
       else:
           actions = split_orchestrator.fallback_actions()

9. **TELEMETRY & HOOKS** (A2C2 correction, RTC merge, etc.)
   - Record latency percentiles
   - Apply A2C2 residual correction (if enabled)
   - Merge into RTC buffer (if enabled)
   - Write JSONL record

10. **RESPONSE**
    200 OK {
      "actions": [[a0, a1, ...], [a0, a1, ...], ...],
      "num_actions": 50,
      "action_dim": 7,
      "latency_ms": 42.5,
      "hz": 23.5,
      "denoising_steps": 7,
      "safety_violations": 0,
      "deadline_exceeded": false,
      "model_hash": "abc123...",
      "config_hash": "def456...",
      ...
    }
```

---

## Configuration Reference

### Startup Flags

| Flag                      | Type  | Default     | Description                                                                                           |
| ------------------------- | ----- | ----------- | ----------------------------------------------------------------------------------------------------- |
| `--device`                | str   | `cuda`      | `cuda` or `cpu` — default execution provider                                                          |
| `--num-denoising-steps`   | int   | `10`        | Fixed denoise steps (without `--adaptive-steps`)                                                      |
| `--providers`             | str   | auto-detect | Explicit ORT execution providers, comma-separated. E.g., `CUDAExecutionProvider,CPUExecutionProvider` |
| `--no-strict-providers`   | bool  | `false`     | Disable strict provider loading and allow CPU fallback if requested providers fail to load |
| `--safety-config`         | path  | `None`      | Path to `safety.json` from `reflex guard init`. Enables ActionGuard.                                  |
| `--adaptive-steps`        | bool  | `false`     | Enable TurboOptimizer velocity convergence early-stop                                                 |
| `--cloud-fallback`        | str   | `""`        | Cloud endpoint URL (e.g., `http://cloud-vla:8000`). Enables Split setup.                              |
| `--deadline-ms`           | float | `0.0`       | Deadline in milliseconds. Default `0.0` means disabled (internally treated as `None`).                |
| `--max-batch`             | int   | `1`         | ⚠️ **Deprecated**. Use `--max-batch-cost-ms` instead.                                                 |
| `--max-batch-cost-ms`     | float | `100`       | GPU-ms cost budget per batch flush (see `docs/batching.md`)                                           |
| `--batch-timeout-ms`      | float | `5.0`       | Max time a request waits in queue before forced flush                                                 |

### Runtime APIs

#### GET `/health`

Health check (usable without API key).

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "ok",
  "state": "ready",
  "model_loaded": true,
  "inference_mode": "onnx_cuda",
  "export_dir": "/models/pi0-export",
  "vlm_loaded": true,
  "consecutive_crashes": 0,
  "max_consecutive_crashes": 5,
  "robot_id": ""
}
```

#### POST `/act`

Main inference endpoint (requires API key).

Request:

```json
{
  "image": "base64-encoded RGB image",
  "instruction": "pick up cup",
  "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "episode_id": "ep_001"
}
```

Response:

```json
{
  "actions": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    [0.15, 0.22, 0.31, 0.42, 0.5, 0.62, 0.7],
    ...
  ],
  "num_actions": 50,
  "action_dim": 7,
  "latency_ms": 42.5,
  "hz": 23.5,
  "denoising_steps": 7,
  "inference_mode": "onnx_cuda",
  "vlm_conditioning": "real",
  "model_hash": "abc123def456...",
  "config_hash": "xyz789...",
  "safety_violations": 0,
  "guard_summary": {
    "violations": [],
    "clamped": false,
    "clamp_count": 0
  },
  "deadline_exceeded": false,
  "latency_p50_ms": 40.2,
  "latency_p95_ms": 45.1,
  "latency_p99_ms": 48.3,
  "jitter_ms": 8.1
}
```

#### GET `/guard/status`

Check ActionGuard state (no API key required).

```bash
curl http://localhost:8000/guard/status
```

Response:

```json
{
  "enabled": true,
  "tripped": false,
  "trip_reason": null,
  "consecutive_clamps": 2,
  "max_consecutive_clamps": 10,
  "inference_count": 1234
}
```

#### POST `/guard/reset`

Clear a tripped guard and resume inference.

```bash
curl -X POST http://localhost:8000/guard/reset
```

Response:

```json
{
  "reset": true,
  "was_tripped": true
}
```

---

## Observability & Telemetry

### Prometheus Metrics

The server emits the following Prometheus series (when `prometheus-client` is installed):

| Metric                               | Type      | Labels                                  | Description                                                                   |
| ------------------------------------ | --------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| `reflex_act_latency_seconds`         | Histogram | `embodiment`, `model_id`, `policy_slot` | End-to-end `/act` latency (p50, p95, p99)                                     |
| `reflex_safety_violations_total`     | Counter   | `embodiment`, `violation_kind`          | Total violations by type (`joint_clamp`, `non_finite`)                        |
| `reflex_fallback_invocations_total`  | Counter   | `embodiment`, `fallback_target`         | Total fallback invocations by target (`previous_chunk`, `hold_position`, `abort`), including deadline-triggered fallbacks |
| `reflex_batch_size_per_flush`        | Histogram | `embodiment`, `policy_slot`             | Requests per batch flush                                                      |
| `reflex_batch_cost_per_flush_ms`     | Histogram | `embodiment`, `policy_slot`             | GPU-ms cost per batch                                                         |
| `reflex_batch_flush_total`           | Counter   | `embodiment`, `policy_slot`, `reason`   | Flushes by reason (`budget_reached`, `timeout`, `single_request_over_budget`) |
| `reflex_policy_runtime_queue_depth`  | Gauge     | `embodiment`, `policy_slot`             | Pending requests in queue                                                     |

### OpenTelemetry Spans

When `[tracing]` extra is installed, `/act` emits structured OTel spans:

```
Span: act
├─ gen_ai.operation.name = "act"
├─ gen_ai.request.model = "/path/to/export"
├─ gen_ai.action.embodiment = "franka"
├─ gen_ai.action.chunk_size = 50
├─ gen_ai.action.denoise_steps = 7
├─ reflex.instruction = "pick up cup"
├─ reflex.state_dim = 6
├─ reflex.image_bytes = 921600
├─ reflex.inference_ms = 42.5
├─ reflex.inference_mode = "onnx_cuda"
├─ reflex.guard.violation_count = 0
├─ reflex.a2c2.applied = false
└─ reflex.record.seq = 12345
```

### JSONL Recording

When enabled via `--record <dir>`, the server writes one line per request to a timestamped session file named `<YYYYMMDD-HHMMSS>-<model_hash>-<session_id>.jsonl[.gz]` (for example, `20260510-143022-abc123def-session42.jsonl.gz`):

```json
{
  "seq": 1,
  "timestamp": "2026-05-10T14:30:45.123Z",
  "image_bytes": 921600,
  "instruction": "pick up cup",
  "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "actions": [[...], [...], ...],
  "action_dim": 7,
  "latency_total_ms": 42.5,
  "mode": "onnx_cuda",
  "error": null,
  "routing": {
    "slot": "prod",
    "routing_key": "abc123",
    "degraded": false,
    "cached": false
  },
  "guard": {
    "violations": [],
    "clamped": false,
    "clamp_count": 0
  }
}
```

---

## Tuning & Debugging

### Latency Too High?

| Symptom                       | Likely cause                | Fix                                                     |
| ----------------------------- | --------------------------- | ------------------------------------------------------- |
| p99 > 100ms                   | Model + overhead too slow   | Run benchmark: `reflex bench ./export --iterations 100` |
| p99 spikes                    | Large batches stalling      | Lower `--max-batch-cost-ms` (e.g., 50 ms)               |
| Deadlines missed              | Inference > deadline        | Raise `--deadline-ms` or optimize model                 |
| Adaptive-steps not triggering | Model doesn't converge fast | Check per-model benchmarks; disable for now             |

### Safety Violations Every Request?

| Symptom                                   | Likely cause                       | Fix                                                                          |
| ----------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------- |
| `safety_violations: 50+` per request      | Safety limits too tight            | Regenerate: `reflex guard init --urdf robot.urdf --margin 0.2`               |
| `guard_tripped` errors                    | Max clamps exceeded                | `POST /guard/reset` and investigate upstream (sensor drift, bad instruction) |
| `safety_detail: "action 0: X violations"` | Specific action index always fails | Check if VLM conditioning is broken (bad image/instruction pair)             |

### Cloud Fallback Not Working?

| Symptom                                         | Likely cause           | Fix                                                                     |
| ----------------------------------------------- | ---------------------- | ----------------------------------------------------------------------- |
| `split_enabled: true` but never routes to cloud | Cloud unhealthy        | `curl http://cloud-vla:8000/health` — verify endpoint is running        |
| Cloud routes on every request                   | Network latency high   | Lower `--cloud-latency-budget-ms` or increase `health_check_interval_s` |
| Fallback returns zeros constantly               | Both edge + cloud down | Restart edge server and verify cloud is reachable                       |

### Deadline Enforcement Unpredictable?

| Symptom                                        | Likely cause                  | Fix                                                              |
| ---------------------------------------------- | ----------------------------- | ---------------------------------------------------------------- |
| Deadline exceeded on fast requests             | Timer starting at wrong point | Check logs; verify deadline includes network I/O                 |
| No deadline misses, but latency p99 near limit | Deadline threshold perfect    | Good! Monitor `deadline_misses_total` for sustained issues       |
| Deadline returns zeros every time              | No last-good action cached    | Ensure first request succeeds; subsequent requests will cache it |

---

## Composition Patterns

### Development (No Safety)

Good for iterating on model without constraints:

```bash
reflex serve ./pi0 \
  --num-denoising-steps 10
  # No safety, no split, no deadline
```

### Edge Deployment (Safety + Deadline, No Cloud)

Typical Jetson/Orin setup:

```bash
reflex serve ./pi0 \
  --safety-config ./safety.json \
  --deadline-ms 100 \
  --device cuda
```

### Cloud Deployment (Adaptive + Split)

Large model on GPU farm:

```bash
reflex serve ./gr00t \
  --safety-config ./safety.json \
  --adaptive-steps \
  --device cuda
```

### Redundant Edge + Cloud

Edge-first with cloud as hot standby:

```bash
# Edge node
reflex serve ./pi0 \
  --safety-config ./safety.json \
  --cloud-fallback http://cloud-vla:8000 \
  --deadline-ms 100

# Cloud node
reflex serve ./gr00t \
  --safety-config ./safety.json \
  --adaptive-steps \
  --device cuda
```

---

## Architectural Decisions

**ADR: Composition over Pipeline**

The Wedge pattern is **composition** (orthogonal toggles), not a monolithic pipeline. Each wedge:

- Has a single responsibility (safety, routing, latency, time limits)
- Declares its dependencies upfront (safety needs URDF → `safety.json`)
- Can be tested independently
- Fails gracefully (safety error → clamp, not crash)

This avoids the "feature creep" anti-pattern where a single monolithic `/act` handler balloons with conditional logic for every edge case.

**Deadlines Before Fallback**

The Deadline wedge runs _before_ Split fallback. Why? If edge is slow, returning a fast cloud action is better than a fresh inference; but returning on deadline is better than waiting forever for cloud. Order matters:

```
1. denoise (with adaptive stop)
2. safety check
3. deadline check ← return fast, even if imperfect
4. split fallback ← last resort (slow + maybe stale)
```

**Safety Clamp, Not Reject**

Default mode is `clamp`, not `reject`. Why?

- Robot doesn't stop mid-motion; graceful degrade is safer.
- Client can inspect `safety_violations` count and decide (log, alert, reset guard, etc.).
- Easier to debug: clamped action visible in JSONL record.

Use `mode: "reject"` only if your task cannot tolerate ANY constraint violation (rare).

---

## Further Reading

- [Batching & throughput tuning](batching.md)
- [CUDA Graphs & performance](cuda_graphs.md)
- [Safety & ActionGuard](../scripts/emit_embodiment_presets.py) — code for guard initialization
- [A2C2 Correction](self_distilling_serve.md) — post-hoc action correction based on task success rate
- [Policy Versioning](policy_versioning.md) — A/B testing and policy slots
