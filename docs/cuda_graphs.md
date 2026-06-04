# CUDA graphs

`tether serve --cuda-graphs` captures the two ONNX Runtime sessions on the decomposed path (`vlm_prefix` + `expert_denoise`) into CUDA graphs at startup and replays them thereafter. Replay is ~3-4× faster than eager execution because the GPU skips kernel-launch overhead for every call.

Tier-aware out of the box:

| GPU | vlm_prefix | expert_denoise | Combined effect |
|---|---|---|---|
| A100-40GB / A100-80GB | Captured | Captured | Full speedup on every /act |
| A10G (24 GB) | **Eager** (graceful degrade at init) | Captured | Expert-only; ~95% of total speedup since expert fires 10× per /act vs vlm_prefix 1× per cache-miss |
| Jetson Orin Nano (8 GB) | Not validated | Not validated | Run without `--cuda-graphs` for now — research tier |
| Jetson AGX Orin (64 GB) | Expected to capture | Expected to capture | Validate before production |

Measured on Modal 2026-04-25 against an SnapFlow pi0.5 decomposed export (franka): A100 hit 4.44× on vlm_prefix + 3.03× on expert_denoise; A10G hit 3.76× on expert_denoise and fell back to eager on vlm_prefix (16 MB BFCArena alloc exception — vLLM #5517 memory-overhead pattern). See `reflex_context/03_experiments/2026-04-25-cuda-graphs-ort-spike-modal.md`.

## Quick start

```bash
tether serve ./my-export/ --cuda-graphs
```

That's the whole knob. Run, hit `/act`, observe latency drop. The first /act per session is slower than eager (graph capture cost, ~50-400 ms depending on session); every subsequent /act hits the fast-path replay.

## What happens at startup

1. Server constructs `vlm_prefix` ONNX session with `enable_cuda_graph=1`.
2. Probes capture by running one synthetic forward pass.
3. On **success**: wraps with `CudaGraphWrapper`, emits `reflex_cuda_graph_captured_total{session=vlm_prefix}`.
4. On **capture failure** (OOM, unsupported op): rebuilds the session without `enable_cuda_graph`, wraps with `EagerSessionWrapper`, emits `reflex_cuda_graph_capture_failed_at_init_total{session=vlm_prefix,reason=<ExceptionClass>}` ONCE, logs an INFO line. This is the A10G vlm_prefix path today.
5. Same sequence for `expert_denoise`.

Request handling then proceeds transparently — the wrappers expose the same `.run()` API whether the session is captured or eager.

## Metrics

Scope is bounded-enum to stay within the Prometheus cardinality budget. Labels: `embodiment` (franka / so100 / ur5 / custom) × `session` (vlm_prefix / expert_denoise) × `reason` (capture_failed / replay_failed / capture_failed_oom) where applicable.

| Metric | Type | Labels | Fires when |
|---|---|---|---|
| `reflex_cuda_graph_captured_total` | Counter | embodiment, model_id, session | First successful capture per session |
| `reflex_cuda_graph_replayed_total` | Counter | embodiment, model_id, session | Every replay (so ratio reveals traffic shape) |
| `reflex_cuda_graph_eager_fallback_total` | Counter | embodiment, model_id, reason | In-request capture or replay raised |
| `reflex_cuda_graph_capture_failed_at_init_total` | Counter | embodiment, model_id, session, reason | Session-init probe failed → eager-for-process-lifetime. **A10G vlm_prefix hits this at startup.** |
| `reflex_cuda_graph_capture_seconds` | Histogram | embodiment, session | Capture wall-clock (first run) |
| `reflex_cuda_graph_replay_seconds` | Histogram | embodiment, session | Replay wall-clock |

Distinguishing `capture_failed_at_init` from `eager_fallback` matters: the init-failure is a hardware-tier signal (this process will NEVER capture this session), while the replay-time fallback is a per-request error signal. Operators want to see them separately.

## Troubleshooting

**"`reflex_cuda_graph_capture_failed_at_init_total` fires at startup on my A10G"**
Expected for `session=vlm_prefix`. A10G's CUDA-graph memory overhead (~128 MB reserved per captured graph) exceeds the BFCArena pre-allocated pool for the vlm_prefix model. The session runs eager instead; `expert_denoise` still captures and gives you ~95% of the total speedup. No customer-facing action.

**I see `eager_fallback_total` climbing during traffic**
Replay-time failure — rarer. Likely causes: (a) CUDA context contention with another process on the same GPU, (b) input shape changed (should be impossible on a static-shape Tether export — file a bug), (c) GPU memory pressure from a concurrent workload. Check `reflex_cuda_graph_eager_fallback_total{reason=replay_failed}` for the class of failure.

**No speedup after enabling `--cuda-graphs`**
Your serve backend isn't the decomposed dispatch path. Today's production `tether serve` uses the legacy `ReflexServer` backend that doesn't consume `cuda_graphs_enabled`. The flag applies to `Pi05DecomposedInference` dispatch (used in Modal scripts today; production wire-up pending the decomposed-dispatch fix tracked by `chunk-budget-batching`). In that case, the startup log reads `"--cuda-graphs was set but this backend (ReflexServer legacy decomposed path) does not consume the flag."` Watch for that line.

**Latency spike on the first request after startup**
That's the capture cost. Expected — ~100-400 ms extra on the first /act. If you can't tolerate first-request spikes (robot demo, live customer), keep `--prewarm` enabled (default) so capture runs during startup, not on the first user request.

## When NOT to enable

- You're not on the decomposed dispatch path (see above — no-op).
- You're on a hardware tier we haven't validated (Orin Nano, Thor, Hopper/Blackwell custom silicon). Baseline without `--cuda-graphs` first, then enable + compare p99.
- You depend on torch.cuda.graph semantics — Tether uses ORT-native capture (per ADR `2026-04-24-cuda-graphs-architecture`). Phase 2 may add a torch-native path if a customer blocks on it.

## What's locked (ADR)

Per `01_decisions/2026-04-24-cuda-graphs-architecture.md`:
- ORT-native capture, not `torch.cuda.graph`
- Two separate captured graphs per model (vlm_prefix + expert_denoise) — not a single combined graph (the cache layer makes that impossible without destroying the 9× episode-cache moat)
- One shape per (model × embodiment) pair — ONNX is shape-specialized at export, so this is free
- Opt-in customer flag for Phase 1; flip to default-on in Phase 2 after customer-deploy telemetry
- Orin Nano validation deferred pending hardware

Modal validation run: `reflex_context/03_experiments/2026-04-25-cuda-graphs-ort-spike-modal.md`.
