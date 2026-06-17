# Auto-calibration

`tether serve --auto-calibrate` picks the right pre-shipped (variant × provider × NFE × chunk_size) configuration for your hardware + embodiment, then passively learns `latency_compensation_ms` from real `/act` traffic. One-flag first-deploy DX win.

Per ADR `2026-04-25-auto-calibration-architecture` — SELECTION not tuning, strict partial order resolver, passive actuator-latency observation (no boot-time probe).

## Quick start

```bash
# Calibrate + serve. First run takes ~5-7s for the measurement pass; subsequent
# runs hit the cache and start instantly.
tether serve ./my-export/ --embodiment franka --auto-calibrate

# Override cache location (e.g., to ship a frozen cache inside a container)
tether serve ./my-export/ --auto-calibrate \
    --calibration-cache /opt/tether/calibration.json

# Force re-measurement on next start (after hardware swap or driver upgrade)
tether serve ./my-export/ --auto-calibrate --calibrate-force
```

When `--auto-calibrate` is unset, behavior is unchanged from baseline; calibration is fully opt-in for Phase 1. Phase 1.5 will flip to default-on after telemetry from first deploys.

## What gets selected

Five parameters, in strict partial order (each downstream choice narrows by the upstream choice):

| Parameter | Choices | Selected by |
|---|---|---|
| `variant` | fp16, int8, fp8 | Hardware + which `.onnx` files exist in your export. fp8 only on sm_89+ (Hopper / Ada Lovelace / Blackwell — H100, 4090, L40, etc.) |
| `provider` | TensorRT-EP, CUDA-EP, CPU-EP | Variant + `--max-batch`. TRT-EP requires fp16 + batch=1 (per ADR `2026-04-14-disable-trt-when-batch-gt-1`); CUDA-EP otherwise. |
| `nfe` (denoise steps) | 1, 2, 4, 8, 10 | Largest NFE such that `nfe × measured_expert_step_ms ≤ chunk_period × 0.7`. Falls back to NFE=1 (forces SnapFlow distill path) when no candidate fits. |
| `chunk_size` | embodiment default (franka=50, so100=30, ur5=50) | Default unless even NFE=1 doesn't fit; then halves. |
| `latency_compensation_ms` | embodiment cold-start (franka=40, so100=60, ur5=40) | Cold-start at startup; warm-update from real `/act` p95 after 30s of traffic. |

The resolver runs in single-pass at startup. The warm-up tracker continuously refines `latency_compensation_ms` and writes back to the cache when stable.

## The cache

Lives at `~/.tether/calibration.json` by default. JSON schema v1:

```json
{
  "schema_version": 1,
  "reflex_version": "0.5.0",
  "calibration_date": "2026-04-25T16:30:00.000000Z",
  "hardware_fingerprint": {
    "gpu_uuid": "GPU-abc123...",
    "gpu_name": "NVIDIA A10G",
    "driver_version_major": 535,
    "driver_version_minor": 129,
    "cuda_version_major": 12,
    "cuda_version_minor": 2,
    "kernel_release": "6.1.0",
    "cpu_count": 8,
    "ram_gb": 32,
    "reflex_version": "0.5.0"
  },
  "entries": {
    "franka::pi05_decomposed_libero_v3": {
      "chunk_size": 50,
      "nfe": 4,
      "latency_compensation_ms": 42.5,
      "provider": "TensorrtExecutionProvider",
      "variant": "fp16",
      "measurement_quality": {
        "warmup_iters": 10, "measurement_iters": 100,
        "median_ms": 38.2, "p99_ms": 47.1,
        "n_outliers_dropped": 10, "quality_score": 0.94
      },
      "measurement_context": {...},
      "timestamp": "2026-04-25T16:30:00.000000Z"
    }
  }
}
```

Keyed by `{embodiment}::{model_hash}` so multi-embodiment / multi-model deployments coexist in one cache file.

### Inspecting the cache

```bash
# Pretty-print
tether doctor --show-calibration

# Machine-readable (for CI / scripts)
tether doctor --show-calibration --format json | jq .

# Custom cache path
tether doctor --show-calibration --calibration-cache /opt/tether/calibration.json
```

Output flags:
- `hardware_fingerprint: matches current host` — cache is valid for this machine
- `hardware_fingerprint: STALE — hardware/version mismatch or > 30d old` — re-calibration will run on next `--auto-calibrate` start

### Staleness

The cache is stale when ANY of these is true:

- Hardware fingerprint mismatches the current host (GPU swap, driver major-version bump, reflex_version change)
- Cache age > 30 days
- `--calibrate-force` is set

Staleness triggers re-measurement on the next `--auto-calibrate` startup. Driver patch-version bumps are ignored (kernel patches shouldn't invalidate the cache).

## Override ladder

Explicit CLI flags > calibration cache > embodiment_config default.

If you pass `--chunk-size 40` AND `--auto-calibrate`, the explicit `40` wins; calibration is overridden. A yellow stderr warning makes this visible at startup:

```
[auto-calibrate] ignoring cached chunk_size=50 because --chunk-size=40
```

Same pattern for `--nfe`, `--deadline-ms`, `--providers`.

## What's the warm-up?

Per the ADR, **no active probe at boot** — that would risk an unintended first-move on a robot in an unsafe pose. Instead:

1. Server starts with the embodiment cold-start default for `latency_compensation_ms` (franka=40ms, so100=60ms, ur5=40ms).
2. As real `/act` traffic flows, the warmup tracker records each request's wall-clock latency in a rolling 100-sample window.
3. Once 30+ samples accumulate AND the rolling p95 has been stable within 5ms for 3 consecutive checks, the tracker writes the new value to the cache + persists atomically.
4. Subsequent server restarts pick up the warmed-up value from the cache.

Customer experience: works correctly from the first request (cold-start default is a reasonable bound), gets steadily more accurate over the first ~30 seconds.

## A note on `rtc_execution_horizon` migration

Per ADR `2026-04-25-auto-calibration-architecture` decision #8, the embodiment JSON's `control.rtc_execution_horizon` is now an **integer count of actions**, not a fraction of `chunk_size`. Legacy fractional values (e.g., `0.5` in shipped franka.json) auto-migrate at load time:

```
embodiment franka at /path/to/franka.json stores rtc_execution_horizon as a
fraction (0.5) — converted to integer count 25 via chunk_size=50. Update
the JSON to integer count to silence this warning.
```

Update your JSON when convenient: `"rtc_execution_horizon": 0.5` → `"rtc_execution_horizon": 25`. Schema v2 will reject fractional values.

## Hardware tier expectations

Calibration outputs vary materially by hardware. Approximate expectations on franka:

| Hardware | variant | provider | NFE | chunk_size | latency_comp_ms (warmed) |
|---|---|---|---|---|---|
| Cloud A100-80GB | fp16 | TRT-EP | 10 | 50 | 25-40 |
| Cloud A10G-24GB | fp16 | TRT-EP | 4-8 | 50 | 35-50 |
| Jetson AGX Orin (64GB) | fp16 | TRT-EP | 4 | 50 | 50-70 |
| Jetson Orin Nano (8GB) | fp16 | CUDA-EP | 1 (SnapFlow path) | 50 | 80-120 |
| H100 (Hopper) | fp8 | TRT-EP | 10 | 50 | 20-30 |

These are illustrative — your numbers depend on the export, the model, and the customer-specific traffic shape. The bundled `calibration_defaults.json` (Phase 1.5) ships with measured baselines so the first run is instant on common hardware.

## Troubleshooting

**`auto-calibrate: cache stale (fingerprint mismatch ...)`**
Expected after hardware swap or driver upgrade. Re-measurement runs on the next start. To skip the cache entirely and re-measure, pass `--calibrate-force`.

**`No NFE fits budget — falling back to NFE=1`**
Your hardware × embodiment × model combo has no legal NFE that fits the chunk-period budget. For most setups this means: switch to a SnapFlow-distilled student (which is single-step inherently), OR lower the embodiment's `control.frequency_hz`. Validation experiment: A10G × franka × pi0.5-teacher × NFE=10 reproducibly hits this falsifiable claim — see ADR.

**`auto-calibrate: warmup tracker armed for ... (writes back when 30+ samples + p95 stable)`**
Working as designed. First /act traffic populates the rolling window; the cache updates once the p95 settles.

**`tether doctor --show-calibration` says "No calibration cache found"**
You haven't run `tether serve --auto-calibrate` yet on this host. Or the cache lives at a different path — pass `--calibration-cache <path>`.

**Cache size**
A typical multi-embodiment cache is < 4 KB. No concern about disk pressure.

## Architectural commitments

Per ADR `2026-04-25-auto-calibration-architecture`:

- SELECTION not tuning — picks among pre-shipped variants + bucketed values, never runtime-tunes kernel shapes
- Greedy resolver order locked: variant → provider → NFE → chunk_size → latency_compensation_ms (no parallel optimization)
- Schema v1 first-field locked: `schema_version`. Phase 2 evolution is additive-only — no rename, no remove of v1 fields
- Passive observation only for actuator latency — no active boot-time probe (avoids unintended first-move)
- Cache path `~/.tether/calibration.json` locked (customers grep for this in scripts)
- `policy_slot` label vocabulary (`a` | `b` | `prod` | `shadow`) locked when policy-versioning composes here

Validation test surface: 108 unit tests (substrate + harness + resolver + CLI + warmup + doctor flow). Modal cross-hardware integration is Phase 1 Day 7 (user-authorized when ready).

## What's not in Phase 1

- **Default-on**: Phase 1 ships opt-in (`--auto-calibrate`); Phase 1.5 flips to default-on after telemetry shows no regressions on first deploys.
- **Per-policy calibration**: composes with policy-versioning's per-policy state (Phase 2).
- **Continuous online retune** during steady state (thermal drift, load changes) — Phase 2 self-tuning.
- **Bandit-driven NFE / chunk_size adjustment** — Phase 2.5 refinement.
- **Bundled `calibration_defaults.json`** — Day 9, Modal-bound (user-authorized when ready). Customers without the bundle pay the ~5-7s first-run measurement.
- **Customer-data fine-tuning** — composes with `self-distilling-serve` (Phase 1 separate feature).
