# Tether

> by [FastCrest](https://fastcrest.com) — deployment infrastructure for vision-language-action models.

[![PyPI](https://img.shields.io/pypi/v/fastcrest-tether.svg)](https://pypi.org/project/fastcrest-tether/)
[![Python](https://img.shields.io/pypi/pyversions/fastcrest-tether.svg)](https://pypi.org/project/fastcrest-tether/)
[![License](https://img.shields.io/pypi/l/fastcrest-tether.svg)](https://github.com/FastCrest/tether/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/fastcrest-tether.svg)](https://pypi.org/project/fastcrest-tether/)

![Tether — pip install + tether doctor + tether --help on Modal A10G with TRT EP active](assets/tether-tweet.gif)

**Deployment confidence for VLA robot policies** — Tether answers one production question: can this policy safely move forward?

**Verified parity across ALL four major open VLAs.** Tether's monolithic ONNX export matches the reference PyTorch policy to **cos = +1.000000** end-to-end on SmolVLA, pi0, pi0.5 (canonical 10-step flow-matching unrolled) and GR00T N1.6 (canonical 4-step DDIM loop external to the ONNX). Per-model first-action max_abs: SmolVLA 5.96e-07, pi0 2.09e-07, pi0.5 2.38e-07, GR00T 8.34e-07 — all at machine precision, shared seeded inputs. Full claim ledger in [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

The public workflow is intentionally small:

```bash
tether chat
tether prove ./export --output-dir ./tether-deploy-proof
tether promote ./tether-deploy-proof --profile warehouse-safe
```

Everything else in the CLI feeds evidence into that answer.

## Install

**Recommended** — runs hardware + Python checks first, picks the right install extras for your machine:

```bash
curl -fsSL https://fastcrest.com/install | sh
```

The bootstrap installer detects your platform (Mac / Jetson Orin / NVIDIA GPU / CPU) and chooses the right extras automatically. It also bails early with a useful message on unsupported hardware (e.g. original 4 GB Jetson Nano — Maxwell GPU + JetPack 4.6 / Python 3.6, too old for VLAs).

**Manual install** if you know what you want:

```bash
pip install fastcrest-tether                            # core
pip install 'fastcrest-tether[serve,gpu,monolithic]'    # GPU production path
pip install 'fastcrest-tether[serve,onnx]'              # Mac / CPU runtime
```

Requires Python ≥ 3.10.

### What's new in v0.11.2 (2026-05-29)

- **`tether connect` works on a clean install** — `requests` is now a core dependency, so `tether connect status` no longer raises `ModuleNotFoundError` on `pip install fastcrest-tether` without extras (it had been an undeclared import that only resolved transitively).
- **`--fast-kernels` cleared the formal N=100/task L3 LIBERO parity gate** — on Pi0.5 LIBERO-10 tasks 0-2 (600 episodes), Triton fast kernels scored 91.3% (274/300) vs native ORT 85.3% (256/300) — 6.0pp *ahead* of native, so kill-trigger 3 stays clear and the opt-in Triton runtime stays on.
- **Hardened monolithic serve/bench path, with external-data ONNX** — dedicated ORT provider-options + tokenizer-loading modules are extracted from the request hot path; ONNX models with external weight data (`.onnx` + `.onnx_data`, required once a graph exceeds the 2 GB protobuf limit) now load in both serve and the weight-fusion export pass.
- **Cleaner streams** — integration-command errors route to stderr, so `--json` consumers and shell pipelines get a clean stdout.

### What's new in v0.11.1 (2026-05-27)

- **Triton fast kernels** — 2.5x PyTorch, ~12x ORT on A100. Opt in with `tether serve --fast-kernels` (requires the `[fast-kernels]` extra). Falls back to ORT silently when Triton is unavailable.
- **ZMQ transport** — `tether serve --transport zmq` for low-latency robot communication. JPEG-on-wire image serialization via msgpack. Ships with the `[serve]` extra (pyzmq + msgpack included).
- **DreamZero WAM** — 6th VLA family on the BaseVLA spine. NVIDIA Research's world-action model: joint video + action diffusion on Wan 2.1 DiT backbone. 94.65% LIBERO average. Apache-2.0 via FluxVLA.
- **Safetensors-direct loading** — 67% RSS reduction, no nn.Module overhead. Weights load from safetensors into flat dicts without constructing a PyTorch module tree.
- **FluxVLA pi0.5 LIBERO-10 checkpoint** — LimX Dynamics' finetuned pi0.5 (Apache-2.0), published 97.85% LIBERO-10 average. Registry ID: `pi05-libero10-fluxvla`.
- **transformers 5.x DynamicCache compat** — fixes compatibility with transformers 5.x DynamicCache API changes.
- **ROS2 robot adapter starter kits** — Aloha + UR3 templates in `contrib/ros2/` for integrating Tether with real robot hardware.

### What's new in v0.10.0 (2026-05-22)

**BaseVLA spine refactor** — every VLA family is now a thin (~100 LOC) composition class declaring which of 6 component slots it uses (`vision_backbone`, `llm_backbone`, `vlm_backbone`, `projector`, `vla_head`, `text_encoder`). Adding a new VLA backbone is now a composition-class file + a registry entry. See `src/tether/models/vlas/{pi0,pi05,smolvla,gr00t}.py` for worked examples.

Validated bit-identical to lerobot's reference on real checkpoints — pi0 (max 1.13e-6), pi0.5 (max 2.74e-6), SmolVLA (synthetic max 0.0), GR00T N1.6 (max 0.0). 6 silent ONNX export bugs fixed along the way (PR #156).

Breaking: module renames — `tether.exporters.{pi0,smolvla,gr00t}_exporter` → `tether.exporters.{pi0,smolvla,gr00t}`. Update import statements.

### Upgrading

We ship patches frequently — make sure you're on the latest:

```bash
pip install --upgrade fastcrest-tether              # pip
uv add --refresh fastcrest-tether                   # uv (the --refresh flag is required;
                                              # uv caches the package index aggressively
                                              # and won't see new releases without it)
```

After upgrading, if you've previously run `tether go` and the server fails to start, your
export cache may be stale (built by an older version, schema mismatch, etc.):

```bash
rm -rf ~/.cache/tether/exports/<model_id>     # forces a fresh export on next tether go
```

v0.5.4+ does this automatically when it detects a version mismatch — the manual step is
only needed for caches built by v0.5.3 or earlier.

## Performance

`fastcrest-tether[serve,gpu]` uses ONNX Runtime's TensorRT execution provider out of the box. Measured on Modal A10G (Ampere, sm_8.6) on 2026-04-29 against SmolVLA monolithic (5 warmup + 20 measured forward passes, batch=1):

| Provider | Mean latency | p95 |
|---|---|---|
| `CUDAExecutionProvider` (ORT-CUDA fallback) | 108.11 ms | 108.68 ms |
| **`TensorrtExecutionProvider` (default)** | **19.49 ms** | 19.71 ms |

**5.55x faster.** The win comes from TensorRT's FP16 kernels + engine fusion. The `[serve,gpu]` extras pull `tensorrt>=10` and `tether` patches `LD_LIBRARY_PATH` automatically at import.

### How to verify you're getting the win

```bash
tether doctor
```

Look for a **green ✓** on these checks (all four must pass):
- `TensorRT runtime (libnvinfer.so.10)` — loadable
- `CUDA cuBLAS (libcublas.so.12)` — loadable
- `CUDA cuDNN (libcudnn.so.9)` — loadable
- `ORT-TRT EP active` — session created with TRT EP in active providers

If any are red ✗, the remediation hint says exactly which `pip install` to run. The most common cause is that you used `[serve,gpu-min]` or an older release that didn't pull `tensorrt` automatically.

### Reproduce the measurement

```bash
modal profile activate <your-profile>
modal run scripts/modal_v07_runtime_spike.py
```

Full reproducer + 9-iteration debug log: [`reflex_context/03_experiments/2026-04-29-v07-runtime-spike.md`](reflex_context/03_experiments/2026-04-29-v07-runtime-spike.md).

### Caveats

- Measured only on A10G + SmolVLA monolithic so far. Other model architectures (pi0.5 decomposed, GR00T) and other hardware tiers (Orin Nano, T4, H100) may show different ratios.
- **Blackwell (RTX 50-series, B200, GB200)** — ORT 1.25.1+ ships sm_120 kernels. Smoke validation recommended before declaring fully production-ready (open threading issue #27621).

### Opt-out

Adds ~2 GB to `[serve,gpu]` install (the `tensorrt` package + bundled libs). If you don't want it:

```bash
pip install 'fastcrest-tether[serve,gpu-min]'   # ORT-CUDA only, ~5x slower on transformers
```

Or disable the `LD_LIBRARY_PATH` patch (e.g. if it conflicts with another env-aware tool):

```bash
TETHER_NO_LD_LIBRARY_PATH_PATCH=1 tether go ...
```

## Quickstart — chat to it

```bash
tether chat
```

```
you › what version am I running and what hardware can I deploy to?

  → show_version({})    → tether --version    → "tether 0.2.0"
  → list_targets({})    → tether targets      → [orin-nano, orin, orin-64, thor, desktop]

You're running tether 0.2.0. Supported targets:
  - orin-nano — Jetson Orin Nano: 8 GB, fp16
  - orin — Jetson AGX Orin 32GB: 32 GB, fp16
  - orin-64 — Jetson AGX Orin 64GB: 64 GB, fp16
  - thor — Jetson Thor: 128 GB, fp8
  - desktop — Desktop GPU (RTX 4090 / A100 / H100): 24 GB, fp16

Want me to show which models support each target, or run tether doctor?
```

Chat wraps the real `tether` CLI tools and runs them as subprocesses on your behalf. Ask for outcomes, not flags: "prove ./export is ready for franka", "can I promote this proof packet?", or "why did my last /act fail?". Powered by GPT-5 Mini through a proxy hosted at `chat.fastcrest.com` — free tier is 100 calls/day per machine, no signup, no API key.

> Bring your own key? `export FASTCREST_PROXY_URL=https://api.openai.com/v1`

## Quickstart — explicit deploy

```bash
# Browse the curated model registry
tether models list

# Smoke test — probe hardware → resolve model → pull → export → serve
tether go --model smolvla-base

# Now make it real with per-robot normalization (ships with franka, so100, ur5, quadcopter)
tether go --model smolvla-base --embodiment franka
# Or for a drone:
tether go --model smolvla-base --embodiment quadcopter

# Or with explicit hardware override
tether go --model pi05-libero --embodiment franka --device-class a10g
```

Drop `--embodiment` for a quick smoke test — the server starts cleanly and `/act` returns
unscaled raw actions. Add `--embodiment <preset>` (or `--custom-embodiment-config <path>`)
when you're ready for per-robot normalization + ActionGuard clamping.

Then from your code:

```python
from tether.client import ReflexClient

with ReflexClient("http://localhost:8000") as client:
    with client.episode() as ep:                       # auto episode_id, RTC reset
        result = ep.act(image=numpy_frame, state=[0.1, 0.2, ...])
        print(result["actions"])                       # list of action chunks
        # 503-warming retried automatically; guard violations surface as fields
```

Or with curl:

```bash
curl -X POST http://localhost:8000/act -H 'content-type: application/json' \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

```json
{
  "actions": [[...], [...], ...],          // 50 × action_dim chunk
  "latency_ms": 11.9,                      // smolvla on A10G, 10-step denoise
  "inference_mode": "onnx_trt_fp16",       // automatic — no engine flags needed
  "guard_clamped": false                    // ActionGuard didn't have to clamp anything
}
```

`tether go` auto-detects your hardware (NVIDIA GPU / Jetson / CPU), picks the right model variant for that device, downloads weights from HuggingFace, and starts the /act endpoint. **No editing configs, no separate `tether export` step, no manual variant selection.** For models that ship as raw PyTorch weights, you get the export command to run next.

### Security — production auth

For production deployments, require an API key on `/act` and `/config`:

```bash
export TETHER_API_KEY="$(openssl rand -hex 32)"
tether serve ./p0 --host 0.0.0.0 --port 8000 --api-key "$TETHER_API_KEY"
```

Authenticated clients should send either the preferred bearer token header or the compatible `X-Tether-Key` header:

```bash
curl -X POST http://localhost:8000/act \
  -H "Authorization: Bearer $TETHER_API_KEY" \
  -H "content-type: application/json" \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'

curl -X POST http://localhost:8000/act \
  -H "X-Tether-Key: $TETHER_API_KEY" \
  -H "content-type: application/json" \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

The Python client sets `X-Tether-Key` when `api_key` is provided:

```python
import os

from tether.client import ReflexClient

with ReflexClient("http://localhost:8000", api_key=os.environ["TETHER_API_KEY"]) as client:
    result = client.act(
        image=numpy_frame,
        state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        instruction="pick up the red cup",
    )
```

`/health` stays unauthenticated so load balancers and orchestrators can probe readiness without credentials.

### The product surface

For new users, Tether is three verbs:

```bash
tether chat             # ask for the outcome in plain English
tether prove ./export   # collect a deployment proof packet
tether promote ./proof  # return PROMOTE, BLOCK, or ROLLBACK
tether profiles list    # choose a built-in promotion profile
```

The rest of the CLI is supporting machinery:

- Runtime evidence: `go`, `serve`, `doctor`, `smoke`.
- Rollout evidence: `policy diff`, `traces`, `replay`, `eval`.
- Model workflow: `models`, `validate`, `train`.
- Enterprise/admin workflow: `pro`, `contribute`, `curate`, `data`, `comply`.

Advanced/SO-100/internal commands stay callable directly (`config`, `calibrate`, `bench-game`, `status`, `inspect bench/targets/guard/doctor`), but stay out of the first-run path. Hidden legacy commands (`export`, `bench`, `replay`, etc.) stay callable as alias bridges.

### Install notes

- `[monolithic]` extra is required for the cos=+1.000000 verified export path (pins transformers==5.3.0)
- CPU-only: `pip install 'fastcrest-tether[serve,onnx,monolithic]'`
- GPU install needs the FULL cuDNN 9 system library (not just the pip wheel). Easiest path: NVIDIA's container `docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3`, then `apt-get install -y clang` (for lerobot→evdev), then the pip install
- `tether serve` errors loudly if cuDNN can't load — no silent CPU fallback
- First `tether go` downloads weights (~1-14 GB depending on model) — cached on subsequent runs
- First serve takes 10-70s warmup; `/health` returns HTTP 503 until ready, HTTP 200 after — load balancers correctly skip the server during warmup
- `tether chat` works on the base install — no extras required. Network access required (calls FastCrest's hosted proxy).

### Docker — zero-install serve

```bash
# x86_64 CUDA runtime (cloud GPUs, dev workstations)
docker pull ghcr.io/fastcrest/tether:latest
docker run --gpus all \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/fastcrest/tether:latest

# Jetson Orin / Orin Nano / Thor (arm64 + nvidia container runtime)
docker pull ghcr.io/fastcrest/tether:latest-arm64
docker run --runtime=nvidia \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/fastcrest/tether:latest-arm64
```

The container's default command is `tether serve /exports --host 0.0.0.0 --port 8000`. Override with any `tether` subcommand: `docker run ... ghcr.io/fastcrest/tether:latest export <hf_id>` etc.

Jetson arm64 image: built via QEMU cross-compile on tag push (`v*`). Bring-your-own-CUDA — the image deliberately doesn't bundle CUDA/cuDNN/TensorRT (those live on the Jetson under `/usr/local/cuda` and are ABI-locked to the host's JetPack version; the nvidia container runtime exposes them into the container).

### ROS2 — `tether ros2-serve`

Wraps the inference loop as a ROS2 node. Subscribes to `sensor_msgs/Image`, `sensor_msgs/JointState`, and `std_msgs/String`; publishes action chunks as `std_msgs/Float32MultiArray` at a configurable rate.

```bash
# rclpy is NOT pip-installable. Install ROS2 via apt or robostack first:
source /opt/ros/humble/setup.bash   # or iron / jazzy

# Hidden alias — kept for back-compat through v0.2; will fold into
# `tether serve --transport ros2` in a future release.
tether ros2-serve ./my_export \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --task-topic  /tether/task \
  --action-topic /tether/actions \
  --rate-hz 20
```

Inference respects `--safety-config` (same limits file as HTTP serve).

When `onnxruntime-gpu` ships with the TensorRT execution provider (it does in v1.20+), `tether serve` uses TRT FP16 automatically and caches the engine in `<export_dir>/.trt_cache` so subsequent server starts skip the engine-build cost. The first `tether serve` takes ~30-90s to warm up; restart is ~1-2s.

## Pre-flight validation

Before deploying, validate your dataset (will it train?) and your export (does it serve cleanly?):

```bash
# Dataset: 8 falsifiable checks against your LeRobot v3.0 corpus
tether validate dataset /path/to/lerobot_data --embodiment franka --strict

# Export: round-trip ONNX vs PyTorch parity at machine-precision threshold
tether validate export ./p0 --model lerobot/pi0_base --threshold 1e-4
```

Sample passing output (abbreviated):

```
Per-fixture results
fixture_idx  max_abs_diff  mean_abs_diff  passed
0            3.21e-06      8.40e-07       PASS
1            2.98e-06      7.92e-07       PASS
...
Summary
max_abs_diff_across_all  3.21e-06
passed                   PASS
```

Exit codes: `0` pass, `1` fail (any fixture above threshold), `2` error (missing ONNX, bad config). Pipe `--output-json` for CI consumption, or run `tether validate --init-ci` to scaffold a GitHub Actions workflow at `.github/workflows/tether-validate.yml`.

### Deployment proof packet

`tether prove ./p0` turns a real export into a local acceptance packet. `tether deploy-proof` is the explicit backend command and remains supported for scripts:
doctor diagnostics, `/health`, authenticated `/act` samples, TTFA, p50/p95/p99,
jitter, control-budget misses, API-key boundary checks, `/metrics` scrape,
optional trace recording, ActionGuard stress checks, export file hashes, and a
hashed `MANIFEST.json`.

Before promoting a new or shadow policy, include a policy diff in the proof
packet with `--policy-diff-baseline` plus `--policy-diff-candidate` or
`--policy-diff-shadow` to gate action deltas, latency regressions, shape
mismatches, and guard regressions.

Use `tether serve ./current --shadow-policy ./candidate --record ./traces/shadow`
to collect a single shadow trace where production actions stay live and
candidate actions are appended as `shadow_result` evidence for
`tether policy diff --shadow`.

For the self-serve rollout decision, run the rollout gate directly:

```bash
tether rollout gate ./traces/shadow.jsonl.gz \
  --packet-dir /tmp/tether-shadow-rollout \
  --profile lab-shadow \
  --min-compared 100 \
  --wait-timeout-s 5
```

It writes a hashed packet with `policy-diff.json` and
`promotion-decision.json`, then exits `0` for `PROMOTE`, `1` for `HOLD`, and
`4` for `ROLLBACK`.

For the operator-facing release decision, use the composed assurance command.
It consumes an existing proof packet, optionally adds realtime/action-execution
and shadow evidence, and writes one `release-assurance.json` / Markdown report:

```bash
tether release assure /tmp/tether-deploy-proof \
  --profile warehouse-safe \
  --control-hz 20 \
  --execution-cert \
  --shadow-trace ./traces/shadow.jsonl.gz \
  --output-dir /tmp/tether-release-assurance
```

The top-level decision is `PROMOTE`, `HOLD`, or `ROLLBACK`. The report includes
component decisions, risk signals such as policy action delta, latency
regression, stale action window, chunk-boundary delta, velocity discontinuity,
and open evidence gaps for production rollout.

For serving-specific deployment confidence, turn the same proof packet into a
realtime certificate. This answers whether the measured `/act` path fits the
robot control loop on the target hardware/cell:

```bash
tether prove ./p0 \
  --profile production.yml \
  --control-hz 20 \
  --samples 100 \
  --output-dir /tmp/tether-deploy-proof

tether bench realtime /tmp/tether-deploy-proof \
  --target agx-orin-cell-a \
  --execution-cert \
  --output-dir /tmp/tether-realtime-cert
```

The certificate gates roundtrip p95 against the control period, deadline
misses, `/act` errors, and control-budget misses, then writes
`realtime-serving-cert.json`, Markdown, and a hashed `MANIFEST.json`.
Add `--execution-cert` when the proof packet carries action chunks plus
`action_execution` telemetry; it also gates stale-action window, chunk-boundary
continuity, boundary velocity jump, and scheduler/cache/adaptive-horizon
attribution.

```bash
tether prove ./p0 \
  --embodiment franka \
  --api-key "$TETHER_API_KEY" \
  --record-dir /tmp/tether-proof-traces \
  --policy-diff-baseline ./traces/current.jsonl.gz \
  --policy-diff-candidate ./traces/candidate.jsonl.gz \
  --profile production.yml \
  --samples 100 \
  --output-dir /tmp/tether-deploy-proof
```

Then turn the packet into an operator decision:

```bash
tether promote /tmp/tether-deploy-proof --profile warehouse-safe
```

Use a built-in promotion profile directly, or copy one to an editable file:

```bash
tether profiles list
tether profiles show warehouse-safe
tether profiles init warehouse-safe --output warehouse-safe.yml
tether promote /tmp/tether-deploy-proof --profile warehouse-safe
```

Profiles are JSON/YAML and override default thresholds:

```yaml
name: production
thresholds:
  require_auth: true
  require_record_trace: true
  require_guard: true
  control_hz: 20
  max_warm_roundtrip_p95_ms: 40
  max_missed_control_budget: 0
```

## Evidence knobs, not extra products

Advanced teams can enable more runtime evidence with `tether serve` flags:

```bash
tether serve ./p0 \
  --safety-config ./robot_limits.json \    # joint-limit clamping + EU AI Act audit log
  --adaptive-steps \                        # stop denoise loop early when velocity converges
  --deadline-ms 33 \                        # return last-known-good action if over budget
  --cloud-fallback http://cloud:8000 \     # edge-first with cloud backup
  --action-similarity-threshold 0.05 \     # FlashVLA: skip expert when consecutive chunks are L2-similar
  --max-similar-skips 3                    # cap on consecutive cached returns (anti-drift safety)
```

The response JSON surfaces telemetry from each enabled knob so proof packets can show what actually happened (`safety_violations`, `deadline_exceeded`, `adaptive_enabled`, etc.). Skip-count from the action-similarity fast path lands on the `reflex_action_skip_total` Prometheus counter at `/metrics`.

## Trace archive — search + aggregate recorded `/act` traces

Pair `tether serve --record /tmp/traces` with `tether traces` to debug deployments:

```bash
# Run any /act traffic with recording on
tether serve ./p0 --record /tmp/traces &

# Filter recorded calls — failed pick-cube attempts in the last 7 days
tether traces query --dir /tmp/traces \
  --task pick-cube --status failed --since 7d \
  --output failures.json

# Aggregate by task → success-rate + p50/p95/p99/max latency per bucket
tether traces summary --dir /tmp/traces --since 24h --by task

# Group by model_hash to compare two deployed model versions
tether traces summary --dir /tmp/traces --by model --since 7d --output v1_vs_v2.csv
```

Filter dimensions: `--since` (`7d` / `24h` / `30m`), `--task` (case-insensitive substring), `--status` (`success` / `failed` / `any` — failed = `error` field present), `--model` (substring on `model_hash`), `--limit`. Output formats: rich table (default), JSON or CSV via `--output FILE` + auto-detected from suffix.

## Supported VLA models

| Model | HF ID | Params | Export status |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | 450M | ONNX + validated (max_diff=3.3e-06) |
| pi0 | `lerobot/pi0_base` | 3.5B | ONNX + validated (max_diff=6.0e-08) |
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX + validated (max_diff=2.38e-07) |
| pi0.5 LIBERO-10 (FluxVLA) | `Rylinjames/pi05-libero10-finetune-v1` | 3.62B | ONNX + validated. 97.85% LIBERO-10 avg. Apache-2.0. |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + validated (max_diff=8.34e-07, **live VLM conditioning**) |
| DreamZero WAM | `limxdynamics/FluxVLAEngine` | ~14B | ONNX + export. Joint video + action diffusion. 94.65% LIBERO avg. |
| OpenVLA | `openvla/openvla-7b` | 7.5B | `optimum-cli export onnx` + `tether.postprocess.openvla.decode_actions` |

`tether models list` browses the curated registry; `tether models info <id>` shows benchmarks; `tether models pull <id>` downloads. OpenVLA is a vanilla Llama-2-7B VLM — there's no custom action expert to reconstruct, so we defer to the standard HuggingFace export path and ship only the bin-to-continuous postprocess helper.

## Hardware targets

| Target | Hardware | Memory | Precision |
|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | fp16 |
| `orin` | Jetson Orin | 32 GB | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | fp16 |
| `thor` | Jetson Thor | 128 GB | fp8 |
| `desktop` | RTX / A100 | 40 GB | fp16 |

**Memory fit (monolithic ONNX on disk, FP32):** SmolVLA 1.6GB, pi0 12.5GB, pi0.5 13.0GB, GR00T 4.4GB, DreamZero ~28GB. SmolVLA fits comfortably on Orin Nano 8GB; **pi0 realistically needs Orin 16GB+ or a desktop NVIDIA GPU** — the 12.5GB monolithic ONNX cannot load on the 8GB Orin Nano even in FP16 (~6GB weights plus activations + OS). DreamZero (~14B params) requires A100/H100-class hardware (40GB+ VRAM).

`tether inspect targets` lists current profiles.

### Supported GPU architectures

| Architecture | Compute | Status | Notes |
|---|---|---|---|
| Ampere (RTX 30-series, A10G, A100) | sm_8.0–8.6 | ✅ Supported | Tested on Modal A10G + A100, RTX 4090 |
| Ada Lovelace (RTX 40-series, L4) | sm_8.9 | ✅ Supported | |
| Hopper (H100, H200) | sm_9.0 | ✅ Supported | |
| Jetson Orin (Orin Nano / NX / AGX) | sm_8.7 | ✅ Supported | JetPack 6.x |
| Jetson Thor | sm_10.x | ⚠️ Untested | Same Blackwell silicon as desktop; ORT ≥1.25.1 ships those kernels. Untested only for lack of hardware. |
| **Blackwell desktop (RTX 5090, RTX PRO 6000, B200, GB200)** | **sm_10.0 / 12.0** | **⚠️ Supported (smoke-validate)** | The pinned `onnxruntime-gpu>=1.25.1` ships Blackwell sm_120 kernels, so the earlier `InferenceSession`-init segfault is resolved. Smoke-validation recommended before declaring fully production-ready (open ORT threading issue #27621). On ORT < 1.25.1 the server still segfaults — `tether doctor` and the `tether go` Blackwell guard detect this and print the upgrade path. |
| Older NVIDIA (Turing RTX 20, GTX 16) | sm_7.5 | ⚠️ Best-effort | Should work but not in CI matrix |
| Pre-Tensor-Core (Maxwell Jetson Nano 4GB, GTX 9-series) | sm_5.x | ❌ Not supported | NVIDIA EOL'd this hardware at JetPack 4.6 (Python 3.6) — too old for modern ML stacks regardless. The bootstrap installer auto-detects and bails fast with redirect instructions. |

**For Blackwell users:** the default install pins `onnxruntime-gpu>=1.25.1`, which ships Blackwell sm_120 kernels — so `tether go` serves on RTX 50-series / B200 / GB200 hardware. The earlier `InferenceSession`-init segfault only occurs on ORT < 1.25.1; `tether doctor` and the `tether go` Blackwell guard detect that and print the upgrade path. Smoke-validate your model on-device before production (open ORT threading issue #27621).

A native TensorRT-LLM path (sm_100 / sm_120) is tracked upstream as an additional Blackwell runtime.

## Composable runtime wedges

Each wedge is a flag on `tether serve` (also flowed through `tether go`):

```bash
tether serve ./p0 \
  --embodiment franka \                   # per-robot action ranges + ActionGuard clamping
  --safety-config ./robot_limits.json \   # URDF-derived joint limits + EU AI Act audit log
  --adaptive-steps \                      # stop denoise loop early on velocity convergence
  --deadline-ms 33 \                      # return last-known-good action if over budget
  --cloud-fallback http://cloud:8000 \    # edge-first with cloud backup
  --inject-latency-ms 0 \                 # synthetic delay (B.4 A2C2 gate methodology)
  --record /tmp/traces \                  # JSONL request/response capture for replay
  --max-consecutive-crashes 5             # circuit breaker (503 + Retry-After: 60 on trip)
```

Every response surfaces telemetry from each enabled wedge (`guard_clamped`, `guard_violations`, `injected_latency_ms`, `inference_mode`, etc.).

## What Tether is and isn't

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export verified at cos=+1.0000000 on four VLA families — SmolVLA + pi0 + pi0.5 (flow-matching, num_steps=10) + GR00T N1.6 (DDPM DiT, num_steps=4, **with Eagle 2.5 VL backbone producing live image+language KV**). DreamZero (world-action model — config + PyTorch runtime today, DiT ONNX in progress) and OpenVLA (optimum-cli shim + postprocess helper) are supported but not yet numerically verified through a Tether ONNX export. Plus a composable runtime (serve + safety + turbo + split), edge-first design targeting Jetson + desktop NVIDIA GPUs.

**Isn't:** a training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Tether's moat is the deployment toolchain: cross-framework ONNX with verified numerical parity, composable safety wedges, ROS2 + Docker + HTTP serving, and a deterministic export receipt (`VERIFICATION.md`) your QA team can audit.

## Verified parity (the only load-bearing numbers)

Four ONNX artifacts in production, measured against PyTorch on shared seeded inputs:

| Artifact | Reference | first-action max_abs | verdict |
|---|---|---|---|
| **SmolVLA ONNX, num_steps=10** (production default) | `sample_actions(num_steps=10)` | **5.96e-07** | ✅ machine precision |
| **pi0 ONNX, num_steps=10** (production default) | `sample_actions(num_steps=10)` | **2.09e-07** | ✅ **machine precision** |
| **pi0.5 ONNX, num_steps=10** (production default) | `sample_actions(num_steps=10)` | **2.38e-07** | ✅ **machine precision** |
| **GR00T N1.6 ONNX, single-step DiT** (DDPM, loop external) | `GR00TFullStack.forward` | **8.34e-07** | ✅ **machine precision** |
| **GR00T N1.6 end-to-end 4-step denoise loop** | Python loop over PyTorch ref | **4.77e-07** | ✅ **machine precision** |
| **GR00T N1.6 Eagle VLM ONNX** (SigLIP + Qwen3 + mlp1, 1.87B) | `EagleExportStack` PyTorch | **4.25e-04** | ✅ machine precision |
| **GR00T N1.6 DiT with real VLM KV** (5-input `expert_stack_with_vlm.onnx`) | `GR00TFullStack(state, vlm_kv)` | **1.78e-05** | ✅ machine precision |
| **GR00T N1.6 end-to-end two-ONNX chain** (Eagle → DiT) | same chain in PyTorch | **1.90e-05** | ✅ parity + image-driven sensitivity verified (max_abs=0.21 on actions when input image changes) |
| SmolVLA ONNX, num_steps=1 | `sample_actions(num_steps=1)` | 1.55e-06 | ✅ machine precision |
| pi0 ONNX, num_steps=1 | `sample_actions(num_steps=1)` | 1.43e-06 | ✅ machine precision |

Plus PyTorch-level native-path sanity checks (`SmolVLAPolicy` with DecomposedRMSNorm swap vs reference = cos=1.0; `PI0Policy.predict_action_chunk` vs raw `sample_actions` = bit-exact).

**About the production defaults**: flow-matching VLAs (SmolVLA, pi0, pi0.5) canonically integrate the velocity field with 10 Euler steps — the ONNX bakes in the unrolled loop. GR00T is DDPM-style diffusion with 4 canonical steps — the ONNX exports one velocity step, and `tether serve` wraps it in the loop. All four match canonical PyTorch to machine precision. Getting pi0 / pi0.5 there required three interacting patches under `torch.export` (F.pad causal mask, frozen `DynamicLayer.update`, `past_kv.get_seq_length()` for mask assembly); GR00T's simpler DiT graph (no DynamicCache, no PaliGemma masking) traces cleanly via plain `torch.onnx.export(opset=19)` — no patches needed. Details in `reflex_context/01_architecture/pi0_monolithic_wrap_pattern.md`.

Full ledger: [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

**Latency numbers are intentionally not in the README yet** — earlier TRT FP16 tables were measured on a now-abandoned decomposed-ONNX path. `tether bench <export_dir>` reproduces on any hardware.

<!-- BEGIN:jetson-latency-table -->
<!-- Auto-populated by scripts/publish_jetson_latency.py from `tether bench realtime` certificates. Empty until certs are published. -->
<!-- END:jetson-latency-table -->

Reproduce on your own GPU with one command:

```bash
tether bench ./pi0 --iterations 100
```

### Multi-robot batching (`tether serve --max-batch N`)

Continuous batching on the HTTP layer: each `/act` request enters an asyncio queue; the server flushes the queue every `--batch-timeout-ms` (default 5ms) into one batched ONNX inference. Earlier measurements on the decomposed-ONNX path showed 2.3-2.9x throughput scaling at batch sizes 4-16.

## Status

**v0.11.2 — source-available under BSL 1.1.** Active development. Install, kick the tires, open issues loudly. Six VLA families (SmolVLA, pi0, pi0.5, GR00T, OpenVLA, DreamZero), Triton fast kernels, ZMQ transport, safetensors-direct loading.

## License

Source-available under the [Business Source License 1.1](LICENSE) — same model HashiCorp, MongoDB, Sentry, Cockroach, and Couchbase use. Free for any non-competitive use (personal, commercial, internal); restricts only competing hosted/embedded offerings. Auto-converts to Apache 2.0 in 4 years.

For commercial licensing inquiries (offering Tether as a hosted service to compete with FastCrest, OEM/embedded use, etc.): hello@fastcrest.com

---

Tether is built by [FastCrest](https://fastcrest.com). No signup, no telemetry by default.

Made with 🔥 Passion in San Francisco
