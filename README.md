# Reflex

> by [FastCrest](https://fastcrest.com) — deployment infrastructure for vision-language-action models.

[![PyPI](https://img.shields.io/pypi/v/reflex-vla.svg)](https://pypi.org/project/reflex-vla/)
[![Python](https://img.shields.io/pypi/pyversions/reflex-vla.svg)](https://pypi.org/project/reflex-vla/)
[![License](https://img.shields.io/pypi/l/reflex-vla.svg)](https://github.com/FastCrest/reflex-vla/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/reflex-vla.svg)](https://pypi.org/project/reflex-vla/)

![Reflex VLA — pip install + reflex doctor + reflex --help on Modal A10G with TRT EP active](assets/reflex-tweet.gif)

**The deployment layer for VLAs** — take a Vision-Language-Action model off the training cluster and onto a robot. Now with **`reflex chat`** — talk to your robot fleet in plain English.

**Verified parity across ALL four major open VLAs.** Reflex's monolithic ONNX export matches the reference PyTorch policy to **cos = +1.000000** end-to-end on SmolVLA, pi0, pi0.5 (canonical 10-step flow-matching unrolled) and GR00T N1.6 (canonical 4-step DDIM loop external to the ONNX). Per-model first-action max_abs: SmolVLA 5.96e-07, pi0 2.09e-07, pi0.5 2.38e-07, GR00T 8.34e-07 — all at machine precision, shared seeded inputs. Full claim ledger in [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

One CLI, seven verbs, plus a chat agent.

## Install

**Recommended** — runs hardware + Python checks first, picks the right install extras for your machine:

```bash
curl -fsSL https://fastcrest.com/install | sh
```

The bootstrap installer detects your platform (Mac / Jetson Orin / NVIDIA GPU / CPU) and chooses the right extras automatically. It also bails early with a useful message on unsupported hardware (e.g. original 4 GB Jetson Nano — Maxwell GPU + JetPack 4.6 / Python 3.6, too old for VLAs).

**Manual install** if you know what you want:

```bash
pip install reflex-vla                            # core
pip install 'reflex-vla[serve,gpu,monolithic]'    # GPU production path
pip install 'reflex-vla[serve,onnx]'              # Mac / CPU runtime
```

Requires Python ≥ 3.10.

### What's new in v0.7.3 (2026-04-30)

Five production-relevant fixes from per-step expert ONNX export validation work. All take effect on `pip install --upgrade reflex-vla` + a re-export of any decomposed pi0.5 ONNX:

- **pi0.5 decomposed export precision** — freeze-flag for the `DynamicLayer.update` cache mutation, now applied to `PI05Pytorch.denoise_step` (was only applied to pi0). Removes stale-suffix K/V from attention in the baked-loop trace.
- **Expert ONNX precision** — `optimize=False` on `torch.onnx.export` to skip the constant-folding pass that evaluates fp64 sin/cos in fp32 (3e-5 max_abs error).
- **CUDA EP loadchain** — eager-dlopen `libcurand` / `libcufft` / `libcusparse` / `libnvJitLink`. Independently load-bearing for any consumer on a fresh image where torch's transitive curand path doesn't happen to be loadable.
- **`--rtc` actually works on decomposed servers** — pre-fix, `--rtc` constructed `RtcAdapter` with `action_buffer=None` whenever `--replan-hz`/`--execute-hz` weren't also passed, and `merge_and_update` silently no-op'd RTC's carry-forward. Now auto-builds the buffer.
- **Per-step expert ORT IOBinding** (preview for v0.8.0 per-step export feature) — pin past_kvs to device once per chunk via `OrtValue.ortvalue_from_numpy`. Drops per-step Euler loop overhead from +36% to +13%.

Full per-fix details + commit refs in [CHANGELOG.md](CHANGELOG.md).

### Upgrading

We ship patches frequently — make sure you're on the latest:

```bash
pip install --upgrade reflex-vla              # pip
uv add --refresh reflex-vla                   # uv (the --refresh flag is required;
                                              # uv caches the package index aggressively
                                              # and won't see new releases without it)
```

After upgrading, if you've previously run `reflex go` and the server fails to start, your
export cache may be stale (built by an older version, schema mismatch, etc.):

```bash
rm -rf ~/.cache/reflex/exports/<model_id>     # forces a fresh export on next reflex go
```

v0.5.4+ does this automatically when it detects a version mismatch — the manual step is
only needed for caches built by v0.5.3 or earlier.

## Performance

`reflex-vla[serve,gpu]` (v0.7+) uses ONNX Runtime's TensorRT execution provider out of the box. Measured on Modal A10G (Ampere, sm_8.6) on 2026-04-29 against SmolVLA monolithic (5 warmup + 20 measured forward passes, batch=1):

| Provider | Mean latency | p95 |
|---|---|---|
| `CUDAExecutionProvider` (ORT-CUDA fallback) | 108.11 ms | 108.68 ms |
| **`TensorrtExecutionProvider` (default in v0.7+)** | **19.49 ms** | 19.71 ms |

**5.55× faster.** The win comes from TensorRT's FP16 kernels + engine fusion. Older releases silently fell back to ORT-CUDA on most installs because `libnvinfer.so.10` and CUDA libs weren't on `LD_LIBRARY_PATH` — v0.7's `[serve,gpu]` extras pull `tensorrt>=10` and `reflex` patches `LD_LIBRARY_PATH` automatically at import.

### How to verify you're getting the win

```bash
reflex doctor
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

- Measured only on A10G + SmolVLA monolithic so far. Other model architectures (pi0.5 decomposed, GR00T) and other hardware tiers (Orin Nano, T4, H100) may show different ratios — broader matrix planned for v0.7.x.
- **Blackwell (RTX 50-series, B200, GB200) still not supported** in v0.7 — ORT's bundled cuBLAS/cuDNN don't ship sm_120 kernels yet. v0.5.5 documents this in detail; nothing changed here. Tracking ORT upstream for the fix.

### Opt-out

Adds ~2 GB to `[serve,gpu]` install (the `tensorrt` package + bundled libs). If you don't want it:

```bash
pip install 'reflex-vla[serve,gpu-min]'   # ORT-CUDA only, ~5x slower on transformers
```

Or disable the `LD_LIBRARY_PATH` patch (e.g. if it conflicts with another env-aware tool):

```bash
REFLEX_NO_LD_LIBRARY_PATH_PATCH=1 reflex go ...
```

## Quickstart — chat to it

```bash
reflex chat
```

```
you › what version am I running and what hardware can I deploy to?

  → show_version({})    → reflex --version    → "reflex 0.2.0"
  → list_targets({})    → reflex targets      → [orin-nano, orin, orin-64, thor, desktop]

You're running reflex 0.2.0. Supported targets:
  - orin-nano — Jetson Orin Nano: 8 GB, fp16
  - orin — Jetson AGX Orin 32GB: 32 GB, fp16
  - orin-64 — Jetson AGX Orin 64GB: 64 GB, fp16
  - thor — Jetson Thor: 128 GB, fp8
  - desktop — Desktop GPU (RTX 4090 / A100 / H100): 24 GB, fp16

Want me to show which models support each target, or run reflex doctor?
```

Chat understands 16 reflex commands (export, serve, bench, eval, distill, finetune, traces, doctor, etc.) and runs them as subprocess on your behalf. Powered by GPT-5 Mini through a proxy hosted at `chat.fastcrest.com` — free tier is 100 calls/day per machine, no signup, no API key.

> Bring your own key? `export FASTCREST_PROXY_URL=https://api.openai.com/v1` (coming in v0.3 — for now, the hosted proxy is the path).

## Quickstart — explicit deploy

```bash
# Browse the curated model registry
reflex models list

# Smoke test — probe hardware → resolve model → pull → export → serve
reflex go --model smolvla-base

# Now make it real with per-robot normalization (ships with franka, so100, ur5)
reflex go --model smolvla-base --embodiment franka

# Or with explicit hardware override
reflex go --model pi05-libero --embodiment franka --device-class a10g
```

Drop `--embodiment` for a quick smoke test — the server starts cleanly and `/act` returns
unscaled raw actions. Add `--embodiment <preset>` (or `--custom-embodiment-config <path>`)
when you're ready for per-robot normalization + ActionGuard clamping.

Then from your code:

```python
from reflex.client import ReflexClient

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

`reflex go` auto-detects your hardware (NVIDIA GPU / Jetson / CPU), picks the right model variant for that device, downloads weights from HuggingFace, and starts the /act endpoint. **No editing configs, no separate `reflex export` step, no manual variant selection.** For models that ship as raw PyTorch weights, you get the export command to run next.

### The verb surface

```
reflex chat             # NEW — natural-language interface to every command below
reflex go               # one-command-deploy: probe → resolve → pull → serve
reflex serve            # explicit-config server (full flag surface)
reflex doctor           # diagnose env + GPU + per-deploy issues
reflex models {list, pull, info, export}    # curated registry + lifecycle
reflex train  {finetune, distill}           # training operations
reflex validate {dataset, export}           # pre-flight checks
reflex inspect {replay, traces}             # forensic tools (legacy diagnostics: --advanced)
reflex traces {query, summary}              # search + aggregate recorded /act traces
reflex pro     {activate, status, deactivate}   # Pro tier license
reflex contribute  {opt-in, opt-out, status}    # Curate data contribution
```

11 visible verbs. 8 advanced/SO-100/internal commands stay callable directly (config, calibrate, bench-game, status, inspect bench/targets/guard/doctor) but hidden from `--help` to reduce cognitive load. Power-users can still invoke them; they just don't crowd the discovery surface.

Hidden legacy commands (`export`, `bench`, `replay`, etc.) stay callable for one release as alias bridges. Removed in v0.3.

### Install notes

- `[monolithic]` extra is required for the cos=+1.000000 verified export path (pins transformers==5.3.0)
- CPU-only: `pip install 'reflex-vla[serve,onnx,monolithic]'`
- GPU install needs the FULL cuDNN 9 system library (not just the pip wheel). Easiest path: NVIDIA's container `docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3`, then `apt-get install -y clang` (for lerobot→evdev), then the pip install
- `reflex serve` errors loudly if cuDNN can't load — no silent CPU fallback
- First `reflex go` downloads weights (~1-14 GB depending on model) — cached on subsequent runs
- First serve takes 10-70s warmup; `/health` returns HTTP 503 until ready, HTTP 200 after — load balancers correctly skip the server during warmup
- `reflex chat` works on the base install — no extras required. Network access required (calls FastCrest's hosted proxy).

### Docker — zero-install serve

```bash
# x86_64 CUDA runtime (cloud GPUs, dev workstations)
docker pull ghcr.io/fastcrest/reflex-vla:latest
docker run --gpus all \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/fastcrest/reflex-vla:latest

# Jetson Orin / Orin Nano / Thor (arm64 + nvidia container runtime)
docker pull ghcr.io/fastcrest/reflex-vla:latest-arm64
docker run --runtime=nvidia \
  -v $(pwd)/p0:/exports \
  -p 8000:8000 \
  ghcr.io/fastcrest/reflex-vla:latest-arm64
```

The container's default command is `reflex serve /exports --host 0.0.0.0 --port 8000`. Override with any `reflex` subcommand: `docker run ... ghcr.io/fastcrest/reflex-vla:latest export <hf_id>` etc.

Jetson arm64 image: built via QEMU cross-compile on tag push (`v*`). Bring-your-own-CUDA — the image deliberately doesn't bundle CUDA/cuDNN/TensorRT (those live on the Jetson under `/usr/local/cuda` and are ABI-locked to the host's JetPack version; the nvidia container runtime exposes them into the container).

### ROS2 — `reflex ros2-serve`

Wraps the inference loop as a ROS2 node. Subscribes to `sensor_msgs/Image`, `sensor_msgs/JointState`, and `std_msgs/String`; publishes action chunks as `std_msgs/Float32MultiArray` at a configurable rate.

```bash
# rclpy is NOT pip-installable. Install ROS2 via apt or robostack first:
source /opt/ros/humble/setup.bash   # or iron / jazzy

# Hidden alias — kept for back-compat through v0.2; will fold into
# `reflex serve --transport ros2` in a future release.
reflex ros2-serve ./my_export \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --task-topic  /reflex/task \
  --action-topic /reflex/actions \
  --rate-hz 20
```

Inference respects `--safety-config` (same limits file as HTTP serve).

When `onnxruntime-gpu` ships with the TensorRT execution provider (it does in v1.20+), `reflex serve` uses TRT FP16 automatically and caches the engine in `<export_dir>/.trt_cache` so subsequent server starts skip the engine-build cost. The first `reflex serve` takes ~30-90s to warm up; restart is ~1-2s.

## Pre-flight validation

Before deploying, validate your dataset (will it train?) and your export (does it serve cleanly?):

```bash
# Dataset: 8 falsifiable checks against your LeRobot v3.0 corpus
reflex validate dataset /path/to/lerobot_data --embodiment franka --strict

# Export: round-trip ONNX vs PyTorch parity at machine-precision threshold
reflex validate export ./p0 --model lerobot/pi0_base --threshold 1e-4
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

Exit codes: `0` pass, `1` fail (any fixture above threshold), `2` error (missing ONNX, bad config). Pipe `--output-json` for CI consumption, or run `reflex validate --init-ci` to scaffold a GitHub Actions workflow at `.github/workflows/reflex-validate.yml`.

## Composable wedges

Every wedge is a flag on `reflex serve`:

```bash
reflex serve ./p0 \
  --safety-config ./robot_limits.json \    # joint-limit clamping + EU AI Act audit log
  --adaptive-steps \                        # stop denoise loop early when velocity converges
  --deadline-ms 33 \                        # return last-known-good action if over budget
  --cloud-fallback http://cloud:8000 \     # edge-first with cloud backup
  --action-similarity-threshold 0.05 \     # FlashVLA: skip expert when consecutive chunks are L2-similar
  --max-similar-skips 3                    # cap on consecutive cached returns (anti-drift safety)
```

The response JSON surfaces telemetry from each enabled wedge so you can see what's actually happening (`safety_violations`, `deadline_exceeded`, `adaptive_enabled`, etc.). Skip-count from the action-similarity fast path lands on the `reflex_action_skip_total` Prometheus counter at `/metrics`.

## Trace archive — search + aggregate recorded `/act` traces

Pair `reflex serve --record /tmp/traces` with `reflex traces` to debug deployments:

```bash
# Run any /act traffic with recording on
reflex serve ./p0 --record /tmp/traces &

# Filter recorded calls — failed pick-cube attempts in the last 7 days
reflex traces query --dir /tmp/traces \
  --task pick-cube --status failed --since 7d \
  --output failures.json

# Aggregate by task → success-rate + p50/p95/p99/max latency per bucket
reflex traces summary --dir /tmp/traces --since 24h --by task

# Group by model_hash to compare two deployed model versions
reflex traces summary --dir /tmp/traces --by model --since 7d --output v1_vs_v2.csv
```

Filter dimensions: `--since` (`7d` / `24h` / `30m`), `--task` (case-insensitive substring), `--status` (`success` / `failed` / `any` — failed = `error` field present), `--model` (substring on `model_hash`), `--limit`. Output formats: rich table (default), JSON or CSV via `--output FILE` + auto-detected from suffix.

## Supported VLA models

| Model | HF ID | Params | Export status |
|---|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | 450M | ONNX + validated (max_diff=3.3e-06) |
| pi0 | `lerobot/pi0_base` | 3.5B | ONNX + validated (max_diff=6.0e-08) |
| pi0.5 | `lerobot/pi05_base` | 3.62B | ONNX + validated (max_diff=2.38e-07) |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 3.29B | ONNX + validated (max_diff=8.34e-07, **live VLM conditioning**) |
| OpenVLA | `openvla/openvla-7b` | 7.5B | `optimum-cli export onnx` + `reflex.postprocess.openvla.decode_actions` |

`reflex models list` browses the curated registry; `reflex models info <id>` shows benchmarks; `reflex models pull <id>` downloads. OpenVLA is a vanilla Llama-2-7B VLM — there's no custom action expert to reconstruct, so we defer to the standard HuggingFace export path and ship only the bin-to-continuous postprocess helper.

## Hardware targets

| Target | Hardware | Memory | Precision |
|---|---|---|---|
| `orin-nano` | Jetson Orin Nano | 8 GB | fp16 |
| `orin` | Jetson Orin | 32 GB | fp16 |
| `orin-64` | Jetson Orin 64 | 64 GB | fp16 |
| `thor` | Jetson Thor | 128 GB | fp8 |
| `desktop` | RTX / A100 | 40 GB | fp16 |

**Memory fit (monolithic ONNX on disk, FP32):** SmolVLA 1.6GB, pi0 12.5GB, pi0.5 13.0GB, GR00T 4.4GB. SmolVLA fits comfortably on Orin Nano 8GB; **pi0 realistically needs Orin 16GB+ or a desktop NVIDIA GPU** — the 12.5GB monolithic ONNX cannot load on the 8GB Orin Nano even in FP16 (~6GB weights plus activations + OS). FP16 engine rebuild + Orin Nano fit work is tracked for v0.3.

`reflex inspect targets` lists current profiles.

### Supported GPU architectures

| Architecture | Compute | Status | Notes |
|---|---|---|---|
| Ampere (RTX 30-series, A10G, A100) | sm_8.0–8.6 | ✅ Supported | Tested on Modal A10G + A100, RTX 4090 |
| Ada Lovelace (RTX 40-series, L4) | sm_8.9 | ✅ Supported | |
| Hopper (H100, H200) | sm_9.0 | ✅ Supported | |
| Jetson Orin (Orin Nano / NX / AGX) | sm_8.7 | ✅ Supported | JetPack 5.x or 6.x |
| Jetson Thor | sm_10.x | ⚠️ Untested | Should work — same Blackwell silicon as desktop, but ORT-bundled CUDA EP needs Blackwell support (see below) |
| **Blackwell desktop (RTX 5090, RTX PRO 6000, B200, GB200)** | **sm_10.0** | **❌ Not yet supported** | ORT's bundled cuBLAS/cuDNN don't ship sm_100 kernels. Server segfaults at `InferenceSession` init. **Workaround:** use `reflex chat` (no GPU needed), or `/act` testing on Modal cloud or non-Blackwell GPU until ORT updates ship. Tracking: [microsoft/onnxruntime#blackwell](https://github.com/microsoft/onnxruntime/issues) |
| Older NVIDIA (Turing RTX 20, GTX 16) | sm_7.5 | ⚠️ Best-effort | Should work but not in CI matrix |
| Pre-Tensor-Core (Maxwell Jetson Nano 4GB, GTX 9-series) | sm_5.x | ❌ Not supported | NVIDIA EOL'd this hardware at JetPack 4.6 (Python 3.6) — too old for modern ML stacks regardless. The bootstrap installer auto-detects and bails fast with redirect instructions. |

**For Blackwell users right now:** the bootstrap installer accepts your hardware and the package installs cleanly, but `reflex go` will segfault at server startup. The real fix requires ORT to ship Blackwell-aware bundled binaries (no published timeline). Workarounds: chat-only mode (no GPU needed), `reflex doctor`, `reflex models list` all work fine. `/act` and TRT-engine inference need a non-Blackwell GPU temporarily.

A Blackwell-specific runtime path via TensorRT-LLM (which DOES support sm_100) is planned for v0.7+ — see [reflex_context ADR queue](https://github.com/FastCrest/reflex-vla/issues).

## Composable runtime wedges

Each wedge is a flag on `reflex serve` (also flowed through `reflex go`):

```bash
reflex serve ./p0 \
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

## What Reflex is and isn't

**Is:** the deployment layer between a trained VLA and a real robot. Cross-framework export verified at cos=+1.0000000 on all four major open VLAs — SmolVLA + pi0 + pi0.5 (flow-matching, num_steps=10) + GR00T N1.6 (DDPM DiT, num_steps=4, **with Eagle 2.5 VL backbone producing live image+language KV**) — plus a composable runtime (serve + safety + turbo + split), edge-first design targeting Jetson + desktop NVIDIA GPUs.

**Isn't:** a training framework (PyTorch/JAX own that) or a cloud inference provider (vLLM/Baseten own that). Reflex's moat is the deployment toolchain: cross-framework ONNX with verified numerical parity, composable safety wedges, ROS2 + Docker + HTTP serving, and a deterministic export receipt (`VERIFICATION.md`) your QA team can audit.

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

**About the production defaults**: flow-matching VLAs (SmolVLA, pi0, pi0.5) canonically integrate the velocity field with 10 Euler steps — the ONNX bakes in the unrolled loop. GR00T is DDPM-style diffusion with 4 canonical steps — the ONNX exports one velocity step, and `reflex serve` wraps it in the loop. All four match canonical PyTorch to machine precision. Getting pi0 / pi0.5 there required three interacting patches under `torch.export` (F.pad causal mask, frozen `DynamicLayer.update`, `past_kv.get_seq_length()` for mask assembly); GR00T's simpler DiT graph (no DynamicCache, no PaliGemma masking) traces cleanly via plain `torch.onnx.export(opset=19)` — no patches needed. Details in `reflex_context/01_architecture/pi0_monolithic_wrap_pattern.md`.

Full ledger: [reflex_context/measured_numbers.md](reflex_context/measured_numbers.md).

**Latency numbers are intentionally not in the README yet** — earlier TRT FP16 tables were measured on a now-abandoned decomposed-ONNX path. Desktop GPU + Jetson latency re-measurement is tracked for v0.3. `reflex bench <export_dir>` reproduces on any hardware.

Reproduce on your own GPU with one command:

```bash
reflex bench ./pi0 --iterations 100
```

### Multi-robot batching (`reflex serve --max-batch N`)

Continuous batching on the HTTP layer: each `/act` request enters an asyncio queue; the server flushes the queue every `--batch-timeout-ms` (default 5ms) into one batched ONNX inference. Earlier measurements on the decomposed-ONNX path showed 2.3-2.9× throughput scaling at batch sizes 4-16; those numbers are being re-measured on the monolithic path for v0.3.

## License

Apache 2.0

## Status

**v0.5 — source-available under BSL 1.1.** Active development. Install, kick the tires, open issues loudly. We're looking for the first 20 robotics teams actually deploying this; your feedback shapes v0.6.

## License

Source-available under the [Business Source License 1.1](LICENSE) — same model HashiCorp, MongoDB, Sentry, Cockroach, and Couchbase use. Free for any non-competitive use (personal, commercial, internal); restricts only competing hosted/embedded offerings. Auto-converts to Apache 2.0 in 4 years.

For commercial licensing inquiries (offering Reflex as a hosted service to compete with FastCrest, OEM/embedded use, etc.): hello@fastcrest.com

---

Reflex is built by [FastCrest](https://fastcrest.com). No signup, no telemetry by default.

Made with 🔥 Passion in San Francisco

