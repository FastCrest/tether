# Getting started with Reflex

A 30-minute walkthrough of the typical first hour after `pip install reflex-vla[serve,gpu]`.

This guide assumes a Linux box with an NVIDIA GPU. CPU-only deployments work with `[serve]` instead of `[serve,gpu]` — every example below applies, just replace `--device cuda` with `--device cpu`.

---

## Install

```bash
# GPU box (requires CUDA 12 + cuDNN 9 — easiest via nvcr.io/nvidia/tensorrt container):
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/FastCrest/reflex-vla'

# CPU-only box:
pip install 'reflex-vla[serve,onnx] @ git+https://github.com/FastCrest/reflex-vla'
```

Or for development from source:

```bash
git clone https://github.com/FastCrest/reflex-vla
cd reflex-vla
pip install -e '.[serve,gpu,dev]'  # or [serve,onnx,dev] for CPU
```

If `--device cuda` errors with "CUDAExecutionProvider not available," see [Troubleshooting](#troubleshooting) below.

---

## 1. Pick a model and export it

Reflex auto-detects model type from the HuggingFace repo. Pick one based on what you have:

```bash
# Orin Nano-safe first try. Export on a dev/cloud box, then copy to Jetson.
reflex export lerobot/smolvla_base --target orin-nano --output ./smolvla

# pi0 / pi0.5 are datacenter targets in Reflex's registry today.
reflex export lerobot/pi0_base --target desktop --output ./pi0

reflex export lerobot/pi05_base --target desktop --output ./pi05

# GR00T N1.6 needs AGX Orin/Thor/datacenter-class memory, not Orin Nano.
reflex export nvidia/GR00T-N1.6-3B --target orin --output ./gr00t
```

Each command:
1. Downloads the checkpoint from HuggingFace (cached after first run)
2. Runs the model-specific exporter (auto-dispatches based on key prefix)
3. Writes ONNX + reflex_config.json into `--output`
4. Validates the ONNX numerically against the PyTorch reference (max_diff < 1e-5)
5. If trtexec is available, builds a TensorRT engine

The `--target` flag picks per-hardware tuning (FP16 on Orin, FP8 on Thor, etc.). It does not make an oversized model fit a small device. Use `reflex models list --device orin_nano` before targeting Jetson Orin Nano.

After export your output directory looks like:

```
./pi0/
├── expert_stack.onnx       # the graph (1.25MB)
├── expert_stack.onnx.data  # the weights (~1.3GB for pi0)
├── reflex_config.json      # model meta — used by serve
└── expert_stack.trt        # TRT engine (only if trtexec was available)
```

---

## 2. Serve it

```bash
reflex serve ./pi0 --port 8000
```

Reflex now listens on `http://localhost:8000` with three endpoints:

- `POST /act` — send `{instruction, state, image?}`, get back a 50-step action chunk
- `GET /health` — returns `{status, model_loaded, inference_mode}`
- `GET /config` — returns the saved reflex_config

Test it:

```bash
curl -X POST http://localhost:8000/act \
  -H 'content-type: application/json' \
  -d '{"instruction":"pick up the red cup","state":[0.1,0.2,0.3,0.4,0.5,0.6]}'
```

> **For drone deployments:** send IMU/attitude state instead of joint positions, e.g., `"state": [roll, pitch, yaw, altitude]`.

You'll get back something like:

```json
{
  "actions": [[...], [...], ...],
  "num_actions": 50,
  "action_dim": 32,
  "latency_ms": 47.3,
  "hz": 21.1,
  "denoising_steps": 10,
  "inference_mode": "onnx_gpu"
}
```

`actions` is a 50-step chunk × 32-dim per step. Your robot picks the first action, dispatches it, then either re-queries or works through more of the chunk depending on your control loop.

---

## 3. Add safety limits (`reflex guard`)

Most production deployments want the server to clamp actions before they reach the actuators. Reflex has a built-in guard for this.

```bash
# Generate a SafetyLimits config from your robot's URDF
reflex guard init --urdf ./my_robot.urdf --output ./safety.json

# Or hand-write one for a 6-joint arm (defaults to ±π on each joint)
reflex guard init --num-joints 6 --output ./safety.json
```

Then re-launch serve with the guard enabled:

```bash
reflex serve ./pi0 --port 8000 --safety-config ./safety.json
```

Now every `/act` response includes a `safety_violations` count and (if any clamps fired) a `safety_detail` field. Your robot gets clamped values — never the raw model output. The guard logs every check with a timestamp and input hash, satisfying EU AI Act Article 12 for high-risk AI systems.

To test the safety check standalone:

```bash
reflex guard check --config ./safety.json --num-joints 6
```

---

## 4. Fleet mode — serve N robots through one GPU

If you're running more than one robot, batch them:

```bash
reflex serve ./pi0 \
  --port 8000 \
  --max-batch 8 \
  --batch-timeout-ms 10
```

Now up to 8 concurrent `/act` requests get collected within a 10ms window and served in one batched ONNX inference. Verified on Modal A10G with pi0:

| `--max-batch` | Throughput | Speedup vs serial |
|---|---|---|
| 1 (baseline) | 17.1 qps | 1.00× |
| 4 | 40.0 qps | 2.34× |
| 8 | 46.6 qps | 2.73× |
| 16 | 49.3 qps | 2.88× |

Per-request latency *drops* with batching (no more queueing serially through the model). This is the single biggest scale lever for fleet operators.

---

## 5. Stack everything together

```bash
reflex serve ./pi0 \
  --port 8000 \
  --device cuda \
  --safety-config ./safety.json \
  --adaptive-steps \
  --deadline-ms 33 \
  --cloud-fallback http://cloud-vla-cluster:8000 \
  --max-batch 8 \
  --batch-timeout-ms 10
```

That's the full production-server invocation:
- `--safety-config` clamps unsafe actions
- `--adaptive-steps` skips denoise iterations on easy tasks (validated on pi0 — ~58% savings, action diff 0.07; unvalidated for other models in v0.1, server prints a warning)
- `--deadline-ms 33` returns last-known-good action if inference doesn't finish in 33ms
- `--cloud-fallback` routes through a remote cluster on edge failure
- `--max-batch 8` shares the GPU across up to 8 robots

Each enabled wedge surfaces telemetry in the `/act` response so you can monitor what's actually happening.

---

## 6. Pre-flight check before you ship

Before deploying a new checkpoint to a robot fleet, run:

```bash
reflex check ./pi0 --target orin-nano
```

Five quick validations (loadable, size fits target, key structure, dtype OK, no NaN/Inf). Exits non-zero on failure. Add to your CI.

---

## Common workflows

### Deploy SmolVLA to a Jetson Orin Nano

```bash
# On your dev box:
reflex export lerobot/smolvla_base --target orin-nano --output ./sv
scp -r ./sv jetson:~/sv

# On the Jetson (Jetpack 6.x with TensorRT preinstalled):
pip install 'reflex-vla[serve] @ git+https://github.com/FastCrest/reflex-vla'
reflex serve ./sv --port 8000 --device cuda
```

### Switch model without restarting the robot

Currently requires server restart — hot reload lands in v0.2.

### Test on simulation before real hardware

The `/act` endpoint is HTTP, so any sim that can do an HTTP POST works. MuJoCo + a small Python wrapper is the typical setup. Reflex doesn't ship a sim integration; that's intentionally outside our scope.

---

## Troubleshooting

**First thing to run when something is off:**

```bash
reflex doctor
```

Reports your Python / torch / ORT / TRT versions, which execution providers loaded, free disk, and the most common install-time gotchas. Often pinpoints the issue in one screen.

### `CUDAExecutionProvider not available`

ORT 1.20+ requires CUDA 12.x + cuDNN 9.x. The pip-installed `nvidia-cudnn-cu12` wheel is missing `libcudnn_adv.so.9`. Easiest fix is the NVIDIA TensorRT container:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:24.10-py3
# inside the container:
pip install 'reflex-vla[serve,gpu] @ git+https://github.com/FastCrest/reflex-vla'
reflex serve ...
```

Or pass `--device cpu` to explicitly run on CPU, or `--no-strict-providers` to allow silent CPU fallback (not recommended for benchmarks).

### `trtexec not found` warning during export

Normal on a dev box without TensorRT installed. ONNX is exported regardless. The TRT engine build is skipped — happens automatically when you run on a target with TensorRT (Jetson Jetpack, or x86 with `nvidia-tensorrt` installed).

### Server starts but `/act` returns `{"error": "Model not loaded"}`

The lifespan handler hasn't completed yet. Big models (pi0, gr00t) take 30-60s for ORT-GPU session creation. Wait for `GET /health` to return `model_loaded: true`.

### Action values look random / nonsensical

Expected in v0.1. The current ONNX export covers the action-expert denoising loop with random VLM conditioning. Real per-image conditioning lands when the VLM prefix encoder is wired (Phase II.4 / v0.2). For now you're getting valid action shapes but not task-relevant actions.

---

## What's next

If you have a Jetson Orin Nano and run this on real hardware, please open an issue with your latency numbers — that's the headline benchmark we don't have yet.

If you find a model that doesn't auto-detect correctly, check `reflex models` for current support, then open an issue with the HF ID.

If you want to add a new VLA family, see `src/reflex/exporters/` — each model has its own builder file (~200 lines). Pattern-match against `pi0_exporter.py` for flow-matching variants.
