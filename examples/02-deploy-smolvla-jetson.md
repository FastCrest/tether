# 02 — Deploy SmolVLA on a Jetson Orin Nano

**What you'll see:** pull SmolVLA from HuggingFace, export it to ONNX for the Orin Nano, start the inference server, hit `/act`.

**Requires:** Jetson Orin Nano (8 GB) running JetPack 6.x with `nvidia-container-runtime`. About 2 GB of free disk for weights + ONNX. Network for the initial pull.

## Install on the Jetson

> **Two things that will break your install if you skip them:**
> 1. **Pin `numpy<2` before installing anything else.** The Jetson AI Lab torch and onnxruntime-gpu wheels are compiled against NumPy 1.x. If pip pulls NumPy 2.x, both libraries will crash on import with *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"*.
> 2. **Do NOT use `[gpu]` from standard PyPI.** Those wheels are `x86_64`-only and will fail with `ResolutionImpossible` on `aarch64`.

### Recommended: bootstrap installer

```bash
./install.sh
```

### Manual install

```bash
# 0. Create a clean venv (recommended)
python3 -m venv ~/tether-orin && source ~/tether-orin/bin/activate
pip install -U pip setuptools wheel

# 1. Pin numpy<2 FIRST — before torch or ort
pip install 'numpy<2'

# 2. Install Jetson-native torch + ort from the Jetson AI Lab index
pip install torch torchvision \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

pip install onnxruntime-gpu \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 3. Install tether with [serve] only (NOT [gpu], NOT [monolithic])
pip install 'fastcrest-tether[serve]'
```

> **Why not `[monolithic]`?** The monolithic export extra depends on `lerobot==0.5.1`, which requires **Python ≥ 3.12**. JetPack 6 ships Python 3.10. Export your model on a desktop/cloud machine with Python 3.12+, then copy the ONNX to the Jetson and serve it.

### Adding `tether` to your PATH (non-venv installs)
If you installed without a venv and see `tether: command not found`, add `~/.local/bin`:
```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### What each piece provides:
- `numpy<2` — ABI compatibility with Jetson AI Lab pre-built wheels
- `torch` / `onnxruntime-gpu` (from Jetson AI Lab) — GPU-accelerated inference, compiled for `aarch64` + JetPack CUDA
- `fastcrest-tether[serve]` — FastAPI + uvicorn HTTP inference server + embodiment validation

This pulls ~2 GB of dependencies. Takes 5-10 minutes on the Jetson.

## Deploy

Since monolithic export requires Python 3.12+ (for `lerobot`), the typical Jetson workflow is **export on a desktop/cloud host, serve on-device**.

### Step 1: Export on a Python 3.12+ machine

```bash
# On your desktop / cloud GPU (Python 3.12+)
pip install 'fastcrest-tether[serve,monolithic]'
tether export --model smolvla-base --out ./smolvla-export/
```

Then copy the export directory to the Jetson:
```bash
scp -r ./smolvla-export/ aihpc@<jetson-ip>:~/smolvla-export/
```

### Step 2: Serve on the Jetson

```bash
# On the Jetson (Python 3.10, [serve] only)
tether serve ~/smolvla-export/
```

What happens:
```
Loading ONNX into onnxruntime-gpu (CUDAExecutionProvider)...
TRT engine build (first time)...   ~60-90 sec
Warmup inference...                ~5 sec
✓ Server ready on http://0.0.0.0:8000
```

> **If `tether go` is available** (i.e. you have a pre-exported ONNX cached from a prior session), `tether go --model smolvla-base` will skip export (cache hit) and go straight to serve in ~2 sec.

## Or use the chat

If you'd rather have the chat agent run all this for you:

```bash
tether chat
you › deploy smolvla to my orin nano
```

Watch it call `list_targets`, `pull_model`, `export_model`, `serve_model` in sequence.

## Troubleshooting

- **"Missing dependencies for monolithic export"** — export requires Python 3.12+ with `[monolithic]`; run on a desktop/cloud host
- **"CUDA unavailable"** — confirm `nvidia-container-runtime` is set up on the Jetson; `tether doctor` will tell you which check failed
- **TRT engine build fails** — try `--no-trt` to fall back to plain CUDAExecutionProvider; usually means `trtexec` isn't on PATH
- **Disk full** — SmolVLA needs ~2 GB free for weights + ONNX. `tether inspect targets` shows memory budgets per hardware tier.
