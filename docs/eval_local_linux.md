# SmolVLA matched evaluation on Linux

Tether’s real LIBERO evaluator loads the checkpoint named on the command line. It accepts a full local checkpoint, a Hugging Face repository pinned to a commit, or a SmolVLA LoRA adapter with an explicit base checkpoint. A missing or incompatible checkpoint stops the run; the evaluator does not substitute a reference policy.

## Host contract

- Linux x86-64 with a working NVIDIA driver and CUDA visible to PyTorch
- Python 3.12 and `uv`
- Enough GPU memory for SmolVLA, the LoRA adapter, observations, and the LIBERO simulator
- `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`

Run `scripts/setup_libero_eval_linux.sh`. On Debian or Ubuntu it installs the required compiler, FFmpeg, EGL, GL and OSMesa system packages. It then creates `.venv-eval-linux`, installs Tether's `eval-local` dependencies, and checks out LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. The script prints the resolved PyTorch, CUDA and GPU versions. It does not start a cloud instance.

On another Linux distribution, install equivalents for `build-essential`, `cmake`, `ffmpeg`, `git`, `libegl1`, `libgl1`, `libglib2.0-0`, `libglvnd0`, `libosmesa6` and `libosmesa6-dev` before running the script.

For the September 2026 qualification run, pin the public inputs to these measured revisions:

```bash
parent=lerobot/smolvla_base
parent_revision=c83c3163b8ca9b7e67c509fffd9121e66cb96205
dataset=lerobot/libero
dataset_revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4

.venv-eval-linux/bin/python -m tether.cli train finetune \
  --base "$parent" --base-revision "$parent_revision" \
  --dataset "$dataset" --dataset-revision "$dataset_revision" \
  --output evidence/training --steps 5 --batch-size 1 \
  --lora-rank 8 --mode lora --precision bf16 --skip-export
```

Five steps are a loader and evidence smoke test, not a useful adaptation claim. Tether downloads the named base revision before training, passes the dataset revision to LeRobot, and writes `training-source.json`. Keep that receipt with the adapter checkpoint and evaluation reports.

## End-to-end smoke test

Use an exact parent revision and the adapter directory produced by `tether finetune`:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python=.venv-eval-linux/bin/python
suite=libero_10
tasks=0
episodes=1
seed=7001
parent=lerobot/smolvla_base
revision=c83c3163b8ca9b7e67c509fffd9121e66cb96205
adapter=/absolute/path/to/checkpoints/<step>/pretrained_model

$python -m tether.cli eval "$parent" \
  --checkpoint-revision "$revision" --runtime local --tasks "$suite" \
  --task-indices "$tasks" --num-episodes "$episodes" --seed "$seed" \
  --output evidence/development/parent

$python -m tether.cli eval "$adapter" \
  --checkpoint-kind smolvla-lora --adapter-base "$parent" --adapter-base-revision "$revision" \
  --runtime local --tasks "$suite" --task-indices "$tasks" \
  --num-episodes "$episodes" --seed "$seed" \
  --output evidence/development/candidate
```

Confirm that both `report.json` files contain the requested `checkpoint.identity`, identical task and episode keys, and real per-episode outcomes. Repeat with new seeds in a separate `evidence/holdout` directory. Do not reuse development seeds for held-out evidence.

Studio automates this two-arm sequence. Its review blocks execution on macOS, without CUDA, when the parent revision is unpinned, when a candidate is missing, or when held-out seeds overlap development evidence. A completed run records both checkpoint identities, exact cases, and report hashes. Selecting the candidate requires a completed held-out comparison and an explicit decision.

## Current qualification state

The checkpoint contract and orchestration are covered by local tests. The SmolVLA/LIBERO path has not been executed on this Apple Silicon host, so GPU readiness and task success remain untested until the smoke procedure above completes on a compatible Linux system.
