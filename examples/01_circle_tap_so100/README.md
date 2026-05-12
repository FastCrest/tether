# SO-ARM 100 + tablet circle-tap (end-to-end)

A complete recipe for the canonical Reflex bench game: SO-ARM 100 taps a green
circle on an Android tablet. Walks you through calibration, data collection,
ACT training (from scratch), deployment via `reflex serve`, and live eval.

Substrate vendored from [auto_soarm](https://github.com/0o8o0-blip/auto_soarm)
by 0o8o0 (MIT) per ADR `reflex_context/01_decisions/2026-05-06-vendor-auto-soarm.md`.

## Hardware

- SO-ARM 100 follower (or SO-101) on `/dev/ttyACM0`
- Android tablet, USB-connected, USB debugging enabled
- Capacitive stylus taped to the gripper
- Camera (any OpenCV-compatible) — required for collection only

## Software

```bash
# Install Reflex with the so100 + bench-game extras
pip install 'reflex-vla[so100]'           # adds scservo_sdk for the Pi-side driver
pip install lerobot==0.5.1                # for the from-scratch ACT training step
```

## Step 1 — preflight + calibration

Check the rig is ready:

```bash
reflex calibrate so100 preflight
```

If you don't have a camera attached yet:

```bash
reflex calibrate so100 preflight --skip-camera
```

Then run the full calibration sequence (corners + surface + tap model):

```bash
reflex calibrate so100 all
```

This will:
1. Move the arm to a smooth home pose.
2. Launch the tablet corner page over ADB.
3. Hand-guide the stylus to the four numbered tablet markers.
4. Probe the tablet surface for tap depth.
5. Fit the calibrated tap model.

Outputs land at `~/.reflex/calibration/so100/<id>/{corners,surface,model}.json`.

## Step 2 — collect demonstrations

```bash
reflex bench-game circle_lr collect --episodes 30
```

Records 30 clean demonstrations of the arm tapping circles. Output is a
LeRobot v3 dataset at `~/.reflex/bench/circle_lr/data/collections/<run>/`.

If you've opted into the contribution program (`reflex contribute --opt-in`),
each episode also flows into the Curate queue at
`~/.reflex/contribute/queue/`. Quality-scored, deduped, auto-tagged + uploaded
to the Reflex corpus on the next uploader pass.

## Step 3 — train ACT from scratch

```bash
reflex finetune \
    --policy act \
    --mode full \
    --chunk-size 31 \
    --dataset ~/.reflex/bench/circle_lr/data/collections/<run> \
    --output ~/.reflex/bench/circle_lr/artifacts/circle_lr_001 \
    --steps 30000 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --seed 1 \
    --skip-export    # skip auto-export; we'll deploy via reflex serve next
```

Recipe matches the auto_soarm baseline. ~30 minutes on an RTX 3090; longer on
slower hardware. The Pi can't realistically train; do this on a GPU machine.

The trained checkpoint lands at
`<output>/training/checkpoints/030000/pretrained_model/`.

## Step 4 — deploy + eval

On the GPU machine, serve the trained policy:

```bash
reflex serve <output>/training/checkpoints/030000/pretrained_model \
    --port 8000
```

On the Pi (with the arm + tablet), run eval against the served policy:

```bash
reflex bench-game circle_lr eval \
    --ckpt <output>/training/checkpoints/030000/pretrained_model \
    --episodes 8 \
    --remote-host <gpu-host> --remote-port 8000
```

Eval prints per-episode hit/miss + a summary at the end.

## Tuning + iteration

- **Lower hit rate?** Re-run calibration; check the stylus is firmly attached.
- **Slow inference?** Pull the [serve,gpu] extra on the GPU machine for the
  ORT-TRT EP path (5.55× speedup vs ORT-CUDA on SmolVLA-class workloads;
  ACT is smaller but the path matters once you scale chunk sizes).
- **Failure modes?** `reflex contribute --status` shows your contribution stats
  + failure-mode distribution. The Failure Corpus picks up the patterns
  automatically.

## Why this recipe?

- **From-scratch ACT** because circle-tap doesn't need a pretrained VLA's
  language understanding. ACT (Action Chunking Transformer) is small +
  trains fast on a couple thousand demonstrations.
- **Chunk size 31** matches auto_soarm's empirically-tuned recipe — long
  enough to capture the full press-hold-retract motion, short enough to
  avoid overfitting on the chunk boundary.
- **lr=1e-5** (not the usual 1e-4) because from-scratch ACT is sensitive
  to gradient explosions with a higher LR on small datasets.
- **30K steps** with `--save-freq 2500` gives 12 checkpoints to compare;
  the last few usually win.

## Attribution

Original recipe + scaffolding by 0o8o0 in
[auto_soarm](https://github.com/0o8o0-blip/auto_soarm) (MIT). Vendoring scope
+ adaptation rationale in
`reflex_context/01_decisions/2026-05-06-vendor-auto-soarm.md`.
