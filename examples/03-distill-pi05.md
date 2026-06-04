# 03 — Distill pi0.5 to a 1-NFE student with SnapFlow

**What you'll see:** train a 1-step student from a 10-step pi0.5 teacher using SnapFlow self-distillation, then export the student to ONNX.

**Requires:** A100 (or H100) with ≥40 GB VRAM, ~6-8 hours wall clock, ~$30-60 in compute (cheaper if you self-host). The base `[monolithic]` extras + a LeRobot-format dataset.

## Why distill?

A pi0.5 model evaluates a 10-step Euler flow-matching loop on every `/act` call. The student we're about to train collapses that into **1 step** — the policy still produces a valid action chunk, but inference latency drops ~10× with minimal task-success regression (often *zero* on smooth-trajectory tasks; see [our 2026-04-26 LIBERO experiment](https://github.com/FastCrest/reflex-vla/blob/main/reflex_context/03_experiments/2026-04-26-self-distilling-serve-libero-regression-student-beats-teacher.md) where a 1-NFE student *beat* the 10-step teacher 32/50 vs 28/50).

SnapFlow ([arxiv 2604.05656](https://arxiv.org/abs/2604.05656)) is the canonical training-free self-distillation method. Tether was the first open-source reproduction.

## Install

```bash
pip install 'tether[monolithic]'
```

## The command

```bash
tether train distill \
    --teacher lerobot/pi05_libero \
    --dataset lerobot/libero_object \
    --steps 10000 \
    --batch-size 8 \
    --lr 1e-5 \
    --output ~/distilled/pi05_libero_1nfe
```

What you'll see (approximate):

```
SnapFlow distillation
  teacher:  lerobot/pi05_libero
  dataset:  lerobot/libero_object
  output:   /home/.../distilled/pi05_libero_1nfe
  steps:    10000
  ...
[step 100/10000]   loss=6.42   step_time=2.1s   eta=5h47m
[step 500/10000]   loss=2.18   step_time=2.0s   eta=5h12m
[step 1000/10000]  loss=0.94   step_time=2.0s   eta=4h59m
[step 5000/10000]  loss=0.21   step_time=2.0s   eta=2h47m
[step 10000/10000] loss=0.18   step_time=2.0s   done
Saving checkpoint... ~/distilled/pi05_libero_1nfe/{model.safetensors,config.json,training_args.json}
```

Total: ~5-6 hours on a single A100-80 GB.

## Export the student

```bash
tether export ~/distilled/pi05_libero_1nfe --target desktop --from-distilled
```

`--from-distilled` tells the exporter to treat the input as a SnapFlow student checkpoint — auto-detects pi0 vs pi0.5 from `config.json`, exports at 1-NFE with `target_time=1` baked into the ONNX. Output has the same I/O signature as the matching teacher family's monolithic export, so `tether serve` loads it through the standard path with no special flags.

```
Tether Export (SnapFlow student, 1-NFE)
  Model:      ~/distilled/pi05_libero_1nfe
  Output:     ./tether_export
Loading PyTorch model...
Tracing torch.export...
Writing ONNX (1.34 GB)...
Validating cos=+1.0 vs PyTorch reference...
Monolithic export complete in 187.3s
  ONNX: ./tether_export/model.onnx
  Size: 1342.7 MB
  Verification manifest: ./tether_export/VERIFICATION.md
```

## Serve it

```bash
tether serve ./tether_export --port 8000
```

Hit `/act` exactly like in [02-deploy-smolvla-jetson.md](02-deploy-smolvla-jetson.md). The `/act` response will show a `latency_ms` ~10× lower than the 10-step teacher.

## Validate the regression

Before deploying the student, run a LIBERO eval to confirm task success doesn't regress:

```bash
tether eval ./tether_export --suite libero_object --num-episodes 50
```

Expect: parity with the teacher (or sometimes better — see linked experiment). If success rate drops more than ~5pp, the distillation hyperparams need tuning (most often: more steps, lower LR).

## Where it goes wrong

- **Out of memory at step 0** — pi0.5 + a batch of 8 LIBERO episodes wants ~50 GB VRAM. Drop `--batch-size 4` or swap to A100-80.
- **Loss plateaus at ~3-5** — your dataset may not match the teacher's training distribution (action ranges, image sizes). Run `tether validate dataset <path>` first.
- **`--from-distilled` complains about config.json** — the student dir needs both `model.safetensors` (with `target_time_embed_mlp.*` keys) and `config.json` (specifying `model_type: pi05` or `pi0`). The trainer writes both; if you copied files manually, check both are present.

## Next

- [04-record-and-replay.md](04-record-and-replay.md) — capture live `/act` traces on the student and replay them against the teacher to confirm action-chunk parity in production.
