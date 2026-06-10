# Draft post for huggingface/lerobot#3146

> Issue title (paraphrased): "Add ONNX export and TensorRT support for VLA models so they can be deployed to edge hardware (Jetson, etc.)"

---

## Draft v2 — to be posted as a comment on issue #3146

Hi all — picking this up from the outside.

I've been building a standalone export + serve toolchain for flow-matching VLAs. Apache 2.0, single maintainer. Focused on the two most-used LeRobot models:

| Model | HF ID | ONNX status |
|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | ✅ cos=+1.0000000 vs PyTorch @ num_steps=10, max_abs 5.96e-07 |
| pi0 | `lerobot/pi0_base` | ✅ cos=+1.0000000 vs PyTorch @ num_steps=10, max_abs 2.09e-07 |
| pi0.5 | `lerobot/pi05_base` | ✅ cos=+1.0000000 vs PyTorch @ num_steps=10, max_abs 2.38e-07 |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | ✅ cos=+1.0000000 vs PyTorch (DiT+AdaLN single-step), max_abs 8.34e-07 |

Repo: https://github.com/FastCrest/tether

### What's actually verified

One ONNX artifact per model, measured against PyTorch on shared seeded inputs:

**Flow-matching VLAs (SmolVLA, pi0, pi0.5) at canonical num_steps=10**: the ONNX unrolls the 10-step Euler loop at trace time. Matches PyTorch `sample_actions(num_steps=10)` to machine precision.
- SmolVLA num_steps=10 ONNX: cos=+1.0000000, first-action max_abs=5.96e-07
- pi0 num_steps=10 ONNX: cos=+1.0000000, first-action max_abs=2.09e-07
- pi0.5 num_steps=10 ONNX: cos=+1.0000000, first-action max_abs=2.38e-07

**GR00T N1.6 (DDPM DiT)**: the ONNX is the *per-step velocity function*; `tether serve` runs the canonical 4-step DDIM loop around it. Both single-step and end-to-end 4-step loop match PyTorch at machine precision.
- GR00T single-step ONNX: cos=+1.0000000, first-action max_abs=8.34e-07
- GR00T 4-step denoise loop: cos=+1.0000000, first-action max_abs=4.77e-07

**How we got pi0 / pi0.5 to cos=1.0 at num_steps=10**: three interacting patches under `torch.export`:
1. Replace `torch.cat` of the block-causal mask with `F.pad + logical AND` (cat loses the suffix dim under FakeTensor tracing)
2. Freeze `DynamicLayer.update` during the Euler loop so the cache doesn't grow across unrolled iterations
3. Use `past_kv.get_seq_length()` (not the pad-mask shape) for mask assembly

GR00T's simpler DiT graph (no DynamicCache, no PaliGemma masking) traces cleanly via plain `torch.onnx.export(opset=19)` with no patches.

**pi0 native-path sanity**: `PI0Policy.predict_action_chunk` wrapper vs raw `sample_actions` = bit-exact (max_abs = 0.0).

Exporter uses onnx-diagnostic's `torch_export_patches(patch_transformers=True)` under `transformers==5.3.0` for pi-family (5.4+ has a `q_length` scalar regression). GR00T uses stock `torch.onnx.export`. Reproducers: `scripts/modal_{smolvla,pi0,pi05,gr00t}_monolithic_export.py --parity`.

### How to try it

```bash
# HTTP serve
pip install 'fastcrest-tether[serve,gpu] @ git+https://github.com/FastCrest/tether'
tether export lerobot/smolvla_base --target desktop --output ./smol
tether serve ./smol --port 8000

# Docker
docker run --gpus all -v $(pwd)/smol:/exports -p 8000:8000 \
  ghcr.io/fastcrest/tether:latest

# ROS2 (after sourcing ROS2 humble/iron/jazzy)
tether ros2-serve ./smol --rate-hz 20
```

Every export directory gets a `VERIFICATION.md` with sha256 of every file, ONNX opset, and (after `tether validate`) per-fixture cos/L2 numbers.

### What's in scope vs not

**In scope for v0.2:** SmolVLA + pi0 + pi0.5 + GR00T cos-verified monolithic ONNX, Docker image, ROS2 bridge, NaN/Inf kill-switch, auto-generated VERIFICATION.md.

**Not in scope yet:** Jetson latency numbers — CloudJetson only has AGX Orin 64GB, Orin Nano is waitlisted; latency re-measurement on the monolithic path (earlier TRT FP16 tables were from a now-abandoned decomposed-ONNX path). **GR00T VLM conditioning (Eagle backbone)** — currently zero-stubbed (same convention as pi0/SmolVLA's prefix=None); real multimodal control is a v0.3 item. **Memory note:** pi0 / pi0.5 monolithic ONNX is 12.5–13GB and does not fit on Orin Nano 8GB — SmolVLA (1.6GB) and GR00T (4.4GB) are smaller. Large pi-family models currently need Orin 16GB+ or desktop GPU; FP16 engine rebuild for Orin Nano fit is v0.3.

### Asks

1. **Testers welcome.** Install it, point at your robot, open issues. Response time <24h.
2. **Jetson Orin Nano benchmark contributor.** If you've got a dev kit and 30 min, I'd love a real-hardware latency number. Will credit + send a small thank-you.
3. **Architectural feedback.** Happy to upstream subsets into LeRobot if maintainers want, or stay separate — whichever fits best.

Honest disclaimer: alpha, single maintainer, no funding. If it works for you, great; if not, please tell me how it broke.

— @rylinjames
