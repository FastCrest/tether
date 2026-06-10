# Draft: r/robotics post

## Title

**Open-source tool to export & serve SmolVLA / pi0 on Jetson or desktop (cos=1.0 verified)**

## Body (markdown)

Hi r/robotics —

I built [Tether](https://github.com/FastCrest/tether), an open-source CLI for taking a trained Vision-Language-Action model from a HuggingFace checkpoint to a working inference server you can hit from a robot.

**Verified today**: all four major open VLAs exported as monolithic ONNX, measured against PyTorch on shared seeded inputs:

| Model | Comparison | cos | first-action max_abs |
|---|---|---|---|
| SmolVLA (flow matching) | `sample_actions(num_steps=10)` | **+1.0000000** | **5.96e-07** (machine precision) |
| pi0     (flow matching) | `sample_actions(num_steps=10)` | **+1.0000000** | **2.09e-07** (machine precision) |
| pi0.5   (flow matching) | `sample_actions(num_steps=10)` | **+1.0000000** | **2.38e-07** (machine precision) |
| GR00T N1.6 (DDPM DiT, per-step) | `GR00TFullStack.forward` | **+1.0000000** | **8.34e-07** (machine precision) |
| GR00T N1.6 (DDPM, 4-step end-to-end denoise loop) | Python loop over PyTorch ref | **+1.0000000** | **4.77e-07** (machine precision) |

Getting pi0 / pi0.5 to cos=1.0 at num_steps=10 needed three interacting patches under `torch.export` + DynamicCache (F.pad causal mask, frozen `DynamicLayer.update` during the unrolled Euler loop, and `past_kv.get_seq_length()` for mask assembly). GR00T's simpler DiT graph traces cleanly with plain `torch.onnx.export` — no patches needed. Details in the repo's architecture doc.

Three commands from zero to serving:

```bash
pip install 'tether[serve,gpu] @ git+https://github.com/FastCrest/tether'
tether export lerobot/pi0_base --output ./p0
tether serve ./p0 --port 8000
```

Then `POST /act` returns flow-matching action chunks. Composable wedges let you build a real production pipeline without writing your own runtime:

- `--safety-config` — joint limits + NaN/Inf kill-switch + EU AI Act audit logging
- `--adaptive-steps` — early-stop denoising
- `--deadline-ms` — WCET fallback
- `--max-batch` — fleet serving with HTTP-layer continuous batching

**Plus:**
- **Docker image** at `ghcr.io/fastcrest/tether:latest` (x86 CUDA) — zero install
- **ROS2 bridge** (`tether ros2-serve`) — subs `sensor_msgs/Image` + `sensor_msgs/JointState` + `std_msgs/String`, pubs action chunks as `Float32MultiArray` at configurable Hz
- **`VERIFICATION.md`** — every export directory gets an auto-generated manifest (sha256 of every file, ONNX opset, parity results after `tether validate`) that your QA team can audit

**Honest disclaimers:**
- Alpha, single maintainer, Apache 2.0
- Jetson Orin Nano numbers not yet published — CloudJetson waitlisted, Orin Nano dev kit not on hand. Launch latency data is from Modal A10G; real Jetson numbers land when someone runs `tether bench` on a dev kit (happy to credit + thank-you gift)
- **pi0 / pi0.5 monolithic ONNX (12.5–13GB) doesn't fit on Orin Nano 8GB.** SmolVLA (1.6GB) and GR00T (4.4GB) are smaller. pi-family models currently want Orin 16GB+ or desktop GPU; FP16 engine rebuild for Orin Nano fit is v0.3
- **GR00T VLM conditioning (Eagle backbone) is zero-stubbed** — DiT + AdaLN action stack matches PyTorch at machine precision, but full multimodal control needs VLM export (v0.3)
- Earlier TRT FP16 latency tables were on a decomposed-ONNX path that's now abandoned; latency re-measurement on the monolithic path is a v0.3 item

What I'm specifically asking for:

1. **Testers** — install, point at your robot, open issues. <24h response commitment.
2. **Jetson Orin Nano benchmark contributor** — 30 min on a dev kit + you get real edge numbers published with your credit.
3. **Wedge feedback** — does `--safety-config / --adaptive-steps / --deadline-ms / --max-batch` match how you actually want to deploy?

Repo: https://github.com/FastCrest/tether
Verified numbers ledger: [measured_numbers.md](https://github.com/FastCrest/tether/blob/main/reflex_context/measured_numbers.md)

Happy to answer questions in the thread.

---

## Posting notes

- r/robotics values implementation depth + honesty over hype
- Lead with verified cos numbers — that's the differentiator
- Mention Apache 2.0 early
- Be ready for "why not just use [X]" — answer with specific code links
- Respond to every comment within 24h post-launch
