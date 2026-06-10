# Draft: Show HN — Tether (VLA deployment toolchain)

## Title (~80 chars)

**Show HN: Tether — ONNX export + serve for Vision-Language-Action models (cos=1.0 verified)**

## Body (text post)

Hi HN —

I built Tether because the path from "we have a trained Vision-Language-Action model" to "it runs on a real robot" is painful. Every VLA team writes their own export pipeline. Most break silently under FP16 / TRT / Jetson constraints.

**What's verified today:** I export all four major open VLAs as monolithic ONNX — SmolVLA (HuggingFace LeRobot), pi0, pi0.5 (Physical Intelligence, via lerobot), and GR00T N1.6 (NVIDIA). Measured on shared seeded inputs against PyTorch eager:

**Flow-matching VLAs (SmolVLA, pi0, pi0.5), 10-step Euler loop unrolled into the ONNX:**
- **SmolVLA num_steps=10 ONNX**: cos = +1.0000000, max_abs = 5.96e-07. **Machine precision.**
- **pi0 num_steps=10 ONNX**: cos = +1.0000000, max_abs = 2.09e-07. **Machine precision.**
- **pi0.5 num_steps=10 ONNX**: cos = +1.0000000, max_abs = 2.38e-07. **Machine precision.**

**DDPM VLA (GR00T N1.6), per-step DiT exported + loop external to ONNX:**
- **GR00T single-step**: cos = +1.0000000, max_abs = 8.34e-07 vs `GR00TFullStack.forward`. **Machine precision.**
- **GR00T 4-step denoise loop** (Python loop around ONNX vs same loop around PyTorch): cos = +1.0000000, max_abs = 4.77e-07. **Machine precision.**

Getting pi0 / pi0.5 to cos=1.0 at num_steps=10 required three interacting patches (under `torch.export` + `transformers==5.3.0` + DynamicCache): (1) F.pad + logical AND for the block-causal mask instead of `torch.cat` (cat loses the suffix dim under FakeTensor); (2) freeze `DynamicLayer.update` during the unrolled Euler loop so the cache doesn't grow across iterations; (3) use `past_kv.get_seq_length()` instead of the pad-mask shape for mask assembly. GR00T's simpler DiT graph (no DynamicCache, no PaliGemma masking) traces cleanly with plain `torch.onnx.export(opset=19)` — no patches needed.

```bash
pip install 'tether[serve,gpu] @ git+https://github.com/FastCrest/tether'
tether export lerobot/smolvla_base --output ./smol
tether serve ./smol --port 8000
# POST /act returns 50-step action chunks
```

**Also ships:**
- Docker image published to GHCR (`ghcr.io/fastcrest/tether:latest`) — no CUDA driver wrangling
- ROS2 bridge (`tether ros2-serve`) — subs image/state/task, pubs action chunks
- Safety guard with NaN/Inf rejection + consecutive-clamp kill-switch
- Auto-generated `VERIFICATION.md` per export directory — sha256 of every file, opset, and (after `tether validate`) per-fixture cos/L2 numbers for audit

**The hard parts were non-obvious.** Three issues under transformers 5.x took most of the session to isolate:

1. `transformers 5.4+` has a `q_length` scalar regression in `masking_utils.sdpa_mask` that breaks onnx-diagnostic patches. Pinning `transformers==5.3.0` fixes it.
2. SmolVLM2's vision embedder does `torch.where(bool_mask, torch.full(fill_value=0), float_tensor)` where `fill_value=0` creates an int64 branch. `torch.export` traces this with mismatched dtypes and the resulting ONNX `Where` op is rejected by onnxruntime at load time. Fix: wrap `torch.where` to insert explicit `torch.promote_types`.
3. Even with a clean aten graph, `torch.onnx.export` sometimes lowers `index_put` to a `Where(bool, int64, float)` ONNX node. Fix: post-export pass that walks Where nodes and inserts Cast nodes targeting the declared output dtype.

**What's explicitly NOT done:**
- **GR00T VLM conditioning (Eagle backbone).** The current GR00T export covers the DiT + AdaLN action stack with a zero VLM-KV placeholder (same convention pi0/SmolVLA use for prefix=None). Full multimodal control needs Eagle VLM export + per-step conditioning KV — v0.3
- Jetson latency numbers — CloudJetson has only AGX Orin 64GB available; Orin Nano waitlisted. Launch numbers are from Modal A10G/A100; real Jetson data comes when someone runs `tether bench` on a dev kit
- **Orin Nano 8GB fit for pi0 / pi0.5.** The pi0 / pi0.5 monolithic ONNX is 12.5–13GB (FP32) and does not fit on Orin Nano 8GB in any precision once activations + OS are counted. SmolVLA (1.6GB) fits fine; GR00T (4.4GB) likely does in FP16 but is unverified on the Nano. pi-family models realistically need Orin 16GB+ or a desktop NVIDIA GPU. FP16 engine rebuild + Orin Nano fit is a v0.3 item
- Earlier TRT FP16 latency tables were on a now-abandoned decomposed-ONNX path; latency re-measurement on the monolithic path is in v0.3

Repo: https://github.com/FastCrest/tether
Verified numbers ledger: [reflex_context/measured_numbers.md](https://github.com/FastCrest/tether/blob/main/reflex_context/measured_numbers.md)

Apache 2.0, single maintainer. Looking for testers — especially anyone with a real robot or a Jetson Orin Nano dev kit. Open an issue, I respond fast.

---

## Tone notes

- Lead with verified cos numbers; avoid hype
- Disclose what's NOT done before commenters ask
- Don't oversell (no "revolutionary" / "game-changing")
- No "we" — single maintainer
