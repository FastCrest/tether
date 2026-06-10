# Tether VLA v0.2 — the deployment layer for Vision-Language-Action models

*Draft announcement. Master post; channel-specific variants live in
`launch/{lerobot_3146,show_hn,reddit_robotics}_draft.md`.*

## TL;DR

Tether takes a trained VLA checkpoint and produces a monolithic ONNX
that **matches the reference PyTorch policy to machine precision** —
verified on all four major open VLAs (SmolVLA, pi0, pi0.5, GR00T N1.6)
at cos=+1.0000000. Plus a FastAPI server, Docker image, ROS2 bridge,
safety kill-switch, and an auto-generated verification receipt.
Apache 2.0, works today on x86 CUDA + desktop GPUs, Jetson support
coming in v0.3.

```bash
pip install 'fastcrest-tether[serve,gpu] @ git+https://github.com/FastCrest/tether'
tether export --monolithic lerobot/smolvla_base --output ./smol
tether serve ./smol
# POST http://localhost:8000/act → 50-step action chunks
```

## What's verified in v0.2 (the only numbers in any pitch)

| Artifact | vs PyTorch | max_abs | cos |
|---|---|---|---|
| SmolVLA monolithic ONNX, num_steps=10 (production default) | `sample_actions(num_steps=10)` | **5.96e-07** | **+1.0000000** |
| pi0 monolithic ONNX, num_steps=10 (production default) | `sample_actions(num_steps=10)` | **2.09e-07** | **+1.0000000** |
| pi0.5 monolithic ONNX, num_steps=10 (production default) | `sample_actions(num_steps=10)` | **2.38e-07** | **+1.0000000** |
| GR00T N1.6 monolithic ONNX, single-step DiT (DDPM) | `GR00TFullStack.forward` | **8.34e-07** | **+1.0000000** |
| GR00T N1.6 end-to-end 4-step denoise loop | Python loop over PyTorch ref | **4.77e-07** | **+1.0000000** |
| pi0 native wrapper vs raw sample_actions | raw `sample_actions` | **0.000 (bit-exact)** | 1.0 |

All reproducible: `modal run scripts/modal_{smolvla,pi0,pi05,gr00t}_monolithic_export.py --parity`. Full ledger: `reflex_context/measured_numbers.md`.

## 9 regression gates

Every release is held against these; `tests/test_*.py` has receipt-based markers. Today 9/9 pass:

1. fresh-install from git + GHCR image
2. CUDAExecutionProvider parity vs CPU (cos ≥ 0.9999 both models)
3. num_steps=1 vs num_steps=10 quality gap characterized
4. docker-run smoke via GH Actions workflow
5. ros2-bridge-live (real rclpy, ros:humble container, not mocked)
6. `tether export --monolithic` CLI produces a working export
7. FastAPI `POST /act` returns valid action chunks end-to-end
8. Runtime correctly serves num_steps=10 artifacts (new `SmolVLAOnnxServer`)
9. ActionGuard kill-switch propagates to `/act` 503 + `/guard/reset` clears

## Honest disclaimers

- **pi0 / pi0.5 monolithic ONNX (12.5–13GB) does not fit on Orin Nano 8GB.** SmolVLA (1.6GB) does; GR00T (4.4GB) likely does in FP16 but has not been verified on the Nano. Large pi-family models need Orin 16GB+ or desktop GPU for v0.2. FP16 engine rebuild for Orin Nano fit is v0.3.
- **Jetson latency numbers — none.** CloudJetson's Orin Nano is waitlisted; no customer hardware yet. Numbers ship when a community benchmark lands.
- **GR00T VLM conditioning is zero-stubbed.** Same convention as pi0/SmolVLA's prefix=None convention — the exported DiT + AdaLN stack matches PyTorch at machine precision, but full multimodal control requires Eagle VLM backbone export (v0.3 item). For DiT-only validation on seeded inputs, the current export is canonical.
- **Earlier TRT FP16 latency tables were from a now-abandoned decomposed-ONNX path.** Desktop GPU + Jetson latency re-measurement tracked for v0.3.

## v0.3 roadmap (ordered by customer signal + cost)

1. **Jetson latency** — publish real ms/step numbers once hardware is available
2. **GR00T VLM conditioning** — Eagle backbone export so full multimodal control is shippable (DiT-only already at machine precision)
3. **FP16 engine rebuild** for Orin Nano 8GB fit (pi0/pi0.5)
4. **Docker arm64 image** for Jetson deployment

## Try it + feedback

- Repo: https://github.com/FastCrest/tether
- Docker: `ghcr.io/fastcrest/tether:0.2.0`
- Verified numbers: [`reflex_context/measured_numbers.md`](https://github.com/FastCrest/tether/blob/main/reflex_context/measured_numbers.md)
- Issues: respond within 24h

Apache 2.0. Single maintainer. Looking especially for:
- Jetson Orin Nano benchmark contributor (30 min on a dev kit = real edge numbers published with your credit)
- Anyone deploying SmolVLA or pi0 to a real robot
- Wedge feedback (`--safety-config`, `--adaptive-steps`, `--deadline-ms`, `--max-batch`)

---

## Launch sequencing (after user approval)

1. **LeRobot #3146 comment first** — strategic audience (active VLA users)
2. **48-72h later: Show HN** — broader tech audience
3. **Same day or next: r/robotics** — third audience
4. Direct outreach to 3 named companies during weeks 5-6

## Pre-launch checklist (from `launch/README.md`)

- [x] SmolVLA + pi0 + pi0.5 + GR00T ONNX parity verified (cos=+1.000000, machine precision across all four major open VLAs)
- [x] pi0 native wrapper parity bit-exact
- [x] README reframed around verified cos numbers (not unverified TRT)
- [x] Docker workflow landed + v0.2.0 image published + smoke-tested
- [x] ROS2 bridge shipped with live rclpy test
- [x] Safety kill-switch + NaN/Inf guard shipped
- [x] Auto-generated VERIFICATION.md receipt per export dir
- [x] Token scrubbed from git history
- [x] Launch drafts updated for Docker + ROS2 + F.pad findings
- [ ] Jetson benchmark — **explicit v0.3 deferral** (disclosed in every draft)
- [ ] Fresh-box install re-tested on fresh Mac + Linux box (optional polish — CI + Modal fresh-install gate already cover this)
- [ ] GitHub Issues open + <24h response commitment set in profile
- [ ] (Optional) Discord or Slack link added to README
