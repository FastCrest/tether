# launch/

Drafts for public posts. **Nothing here is published yet — all need user approval before going live.**

| File | Where it goes | Status |
|---|---|---|
| `lerobot_3146_draft.md` | Comment on huggingface/lerobot#3146 | Draft v1 |
| `show_hn_draft.md` | news.ycombinator.com Show HN | Draft v1 |
| `reddit_robotics_draft.md` | reddit.com/r/robotics | Draft v1 |

## Sequencing

When ready to launch:

1. **Post LeRobot #3146 first** (most strategic — that's where the actual VLA users live)
2. **48-72h later, post Show HN** (orthogonal audience, broader tech)
3. **Same day or next day, post r/robotics** (third audience)

Don't post all three the same day — reduces signal in each, and means you can't respond to comments in any of them.

## Pre-launch checklist

- [x] SmolVLA + pi0 ONNX parity verified at cos=+1.0000000 (2026-04-18)
- [x] pi0 native-path parity verified bit-exact (2026-04-18)
- [x] README.md reframed around verified cos parity (not unverified TRT numbers)
- [x] Docker workflow landed — `git tag v0.2.0 && git push --tags` publishes to GHCR
- [x] ROS2 bridge shipped (`tether ros2-serve`)
- [x] Safety kill-switch + NaN/Inf guard shipped
- [x] Auto-generated `VERIFICATION.md` receipt per export dir
- [ ] **Jetson benchmark** — explicitly deferred to v0.3. Launch pitch reframes around A10G + Docker; Orin Nano numbers land when community / first customer runs them.
- [ ] Tag `v0.2.0` + push → CI publishes `ghcr.io/fastcrest/tether:0.2.0` + `:latest`
- [ ] `pip install ... @ git+https://...` install path re-tested on a fresh Mac + Linux box
- [ ] GitHub Issues open + <24h response commitment set in profile
- [ ] (Optional) Discord or Slack link added to README
