# Tether examples

Each example is a self-contained walkthrough you can paste into a terminal. Start with `01-chat-quickstart.md` if you've never used Tether before — it gets you to a working `tether chat` session in under 60 seconds.

| File | What it covers | Time |
|---|---|---|
| [01-chat-quickstart.md](01-chat-quickstart.md) | `pip install` → `tether chat` → ask three questions, see tool calls happen | ~3 min |
| [02-deploy-smolvla-jetson.md](02-deploy-smolvla-jetson.md) | Full path: pull → export → serve SmolVLA on a Jetson Orin Nano | ~15-25 min |
| [03-distill-pi05.md](03-distill-pi05.md) | Train a 1-NFE SnapFlow student from a pi0.5 teacher; export the student to ONNX | ~2-4 hrs (mostly GPU-bound) |
| [04-record-and-replay.md](04-record-and-replay.md) | Record live `/act` traces, browse them with `tether inspect traces`, replay against a different model | ~10 min |

All examples assume you've installed at least the base package:

```bash
pip install tether
```

Some examples need the GPU or monolithic export extras — each example flags what it needs at the top.
