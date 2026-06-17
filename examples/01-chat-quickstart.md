# 01 — `tether chat` in 60 seconds

**What you'll see:** install Tether, open the chat agent, ask three questions, watch it call CLI tools on your behalf.

**Requires:** Python ≥ 3.10, network access. No GPU needed for this example.

## Install

```bash
pip install fastcrest-tether
```

(~40 seconds; pulls torch + transformers + onnx as deps. Yes, it's chunky — the chat agent itself is ~50 KB, but it shares the install with the rest of Tether.)

## Open the chat

```bash
tether chat
```

You'll see:

```
connected: chat.fastcrest.com (model=gpt-5-mini)
tether chat — Ctrl+C or 'exit' to quit. /reset to clear.

you ›
```

The chat agent runs against a hosted Cloudflare Worker proxy (`chat.fastcrest.com`) that forwards to GPT-5 Mini. Free tier is 100 calls/day per machine. No signup, no API key.

## Ask it things

### "What version am I on?"

```
you › what version of tether am i on?
  → show_version({})
    exit_code=0  tether --version

You are on version "tether 0.12.0".
```

The agent picked the `show_version` tool, ran `tether --version` as a subprocess, and read the result back to you verbatim.

### "Show me what's broken"

```
you › show me what's broken on my install
  → doctor({})
    exit_code=0  tether doctor

Here's what the doctor found:
  ✓ Python 3.11.7
  ✓ tether 0.3.5
  ⚠ torch + CUDA — torch 2.11.0, CUDA unavailable (you're on Apple Silicon)
  ⚠ ONNX Runtime — not installed
    Action: pip install 'fastcrest-tether[serve,onnx]'
  ⚠ fastapi + uvicorn — not installed
    Action: pip install 'fastcrest-tether[serve,onnx]'
```

### "What models can I deploy?"

```
you › what models are available?
  → list_models({})
    exit_code=0  tether models list

Models in your local Tether registry:
  - smolvla-base (900 MB) — small edge VLA, supports orin_nano, agx_orin, a10g, a100, h100
  - smolvla-libero (900 MB) — SmolVLA finetuned on LIBERO
  - pi0-base (14 GB) — pi0 base from Physical Intelligence
  - pi05-base (14 GB) — pi0.5 base, improved pi0
  - pi05-libero (14 GB) — pi0.5 finetuned on LIBERO (Franka 7-DoF)
```

## Quit

`exit`, `quit`, `:q`, or Ctrl+C.

## Next steps

- [02-deploy-smolvla-jetson.md](02-deploy-smolvla-jetson.md) — actually deploy a model with `tether go`
- The chat understands 16 tools — try "deploy smolvla to my desktop GPU" and it'll chain pull → export → serve for you (with `[monolithic]` extras installed).

## Troubleshooting

- **"Could not resolve host: chat.fastcrest.com"** — your DNS resolver hasn't propagated yet. Wait a few minutes or set `FASTCREST_PROXY_URL=https://fastcrest-proxy.fastcrest.workers.dev` to use the workers.dev URL directly.
- **Rate limit** — free tier is 100 calls/day per machine. Reset at UTC midnight.
- **Slow first run after install** — Python is reading ~5000 files from cold OS cache. Subsequent runs are sub-second.
