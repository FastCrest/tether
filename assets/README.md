# Tether — visual assets

Demo gifs for README embeds, social posts, and YC/grant applications. Recorded on Modal A10G with TRT EP active so the doctor checks render green; output is verbatim from the CLI in the container, only the typing animation is synthesized.

> **Stale branding:** both gifs were recorded before the v0.12.0 rename (Reflex → Tether) and show the old `reflex` CLI commands on screen. Re-record under the `tether` CLI before using them in any new public post; the table notes which recordings are pre-rename.

| Asset | Recorded | Version | Length / size | Use for |
|---|---|---|---|---|
| `tether-tweet.gif` | 2026-05-03 | v0.8.0 (pre-rename) | 11.8 s · 152 KB · 1075×873 | X/Twitter posts, tight demo embeds. Shows `reflex --version` → `reflex doctor` (4 green ✓ on TensorRT runtime, cuBLAS, cuDNN, ORT-TRT EP) → `reflex --help` (11-verb listing). |
| `tether-chat-demo.gif` | 2026-04-28 | v0.5.0 (pre-rename) | 37.3 s · 197 KB · 1280×760 | YC application demo upload, longer-form embeds. Shows the `reflex chat` natural-language interface routing through CLI tools. |

## Recording recipe

`scripts/modal_record_demo_gif.py` produces tweet-grade gifs by:

1. Running real tether commands in a Modal A10G container (so the GPU/TRT path renders correctly).
2. Capturing stdout verbatim.
3. Building an asciinema cast programmatically with synthesized typing animation + real captured output.
4. Rendering to gif via [agg](https://github.com/asciinema/agg) (vector text render — stays crisp at any zoom; ~half the file size of QuickTime + ffmpeg at higher quality).

```bash
modal profile activate <your-profile>
modal run scripts/modal_record_demo_gif.py
# saves to ~/Downloads/tether-tweet.gif
```

Cost: ~$0.30 on A10G (~10 min including image cold start).

## Refresh policy

Re-record the tweet gif when:
- A minor version ships that changes `tether doctor` output, the verb listing, or the tagline.
- The README claims a number that's no longer in the gif (e.g., new architectures verified).

The full experiment note for any re-record lives at `reflex_context/03_experiments/YYYY-MM-DD-tweet-gif-*.md`.
