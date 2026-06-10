# Tether — visual assets

Demo gifs for README embeds, social posts, and YC/grant applications. Recorded on Modal A10G; output is verbatim from the CLI in the container, only the typing animation is synthesized.

> **`tether-chat-demo.gif` is still pre-rename** (v0.5.0, shows old `reflex` commands on screen) and is not embedded in the README. Re-record under the `tether` CLI before using it in any new public post.

| Asset | Recorded | Version | Length / size | Use for |
|---|---|---|---|---|
| `tether-tweet.gif` | 2026-06-10 | v0.12.0 | 9.8 s · 126 KB · 1075×873 | X/Twitter posts, tight demo embeds. Shows `tether --version` → `tether inspect targets` (Jetson Orin/Thor + RTX/A100/H100 support table) → `tether --help` (verb listing). |
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
- A minor version ships that changes `tether inspect targets`, the verb listing, or the tagline.
- The README claims a number that's no longer in the gif (e.g., new architectures verified).

> The recording deliberately avoids `tether doctor`: its deploy-diagnostic checks crash on a GPU box with no exported model (`Path(ModelProto)` TypeError) — a real bug tracked separately. Restore the `doctor` panel once that's fixed.

The full experiment note for any re-record lives at `reflex_context/03_experiments/YYYY-MM-DD-tweet-gif-*.md`.
