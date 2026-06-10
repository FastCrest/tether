"""Record a tweet-grade tether VLA demo gif on Modal A10G.

Captures real tether CLI output via subprocess, builds an asciinema cast
with typed-character animation + captured output, renders to gif via agg.
Output is HIGHER quality than QuickTime+ffmpeg (vector-rendered text).

Usage:
    modal profile activate novarepmarketing
    modal run /tmp/modal_record_tweet_gif.py

Cost: ~$0.30-0.60 on A10G in ~5-10 min (mostly image build first run).
Output: ~/Downloads/tether-tweet.gif

Approach justified per CLAUDE.md no-band-aid: real CLI output captured
live in container, not faked. Typing animation is synthesized but every
character of output is verbatim from `tether --version`, `tether doctor`,
`tether --help` running in the container with TRT EP active.
"""
import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "wget", "curl", "fontconfig", "fonts-jetbrains-mono")
    .pip_install("uv")
    .run_commands(
        # Install from the repo until `tether` is published to PyPI (the old
        # pinned 'reflex-vla==0.8.0' / 'tether-vla' package names are dead).
        "uv pip install --system 'fastcrest-tether[serve,gpu] @ git+https://github.com/FastCrest/tether'",
        "uv pip install --system 'tensorrt>=10.0,<11'",
        # agg = official asciinema gif renderer (Rust binary)
        "wget -qO /usr/local/bin/agg https://github.com/asciinema/agg/releases/download/v1.5.0/agg-x86_64-unknown-linux-gnu",
        "chmod +x /usr/local/bin/agg",
    )
    .env({
        "PYTHONFAULTHANDLER": "1",
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/tensorrt_libs:/usr/local/lib/python3.12/site-packages/tensorrt:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "TERM": "xterm-256color",
        # force color output from tether CLI
        "FORCE_COLOR": "1",
        "CLICOLOR_FORCE": "1",
    })
)

app = modal.App("tether-tweet-gif")


@app.function(image=image, gpu="A10G", timeout=900)
def record() -> bytes:
    import subprocess
    import json
    import time
    import pathlib
    import os

    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["FORCE_COLOR"] = "1"
    env["CLICOLOR_FORCE"] = "1"

    def run(cmd: str) -> str:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        return (r.stdout + r.stderr).rstrip("\n")

    print("=== capturing real CLI output ===")
    out_version = run("tether --version")
    out_doctor = run("tether doctor")
    out_help = run("tether --help")
    print(f"version: {out_version!r}")
    print(f"doctor first line: {out_doctor.splitlines()[0] if out_doctor else 'EMPTY'!r}")
    print(f"doctor lines: {len(out_doctor.splitlines())}")
    print(f"help lines: {len(out_help.splitlines())}")

    # ----- build asciinema cast programmatically -----
    width, height = 110, 38
    header = {
        "version": 2,
        "width": width,
        "height": height,
        "timestamp": int(time.time()),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
        "title": "Tether VLA — pip install + edge deploy",
    }

    frames = []
    t = [0.5]  # start with brief blank to let viewer settle
    PROMPT = "[1;36m❯[0m "

    def emit(s: str) -> None:
        frames.append([round(t[0], 3), "o", s])

    def type_command(s: str, char_delay: float = 0.045) -> None:
        emit(PROMPT)
        for c in s:
            t[0] += char_delay
            emit(c)

    def show_output(output: str, lead_pause: float = 0.3, post_pause: float = 1.4) -> None:
        t[0] += lead_pause
        formatted = output.replace("\n", "\r\n")
        emit("\r\n" + formatted + "\r\n")
        t[0] += post_pause

    # demo sequence
    type_command("tether --version")
    show_output(out_version, post_pause=1.0)

    type_command("tether doctor")
    show_output(out_doctor, post_pause=2.5)

    type_command("tether --help")
    show_output(out_help, post_pause=2.0)

    emit(PROMPT)
    t[0] += 0.6

    print(f"\n=== cast built: {len(frames)} frames over {t[0]:.2f}s ===")

    cast_path = pathlib.Path("/tmp/demo.cast")
    with open(cast_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for frame in frames:
            f.write(json.dumps(frame) + "\n")

    gif_path = pathlib.Path("/tmp/tether-tweet.gif")
    print("=== rendering gif via agg ===")
    subprocess.run(
        [
            "agg",
            "--font-size", "16",
            "--speed", "1.0",
            "--theme", "monokai",
            "--cols", str(width),
            "--rows", str(height),
            str(cast_path),
            str(gif_path),
        ],
        check=True,
    )

    size_kb = gif_path.stat().st_size / 1024
    print(f"\n✓ gif: {size_kb:.1f} KB | duration: {t[0]:.1f}s")
    return gif_path.read_bytes()


@app.local_entrypoint()
def main():
    import pathlib
    print("=== recording tether tweet gif on Modal A10G ===")
    gif_bytes = record.remote()
    out = pathlib.Path.home() / "Downloads" / "tether-tweet.gif"
    out.write_bytes(gif_bytes)
    print(f"\n✓ saved {len(gif_bytes)/1024:.1f} KB → {out}")
