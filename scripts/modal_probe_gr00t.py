"""Probe GR00T N1.6 action_encoder and action_decoder weight shapes.

These are per-embodiment MLPs (leading dim 32) that sit around the DiT
expert. The existing Tether GR00T exporter skips them — adding them
lets `tether serve` do full-loop denoising (raw actions in, raw actions
out). This script just dumps shapes so we can build without guessing.

Usage:
    modal run scripts/modal_probe_gr00t.py
"""

import modal

app = modal.App("tether-gr00t-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "safetensors", "huggingface_hub")
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e .")
)


@app.function(image=image, gpu="A100-40GB", timeout=600)
def probe():
    from tether.checkpoint import load_checkpoint

    print("Loading GR00T...", flush=True)
    state_dict, _ = load_checkpoint("nvidia/GR00T-N1.6-3B")

    # Dump all action_head.{action,state}_encoder/decoder/action_decoder shapes
    interesting = [
        k for k in state_dict.keys()
        if any(p in k for p in [
            "action_encoder", "action_decoder", "state_encoder",
            "position_embedding", "vlln", "timestep_encoder",
            "proj_out",
        ])
    ]

    print(f"\nFound {len(interesting)} action-head meta keys:\n", flush=True)
    shapes = {}
    for k in sorted(interesting):
        shape = tuple(state_dict[k].shape)
        shapes[k] = shape
        print(f"  {k}: {shape}", flush=True)

    # Also the block 0 attn for sanity
    print("\nBlock 0 sample shapes:", flush=True)
    for k in sorted(state_dict.keys()):
        if k.startswith("action_head.model.transformer_blocks.0."):
            print(f"  {k}: {tuple(state_dict[k].shape)}", flush=True)

    return shapes


@app.local_entrypoint()
def main():
    shapes = probe.remote()
    print("\n\n=== Summary ===")
    for k, v in shapes.items():
        print(f"  {k}: {v}")
