"""Quick probe — what does lerobot 0.5.1 actually expose for GR00T?

lerobot-0.5.1 on Modal may or may not have GR00T support. The cloned
local lerobot is `main` (newer). Need to reconcile.

Prints the actual module contents of lerobot's groot subpackage.

Usage:
    modal run scripts/modal_gr00t_lerobot_probe.py
"""
import os
import modal

app = modal.App("tether-gr00t-probe")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "clang")
    .pip_install(
        "torch",
        "transformers<5.4,>=4.40",
        "numpy",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "lerobot==0.5.1",
        "num2words",
    )
)


@app.function(image=image, timeout=300)
def probe():
    import importlib
    import inspect
    import pkgutil

    # Check lerobot version
    import lerobot
    print(f"lerobot version: {getattr(lerobot, '__version__', 'unknown')}")
    print(f"lerobot path: {lerobot.__path__}")

    # List policies.groot submodule
    try:
        import lerobot.policies.groot as groot_pkg
        print(f"\ngroot package: {groot_pkg.__path__}")
        for m in pkgutil.iter_modules(groot_pkg.__path__):
            print(f"  module: {m.name}")
    except Exception as e:
        print(f"groot pkg import failed: {e}")
        return {"error": str(e)}

    # Attempt to import groot_n1 + list symbols
    try:
        from lerobot.policies.groot import groot_n1
        print(f"\ngroot_n1 symbols (top-level, non-dunder):")
        for name in sorted(dir(groot_n1)):
            if not name.startswith("_"):
                obj = getattr(groot_n1, name)
                kind = type(obj).__name__
                print(f"  {name}: {kind}")
    except Exception as e:
        print(f"groot_n1 import failed: {e}")

    # Attempt action_head
    try:
        from lerobot.policies.groot.action_head import flow_matching_action_head as fma
        print(f"\nflow_matching_action_head symbols:")
        for name in sorted(dir(fma)):
            if not name.startswith("_"):
                obj = getattr(fma, name)
                if inspect.isclass(obj):
                    print(f"  class {name}")
    except Exception as e:
        print(f"action_head import failed: {e}")

    return {"status": "probed"}


@app.local_entrypoint()
def main():
    probe.remote()
