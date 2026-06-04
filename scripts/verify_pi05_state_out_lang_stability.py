"""Verify the v0.5 state-out preprocessor produces identical lang_tokens
across all frames within a single episode.

This is the load-bearing assumption for the prefix-cache moat: if
lang_tokens is stable per episode, then a lang-keyed cache hits ~100%
of frames after the first. If it drifts (state still leaking into
lang somewhere), the cache moat is structurally broken.

Usage (local, no Modal):
    python scripts/verify_pi05_state_out_lang_stability.py \\
        --proc-ref lerobot/pi05_libero_finetuned_v044

The test fabricates 50 fake observations with the SAME task string and
DIFFERENT proprio state values, runs each through the preprocessor,
hashes the OBS_LANGUAGE_TOKENS output, and reports whether all 50
hashes match.

Pass = lang stable → cache moat is structurally sound.
Fail = lang drifts → bug in preprocessor swap; cache won't hit.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def main(proc_ref: str, num_frames: int = 50) -> int:
    print(f"[verify] proc_ref = {proc_ref}")
    print(f"[verify] simulating {num_frames} frames of one episode")

    # Resolve proc_ref to a local dir (snapshot_download if HF id)
    if not Path(proc_ref).exists():
        from huggingface_hub import snapshot_download
        proc_ref = snapshot_download(proc_ref)
        print(f"[verify] downloaded to: {proc_ref}")

    import torch
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    # Build the BASELINE preprocessor (default — state in lang)
    def to_transition(b):
        from lerobot.processor.pipeline import EnvTransition, TransitionKey
        return EnvTransition(**{TransitionKey.OBSERVATION.value: b, TransitionKey.COMPLEMENTARY_DATA.value: {"task": [b.get("task", "test task")]}})

    def to_output(t):
        return t

    baseline_proc = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=proc_ref,
        config_filename="policy_preprocessor.json",
        to_transition=to_transition,
        to_output=to_output,
        overrides={"device_processor": {"device": "cpu"}},
    )

    # Build the STATE-OUT preprocessor (swap)
    state_out_proc = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=proc_ref,
        config_filename="policy_preprocessor.json",
        to_transition=to_transition,
        to_output=to_output,
        overrides={"device_processor": {"device": "cpu"}},
    )
    from tether.distill.pi05_state_out_processor import swap_prepare_step_in_pipeline
    swap_prepare_step_in_pipeline(state_out_proc, max_state_dim=32)

    print(f"[verify] preprocessors built (baseline + state-out)")

    # Fake an episode: same task, different state per frame
    task_str = "put both the alphabet soup and the tomato sauce in the basket"
    baseline_hashes = []
    state_out_hashes = []

    import numpy as np
    for i in range(num_frames):
        # Vary proprio state per frame (8-dim LIBERO state)
        state = torch.tensor([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i], dtype=torch.float32)

        batch = {
            "observation.state": state.unsqueeze(0),  # (1, 8)
            "observation.images.image": torch.zeros(1, 3, 224, 224, dtype=torch.float32),
            "observation.images.image2": torch.zeros(1, 3, 224, 224, dtype=torch.float32),
            "task": task_str,
        }

        try:
            b_baseline = baseline_proc(batch)
            b_state_out = state_out_proc(batch)
        except Exception as e:
            print(f"[verify] FAILED at frame {i}: {type(e).__name__}: {e}")
            return 2

        bl_tokens = b_baseline[OBS_LANGUAGE_TOKENS].cpu().numpy().tobytes()
        so_tokens = b_state_out[OBS_LANGUAGE_TOKENS].cpu().numpy().tobytes()

        baseline_hashes.append(hashlib.md5(bl_tokens).hexdigest()[:12])
        state_out_hashes.append(hashlib.md5(so_tokens).hexdigest()[:12])

    print(f"\n[verify] BASELINE preprocessor (default):")
    n_unique_baseline = len(set(baseline_hashes))
    print(f"  unique lang_tokens hashes across {num_frames} frames: {n_unique_baseline}")
    print(f"  first 5: {baseline_hashes[:5]}")
    if n_unique_baseline > 1:
        print(f"  → DRIFT confirmed (this is the bug — state in lang)")

    print(f"\n[verify] STATE-OUT preprocessor (swapped):")
    n_unique_state_out = len(set(state_out_hashes))
    print(f"  unique lang_tokens hashes across {num_frames} frames: {n_unique_state_out}")
    print(f"  first 5: {state_out_hashes[:5]}")

    if n_unique_state_out == 1:
        print(f"\n✅ PASS — state-out preprocessor produces stable lang per episode")
        print(f"   The prefix-cache moat is structurally sound. Building Pi05PrefixCache will hit ~100% within an episode.")
        return 0
    else:
        print(f"\n❌ FAIL — state-out preprocessor produces {n_unique_state_out} unique lang hashes across {num_frames} frames")
        print(f"   Something other than state is leaking into lang. Investigate before building cache.")
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proc-ref", default="lerobot/pi05_libero_finetuned_v044")
    p.add_argument("--num-frames", type=int, default=50)
    a = p.parse_args()
    sys.exit(main(a.proc_ref, a.num_frames))
