"""Phase IV: does `reflex turbo --strategy adaptive` actually save time on real VLAs?

The current implementation in src/reflex/kernels/turbo.py and the inline
adaptive-stop in src/reflex/runtime/server.py:_run_denoise stop the denoise
loop early when consecutive velocity-norm deltas drop below 0.01. That
heuristic was only validated on a synthetic 16-hidden toy model.

This benchmark runs the full denoise loop on each real VLA expert (smolvla,
pi0, pi0.5, gr00t) for N trials, and tracks:
  - per-step velocity norms
  - the step at which the heuristic would have triggered (delta < threshold)
  - what the action diff would be vs. fixed-10-step

If the heuristic triggers within 4-7 steps reliably AND the action diff is
small (< 1e-3), adaptive is a real ~30-50% latency win on top of TRT.
If the velocities never converge or the action diff is large, the feature
needs to be removed or rewritten.

Usage:
    modal run scripts/modal_verify_adaptive_real.py
"""

import modal

app = modal.App("reflex-adaptive-real")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu>=1.20,<1.24",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e . --no-deps")
)


@app.function(image=image, gpu="A10G", timeout=2400, scaledown_window=60)
def test_adaptive():
    import gc
    import importlib
    import time
    import numpy as np
    import torch

    from reflex.checkpoint import load_checkpoint

    models = [
        {
            "tag": "smolvla",
            "hf_id": "lerobot/smolvla_base",
            "builder_mod": "reflex.exporters.smolvla_exporter",
            "builder_fn": "build_expert_stack",
            "kwargs": {"head_dim": 64},
        },
        {
            "tag": "pi0",
            "hf_id": "lerobot/pi0_base",
            "builder_mod": "reflex.exporters.pi0",
            "builder_fn": "build_pi0_expert_stack",
            "kwargs": {"head_dim": 128},
        },
        {
            "tag": "pi05",
            "hf_id": "lerobot/pi05_base",
            "builder_mod": "reflex.exporters.pi0",
            "builder_fn": "build_pi05_expert_stack",
            "kwargs": {"head_dim": 128},
        },
        {
            "tag": "gr00t",
            "hf_id": "nvidia/GR00T-N1.6-3B",
            "builder_mod": "reflex.exporters.gr00t",
            "builder_fn": "build_gr00t_full_stack",
            "kwargs": {"embodiment_id": 0},
        },
    ]

    NUM_STEPS = 10
    NUM_TRIALS = 25
    THRESHOLD = 0.01  # current heuristic in server._run_denoise

    results = {}

    for m in models:
        tag = m["tag"]
        print(f"\n{'='*60}\nModel: {tag}\n{'='*60}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        state_dict, _ = load_checkpoint(m["hf_id"])
        builder = getattr(importlib.import_module(m["builder_mod"]), m["builder_fn"])
        stack, meta = builder(state_dict, **m["kwargs"])
        stack = stack.cuda().eval()

        chunk_size = 50
        action_dim = meta.get("raw_action_dim", meta.get("action_dim", 32))
        pos_ids = torch.arange(chunk_size, device="cuda").unsqueeze(0)

        # For each trial: random noisy init, run 10 steps, record velocity norms
        all_norms: list[list[float]] = []
        action_diffs: list[float] = []  # diff between step-N actions and step-10 actions

        for trial in range(NUM_TRIALS):
            torch.manual_seed(trial)
            noisy = torch.randn(1, chunk_size, action_dim, device="cuda")

            x = noisy.clone()
            dt = -1.0 / NUM_STEPS
            norms_this_trial = []
            actions_per_step = [x.clone()]

            for step in range(NUM_STEPS):
                t = torch.tensor([1.0 + step * dt], device="cuda")
                with torch.no_grad():
                    v = stack(x, t, pos_ids)
                norms_this_trial.append(float(v.norm().item()))
                x = x + v * dt
                actions_per_step.append(x.clone())

            all_norms.append(norms_this_trial)

            # Find when adaptive WOULD have triggered: delta < THRESHOLD after step 2
            triggered_at = None
            for step in range(2, NUM_STEPS):
                delta = abs(norms_this_trial[step] - norms_this_trial[step - 1])
                if delta < THRESHOLD:
                    triggered_at = step + 1  # 1-indexed step count
                    break

            if triggered_at is not None:
                # Diff between the actions we'd have shipped (at triggered_at) vs full 10
                shipped = actions_per_step[triggered_at]
                full = actions_per_step[NUM_STEPS]
                action_diff = float((shipped - full).abs().max().item())
                action_diffs.append(action_diff)

        # Stats across trials
        mean_norms = [
            sum(t[s] for t in all_norms) / len(all_norms) for s in range(NUM_STEPS)
        ]
        # Per-step delta (averaged across trials)
        deltas = [
            sum(abs(t[s] - t[s - 1]) for t in all_norms) / len(all_norms)
            for s in range(1, NUM_STEPS)
        ]

        # How many trials would have triggered, and at which step
        triggered_steps = []
        for trial_norms in all_norms:
            for step in range(2, NUM_STEPS):
                delta = abs(trial_norms[step] - trial_norms[step - 1])
                if delta < THRESHOLD:
                    triggered_steps.append(step + 1)
                    break
            else:
                triggered_steps.append(NUM_STEPS)  # would have run all 10

        n_triggered = sum(1 for s in triggered_steps if s < NUM_STEPS)
        mean_step = sum(triggered_steps) / len(triggered_steps)
        latency_savings_pct = round(100 * (NUM_STEPS - mean_step) / NUM_STEPS, 1)

        results[tag] = {
            "params_m": round(meta.get("full_stack_params_m", meta.get("total_params_m", 0)), 1),
            "trials": NUM_TRIALS,
            "threshold": THRESHOLD,
            "fixed_steps": NUM_STEPS,
            "trials_that_triggered": n_triggered,
            "mean_step_when_triggered": round(mean_step, 1),
            "latency_savings_pct_if_adopted": latency_savings_pct,
            "mean_velocity_norm_per_step": [round(v, 3) for v in mean_norms],
            "mean_delta_per_step": [round(d, 4) for d in deltas],
            "mean_action_diff_at_trigger": round(
                sum(action_diffs) / len(action_diffs) if action_diffs else 0.0, 5
            ),
        }
        print(f"  trials_triggered: {n_triggered}/{NUM_TRIALS}", flush=True)
        print(f"  mean step when triggered: {mean_step:.1f}/{NUM_STEPS}", flush=True)
        print(f"  est. latency savings if adopted: {latency_savings_pct:.1f}%", flush=True)
        print(f"  mean action diff at trigger: {results[tag]['mean_action_diff_at_trigger']:.5f}", flush=True)
        print(f"  mean velocity norms: {results[tag]['mean_velocity_norm_per_step']}", flush=True)

        del stack, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*80}", flush=True)
    print("VERDICT: does adaptive denoising actually save time on real VLAs?", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<10} {'triggered':>15} {'mean_step':>12} {'savings':>10} {'action_diff':>14}", flush=True)
    for tag, r in results.items():
        print(f"{tag:<10} {r['trials_that_triggered']}/{r['trials']:<15} "
              f"{r['mean_step_when_triggered']:>12} "
              f"{r['latency_savings_pct_if_adopted']:>9}% "
              f"{r['mean_action_diff_at_trigger']:>14}")

    return results


@app.local_entrypoint()
def main():
    print("Phase IV: validate adaptive denoising on real VLAs\n")
    r = test_adaptive.remote()
    import json
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
