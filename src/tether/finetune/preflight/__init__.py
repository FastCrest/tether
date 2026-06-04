"""Pre-flight validation for tether finetune.

Catches the top customer pain points BEFORE any GPU time is spent:

1. **Schema mismatch** (pain #1, #2): dataset action/state dims
   don't match the base checkpoint. Today customers see 'loss
   converges in 10k steps, eval hits 0%' because the action head
   silently learned on the wrong dims. We refuse the run with an
   actionable error pointing at the offending feature shapes.

2. **Dataset size** (pain #9): per-base-model episode-count floor.
   Warns when a user tries to fine-tune pi0.5 on 50 episodes
   (likely to destabilize norm-stats) without them having to
   discover it through a failed run.

3. **Norm-stats provenance** (pain #6, v0.6): base-checkpoint stats
   reuse vs recompute. Stubbed in v0.5; full implementation lands
   when we have per-base norm-stats manifests.

4. **Memory budget** (pain #5, v0.6): VRAM estimate for the chosen
   mode × precision × gpu_count. Stubbed in v0.5.

Tether philosophy: we'd rather fail in <1s with a clear message
than cost a customer 45 minutes of GPU to discover a config bug.

Full v0.5 validator set defined in
https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/01_architecture/finetune_architecture.md
Section D.
"""
from __future__ import annotations

from tether.finetune.preflight.result import PreflightReport, PreflightCheck
from tether.finetune.preflight.runner import run_preflight

__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "run_preflight",
]
