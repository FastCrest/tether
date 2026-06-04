"""Dataset-size floor check — catches pain #9 ("how much data is enough?").

Uses empirical minimums from the SOTA research (finding 9):
  - SmolVLA: 50-200 episodes is usually fine for LoRA (small LoRA bubble,
    vision+language already good, action head adapts quickly)
  - pi0: 200-500 episodes; PaliGemma backbone is bigger, needs more data
    to not drift
  - pi0.5: ~1000 episodes recommended; quantile norm requires enough
    samples to estimate percentiles reliably (silent-killer case is
    fewer than that and norm stats go unstable)
  - GR00T N1.6: 500+ episodes; DiT action head is data-hungry

Below-floor runs WARN (not fail) — customer might be doing an intentional
small-data experiment. The warning points at what to expect (degraded
task success) and how to proceed safely (rehearsal fraction, frozen
vision encoder).
"""
from __future__ import annotations

import logging
from typing import Any

from tether.finetune.config import FinetuneConfig
from tether.finetune.preflight.result import PreflightCheck

logger = logging.getLogger(__name__)


# Per-base-model episode-count minimums for stable LoRA fine-tune.
# Sources: SOTA research finding 9 + openpi issues #635/672/711.
EPISODE_FLOORS: dict[str, int] = {
    "smolvla": 50,
    "pi0": 200,
    "pi05": 1000,
    "gr00t_n1_5": 500,
}


def _infer_policy_type(base: str) -> str | None:
    """Mirror the inference in run.py. Returns None for unknown bases."""
    base_lower = base.lower()
    if "smolvla" in base_lower:
        return "smolvla"
    if "pi05" in base_lower or "pi0.5" in base_lower or "pi_05" in base_lower:
        return "pi05"
    if "pi0" in base_lower:
        return "pi0"
    if "gr00t" in base_lower or "groot" in base_lower:
        return "gr00t_n1_5"
    return None


def _fetch_dataset_info(dataset_repo_id: str) -> dict[str, Any] | None:
    """Grab `meta/info.json` which carries total_episodes + total_frames.

    Same-as `schema.py::_fetch_dataset_features` — share the cache.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        info_path = hf_hub_download(
            repo_id=dataset_repo_id,
            filename="meta/info.json",
            repo_type="dataset",
        )
    except Exception as e:
        logger.debug("[preflight] couldn't fetch dataset info for %s: %s",
                     dataset_repo_id, e)
        return None

    import json
    with open(info_path) as f:
        return json.load(f)


def check_dataset_size(cfg: FinetuneConfig) -> PreflightCheck:
    """Warn if episode count is below the stability floor for the base model."""
    policy_type = _infer_policy_type(cfg.base)
    if policy_type is None:
        return PreflightCheck(
            name="dataset_size",
            severity="warn",
            summary=f"couldn't infer policy type from base {cfg.base!r}; "
                    f"skipping dataset-size floor check",
        )
    floor = EPISODE_FLOORS.get(policy_type)
    if floor is None:
        return PreflightCheck(
            name="dataset_size",
            severity="ok",
            summary=f"no floor defined for policy_type={policy_type}",
        )

    info = _fetch_dataset_info(cfg.dataset)
    if info is None:
        return PreflightCheck(
            name="dataset_size",
            severity="warn",
            summary=f"couldn't resolve episode count for {cfg.dataset!r}",
            detail={"floor": floor, "policy_type": policy_type},
        )

    num_episodes = info.get("total_episodes") or info.get("num_episodes")
    if num_episodes is None:
        return PreflightCheck(
            name="dataset_size",
            severity="warn",
            summary=f"dataset info has no episode count",
        )

    if num_episodes < floor:
        return PreflightCheck(
            name="dataset_size",
            severity="warn",
            summary=(
                f"dataset has {num_episodes} episodes; {policy_type} typically "
                f"needs ≥{floor} for stable fine-tune. Training will likely "
                f"converge but task success may drop. Options to mitigate: "
                f"(a) add --rehearsal-fraction 0.2 to mix in base-checkpoint "
                f"data; (b) --freeze-vision-encoder to avoid vision drift; "
                f"(c) use a smaller LoRA rank (--lora-rank 8) to limit "
                f"overfitting."
            ),
            detail={
                "dataset_episodes": num_episodes,
                "recommended_floor": floor,
                "policy_type": policy_type,
            },
        )

    return PreflightCheck(
        name="dataset_size",
        severity="ok",
        summary=f"{num_episodes} episodes ≥ {floor} floor for {policy_type}",
        detail={"episodes": num_episodes, "floor": floor},
    )


__all__ = ["check_dataset_size", "EPISODE_FLOORS"]
