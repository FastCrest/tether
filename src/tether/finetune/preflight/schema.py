"""Dataset schema validation — catches pain #1 (loss converges, eval 0%) and #2 (tensor shape 6 vs 7).

Strategy: inspect the dataset's features + the base checkpoint's config
WITHOUT loading any GPU state, then compare dims. If they disagree, fail
loud with an actionable message pointing at the specific feature that's
wrong.

We check action dim first because that's the silent-killer case (the
action head will just project the wrong-dim output into whatever shape
and train happily to 0% eval success). State + image dim mismatches
usually crash lerobot's forward pass inside the first 10 steps, which
is painful but at least obvious.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tether.finetune.config import FinetuneConfig
from tether.finetune.preflight.result import PreflightCheck

logger = logging.getLogger(__name__)


def _fetch_dataset_features(dataset_repo_id: str) -> dict[str, Any] | None:
    """Return the dataset's feature dict (from info.json) without
    downloading episodes. Returns None if we can't resolve.

    LeRobotDataset v3 stores schema at `meta/info.json` on the Hub.
    We grab only that one file; episodes + videos stay remote.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.debug("huggingface_hub not available — skipping schema check")
        return None

    try:
        info_path = hf_hub_download(
            repo_id=dataset_repo_id,
            filename="meta/info.json",
            repo_type="dataset",
        )
    except Exception as e:
        logger.debug("[preflight] couldn't fetch meta/info.json from %s: %s",
                     dataset_repo_id, e)
        return None

    import json
    with open(info_path) as f:
        info = json.load(f)
    return info.get("features") or info.get("feature") or {}


def _fetch_base_config(base_id: str, revision: str | None = None) -> dict[str, Any] | None:
    """Return the base checkpoint's config.json as a dict."""
    # From-scratch policies have no checkpoint. Path("") would read the cwd.
    if not base_id:
        return None
    local = Path(base_id).expanduser()
    if local.is_dir():
        import json
        with (local / "config.json").open() as stream:
            return json.load(stream)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        kwargs = {"revision": revision} if revision else {}
        cfg_path = hf_hub_download(repo_id=base_id, filename="config.json", **kwargs)
    except Exception as e:
        logger.debug("[preflight] couldn't fetch config.json from %s: %s",
                     base_id, e)
        return None

    import json
    with open(cfg_path) as f:
        return json.load(f)


def _extract_dataset_action_dim(features: dict[str, Any]) -> int | None:
    """Find the action feature's shape → scalar dim."""
    # LeRobotDataset v3 features look like:
    #   {"action": {"dtype": "float32", "shape": [8], "names": [...]}}
    action = features.get("action")
    if not action:
        return None
    shape = action.get("shape") or action.get("dims")
    if shape and len(shape) >= 1:
        return int(shape[-1])
    return None


def _extract_base_action_dim(config: dict[str, Any]) -> int | None:
    """Action dim from a base-model config.json.

    SmolVLA / pi0 / pi0.5 use `max_action_dim` (32 for the pads).
    The *original* action dim (before pad) lives in
    `output_features.action.shape` if present. We prefer the native
    dim; fall back to max_action_dim.
    """
    out = config.get("output_features") or {}
    action = out.get("action")
    if action:
        shape = action.get("shape") or action.get("dims")
        if shape and len(shape) >= 1:
            return int(shape[-1])
    # Fallback
    if "max_action_dim" in config:
        return int(config["max_action_dim"])
    return None


def check_schema(cfg: FinetuneConfig) -> PreflightCheck:
    """Compare dataset action dim to base-checkpoint action feature.

    Returns:
        PreflightCheck with severity:
          - `ok` when dims match
          - `fail` when dims differ (the silent-killer case)
          - `warn` when we couldn't resolve either side (network issue,
            local dir, private repo). We prefer a warning over a false
            failure — customers running against local datasets or
            gated models still need a path through.
    """
    if cfg.dataset_root is not None:
        from tether.finetune.local_dataset import validate_local_config, verify_local_export
        validate_local_config(cfg)
        features = verify_local_export(cfg.dataset_root, cfg.dataset_manifest_sha256)["info"]["features"]
    else:
        features = _fetch_dataset_features(cfg.dataset)
    base_config = (_fetch_base_config(cfg.base, cfg.base_revision) if cfg.base_revision
                   else _fetch_base_config(cfg.base))
    local_details = {}
    if cfg.dataset_root is not None:
        expected_modes = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
        if not isinstance(base_config, dict) or base_config.get("type") != "smolvla":
            return PreflightCheck("schema", "fail", "Local admission requires a resolvable SmolVLA base config.")
        # An omitted field uses only the exact LeRobot 0.5.1 class defaults
        # qualified in the verified receipt. Explicit null/contradictory modes
        # are not treated as omission.
        resolved_modes = base_config.get("normalization_mapping", expected_modes)
        if resolved_modes != expected_modes:
            return PreflightCheck("schema", "fail", "Base normalization must match the qualified SmolVLA profile.")
        local_details = {
            "resolved_normalization_mapping": resolved_modes,
            "normalization_mapping_source": (
                "base config.json" if "normalization_mapping" in base_config
                else "pinned LeRobot 0.5.1 SmolVLAConfig defaults"
            ),
        }
        max_state = base_config.get("max_state_dim")
        max_action = base_config.get("max_action_dim")
        if (type(max_state) is not int or max_state <= 0 or type(max_action) is not int
                or max_action <= 0):
            return PreflightCheck("schema", "fail", "Base config must declare its real SmolVLA padding dimensions.")
        if features["observation.state"]["shape"][0] > max_state:
            return PreflightCheck("schema", "fail", "Dataset state exceeds the base max_state_dim.")
        if features["action"]["shape"][0] > max_action:
            return PreflightCheck("schema", "fail", "Dataset action exceeds the base max_action_dim.")

    if features is None:
        return PreflightCheck(
            name="schema",
            severity="warn",
            summary=f"couldn't resolve dataset schema for {cfg.dataset!r} — "
                    f"skipping dim check",
            detail={"dataset": cfg.dataset},
        )
    if base_config is None:
        return PreflightCheck(
            name="schema",
            severity="warn",
            summary=f"couldn't resolve base config for {cfg.base!r} — "
                    f"skipping dim check",
            detail={"base": cfg.base},
        )

    ds_dim = _extract_dataset_action_dim(features)
    base_dim = _extract_base_action_dim(base_config)
    max_action_dim = base_config.get("max_action_dim")
    max_action_dim = int(max_action_dim) if max_action_dim is not None else None

    if ds_dim is None or base_dim is None:
        return PreflightCheck(
            name="schema",
            severity="warn",
            summary="couldn't extract action dims; proceeding anyway",
            detail={"dataset_action_dim": ds_dim, "base_action_dim": base_dim},
        )

    if ds_dim != base_dim:
        if max_action_dim is not None and ds_dim <= max_action_dim:
            return PreflightCheck(
                name="schema",
                severity="ok",
                summary=(
                    f"dataset action is {ds_dim}-D; the base metadata is "
                    f"{base_dim}-D and supports padding to {max_action_dim}-D"
                ),
                detail={
                    "dataset": cfg.dataset,
                    "dataset_action_dim": ds_dim,
                    "base": cfg.base,
                    "base_action_dim": base_dim,
                    "max_action_dim": max_action_dim,
                    "uses_action_padding": True,
                    **local_details,
                },
            )
        return PreflightCheck(
            name="schema",
            severity="fail",
            summary=(
                f"action-dim mismatch: dataset action is {ds_dim}-D but base "
                f"checkpoint's native action is {base_dim}-D. Training would "
                f"appear to converge but eval success drops to ~0% (a silent "
                f"failure mode). Either use a dataset whose action matches, "
                f"or retrain the action head from scratch with a base that's "
                f"compatible."
            ),
            detail={
                "dataset": cfg.dataset,
                "dataset_action_dim": ds_dim,
                "base": cfg.base,
                "base_action_dim": base_dim,
                "max_action_dim": max_action_dim,
            },
        )

    return PreflightCheck(
        name="schema",
        severity="ok",
        summary=f"action dim matches: {ds_dim}-D on both sides",
        detail={"action_dim": ds_dim, **local_details},
    )


__all__ = ["check_schema"]
