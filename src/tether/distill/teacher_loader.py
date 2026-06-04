"""Load a teacher policy for SnapFlow distillation.

The teacher is a tether-exported checkpoint (merged PyTorch, same layout
`_auto_export` consumes): `<dir>/config.json` + `model.safetensors` +
`policy_preprocessor.json` + `policy_postprocessor.json`. Distillation
student starts as a copy of the teacher, so we just load the teacher
twice and train one.

v0.3 scope: pi0 + pi0.5 only. SmolVLA lands in v0.3.1.

## What this module does

- Resolve `teacher_export` (FinetuneConfig field) → loaded nn.Module
- Freeze teacher params + set eval mode
- Infer policy_type from checkpoint config (smolvla / pi0 / pi05)
- Return both the policy + its config for the backend to wire up

## What it does NOT do

- No state-dict surgery (backend handles the student copy)
- No dataset construction (backend's responsibility)
- No velocity-function binding (backend exposes teacher.velocity_fn)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Policy types supported as distillation TEACHERS in v0.3.
# SmolVLA is disabled pending velocity-convergence validation
# (flagged as v0.3.1 kill-gate in scope_decision.md).
V03_TEACHER_ALLOWLIST: frozenset[str] = frozenset({"pi0", "pi05"})


@dataclass
class LoadedTeacher:
    """Loaded teacher handle returned by `load_teacher`."""

    policy: Any
    """The PyTorch policy object (e.g. Pi0Policy instance), eval() +
    all-params-frozen. Kept as Any to avoid importing lerobot at
    teacher_loader import time."""

    config: dict
    """The raw `config.json` dict from the checkpoint."""

    policy_type: str
    """One of 'smolvla' | 'pi0' | 'pi05' | 'gr00t_n1_5'."""

    checkpoint_dir: Path
    """Where the teacher came from. Used by the backend for provenance
    logging in the student's distill_provenance field."""


def resolve_policy_type(config: dict) -> str:
    """Pick the lerobot policy short-name from a checkpoint config.

    Mirrors `tether.finetune.run._infer_policy_type` but reads from the
    config dict rather than a base-model HF id. Uses the `type` field
    if present (draccus writes it when the checkpoint was saved by
    lerobot-train).
    """
    if "type" in config:
        return str(config["type"])
    # Fallback: scan for distinctive keys.
    keys = set(config.keys())
    if "load_vlm_weights" in keys or "resize_imgs_with_padding" in keys:
        # SmolVLA-specific fields
        return "smolvla"
    if "prefix_length" in keys and keys.issuperset({"chunk_size", "num_steps"}):
        # pi0 / pi0.5 both have these; can't distinguish without more fields
        if config.get("num_expert_layers", 0) > 0:
            return "pi05"
        return "pi0"
    raise ValueError(
        "could not infer policy_type from config.json. Pass policy_type "
        "explicitly via FinetuneConfig.extra_lerobot_args."
    )


def load_teacher(
    teacher_export: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "bf16",
    allowlist: frozenset[str] = V03_TEACHER_ALLOWLIST,
) -> LoadedTeacher:
    """Load a frozen teacher policy from a tether-export dir OR HF id.

    Args:
      teacher_export: either
        - a local path to a tether-export dir (the `merged/`
          subdir if source was a LoRA fine-tune, or the monolithic
          `pretrained_model/` dir directly), or
        - an HF repo id like 'lerobot/pi0_base' — the loader will
          snapshot_download it to the HF cache before proceeding.
      device: where to place the teacher. 'cpu' is safe for small
        teachers + dev; 'cuda' for training.
      dtype: 'bf16' | 'fp32'. bf16 saves memory during distillation.
      allowlist: which policy_types are supported. v0.3 default is
        pi0 + pi0.5 only (SmolVLA v0.3.1; GR00T v0.5+).

    Returns a LoadedTeacher. Raises ValueError on unsupported
    policy_type, FileNotFoundError if neither path nor HF resolves.
    """
    import torch

    path = _resolve_teacher_path(teacher_export)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"teacher export missing config.json: {config_path}"
        )
    with config_path.open() as f:
        config = json.load(f)

    policy_type = resolve_policy_type(config)
    if policy_type not in allowlist:
        raise ValueError(
            f"policy_type={policy_type!r} is not supported as a SnapFlow "
            f"teacher in v0.3. Allowlist: {sorted(allowlist)}. "
            f"SmolVLA is v0.3.1 conditional; GR00T is v0.5+."
        )

    logger.info("[teacher_loader] loading %s from %s", policy_type, path)
    # Lazy-import lerobot policy classes only for the types we actually
    # use — keeps the module importable without lerobot installed
    # (useful for CI / unit tests).
    policy = _load_policy_for_type(policy_type, path)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    target_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    policy.to(device).to(target_dtype)
    logger.info("[teacher_loader] teacher %s loaded on %s (%s), frozen",
                policy_type, device, dtype)

    return LoadedTeacher(
        policy=policy,
        config=config,
        policy_type=policy_type,
        checkpoint_dir=path,
    )


def _resolve_teacher_path(teacher_export: str | Path) -> Path:
    """Return a local Path to the teacher's tether-export dir.

    Dispatch:
      1. If the argument is an existing local path → use it.
      2. Else if it looks like an HF repo id ('org/name' pattern, not
         absolute, not starting with './') → snapshot_download to the HF
         cache and return the cached path.
      3. Else → FileNotFoundError with an actionable hint.

    HF downloads are cached by huggingface_hub automatically, so a
    second call with the same id is a no-op. We pull the full
    repo (config.json + model.safetensors + processor files) so the
    downstream `from_pretrained` call can read everything locally.
    """
    path = Path(teacher_export)
    if path.exists():
        return path

    # Heuristic: 'org/repo' is a repo id. Reject obvious path-like strings
    # before attempting an HF download to avoid spurious network calls on
    # typos (e.g. './my_teacher/' with a trailing slash that doesn't exist).
    s = str(teacher_export)
    looks_like_hf_id = (
        "/" in s
        and not s.startswith(("./", "/", "~", ".."))
        and not s.endswith("/")
        and s.count("/") >= 1
    )
    if looks_like_hf_id:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise FileNotFoundError(
                f"teacher export not found locally: {teacher_export}. "
                f"Looks like an HF repo id, but huggingface_hub is not "
                f"installed: {e}. Install it or pass a local path."
            )
        logger.info("[teacher_loader] downloading %s from HF Hub", s)
        cached = snapshot_download(repo_id=s, repo_type="model")
        return Path(cached)

    raise FileNotFoundError(
        f"teacher export not found: {teacher_export}. Expected a "
        f"tether-export dir (model.safetensors + config.json) OR an "
        f"HF repo id like 'lerobot/pi0_base'."
    )


def _load_policy_for_type(policy_type: str, path: Path):
    """Dispatch `from_pretrained` to the right lerobot policy class."""
    if policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy.from_pretrained(str(path))
    if policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy.from_pretrained(str(path))
    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy.from_pretrained(str(path))
    raise ValueError(
        f"No lerobot-policy dispatcher for policy_type={policy_type!r}. "
        f"Add a branch to _load_policy_for_type when expanding the allowlist."
    )


__all__ = [
    "V03_TEACHER_ALLOWLIST",
    "LoadedTeacher",
    "load_teacher",
    "resolve_policy_type",
]
