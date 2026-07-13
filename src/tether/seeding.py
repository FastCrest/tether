"""Shared process seeding helpers for training and evaluation paths."""
from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy, and Torch when available.

    ``PYTHONHASHSEED`` only affects newly spawned Python interpreters after
    process start, but setting it here keeps child training processes aligned
    with the requested run seed.
    """
    seed_int = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    random.seed(seed_int)
    np.random.seed(seed_int)

    report: dict[str, Any] = {
        "seed": seed_int,
        "python": True,
        "numpy": True,
        "torch": False,
        "cuda": False,
        "deterministic_torch": deterministic_torch,
    }

    try:
        import torch
    except ImportError:
        return report

    torch.manual_seed(seed_int)
    report["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
        report["cuda"] = True

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return report


def seeded_subprocess_env(
    seed: int,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with ``PYTHONHASHSEED`` pinned to ``seed``."""
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONHASHSEED"] = str(int(seed))
    return env


__all__ = ["seed_everything", "seeded_subprocess_env"]
