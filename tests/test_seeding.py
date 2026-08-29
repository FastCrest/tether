from __future__ import annotations

import random

import numpy as np
import pytest

from tether.seeding import seed_everything, seeded_subprocess_env


def test_seed_everything_repeats_python_numpy_and_torch():
    torch = pytest.importorskip("torch")

    report = seed_everything(123)
    first = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )

    report_again = seed_everything(123)
    second = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )

    assert first == second
    assert report["seed"] == 123
    assert report["python"]
    assert report["numpy"]
    assert report["torch"]
    assert report_again["torch"]


def test_seeded_subprocess_env_sets_pythonhashseed(monkeypatch):
    monkeypatch.setenv("KEEP_ME", "yes")

    env = seeded_subprocess_env(456)

    assert env["PYTHONHASHSEED"] == "456"
    assert env["KEEP_ME"] == "yes"


def test_seeded_subprocess_env_preserves_explicit_base_env():
    env = seeded_subprocess_env(789, {"EXISTING": "1", "PYTHONHASHSEED": "old"})

    assert env == {"EXISTING": "1", "PYTHONHASHSEED": "789"}
