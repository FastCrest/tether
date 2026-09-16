from pathlib import Path

import pytest

from tether.finetune.resume import ResumeConfigurationError, build_resume_command


def checkpoint(root: Path, step: int, *, config=True):
    item = root / "training" / "checkpoints" / f"{step:06d}"
    model = item / "pretrained_model"
    model.mkdir(parents=True)
    if config:
        (model / "train_config.json").write_text("{}")
    return item


def test_resume_uses_latest_numeric_checkpoint_own_config(tmp_path):
    checkpoint(tmp_path, 100)
    selected = checkpoint(tmp_path, 200)
    (selected.parent / "last").symlink_to(selected.name)

    command, resumed_from = build_resume_command(tmp_path)

    assert resumed_from == selected
    assert command == [
        "lerobot-train",
        f"--config_path={selected / 'pretrained_model' / 'train_config.json'}",
        "--resume=true",
    ]
    assert not any(value.startswith("--policy.path") for value in command)
    assert not any(value.startswith("--dataset.repo_id") for value in command)


def test_resume_refuses_last_only_alias(tmp_path):
    root = tmp_path / "training" / "checkpoints"
    target = root / "005000"
    (target / "pretrained_model").mkdir(parents=True)
    (target / "pretrained_model" / "train_config.json").write_text("{}")
    last = root / "last"
    last.symlink_to(target.name)
    target.rename(tmp_path / "moved")
    with pytest.raises(ResumeConfigurationError, match="numeric checkpoint"):
        build_resume_command(tmp_path)


def test_resume_refuses_checkpoint_missing_saved_train_config(tmp_path):
    checkpoint(tmp_path, 100, config=False)
    with pytest.raises(ResumeConfigurationError, match="train_config"):
        build_resume_command(tmp_path)
