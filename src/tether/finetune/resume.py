from __future__ import annotations

import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class ResumeConfigurationError(ValueError):
    pass


def resolve_resume_checkpoint(output_dir: Path) -> tuple[Path, Path]:
    """Resolve the latest exact numeric LeRobot checkpoint and its saved config."""
    output_dir = Path(output_dir)
    roots = (
        output_dir / "training" / "checkpoints",
        output_dir / "checkpoints",
    )
    checkpoint_root = next((root for root in roots if root.is_dir()), None)
    if checkpoint_root is None:
        raise ResumeConfigurationError(
            f"resume requires checkpoints under {output_dir / 'training' / 'checkpoints'}"
        )
    candidates = [
        item
        for item in checkpoint_root.iterdir()
        if item.is_dir() and not item.is_symlink() and item.name.isdigit()
    ]
    if not candidates:
        raise ResumeConfigurationError(
            "resume requires an exact numeric checkpoint; `last` aliases are not accepted"
        )
    candidates.sort(key=lambda item: int(item.name))
    checkpoint = candidates[-1]
    config = checkpoint / "pretrained_model" / "train_config.json"
    if not config.is_file():
        raise ResumeConfigurationError(
            f"resume checkpoint is missing pretrained_model/train_config.json: {checkpoint}"
        )
    return checkpoint, config


def build_resume_command(output_dir: Path) -> tuple[list[str], Path]:
    """Build the LeRobot v0.5.1 resume command from the checkpoint's own config.

    Do not repeat `--policy.path`, dataset, optimizer, scheduler, seed or other
    training settings. LeRobot's resume contract loads those values from the
    checkpoint config and uses that config to locate the sibling training_state.
    """
    checkpoint, config = resolve_resume_checkpoint(output_dir)
    return [
        "lerobot-train",
        f"--config_path={config}",
        "--resume=true",
    ], checkpoint


def run_lerobot_resume(
    output_dir: Path,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, Path]:
    command, checkpoint = build_resume_command(output_dir)
    logger.info("[finetune] exact resume: %s", " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\n# tether finetune — exact LeRobot checkpoint resume\n")
        log.write(f"# resume_checkpoint: {checkpoint}\n")
        log.write(f"# cmd: {' '.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            logger.info(line.rstrip())
        process.wait()
        return process.returncode, checkpoint


__all__ = [
    "ResumeConfigurationError",
    "resolve_resume_checkpoint",
    "build_resume_command",
    "run_lerobot_resume",
]
