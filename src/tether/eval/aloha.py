"""Local ALOHA evaluation through LeRobot's evaluator and normalization migrator."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tether.eval.checkpoints import CheckpointSpec, resolve_checkpoint


@dataclass(frozen=True)
class AlohaEvalConfig:
    policy: str
    output: Path
    revision: str | None = None
    task: str = "AlohaTransferCube-v0"
    device: str = "cpu"
    seed: int = 1000
    num_episodes: int = 2
    python: str = sys.executable
    timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        if self.num_episodes < 1 or self.seed < 0 or self.timeout_s <= 0:
            raise ValueError("Episodes and timeout must be positive; seed must be nonnegative.")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Device must be cpu, cuda or mps.")
        if self.task not in {"AlohaTransferCube-v0", "AlohaInsertion-v0"}:
            raise ValueError("Unknown ALOHA task. Choose AlohaTransferCube-v0 or AlohaInsertion-v0.")


def needs_migration(checkpoint_dir: Path) -> bool:
    """A checkpoint predating processor pipelines needs LeRobot's migration."""
    return not (checkpoint_dir / "policy_preprocessor.json").is_file()


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """95% Wilson score bounds as fractions, for observed Bernoulli outcomes."""
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials < 1:
        raise ValueError("Wilson interval requires integer counts with 0 <= successes <= trials and trials > 0.")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper


def parse_eval_info(path: Path, *, num_episodes: int, seed: int) -> dict:
    """Read current per-task or legacy per-episode LeRobot results, never aggregates alone.

    LeRobot 0.5.1 drops episode seeds in its per-task format. With batch size 1,
    episode seeds are the explicit start seed plus the episode index.
    """
    info = json.loads(path.read_text())
    if not isinstance(info, dict):
        raise ValueError("LeRobot eval_info.json must contain an object.")
    if "per_task" in info:
        tasks = info["per_task"]
        if (not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict)
                or tasks[0].get("task_group") != "aloha"):
            raise ValueError("Expected results for exactly one ALOHA task.")
        successes = tasks[0]["metrics"]["successes"]
        if not isinstance(successes, list):
            raise ValueError("LeRobot successes must be a list of boolean episode outcomes.")
        rows = [{"episode_index": i, "seed": seed + i, "success": value} for i, value in enumerate(successes)]
        if info["overall"]["n_episodes"] != num_episodes:
            raise ValueError("LeRobot aggregate episode count does not match the request.")
    else:
        rows = info["per_episode"]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("LeRobot per_episode must be a list of episode objects.")
        rows = [
            {"episode_index": i, "seed": row["seed"], "success": row["success"]}
            for i, row in enumerate(rows)
        ]
    if len(rows) != num_episodes or not rows:
        raise ValueError("LeRobot episode count does not match the request.")
    for i, row in enumerate(rows):
        if type(row["success"]) is not bool or type(row["seed"]) is not int or row["seed"] != seed + i:
            raise ValueError("LeRobot results must contain boolean successes and the requested episode seeds.")
    n_success = sum(row["success"] for row in rows)
    lower, upper = wilson_interval(n_success, len(rows))
    return {
        "episodes": rows,
        "n_success": n_success,
        "n_total": len(rows),
        "success_rate": n_success / len(rows),
        "confidence_interval": {"method": "wilson-score", "confidence": 0.95, "lower": lower, "upper": upper},
    }


def build_receipt(config: AlohaEvalConfig, checkpoint: CheckpointSpec, *, lerobot_version: str, migrated: bool) -> dict:
    """Retain the requested case and observed evaluator runtime beside its results."""
    return {
        "schema_version": 1,
        "policy": {"repo": None if Path(checkpoint.source).is_dir() else checkpoint.source, **checkpoint.to_dict()},
        "env": {"type": "aloha", "task": config.task},
        "seed": config.seed,
        "num_episodes": config.num_episodes,
        "lerobot_version": lerobot_version,
        "device": config.device,
        "batch_size": 1,
        "use_amp": False,
        "migrated": migrated,
    }


def _runtime_version(config: AlohaEvalConfig) -> str:
    # LeRobot can silently fall back to CPU. Refuse an unavailable device so the
    # receipt identifies the device actually used by this explicitly configured run.
    probe = subprocess.run(
        [config.python, "-c", "import sys, torch; from importlib.metadata import version; "
         "d=sys.argv[1]; ok=d=='cpu' or (d=='cuda' and torch.cuda.is_available()) or "
         "(d=='mps' and torch.backends.mps.is_available()); "
         "sys.exit('Requested device unavailable: '+d) if not ok else print(version('lerobot'))", config.device],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return probe.stdout.strip()


def run_aloha_eval(config: AlohaEvalConfig) -> dict:
    """Resolve a pinned checkpoint, migrate a local copy if needed, then evaluate."""
    checkpoint = resolve_checkpoint(config.policy, revision=config.revision)
    if checkpoint.kind != "full":
        raise ValueError("ALOHA evaluation requires a full LeRobot checkpoint.")
    source = Path(checkpoint.source)
    if not source.is_dir() and not re.fullmatch(r"[0-9a-fA-F]{40}", checkpoint.revision or ""):
        raise ValueError("ALOHA evaluation requires an exact 40-hex checkpoint revision.")
    output = config.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Evaluation output must be a new or empty directory; choose a fresh --output.")
    version = _runtime_version(config)
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(checkpoint.source, revision=checkpoint.revision))
    migrated = needs_migration(source)
    if migrated:
        migrated_path = output / "migrated-policy"
        cmd = [config.python, "-m", "lerobot.processor.migrate_policy_normalization",
               "--pretrained-path", str(source), "--output-dir", str(migrated_path)]
        if checkpoint.revision:
            cmd.extend(["--revision", checkpoint.revision])
        # No push flag: the source snapshot stays untouched and output is local.
        subprocess.run(cmd, check=True, timeout=config.timeout_s)
        if needs_migration(migrated_path):
            raise ValueError("Migration did not produce policy_preprocessor.json.")
        source = migrated_path
    # This module is the lerobot-eval entry point, using the same Python as migration.
    cmd = [config.python, "-m", "lerobot.scripts.lerobot_eval",
           f"--policy.path={source}", "--env.type=aloha", f"--env.task={config.task}",
           f"--eval.n_episodes={config.num_episodes}", "--eval.batch_size=1",
           f"--policy.device={config.device}", "--policy.use_amp=false", f"--seed={config.seed}",
           f"--output_dir={output}"]
    subprocess.run(cmd, check=True, timeout=config.timeout_s)
    result = parse_eval_info(output / "eval_info.json", num_episodes=config.num_episodes, seed=config.seed)
    receipt = build_receipt(config, checkpoint, lerobot_version=version, migrated=migrated)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "case-identity.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return result
