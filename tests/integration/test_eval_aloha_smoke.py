"""Two real CPU ALOHA episodes, gated by an existing evaluation venv."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

VENV = Path(os.environ.get("TETHER_ALOHA_EVAL_VENV", "/nonexistent"))
PYTHON = VENV / "bin" / "python"
REVISION = "ba73b2766f1371cdc133ca4efb97eb090d744625"
pytestmark = pytest.mark.skipif(not PYTHON.is_file(), reason="Existing ALOHA eval venv is absent; set TETHER_ALOHA_EVAL_VENV.")


def test_aloha_two_episode_customer_flow(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "aloha-results"
    # Keep writes in the test workspace and leave the borrowed runtime untouched.
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1",
               TETHER_SKIP_ONBOARDING="1", TETHER_NO_UPGRADE_CHECK="1",
               TETHER_HOME=str(tmp_path / "tether"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1")
    for name, directory in [("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "cache"),
                            ("TORCH_HOME", "torch"), ("MPLCONFIGDIR", "matplotlib"), ("TMPDIR", "tmp")]:
        cache = tmp_path / directory
        cache.mkdir()
        env[name] = str(cache)
    result = subprocess.run(
        [str(PYTHON), "-m", "tether.cli", "eval", "lerobot/act_aloha_sim_transfer_cube_human",
         "--checkpoint-revision", REVISION, "--suite", "aloha", "--runtime", "local",
         "--env-task", "AlohaTransferCube-v0", "--num-episodes", "2", "--seed", "1000",
         "--device", "cpu", "--output", str(output), "--eval-timeout", "120"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=300,
    )
    (tmp_path / "eval.stdout.log").write_text(result.stdout)
    (tmp_path / "eval.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    report = json.loads((output / "report.json").read_text())
    receipt = json.loads((output / "case-identity.json").read_text())
    assert report["n_total"] == 2
    assert [row["seed"] for row in report["episodes"]] == [1000, 1001]
    assert all(type(row["success"]) is bool for row in report["episodes"])
    assert report["confidence_interval"]["method"] == "wilson-score"
    assert receipt["policy"]["revision"] == REVISION
    assert receipt["lerobot_version"] == "0.5.1" and receipt["device"] == "cpu"
    assert receipt["migrated"] is True
    assert (output / "eval_info.json").is_file()
