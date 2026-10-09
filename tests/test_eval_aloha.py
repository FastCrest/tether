"""Contract checks for local LeRobot ALOHA evaluation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tether.cli import app
from tether.eval import aloha
from tether.eval.checkpoints import resolve_checkpoint

REPO = "lerobot/act_aloha_sim_transfer_cube_human"
REVISION = "ba73b2766f1371cdc133ca4efb97eb090d744625"


def eval_info(successes):
    return {
        "per_task": [{"task_group": "aloha", "task_id": 0, "metrics": {"successes": successes}}],
        "overall": {"n_episodes": len(successes)},
    }


@pytest.fixture
def checkpoint_dir(tmp_path):
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type": "act"}')
    (path / "model.safetensors").write_bytes(b"test weights")
    return path


@pytest.mark.parametrize("preprocessor,expected", [(False, True), (True, False)])
def test_migration_decision(checkpoint_dir, preprocessor, expected):
    if preprocessor:
        (checkpoint_dir / "policy_preprocessor.json").write_text("{}")
    assert aloha.needs_migration(checkpoint_dir) is expected


def test_wilson_published_reproduction():
    lower, upper = aloha.wilson_interval(425, 500)
    assert (round(lower * 100, 1), round(upper * 100, 1)) == (81.6, 87.9)


@pytest.mark.parametrize("successes,trials", [(0, 2), (2, 2), (0, 500), (500, 500)])
def test_wilson_boundaries(successes, trials):
    lower, upper = aloha.wilson_interval(successes, trials)
    assert 0 <= lower <= successes / trials <= upper <= 1


@pytest.mark.parametrize("successes,trials", [(0, 0), (-1, 2), (3, 2), (True, 2), (1, 2.0)])
def test_wilson_rejects_invalid_counts(successes, trials):
    with pytest.raises(ValueError):
        aloha.wilson_interval(successes, trials)


@pytest.mark.parametrize("legacy", [False, True])
def test_parse_per_episode_success(tmp_path, legacy):
    info = eval_info([True, False])
    if legacy:
        info = {"per_episode": [{"success": True, "seed": 1000}, {"success": False, "seed": 1001}]}
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps(info))
    report = aloha.parse_eval_info(path, num_episodes=2, seed=1000)
    assert report["episodes"] == [
        {"episode_index": 0, "seed": 1000, "success": True},
        {"episode_index": 1, "seed": 1001, "success": False},
    ]
    assert report["n_success"] == 1
    assert report["success_rate"] == 0.5
    assert report["confidence_interval"]["lower"] == pytest.approx(0.0945312057)


@pytest.mark.parametrize("info", [
    eval_info([]), eval_info([True]), eval_info([True, "false"]),
    [], {"per_task": [None]},
    {"per_task": [{"task_group": "aloha", "metrics": {"successes": "true"}}]},
    {"overall": {"pc_success": 85.0}},
    {"per_task": [], "overall": {"n_episodes": 2}},
    {"per_episode": [{"success": True, "seed": 999}, {"success": False, "seed": 1001}]},
])
def test_parse_refuses_missing_or_invalid_evidence(tmp_path, info):
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps(info))
    with pytest.raises((ValueError, KeyError)):
        aloha.parse_eval_info(path, num_episodes=2, seed=1000)


def test_receipt_retains_original_pinned_case(tmp_path):
    config = aloha.AlohaEvalConfig(policy=REPO, revision=REVISION, output=tmp_path)
    checkpoint = resolve_checkpoint(REPO, revision=REVISION)
    receipt = aloha.build_receipt(config, checkpoint, lerobot_version="0.5.1", migrated=True)
    assert receipt["policy"]["repo"] == REPO
    assert receipt["policy"]["revision"] == REVISION
    assert receipt["policy"]["identity"] == f"hf:{REPO}@{REVISION}"
    assert receipt["env"] == {"type": "aloha", "task": "AlohaTransferCube-v0"}
    assert (receipt["seed"], receipt["num_episodes"], receipt["lerobot_version"], receipt["device"]) == (1000, 2, "0.5.1", "cpu")
    assert receipt["migrated"] is True
    assert receipt["batch_size"] == 1 and receipt["use_amp"] is False
    changed = replace(config, task="AlohaInsertion-v0", seed=9, num_episodes=3, device="mps")
    other = aloha.build_receipt(changed, checkpoint, lerobot_version="0.6.0", migrated=False)
    assert other["env"]["task"] == "AlohaInsertion-v0"
    assert (other["seed"], other["num_episodes"], other["lerobot_version"], other["device"]) == (9, 3, "0.6.0", "mps")


@pytest.mark.parametrize("legacy", [False, True])
def test_runner_migrates_only_when_needed_and_keeps_source(checkpoint_dir, tmp_path, monkeypatch, legacy):
    if not legacy:
        (checkpoint_dir / "policy_preprocessor.json").write_text("{}")
    original = {p.name: p.read_bytes() for p in checkpoint_dir.iterdir()}
    output = tmp_path / "results"
    config = aloha.AlohaEvalConfig(policy=REPO, revision=REVISION, output=output, python="eval-python")
    calls = []
    monkeypatch.setattr(aloha, "_runtime_version", lambda config: "0.5.1")

    def download(repo, *, revision):
        assert (repo, revision) == (REPO, REVISION)
        return str(checkpoint_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["check"] is True
        assert "--push-to-hub" not in cmd
        assert cmd[0] == "eval-python"
        if cmd[2] == "lerobot.processor.migrate_policy_normalization":
            migrated = Path(cmd[cmd.index("--output-dir") + 1])
            assert migrated == output / "migrated-policy"
            assert cmd[cmd.index("--revision") + 1] == REVISION
            migrated.mkdir()
            (migrated / "policy_preprocessor.json").write_text("{}")
        else:
            expected = output / "migrated-policy" if legacy else checkpoint_dir
            for value in [f"--policy.path={expected}", "--env.type=aloha", "--env.task=AlohaTransferCube-v0",
                          "--policy.device=cpu", "--policy.use_amp=false", "--seed=1000",
                          "--eval.n_episodes=2", "--eval.batch_size=1"]:
                assert value in cmd
            (output / "eval_info.json").write_text(json.dumps(eval_info([True, False])))

    monkeypatch.setattr(aloha.subprocess, "run", run)
    report = aloha.run_aloha_eval(config)
    assert len(calls) == (2 if legacy else 1)
    assert original == {p.name: p.read_bytes() for p in checkpoint_dir.iterdir()}
    assert json.loads((output / "report.json").read_text()) == report
    receipt = json.loads((output / "case-identity.json").read_text())
    assert receipt["migrated"] is legacy
    assert receipt["policy"]["repo"] == REPO and receipt["policy"]["revision"] == REVISION


def test_failed_migration_never_runs_eval_or_writes_receipt(checkpoint_dir, tmp_path, monkeypatch):
    output = tmp_path / "results"
    monkeypatch.setattr(aloha, "_runtime_version", lambda config: "0.5.1")
    calls = []
    monkeypatch.setattr(aloha.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    with pytest.raises(ValueError, match="Migration did not produce"):
        aloha.run_aloha_eval(aloha.AlohaEvalConfig(policy=str(checkpoint_dir), output=output))
    assert len(calls) == 1
    assert not (output / "case-identity.json").exists()


def test_failed_evaluation_writes_no_success_receipt(checkpoint_dir, tmp_path, monkeypatch):
    output = tmp_path / "results"
    (checkpoint_dir / "policy_preprocessor.json").write_text("{}")
    monkeypatch.setattr(aloha, "_runtime_version", lambda config: "0.5.1")

    def fail(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(aloha.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        aloha.run_aloha_eval(aloha.AlohaEvalConfig(policy=str(checkpoint_dir), output=output))
    assert not (output / "report.json").exists()
    assert not (output / "case-identity.json").exists()


def test_refuses_stale_output_before_running(checkpoint_dir, tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    (output / "eval_info.json").write_text("old evidence")
    monkeypatch.setattr(aloha, "_runtime_version", lambda config: pytest.fail("Must reject before runtime probe"))
    with pytest.raises(ValueError, match="fresh --output"):
        aloha.run_aloha_eval(aloha.AlohaEvalConfig(policy=str(checkpoint_dir), output=output))


def test_refuses_mutable_remote_revision(tmp_path):
    with pytest.raises(ValueError, match="40-hex"):
        aloha.run_aloha_eval(aloha.AlohaEvalConfig(policy=REPO, revision="main", output=tmp_path / "out"))


def test_cli_dispatches_aloha_locally(tmp_path, monkeypatch):
    seen = []

    def run(config):
        seen.append(config)
        return {"n_success": 1, "n_total": 2, "success_rate": 0.5,
                "confidence_interval": {"lower": 0.0945, "upper": 0.9055}}

    monkeypatch.setattr("tether.eval.cli.run_aloha_eval", run)
    result = CliRunner().invoke(app, ["eval", REPO, "--suite", "aloha", "--checkpoint-revision", REVISION,
                                    "--num-episodes", "2", "--seed", "1000", "--device", "cpu",
                                    "--eval-python", "eval-python", "--output", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert seen[0].python == "eval-python" and seen[0].seed == 1000
    assert "Wilson 95% CI" in result.output and "case-identity.json" in result.output


@pytest.mark.parametrize("options", [["--runtime", "modal"], ["--tasks", "libero_10"], ["--video"]])
def test_cli_rejects_unsupported_aloha_options(options):
    result = CliRunner().invoke(app, ["eval", REPO, "--suite", "aloha", *options])
    assert result.exit_code == 2


def test_cli_reports_subprocess_failure(tmp_path, monkeypatch):
    def fail(config):
        raise subprocess.CalledProcessError(1, ["python"], stderr="Requested device unavailable: mps")

    monkeypatch.setattr("tether.eval.cli.run_aloha_eval", fail)
    result = CliRunner().invoke(app, ["eval", REPO, "--suite", "aloha", "--output", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "Requested device unavailable: mps" in result.output
