"""Parse the real LoRA argv with LeRobot 0.5.1, offline and without weights.

Mirrors Studio's backend/tests/test_lora_argv_parser.py at the Tether boundary.
Requires LeRobot 0.5.1 and PEFT. Inline resume only parses its flag; saved-config
resume restores the checkpoint path. Neither case runs training or qualifies
dataset admission or interrupted-training recovery.
"""

import importlib
from importlib.metadata import version
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("S-04 must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    pytest.importorskip("lerobot")
    pytest.importorskip("peft")
    assert version("lerobot") == "0.5.1"

    import draccus
    from lerobot.configs import parser, policies
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    run = importlib.import_module("tether.finetune.run")
    source = Path(__file__).resolve().parents[1] / "src"
    assert Path(run.__file__).resolve().is_relative_to(source)
    # Keep the real config loader and policy decoding; stub only hub lookup.
    policy_file = tmp_path / "config.json"
    policy_file.write_text(json.dumps(draccus.encode(SmolVLAConfig(device="cpu"))))

    def hub_config(*args, **kwargs):
        assert kwargs["repo_id"] == "lerobot/smolvla_base"
        assert kwargs["filename"] == "config.json"
        return str(policy_file)

    monkeypatch.setattr(policies, "hf_hub_download", hub_config)
    return SimpleNamespace(draccus=draccus, parser=parser, config_type=TrainPipelineConfig, run=run)


def _config(monkeypatch, tmp_path, rank, dataset="remote", resume=False):
    from tether.finetune.config import FinetuneConfig

    cfg = FinetuneConfig(
        base="lerobot/smolvla_base",
        dataset="fastcrest/parser-fixture",
        dataset_revision="dataset-sha-1234",
        output=tmp_path / "run",
        num_steps=3,
        batch_size=2,
        learning_rate=1e-4,
        lora_rank=rank,
        resume=resume,
    )
    if dataset == "local":
        import tether.finetune.local_dataset as admission
        import tether.finetune.preflight.schema as schema

        root = tmp_path / "export" / "dataset"
        root.mkdir(parents=True)
        cfg.dataset_root = root
        cfg.dataset_manifest_sha256 = "a" * 64
        cfg.dataset_revision = None
        snapshot = dict(dataset_root=str(root), manifest_sha256=cfg.dataset_manifest_sha256)
        # Dataset byte/schema admission has its own tests, outside parser scope.
        def verified_export(actual_root, actual_digest):
            assert actual_root == root
            assert actual_digest == cfg.dataset_manifest_sha256
            return snapshot

        monkeypatch.setattr(admission, "verify_local_export", verified_export)
        monkeypatch.setattr(schema, "check_schema", lambda cfg: SimpleNamespace(severity="ok"))
    return cfg


def _parse(runtime, monkeypatch, command):
    # Match parser.wrap preprocessing; validate() also reads selectors in argv.
    monkeypatch.setattr(sys, "argv", command)
    args = runtime.parser.filter_path_args(runtime.config_type.__get_path_fields__(), command[1:])
    config_path = runtime.parser.parse_arg("config_path", args)
    args = runtime.parser.filter_arg("config_path", args)
    with runtime.draccus.config_type("json"):
        return runtime.draccus.parse(runtime.config_type, config_path=config_path, args=args)


def _assert_policy(parsed, rank, resume, expected_path):
    assert parsed.peft.method_type.upper() == "LORA"
    assert parsed.peft.r == rank
    assert parsed.resume is resume
    # policy.path is a selector consumed by validate(), not a dataclass field.
    assert parsed.policy.pretrained_path == Path(expected_path)
    assert parsed.policy.type == "smolvla"


@pytest.mark.parametrize("rank", [4, 32, 128])
@pytest.mark.parametrize("dataset,resume", [("remote", False), ("local", False), ("remote", True)])
def test_real_lora_argv(runtime, monkeypatch, tmp_path, rank, dataset, resume):
    cfg = _config(monkeypatch, tmp_path, rank, dataset, resume)
    parsed = _parse(runtime, monkeypatch, runtime.run._build_lerobot_command(cfg))
    assert parsed.peft.r == rank
    assert parsed.resume is resume
    parsed.validate()
    _assert_policy(parsed, rank, resume, cfg.base)
    assert parsed.dataset.repo_id == cfg.dataset
    assert parsed.dataset.root == (str(cfg.dataset_root) if cfg.dataset_root else None)
    assert parsed.dataset.revision == cfg.dataset_revision
    assert (parsed.steps, parsed.batch_size, parsed.seed) == (cfg.num_steps, cfg.batch_size, cfg.seed)
    assert parsed.checkpoint_path is None  # Inline resume does not restore a checkpoint.
    assert parsed.output_dir == cfg.output / "training"
    assert parsed.policy.push_to_hub is False
    if not resume:
        from lerobot.optim.optimizers import AdamWConfig
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

        assert isinstance(parsed.optimizer, AdamWConfig)
        assert parsed.optimizer.lr == 1e-4
        assert isinstance(parsed.scheduler, CosineDecayWithWarmupSchedulerConfig)


@pytest.mark.parametrize("rank", [4, 32, 128])
def test_saved_config_resume(runtime, monkeypatch, tmp_path, rank):
    from tether.finetune.resume import build_resume_command

    cfg = _config(monkeypatch, tmp_path, rank)
    parsed = _parse(runtime, monkeypatch, runtime.run._build_lerobot_command(cfg))
    parsed.validate()
    checkpoint = cfg.output / "training" / "checkpoints" / "000003"
    policy_dir = checkpoint / "pretrained_model"
    policy_dir.mkdir(parents=True)
    parsed._save_pretrained(policy_dir)
    command, selected = build_resume_command(cfg.output)
    resumed = _parse(runtime, monkeypatch, command)
    resumed.validate()
    _assert_policy(resumed, rank, True, policy_dir)
    assert resumed.checkpoint_path == selected == checkpoint
    assert (resumed.dataset.repo_id, resumed.dataset.revision, resumed.dataset.root) == (
        parsed.dataset.repo_id, parsed.dataset.revision, parsed.dataset.root,
    )
    assert (resumed.steps, resumed.batch_size, resumed.seed) == (
        parsed.steps, parsed.batch_size, parsed.seed,
    )
    assert resumed.optimizer == parsed.optimizer
    assert resumed.scheduler == parsed.scheduler


def test_unknown_flag_is_rejected(runtime, monkeypatch, tmp_path, capsys):
    cfg = _config(monkeypatch, tmp_path, 32)
    command = runtime.run._build_lerobot_command(cfg) + ["--s04_unknown_flag=1"]
    with pytest.raises(SystemExit) as error:
        _parse(runtime, monkeypatch, command)
    assert error.value.code == 2
    assert "--s04_unknown_flag=1" in capsys.readouterr().err


def test_noninteger_rank_is_rejected(runtime, monkeypatch, tmp_path):
    from draccus.utils import DecodingError

    cfg = _config(monkeypatch, tmp_path, 32)
    cfg.lora_rank = "not-an-integer"
    with pytest.raises(DecodingError, match="Couldn't parse 'not-an-integer' into an int"):
        _parse(runtime, monkeypatch, runtime.run._build_lerobot_command(cfg))


def test_rank_zero_parses(runtime, monkeypatch, tmp_path):
    cfg = _config(monkeypatch, tmp_path, 0)
    parsed = _parse(runtime, monkeypatch, runtime.run._build_lerobot_command(cfg))
    parsed.validate()
    _assert_policy(parsed, 0, False, cfg.base)


def test_validate_replaces_optimizer(runtime, monkeypatch, tmp_path):
    from lerobot.optim.optimizers import AdamConfig, AdamWConfig

    cfg = _config(monkeypatch, tmp_path, 32)
    cfg.learning_rate = 3e-4
    parsed = _parse(runtime, monkeypatch, runtime.run._build_lerobot_command(cfg))
    assert isinstance(parsed.optimizer, AdamConfig)
    assert parsed.optimizer.lr == 3e-4
    parsed.validate()
    assert isinstance(parsed.optimizer, AdamWConfig)
    assert parsed.optimizer.lr == parsed.scheduler.peak_lr == 1e-4
    assert parsed.optimizer.weight_decay == 1e-10
    _assert_policy(parsed, 32, False, cfg.base)
