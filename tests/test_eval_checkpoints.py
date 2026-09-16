import pytest
import subprocess
import sys
import types

from tether.eval.checkpoints import CheckpointError, CheckpointSpec, resolve_checkpoint
from tether.eval.libero import LiberoSuiteConfig
from tether.eval.local_runner import run_local_libero
from tether.eval.local_runner import load_smolvla_checkpoint
from tether.eval.modal_runner import run_libero_on_modal


def test_lora_identity_includes_adapter_files_and_base(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path":"org/base"}')
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    spec = resolve_checkpoint(tmp_path, base_revision="base-rev")
    assert spec.kind == "smolvla-lora"
    assert spec.base == "org/base"
    assert spec.base_revision == "base-rev"
    assert spec.identity.startswith("sha256:")
    assert {item["path"] for item in spec.files} == {"adapter_config.json", "adapter_model.safetensors"}


def test_remote_checkpoint_requires_revision():
    with pytest.raises(CheckpointError, match="revision"):
        resolve_checkpoint("org/model")
    assert resolve_checkpoint("org/model", revision="abc123").identity == "hf:org/model@abc123"


def test_local_runner_passes_exact_cases_and_preserves_real_outcomes():
    captured = {}
    checkpoint = CheckpointSpec("full", "org/model", "hf:org/model@abc", revision="abc")

    def loader(spec):
        assert spec is checkpoint
        return object(), object(), object()

    def rollout(**kwargs):
        captured.update(kwargs)
        return {"per_task": [{"task_idx": 2, "episodes": [
            {"ep": 0, "success": True, "steps": 12},
            {"ep": 1, "success": False, "steps": 220},
        ]}], "errors": []}

    config = LiberoSuiteConfig(tasks=("libero_spatial",), task_indices=(2,), num_episodes=2, seed=41, runtime="local")
    report = run_local_libero(config, checkpoint, loader=loader, rollout=rollout)
    assert captured["task_indices"] == [2]
    assert captured["seed"] == 41
    assert [episode.success for episode in report.results[0].episodes] == [True, False]
    assert report.results[0].episodes[1].n_steps == 220


def test_local_lora_loader_forwards_pinned_adapter_and_base_revisions(monkeypatch):
    calls = {}

    class Policy:
        config = object()

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls["base"] = (source, kwargs)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    class Pipeline:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return cls()

    class Peft:
        @classmethod
        def from_pretrained(cls, policy, source, **kwargs):
            calls["adapter"] = (source, kwargs)
            return policy

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=Peft))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda source, **kwargs: "/adapter"))
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", types.SimpleNamespace(
        PreTrainedConfig=types.SimpleNamespace(from_pretrained=lambda source: "dataset-config")
    ))
    monkeypatch.setitem(sys.modules, "lerobot.processor.converters", types.SimpleNamespace(
        batch_to_transition=object(), policy_action_to_transition=object(),
        transition_to_batch=object(), transition_to_policy_action=object(),
    ))
    monkeypatch.setitem(sys.modules, "lerobot.processor.pipeline", types.SimpleNamespace(PolicyProcessorPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla.modeling_smolvla", types.SimpleNamespace(SmolVLAPolicy=Policy))

    spec = CheckpointSpec(
        "smolvla-lora", "org/adapter", "hf-lora:org/adapter@adapter-rev+org/base@base-rev",
        revision="adapter-rev", base="org/base", base_revision="base-rev",
    )
    load_smolvla_checkpoint(spec)
    assert calls["base"] == ("org/base", {"config": "dataset-config", "revision": "base-rev"})
    assert calls["adapter"] == ("org/adapter", {"revision": "adapter-rev"})


def test_parent_checkpoint_records_dataset_processor_identity(tmp_path):
    processor = tmp_path / "candidate"
    processor.mkdir()
    (processor / "config.json").write_text("{}")
    (processor / "policy_preprocessor.json").write_text("{}")
    spec = resolve_checkpoint(
        "org/base",
        revision="base-rev",
        processor_source=processor,
    )
    assert spec.identity == "hf:org/base@base-rev"
    assert spec.processor_source == str(processor.resolve())
    assert spec.processor_identity.startswith("sha256:")
    assert spec.to_dict()["processor_identity"] == spec.processor_identity


def test_modal_command_names_selected_adapter_and_never_uses_reference(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "modal_libero_lerobot_native.py").write_text("# fixture")
    captured = []

    def invoke(command, timeout):
        captured.append(command)
        return subprocess.CompletedProcess(command, 1, "", "fixture stop")

    spec = resolve_checkpoint("org/candidate", kind="smolvla-lora", base="org/base", revision="adapter-rev", base_revision="base-rev")
    assert spec.identity == "hf-lora:org/candidate@adapter-rev+org/base@base-rev"
    run_libero_on_modal(
        config=LiberoSuiteConfig(tasks=("libero_10",), task_indices=(2,), num_episodes=1),
        checkpoint=spec, repo_root=tmp_path, modal_invoker=invoke,
    )
    command = captured[0]
    assert command[command.index("--model-id") + 1] == "org/candidate"
    assert command[command.index("--adapter-path") + 1] == "org/candidate"
    assert command[command.index("--adapter-base") + 1] == "org/base"
    assert command[command.index("--adapter-base-revision") + 1] == "base-rev"
    assert "HuggingFaceVLA/smolvla_libero" not in command


def test_modal_parent_command_names_dataset_processor_checkpoint(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "modal_libero_lerobot_native.py").write_text("# fixture")
    captured = []

    def invoke(command, timeout):
        captured.append(command)
        return subprocess.CompletedProcess(command, 1, "", "fixture stop")

    spec = resolve_checkpoint(
        "org/base",
        revision="base-rev",
        processor_source="org/candidate",
        processor_revision="candidate-rev",
    )
    run_libero_on_modal(
        config=LiberoSuiteConfig(tasks=("libero_10",), task_indices=(0,), num_episodes=1),
        checkpoint=spec,
        repo_root=tmp_path,
        modal_invoker=invoke,
    )
    command = captured[0]
    assert command[command.index("--preprocessor-ref") + 1] == "org/candidate"
