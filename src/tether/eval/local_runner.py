"""Real Linux/CUDA SmolVLA evaluator built on Tether's LIBERO rollout."""

from __future__ import annotations

import importlib.util
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tether.eval.checkpoints import CheckpointSpec
from tether.eval.libero import EpisodeResult, EvalReport, LiberoSuiteConfig, TaskResult
from tether.eval.episode_evidence import CaptureLimits


class LocalEvaluationUnavailable(RuntimeError):
    pass


def check_local_readiness() -> dict:
    missing = [
        name for name in ("torch", "lerobot", "libero") if importlib.util.find_spec(name) is None
    ]
    if platform.system() != "Linux":
        return {
            "ready": False,
            "reason": "Local LIBERO evaluation is supported on Linux with an NVIDIA GPU.",
            "missing": missing,
        }
    if missing:
        return {
            "ready": False,
            "reason": "Install Tether's eval-local dependencies.",
            "missing": missing,
        }
    import torch

    if not torch.cuda.is_available():
        return {"ready": False, "reason": "CUDA is not available to PyTorch.", "missing": []}
    return {"ready": True, "reason": None, "missing": []}


def load_smolvla_checkpoint(spec: CheckpointSpec):
    """Load exactly the requested policy and its processors."""
    from huggingface_hub import snapshot_download
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    load_source = spec.source
    kwargs = {"revision": spec.revision} if spec.revision else {}
    if spec.kind == "smolvla-lora":
        from peft import PeftModel

        base_kwargs = {"revision": spec.base_revision} if spec.base_revision else {}
        adapter_kwargs = {"revision": spec.revision} if spec.revision else {}
        processor_source = (
            load_source
            if Path(load_source).exists()
            else snapshot_download(load_source, **adapter_kwargs)
        )
        rollout_config = PreTrainedConfig.from_pretrained(processor_source)
        policy = SmolVLAPolicy.from_pretrained(spec.base, config=rollout_config, **base_kwargs)
        policy = PeftModel.from_pretrained(policy, load_source, **adapter_kwargs)
    elif spec.kind == "full":
        if spec.processor_source:
            processor_kwargs = (
                {"revision": spec.processor_revision} if spec.processor_revision else {}
            )
            processor_source = (
                spec.processor_source
                if Path(spec.processor_source).exists()
                else snapshot_download(spec.processor_source, **processor_kwargs)
            )
            rollout_config = PreTrainedConfig.from_pretrained(processor_source)
            policy = SmolVLAPolicy.from_pretrained(load_source, config=rollout_config, **kwargs)
        else:
            policy = SmolVLAPolicy.from_pretrained(load_source, **kwargs)
            processor_source = (
                snapshot_download(load_source, **kwargs)
                if not Path(load_source).exists()
                else load_source
            )
    else:
        raise LocalEvaluationUnavailable(f"Unsupported checkpoint kind: {spec.kind}")
    policy.tether_rollout_config = rollout_config if "rollout_config" in locals() else policy.config
    policy = policy.to("cuda").eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=processor_source,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=processor_source,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def run_local_libero(
    config: LiberoSuiteConfig,
    checkpoint: CheckpointSpec,
    *,
    loader: Callable = load_smolvla_checkpoint,
    rollout: Callable | None = None,
) -> EvalReport:
    readiness = check_local_readiness()
    if loader is load_smolvla_checkpoint and not readiness["ready"]:
        raise LocalEvaluationUnavailable(
            readiness["reason"]
            + (f" Missing: {', '.join(readiness['missing'])}." if readiness["missing"] else "")
        )
    if rollout is None:
        from tether.eval.libero_rollout import run_libero_rollout

        rollout = run_libero_rollout
    policy, preprocessor, postprocessor = loader(checkpoint)
    started = datetime.now(timezone.utc)
    task_results = []
    for suite in config.tasks:
        raw = rollout(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_suite_name=suite,
            task_indices=list(config.task_indices) or None,
            num_episodes=config.num_episodes,
            seed=config.seed,
            save_video_dir=config.output_dir + "/videos" if config.video else "",
            label=checkpoint.identity,
            use_native=True,
            verification_device="cuda",
            evidence_dir=config.output_dir + "/evidence" if config.capture_evidence else "",
            evidence_limits=CaptureLimits(
                max_bytes=config.evidence_max_bytes,
                max_frames=config.evidence_max_frames,
                frame_stride=config.evidence_frame_stride,
            ),
            evidence_provenance={
                "policy": {
                    "kind": checkpoint.kind,
                    "source": checkpoint.source,
                    "revision": checkpoint.revision,
                    "identity": checkpoint.identity,
                    "adapter_base": checkpoint.base,
                    "adapter_base_revision": checkpoint.base_revision,
                }
            },
        )
        errors = {(e.get("task_idx"), e.get("episode")): e for e in raw.get("errors", [])}
        for task in raw.get("per_task", []):
            episodes = []
            for item in task.get("episodes", []):
                ok = bool(item.get("success"))
                error = errors.get((task.get("task_idx"), item.get("ep")))
                reason = (
                    "success"
                    if ok
                    else ("adapter_error" if error or item.get("error") else "timeout")
                )
                episodes.append(
                    EpisodeResult(
                        task_id=f"{suite}_task_{task['task_idx']}",
                        episode_index=int(item["ep"]),
                        success=ok,
                        terminal_reason=reason,
                        wall_clock_s=float(item.get("wall_clock_s", 0)),
                        n_steps=int(item.get("steps", 0)),
                        video_path=item.get("video_path"),
                        error_message=str(error or item.get("error"))
                        if (error or item.get("error"))
                        else (None if ok else "Task did not succeed before the step limit."),
                        evidence_path=item.get("evidence_path"),
                        evidence_complete=item.get("evidence_complete"),
                        evidence_truncated=item.get("evidence_truncated"),
                    )
                )
            task_results.append(
                TaskResult.from_episodes(f"{suite}_task_{task['task_idx']}", episodes)
            )
    return EvalReport.from_task_results(
        suite="libero",
        runtime="local",
        seed=config.seed,
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        results=task_results,
    )
