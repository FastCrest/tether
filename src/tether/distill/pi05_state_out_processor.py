"""Preprocessor step for v0.5 state-out pi0.5 student.

Pi0.5's default preprocessor builds a prompt like::

    "Task: put both the alphabet soup in the basket, State: 134 127 89 45 ...;\\nAction: "

The ", State: ..." segment discretizes proprio state into tokens and
appends them to the lang string, then the downstream tokenizer
produces lang_tokens that include this state suffix. As a result
`lang_tokens` drifts every frame (state changes per frame), and the
prefix KV cache keyed on `lang_hash` never hits in production.

The state-out student reads state from a separate ``state_proj`` input
path instead. This preprocessor step is a drop-in replacement for
``Pi05PrepareStateTokenizerProcessorStep`` that omits the state portion
of the prompt. Lang tokens become stable across frames within an episode
→ cache hits → 9x deployment speedup (matching pi0's prefix cache
behavior).

Design: `reflex_vla/01_architecture/distill_state_out_pi05_design.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _register_step() -> type:
    """Lazy-register the step with lerobot's ProcessorStepRegistry.

    Done in a function so importing this module doesn't force lerobot
    import (matching the pattern in snapflow_pi0_model.py).
    """
    from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
    from lerobot.types import EnvTransition, TransitionKey

    @ProcessorStepRegistry.register(name="pi05_prepare_tokenizer_state_out")
    @dataclass
    class Pi05PrepareTokenizerStateOutStep(ProcessorStep):
        """Omits the ", State: ..." segment of the pi0.5 prompt.

        Produces prompts of the form::

            "Task: put both the alphabet soup in the basket;\\nAction: "

        vs the default::

            "Task: put both the alphabet soup in the basket, State: 134 127 89 ...;\\nAction: "

        Everything else about the pipeline stays identical — this step
        slots into ``make_pi05_pre_post_processors`` in place of
        ``Pi05PrepareStateTokenizerProcessorStep``.
        """

        max_state_dim: int = 32  # accepted for config compat; unused
        task_key: str = "task"

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            transition = transition.copy()
            tasks = (
                transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
            )
            if tasks is None:
                raise ValueError("No task found in complementary data")

            full_prompts = []
            for task in tasks:
                cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                full_prompt = f"Task: {cleaned_text};\nAction: "
                full_prompts.append(full_prompt)

            transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
            return transition

        def transform_features(self, features):
            return features

    return Pi05PrepareTokenizerStateOutStep


_STATE_OUT_STEP_CLS: Any = None


def state_out_step_cls() -> type:
    """Return (and lazily build) the state-out processor step class."""
    global _STATE_OUT_STEP_CLS
    if _STATE_OUT_STEP_CLS is None:
        _STATE_OUT_STEP_CLS = _register_step()
    return _STATE_OUT_STEP_CLS


def swap_prepare_step_in_pipeline(pipeline: Any, max_state_dim: int = 32) -> Any:
    """Find ``Pi05PrepareStateTokenizerProcessorStep`` in the given
    pipeline's steps and replace it in place with the state-out step.

    ``pipeline`` is a ``PolicyProcessorPipeline`` with a ``steps``
    attribute (list of ProcessorStep instances).
    """
    from lerobot.policies.pi05.processor_pi05 import (
        Pi05PrepareStateTokenizerProcessorStep,
    )

    StateOutStep = state_out_step_cls()
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, Pi05PrepareStateTokenizerProcessorStep):
            pipeline.steps[i] = StateOutStep(
                max_state_dim=max_state_dim,
                task_key=step.task_key,
            )
            return pipeline
    raise RuntimeError(
        "No Pi05PrepareStateTokenizerProcessorStep found in pipeline; "
        "already state-out? or wrong pipeline type?"
    )


def make_pi05_state_out_preprocessor(config: Any, dataset_stats: Any = None) -> Any:
    """Return a pi0.5 preprocessor pipeline that emits lang without
    tokenized state. Wrapper around lerobot's factory + step swap."""
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    preproc, postproc = make_pi05_pre_post_processors(
        config=config, dataset_stats=dataset_stats,
    )
    swap_prepare_step_in_pipeline(preproc, max_state_dim=config.max_state_dim)
    return preproc, postproc


__all__ = [
    "state_out_step_cls",
    "swap_prepare_step_in_pipeline",
    "make_pi05_state_out_preprocessor",
]
