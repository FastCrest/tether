"""Regression tests for `tether chat` hallucination behavior.

Diagnoses the failure mode documented in 03_experiments/2026-05-02-chat-hallucination-spike.md:
the agent fabricated model names, sizes, and latencies on research-shaped queries
because (a) `list_models` description suggested it was a local-disk lister rather
than the canonical registry, (b) the SYSTEM_PROMPT had no proactive grounding rule,
and (c) reasoning_effort=minimal kept the agent from making enough tool calls.

These tests pin the structural fixes:
- list_models is described as the registry source-of-truth and accepts filters
- list_models executor passes through --family / --device / --embodiment + --format json
- SYSTEM_PROMPT names a grounding rule before any factual claim

The full N=135 hallucination spike that produced the recommendation is preserved as
~/_spike_chat_hallucination.py and not run as part of this suite (real OpenAI calls,
~$1/run). Run that spike manually if you bump the model or rewrite the prompt.
"""

from __future__ import annotations

from tether.chat.executor import _BUILDERS, _STATIC, _argv_for
from tether.chat.loop import SYSTEM_PROMPT
from tether.chat.schema import by_name


def test_list_models_described_as_registry_source_of_truth() -> None:
    desc = by_name()["list_models"]["function"]["description"].lower()
    assert "canonical registry" in desc
    assert "source of truth" in desc
    assert "available locally" not in desc, (
        "list_models description must not say 'available locally' — it caused "
        "the agent to skip the tool on registry questions and fabricate answers"
    )


def test_list_models_accepts_registry_filters() -> None:
    props = by_name()["list_models"]["function"]["parameters"]["properties"]
    assert {"family", "device", "embodiment"}.issubset(props.keys())


def test_list_models_executor_emits_json_and_filters() -> None:
    assert "list_models" in _BUILDERS
    assert "list_models" not in _STATIC

    default = _argv_for("list_models", {})
    assert default == ["models", "list", "--format", "json"]

    filtered = _argv_for(
        "list_models", {"family": "pi05", "device": "orin_nano", "embodiment": "franka"}
    )
    assert filtered == [
        "models", "list", "--format", "json",
        "--family", "pi05", "--device", "orin_nano", "--embodiment", "franka",
    ]


def test_system_prompt_has_registry_grounding_rule() -> None:
    p = SYSTEM_PROMPT
    assert "registry grounding" in p.lower(), (
        "SYSTEM_PROMPT must include the proactive registry-grounding rule. "
        "Without it the agent at minimal effort fabricates model names + latencies."
    )
    assert "list_models" in p and "model_info" in p, (
        "Grounding rule must name the tools to call before factual claims"
    )
    assert "I don't have that data in the registry" in p, (
        "Grounding rule must give the agent an explicit graceful-unknown phrase"
    )
