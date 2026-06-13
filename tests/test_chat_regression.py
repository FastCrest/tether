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


def test_prove_deployment_tool_routes_to_friendly_alias() -> None:
    tool = by_name()["prove_deployment"]
    props = tool["function"]["parameters"]["properties"]
    assert "export_dir" in props
    assert "embodiment" in props

    argv = _argv_for(
        "prove_deployment",
        {
            "export_dir": "./export",
            "embodiment": "franka",
            "record_dir": "./traces",
            "policy_diff_baseline": "./traces/current.jsonl.gz",
            "policy_diff_candidate": "./traces/candidate.jsonl.gz",
            "policy_diff_fail_on": "any",
            "samples": 5,
            "json": True,
        },
    )
    assert argv == [
        "prove", "./export",
        "--embodiment", "franka",
        "--record-dir", "./traces",
        "--policy-diff-baseline", "./traces/current.jsonl.gz",
        "--policy-diff-candidate", "./traces/candidate.jsonl.gz",
        "--policy-diff-fail-on", "any",
        "--samples", "5",
        "--json",
    ]


def test_diff_policies_tool_routes_to_policy_diff() -> None:
    tool = by_name()["diff_policies"]
    props = tool["function"]["parameters"]["properties"]
    assert "baseline_trace" in props
    assert "candidate_trace" in props
    assert "shadow" in props

    argv = _argv_for(
        "diff_policies",
        {
            "baseline_trace": "./base.jsonl.gz",
            "candidate_trace": "./cand.jsonl.gz",
            "max_action_delta": 0.05,
            "json": True,
        },
    )
    assert argv == [
        "policy", "diff", "./base.jsonl.gz", "./cand.jsonl.gz",
        "--max-action-delta", "0.05",
        "--json",
    ]

    shadow_argv = _argv_for(
        "diff_policies",
        {
            "baseline_trace": "./shadow.jsonl.gz",
            "candidate_trace": "./ignored.jsonl.gz",
            "shadow": True,
        },
    )
    assert shadow_argv == ["policy", "diff", "./shadow.jsonl.gz", "--shadow"]


def test_decide_promotion_tool_routes_to_promote() -> None:
    tool = by_name()["decide_promotion"]
    props = tool["function"]["parameters"]["properties"]
    assert "packet" in props
    assert "candidate_active" in props

    argv = _argv_for(
        "decide_promotion",
        {
            "packet": "./proof",
            "profile": "./warehouse-safe.yml",
            "candidate_active": True,
            "json": True,
        },
    )

    assert argv == [
        "promote", "./proof",
        "--profile", "./warehouse-safe.yml",
        "--candidate-active",
        "--json",
    ]


def test_profile_tools_route_to_profiles_commands() -> None:
    assert "list_promotion_profiles" in _STATIC
    assert _argv_for("list_promotion_profiles", {}) == ["profiles", "list", "--json"]

    tool = by_name()["show_promotion_profile"]
    props = tool["function"]["parameters"]["properties"]
    assert "warehouse-safe" in props["profile"]["enum"]

    assert _argv_for(
        "show_promotion_profile",
        {"profile": "warehouse-safe"},
    ) == ["profiles", "show", "warehouse-safe", "--json"]


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


def test_system_prompt_prefers_prove_for_deployment_readiness() -> None:
    p = SYSTEM_PROMPT
    assert "prove_deployment" in p
    assert "safe, ready, deployable, production-ready" in p
    assert "policy_diff_*" in p
    assert "does not actuate hardware" in p


def test_system_prompt_prefers_policy_diff_for_rollout_questions() -> None:
    p = SYSTEM_PROMPT
    assert "diff_policies" in p
    assert "decide_promotion" in p
    assert "list_promotion_profiles" in p
    assert "promote, block, or roll back" in p
    assert "safe to promote" in p
