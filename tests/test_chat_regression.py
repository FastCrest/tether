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

from tether.chat.executor import _BUILDERS, _STATIC, _argv_for, _argvs_for, execute
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
    assert "control_hz" in props

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
            "control_hz": 20,
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
        "--control-hz", "20",
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


def test_assure_release_tool_routes_to_release_assure() -> None:
    tool = by_name()["assure_release"]
    props = tool["function"]["parameters"]["properties"]
    assert "packet" in props
    assert "control_hz" in props
    assert "shadow_trace" in props
    assert "execution_cert" in props

    argv = _argv_for(
        "assure_release",
        {
            "packet": "./proof",
            "profile": "warehouse-safe",
            "candidate_active": True,
            "control_hz": 20,
            "target": "agx-orin-cell-a",
            "execution_cert": True,
            "shadow_trace": "./shadow.jsonl.gz",
            "min_compared": 100,
            "output_dir": "./release",
            "json": True,
        },
    )

    assert argv == [
        "release", "assure", "./proof",
        "--profile", "warehouse-safe",
        "--candidate-active",
        "--control-hz", "20",
        "--target", "agx-orin-cell-a",
        "--execution-cert",
        "--shadow-trace", "./shadow.jsonl.gz",
        "--min-compared", "100",
        "--output-dir", "./release",
        "--json",
    ]


def test_certify_realtime_serving_routes_to_bench_realtime() -> None:
    tool = by_name()["certify_realtime_serving"]
    props = tool["function"]["parameters"]["properties"]
    assert "proof" in props
    assert "control_hz" in props
    assert "max_roundtrip_p95_ms" in props

    argv = _argv_for(
        "certify_realtime_serving",
        {
            "proof": "./proof",
            "target": "agx-orin-cell-a",
            "control_hz": 20,
            "max_roundtrip_p95_ms": 45,
            "max_jitter_p95_minus_p50_ms": 8,
            "max_deadline_misses": 0,
            "max_control_budget_misses": 0,
            "max_act_errors": 0,
            "output_dir": "./cert",
            "json": True,
        },
    )

    assert argv == [
        "bench", "realtime", "./proof",
        "--target", "agx-orin-cell-a",
        "--control-hz", "20",
        "--max-roundtrip-p95-ms", "45",
        "--max-jitter-p95-minus-p50-ms", "8",
        "--max-deadline-misses", "0",
        "--max-control-budget-misses", "0",
        "--max-act-errors", "0",
        "--output-dir", "./cert",
        "--json",
    ]


def test_certify_realtime_serving_routes_execution_cert_flags() -> None:
    props = by_name()["certify_realtime_serving"]["function"]["parameters"]["properties"]
    assert "execution_cert" in props
    assert "max_stale_action_window_ms" in props

    argv = _argv_for(
        "certify_realtime_serving",
        {
            "proof": "./proof",
            "control_hz": 20,
            "execution_cert": True,
            "max_stale_action_window_ms": 80,
            "max_chunk_boundary_delta": 0.1,
            "max_velocity_discontinuity": 0.15,
            "require_phase_aware_horizon": True,
            "require_runtime_attribution": False,
        },
    )

    assert argv == [
        "bench", "realtime", "./proof",
        "--control-hz", "20",
        "--execution-cert",
        "--max-stale-action-window-ms", "80",
        "--max-chunk-boundary-delta", "0.1",
        "--max-velocity-discontinuity", "0.15",
        "--require-phase-aware-horizon",
        "--no-require-runtime-attribution",
    ]


def test_prove_realtime_deployment_routes_to_deterministic_chain() -> None:
    tool = by_name()["prove_realtime_deployment"]
    props = tool["function"]["parameters"]["properties"]
    assert "export_dir" in props
    assert "control_hz" in props
    assert "proof_output_dir" in props
    assert "cert_output_dir" in props

    argvs = _argvs_for(
        "prove_realtime_deployment",
        {
            "export_dir": "./export",
            "control_hz": 20,
            "target": "agx-orin-cell-a",
            "profile": "warehouse-safe",
            "proof_output_dir": "./proof",
            "cert_output_dir": "./cert",
            "samples": 100,
            "max_roundtrip_p95_ms": 45,
            "json": True,
        },
    )

    assert argvs == [
        [
            "prove", "./export",
            "--profile", "warehouse-safe",
            "--output-dir", "./proof",
            "--samples", "100",
            "--control-hz", "20",
        ],
        [
            "bench", "realtime", "./proof",
            "--target", "agx-orin-cell-a",
            "--control-hz", "20",
            "--max-roundtrip-p95-ms", "45",
            "--output-dir", "./cert",
            "--json",
        ],
    ]


def test_prove_realtime_deployment_defaults_artifact_dirs() -> None:
    argvs = _argvs_for(
        "prove_realtime_deployment",
        {"export_dir": "./export", "control_hz": 50},
    )

    assert "--output-dir" in argvs[0]
    assert argvs[0][argvs[0].index("--output-dir") + 1] == "./tether-deploy-proof"
    assert argvs[1][2] == "./tether-deploy-proof"
    assert "--output-dir" in argvs[1]
    assert argvs[1][argvs[1].index("--output-dir") + 1] == "./tether-realtime-cert"


def test_prove_realtime_deployment_execute_dry_run_shows_chain() -> None:
    result = execute(
        "prove_realtime_deployment",
        {
            "export_dir": "./export",
            "control_hz": 20,
            "target": "agx-orin",
            "proof_output_dir": "./proof",
            "cert_output_dir": "./cert",
            "json": True,
        },
        tether_bin="tether",
        dry_run=True,
    )

    assert result["exit_code"] == 0
    assert result["steps"] == [
        {
            "command": "tether prove ./export --output-dir ./proof --control-hz 20",
            "exit_code": 0,
        },
        {
            "command": "tether bench realtime ./proof --target agx-orin --control-hz 20 --output-dir ./cert --json",
            "exit_code": 0,
        },
    ]
    assert " && " in result["command"]


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
    assert "control_hz" in p
    assert "does not actuate hardware" in p


def test_system_prompt_prefers_realtime_cert_for_control_budget() -> None:
    p = SYSTEM_PROMPT
    assert "certify_realtime_serving" in p
    assert "prove_realtime_deployment" in p
    assert "assure_release" in p
    assert "20 Hz" in p
    assert "control-loop budget" in p
    assert "Use assure_release when the user gives an existing proof packet" in p
    assert "Use certify_realtime_serving only when the user asks specifically" in p


def test_system_prompt_prefers_policy_diff_for_rollout_questions() -> None:
    p = SYSTEM_PROMPT
    assert "diff_policies" in p
    assert "decide_promotion" in p
    assert "list_promotion_profiles" in p
    assert "promote, hold, or roll back" in p
    assert "safe to promote" in p
