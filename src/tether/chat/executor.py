"""Translate LLM tool calls into local Tether CLI invocations.

We shell out to `tether <subcommand>` (the same binary the user installed) so the
chat loop never re-implements logic that already lives in the CLI. Subprocess output
goes back to the LLM as the tool result.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from typing import Any

# Map tool name → callable that returns argv (list[str]) for `tether <subcommand>`.
# Each builder validates required args and ignores unknown keys.

OutputCap = 8000  # truncate stdout/stderr so we don't blow LLM context


def _smart_truncate(text: str, cap: int = OutputCap) -> str:
    """Keep the head + tail when truncating long output.

    Compile errors and stack traces put the actionable info at the END
    (the actual exception line, the failing assertion). Head-only
    truncation loses that. We keep 1/3 head + 2/3 tail with a marker.
    """
    if len(text) <= cap:
        return text
    head_len = cap // 3
    tail_len = cap - head_len - 80  # leave room for the marker
    head = text[:head_len]
    tail = text[-tail_len:]
    dropped = len(text) - head_len - tail_len
    return f"{head}\n... [truncated {dropped} chars in the middle] ...\n{tail}"


def _flag(args: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(f"--{key}")
        return
    args.extend([f"--{key}", str(value)])


def _build_go(p: dict[str, Any]) -> list[str]:
    args = ["go", "--model", str(p["model"])]
    _flag(args, "device-class", p.get("device_class"))
    _flag(args, "embodiment", p.get("embodiment"))
    _flag(args, "port", p.get("port"))
    return args


def _build_export(p: dict[str, Any]) -> list[str]:
    args = ["export", str(p["model"]), "--target", str(p["target"])]
    _flag(args, "output", p.get("output"))
    _flag(args, "precision", p.get("precision"))
    if p.get("decomposed") is True:
        args.append("--decomposed")
    return args


def _build_serve(p: dict[str, Any]) -> list[str]:
    args = ["serve", str(p["export_dir"])]
    _flag(args, "port", p.get("port"))
    _flag(args, "host", p.get("host"))
    return args


def _build_prove(p: dict[str, Any]) -> list[str]:
    args = ["prove", str(p["export_dir"])]
    _flag(args, "embodiment", p.get("embodiment"))
    _flag(args, "profile", p.get("profile"))
    _flag(args, "output-dir", p.get("output_dir"))
    _flag(args, "record-dir", p.get("record_dir"))
    _flag(args, "safety-config", p.get("safety_config"))
    _flag(args, "policy-diff-baseline", p.get("policy_diff_baseline"))
    _flag(args, "policy-diff-candidate", p.get("policy_diff_candidate"))
    if p.get("policy_diff_shadow") is True:
        args.append("--policy-diff-shadow")
    _flag(args, "policy-diff-fail-on", p.get("policy_diff_fail_on"))
    _flag(args, "device", p.get("device"))
    _flag(args, "samples", p.get("samples"))
    _flag(args, "timeout-s", p.get("timeout_s"))
    _flag(args, "control-hz", p.get("control_hz"))
    if p.get("offline") is False:
        args.append("--online")
    if p.get("json") is True:
        args.append("--json")
    return args


def _build_policy_diff(p: dict[str, Any]) -> list[str]:
    args = ["policy", "diff", str(p["baseline_trace"])]
    if p.get("candidate_trace") and not p.get("shadow"):
        args.append(str(p["candidate_trace"]))
    if p.get("shadow") is True:
        args.append("--shadow")
    _flag(args, "min-action-cos", p.get("min_action_cos"))
    _flag(args, "max-action-delta", p.get("max_action_delta"))
    _flag(args, "max-latency-regression-pct", p.get("max_latency_regression_pct"))
    if p.get("json") is True:
        args.append("--json")
    return args


def _build_promote(p: dict[str, Any]) -> list[str]:
    args = ["promote", str(p["packet"])]
    _flag(args, "profile", p.get("profile"))
    if p.get("candidate_active") is True:
        args.append("--candidate-active")
    if p.get("json") is True:
        args.append("--json")
    return args


def _build_release_assurance(p: dict[str, Any]) -> list[str]:
    args = ["release", "assure", str(p["packet"])]
    _flag(args, "profile", p.get("profile"))
    if p.get("candidate_active") is True:
        args.append("--candidate-active")
    _flag(args, "control-hz", p.get("control_hz"))
    _flag(args, "target", p.get("target"))
    if p.get("execution_cert") is True:
        args.append("--execution-cert")
    _flag(args, "shadow-trace", p.get("shadow_trace"))
    _flag(args, "min-compared", p.get("min_compared"))
    _flag(args, "output-dir", p.get("output_dir"))
    if p.get("json") is True:
        args.append("--json")
    return args


def _build_realtime_cert(p: dict[str, Any]) -> list[str]:
    args = ["bench", "realtime", str(p["proof"])]
    _flag(args, "target", p.get("target"))
    _flag(args, "control-hz", p.get("control_hz"))
    _flag(args, "max-roundtrip-p95-ms", p.get("max_roundtrip_p95_ms"))
    _flag(args, "max-jitter-p95-minus-p50-ms", p.get("max_jitter_p95_minus_p50_ms"))
    _flag(args, "max-deadline-misses", p.get("max_deadline_misses"))
    _flag(args, "max-control-budget-misses", p.get("max_control_budget_misses"))
    _flag(args, "max-act-errors", p.get("max_act_errors"))
    if p.get("execution_cert") is True:
        args.append("--execution-cert")
    _flag(args, "max-stale-action-window-ms", p.get("max_stale_action_window_ms"))
    _flag(args, "max-chunk-boundary-delta", p.get("max_chunk_boundary_delta"))
    _flag(args, "max-velocity-discontinuity", p.get("max_velocity_discontinuity"))
    if p.get("require_phase_aware_horizon") is True:
        args.append("--require-phase-aware-horizon")
    if p.get("require_runtime_attribution") is False:
        args.append("--no-require-runtime-attribution")
    _flag(args, "output-dir", p.get("output_dir"))
    if p.get("json") is True:
        args.append("--json")
    return args


def _build_prove_realtime_chain(p: dict[str, Any]) -> list[list[str]]:
    proof_dir = str(p.get("proof_output_dir") or "./tether-deploy-proof")
    cert_dir = str(p.get("cert_output_dir") or "./tether-realtime-cert")
    prove_params = {
        "export_dir": p["export_dir"],
        "embodiment": p.get("embodiment"),
        "profile": p.get("profile"),
        "output_dir": proof_dir,
        "record_dir": p.get("record_dir"),
        "safety_config": p.get("safety_config"),
        "device": p.get("device"),
        "samples": p.get("samples"),
        "timeout_s": p.get("timeout_s"),
        "control_hz": p.get("control_hz"),
        "offline": p.get("offline"),
    }
    cert_params = {
        "proof": proof_dir,
        "target": p.get("target"),
        "control_hz": p.get("control_hz"),
        "max_roundtrip_p95_ms": p.get("max_roundtrip_p95_ms"),
        "max_jitter_p95_minus_p50_ms": p.get("max_jitter_p95_minus_p50_ms"),
        "max_deadline_misses": p.get("max_deadline_misses"),
        "max_control_budget_misses": p.get("max_control_budget_misses"),
        "max_act_errors": p.get("max_act_errors"),
        "execution_cert": p.get("execution_cert"),
        "max_stale_action_window_ms": p.get("max_stale_action_window_ms"),
        "max_chunk_boundary_delta": p.get("max_chunk_boundary_delta"),
        "max_velocity_discontinuity": p.get("max_velocity_discontinuity"),
        "require_phase_aware_horizon": p.get("require_phase_aware_horizon"),
        "require_runtime_attribution": p.get("require_runtime_attribution"),
        "output_dir": cert_dir,
        "json": p.get("json"),
    }
    return [_build_prove(prove_params), _build_realtime_cert(cert_params)]


def _build_show_profile(p: dict[str, Any]) -> list[str]:
    return ["profiles", "show", str(p["profile"]), "--json"]


def _build_bench(p: dict[str, Any]) -> list[str]:
    args = ["bench", str(p["export_dir"])]
    _flag(args, "iterations", p.get("iterations"))
    _flag(args, "batch-size", p.get("batch_size"))
    return args


def _build_eval(p: dict[str, Any]) -> list[str]:
    args = ["eval", str(p["export_dir"]), "--suite", str(p["suite"])]
    _flag(args, "num-episodes", p.get("num_episodes"))
    return args


def _build_pull(p: dict[str, Any]) -> list[str]:
    return ["models", "pull", str(p["model"])]


def _build_list_models(p: dict[str, Any]) -> list[str]:
    args = ["models", "list", "--format", "json"]
    _flag(args, "family", p.get("family"))
    _flag(args, "device", p.get("device"))
    _flag(args, "embodiment", p.get("embodiment"))
    return args


def _build_model_info(p: dict[str, Any]) -> list[str]:
    return ["models", "info", str(p["model"])]


def _build_distill(p: dict[str, Any]) -> list[str]:
    args = ["distill", str(p["teacher"]), "--student-steps", str(p["student_steps"])]
    _flag(args, "output", p.get("output"))
    return args


def _build_finetune(p: dict[str, Any]) -> list[str]:
    args = ["finetune", str(p["model"]), str(p["dataset"])]
    _flag(args, "output", p.get("output"))
    if p.get("lora") is False:
        args.append("--no-lora")
    return args


def _build_traces(p: dict[str, Any]) -> list[str]:
    args = ["inspect", "traces"]
    _flag(args, "since", p.get("since"))
    _flag(args, "task", p.get("task"))
    if p.get("status") and p["status"] != "any":
        _flag(args, "status", p["status"])
    _flag(args, "limit", p.get("limit"))
    return args


def _build_replay(p: dict[str, Any]) -> list[str]:
    return ["replay", str(p["trace_file"]), "--model", str(p["export_dir"])]


# Builders that take no args — just static argv.
_STATIC = {
    "list_targets": ["inspect", "targets"],
    "list_promotion_profiles": ["profiles", "list", "--json"],
    "doctor": ["doctor"],
    "show_status": ["status"],
    "show_config": ["config", "show"],
    "show_version": ["--version"],
}

_BUILDERS = {
    "deploy_one_command": _build_go,
    "export_model": _build_export,
    "serve_model": _build_serve,
    "prove_deployment": _build_prove,
    "diff_policies": _build_policy_diff,
    "decide_promotion": _build_promote,
    "assure_release": _build_release_assurance,
    "certify_realtime_serving": _build_realtime_cert,
    "show_promotion_profile": _build_show_profile,
    "benchmark": _build_bench,
    "evaluate": _build_eval,
    "list_models": _build_list_models,
    "pull_model": _build_pull,
    "model_info": _build_model_info,
    "distill": _build_distill,
    "finetune": _build_finetune,
    "list_traces": _build_traces,
    "replay_trace": _build_replay,
}

_CHAIN_BUILDERS = {
    "prove_realtime_deployment": _build_prove_realtime_chain,
}


def _argv_for(name: str, params: dict[str, Any]) -> list[str]:
    if name in _CHAIN_BUILDERS:
        raise ValueError(f"tool {name!r} expands to multiple commands; use _argvs_for")
    if name in _STATIC:
        return list(_STATIC[name])
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"unknown tool: {name}")
    return builder(params)


def _argvs_for(name: str, params: dict[str, Any]) -> list[list[str]]:
    chain_builder = _CHAIN_BUILDERS.get(name)
    if chain_builder is not None:
        return chain_builder(params)
    return [_argv_for(name, params)]


def execute(name: str, params: dict[str, Any], tether_bin: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Run a tool. Returns dict with stdout, stderr, exit_code, command, dry_run."""
    binary = tether_bin or shutil.which("tether") or "tether"
    argv_list = [[binary] + argv for argv in _argvs_for(name, params)]
    cmd_strs = [" ".join(shlex.quote(a) for a in argv) for argv in argv_list]
    cmd_str = " && ".join(cmd_strs)

    if dry_run:
        return {
            "command": cmd_str,
            "dry_run": True,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "steps": [{"command": cmd, "exit_code": 0} for cmd in cmd_strs],
        }

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    steps: list[dict[str, Any]] = []
    exit_code = 0
    multi_step = len(argv_list) > 1

    for idx, argv in enumerate(argv_list, start=1):
        step_cmd = cmd_strs[idx - 1]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            steps.append({"command": step_cmd, "exit_code": 124})
            return {
                "command": cmd_str,
                "stdout": _smart_truncate("\n".join(stdout_parts)),
                "stderr": _smart_truncate(
                    "\n".join([*stderr_parts, f"step {idx} timeout after 600s"])
                ),
                "exit_code": 124,
                "steps": steps,
            }
        except FileNotFoundError as e:
            steps.append({"command": step_cmd, "exit_code": 127})
            return {
                "command": cmd_str,
                "stdout": _smart_truncate("\n".join(stdout_parts)),
                "stderr": _smart_truncate("\n".join([*stderr_parts, str(e)])),
                "exit_code": 127,
                "steps": steps,
            }

        steps.append({"command": step_cmd, "exit_code": proc.returncode})
        if proc.stdout:
            stdout_parts.append(
                f"--- step {idx}: {step_cmd} ---\n{proc.stdout}" if multi_step else proc.stdout
            )
        if proc.stderr:
            stderr_parts.append(
                f"--- step {idx}: {step_cmd} ---\n{proc.stderr}" if multi_step else proc.stderr
            )
        if proc.returncode != 0:
            exit_code = proc.returncode
            break

    return {
        "command": cmd_str,
        "stdout": _smart_truncate("\n".join(stdout_parts)),
        "stderr": _smart_truncate("\n".join(stderr_parts)),
        "exit_code": exit_code,
        "steps": steps,
    }


def format_tool_result(name: str, result: dict[str, Any]) -> str:
    """Compact tool result for LLM consumption."""
    parts = [f"$ {result['command']}"]
    if result.get("dry_run"):
        parts.append("(dry-run, not executed)")
        return "\n".join(parts)
    parts.append(f"exit_code={result['exit_code']}")
    if result["stdout"]:
        parts.append(f"--- stdout ---\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"--- stderr ---\n{result['stderr']}")
    return "\n".join(parts)
