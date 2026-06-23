"""OpenAI-style tool schemas for the Tether chat agent.

Each tool maps to a `tether <subcommand>` invocation or a short deterministic
command chain. Schemas are kept tight: only the parameters the LLM is likely to
fill correctly. Power users use the CLI directly.
"""

from __future__ import annotations

from typing import Any

# Hardware target choices reused across multiple tools.
_TARGETS = ["orin-nano", "orin", "orin-64", "thor", "desktop"]
_PRECISIONS = ["fp16", "fp8", "int8"]


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                **parameters,
            },
        },
    }


_DEVICE_CLASSES = ["orin_nano", "agx_orin", "thor", "h200", "h100", "a100", "a10g", "cpu"]


TOOLS: list[dict[str, Any]] = [
    _tool(
        "deploy_one_command",
        "One-command deploy: probe hardware → pick model variant → pull weights → export to ONNX → start the inference server. The `tether go` workflow. PREFER THIS over manually chaining pull_model + export_model + serve_model when the user just wants 'deploy X'. The export step requires the [monolithic] extras.",
        {
            "properties": {
                "model": {"type": "string", "description": "Registry id (e.g. 'smolvla-base', 'pi05-libero') or family name ('pi05'/'smolvla'/'pi0')"},
                "device_class": {"type": "string", "enum": _DEVICE_CLASSES, "description": "Override hardware probe. Use when probe misclassifies."},
                "embodiment": {"type": "string", "description": "Robot preset (franka/so100/ur5). Optional but recommended — cross-checks dataset/action shapes."},
                "port": {"type": "integer", "description": "HTTP port for /act + /health (default 8000)"},
            },
            "required": ["model"],
        },
    ),
    _tool(
        "export_model",
        "Export a HuggingFace VLA model (pi0, pi0.5, SmolVLA, GR00T) to ONNX for a target hardware tier. Returns the export directory path.",
        {
            "properties": {
                "model": {"type": "string", "description": "HF model ID or local checkpoint path, e.g. 'lerobot/smolvla_base'"},
                "target": {"type": "string", "enum": _TARGETS, "description": "Hardware tier"},
                "output": {"type": "string", "description": "Output directory (default ./tether_export)"},
                "precision": {"type": "string", "enum": _PRECISIONS, "description": "Numeric precision"},
                "decomposed": {"type": "boolean", "description": "Use 5-stage decomposed export instead of monolithic. Default false."},
            },
            "required": ["model", "target"],
        },
    ),
    _tool(
        "serve_model",
        "Start the Tether inference server for a previously exported model. Returns the server URL.",
        {
            "properties": {
                "export_dir": {"type": "string", "description": "Path to the export directory produced by export_model"},
                "port": {"type": "integer", "description": "HTTP port (default 8000)"},
                "host": {"type": "string", "description": "Bind host (default 0.0.0.0)"},
            },
            "required": ["export_dir"],
        },
    ),
    _tool(
        "prove_deployment",
        "Run `tether prove` on a real export to produce a local deployment proof packet. Use this when the user asks whether an export is safe, ready, deployable, production-ready, or suitable for a robot. This does not actuate hardware; it starts a local server, probes health/config/act/metrics, records optional traces, and writes proof artifacts.",
        {
            "properties": {
                "export_dir": {"type": "string", "description": "Path to the export directory to prove"},
                "embodiment": {"type": "string", "description": "Robot preset such as franka, so100, ur5, or custom. Default custom."},
                "profile": {"type": "string", "description": "Optional JSON/YAML deployment profile with pass/fail thresholds"},
                "output_dir": {"type": "string", "description": "Directory for proof artifacts"},
                "record_dir": {"type": "string", "description": "Optional trace directory to pass as --record-dir"},
                "safety_config": {"type": "string", "description": "Optional SafetyLimits JSON path"},
                "policy_diff_baseline": {"type": "string", "description": "Optional baseline trace to include policy-diff evidence in the proof packet."},
                "policy_diff_candidate": {"type": "string", "description": "Optional candidate trace to compare against policy_diff_baseline."},
                "policy_diff_shadow": {"type": "boolean", "description": "Compare policy_diff_baseline response.actions against routing.shadow_actions."},
                "policy_diff_fail_on": {"type": "string", "enum": ["none", "actions", "latency", "guard", "shape", "any"], "description": "Fail the proof packet on selected policy diff failures. Default any."},
                "device": {"type": "string", "enum": ["cpu", "cuda"], "description": "Runtime device. Default cpu for safe proof runs."},
                "samples": {"type": "integer", "description": "Number of /act samples to measure. Default 20."},
                "timeout_s": {"type": "integer", "description": "Timeout in seconds. Default 30."},
                "control_hz": {"type": "number", "description": "Robot control rate for realtime budget evidence, e.g. 20 for a 20 Hz loop."},
                "offline": {"type": "boolean", "description": "Run in offline mode. Default true."},
                "json": {"type": "boolean", "description": "Emit JSON instead of human output."},
            },
            "required": ["export_dir"],
        },
    ),
    _tool(
        "diff_policies",
        "Compare baseline/candidate recorded traces, or a shadow trace, before promotion. Use this when the user asks whether a new policy differs, regressed, or is safe to promote.",
        {
            "properties": {
                "baseline_trace": {"type": "string", "description": "Baseline trace file. In shadow mode this is the shadow trace."},
                "candidate_trace": {"type": "string", "description": "Candidate trace file. Omit when shadow=true."},
                "shadow": {"type": "boolean", "description": "Compare response.actions with routing.shadow_actions in one trace."},
                "min_action_cos": {"type": "number", "description": "Minimum action cosine similarity before failing."},
                "max_action_delta": {"type": "number", "description": "Max absolute action delta before failing."},
                "max_latency_regression_pct": {"type": "number", "description": "Max candidate latency regression fraction, e.g. 0.10 = 10%."},
                "json": {"type": "boolean", "description": "Emit JSON instead of compact human output."},
            },
            "required": ["baseline_trace"],
        },
    ),
    _tool(
        "decide_promotion",
        "Decide PROMOTE, BLOCK, or ROLLBACK from a deployment proof packet. Use this when the user asks whether a proof packet or rollout should be promoted. The profile can be a built-in name like ci-default, lab-shadow, warehouse-safe, or contact-strict.",
        {
            "properties": {
                "packet": {"type": "string", "description": "Deployment proof packet directory, or deployment-proof.json path."},
                "profile": {"type": "string", "description": "Optional built-in promotion profile name or JSON/YAML path."},
                "candidate_active": {"type": "boolean", "description": "Return ROLLBACK instead of BLOCK when gates fail for an active rollout."},
                "json": {"type": "boolean", "description": "Emit JSON instead of compact human output."},
            },
            "required": ["packet"],
        },
    ),
    _tool(
        "assure_release",
        "Build one release assurance report from an existing proof packet. Use this when the user asks whether a robot policy update/release should promote, hold, or roll back and may also care about realtime serving, action chunk continuity, or shadow rollout evidence.",
        {
            "properties": {
                "packet": {"type": "string", "description": "Deployment proof packet directory, or deployment-proof.json path."},
                "profile": {"type": "string", "description": "Optional built-in promotion profile name or JSON/YAML path."},
                "candidate_active": {"type": "boolean", "description": "Return ROLLBACK instead of HOLD when gates fail for an active rollout."},
                "control_hz": {"type": "number", "description": "Robot control rate. Setting this includes realtime evidence."},
                "target": {"type": "string", "description": "Hardware/cell label, e.g. agx-orin-cell-a."},
                "execution_cert": {"type": "boolean", "description": "Also certify stale action windows, chunk-boundary continuity, velocity discontinuity, and runtime attribution."},
                "shadow_trace": {"type": "string", "description": "Optional shadow trace from `tether serve --shadow-policy --record`."},
                "min_compared": {"type": "integer", "description": "Minimum compared shadow requests required before promotion. Default 1."},
                "output_dir": {"type": "string", "description": "Directory for release-assurance artifacts."},
                "json": {"type": "boolean", "description": "Emit JSON instead of human output."},
            },
            "required": ["packet"],
        },
    ),
    _tool(
        "prove_realtime_deployment",
        "Run a deterministic realtime deployment proof chain for an export: `tether prove` into a known proof directory, then `tether bench realtime` against that same packet. Use this when the user gives an export path and asks whether it can run at 20 Hz, 50 Hz, realtime, or inside a robot control-loop budget.",
        {
            "properties": {
                "export_dir": {"type": "string", "description": "Path to the export directory to prove and certify."},
                "control_hz": {"type": "number", "description": "Robot control rate, e.g. 20 for a 20 Hz loop."},
                "target": {"type": "string", "description": "Hardware/cell label to write into the certificate, e.g. agx-orin-cell-a."},
                "profile": {"type": "string", "description": "Optional JSON/YAML deployment profile with pass/fail thresholds."},
                "embodiment": {"type": "string", "description": "Robot preset such as franka, so100, ur5, or custom."},
                "proof_output_dir": {"type": "string", "description": "Proof packet output directory. Default ./tether-deploy-proof."},
                "cert_output_dir": {"type": "string", "description": "Realtime certificate output directory. Default ./tether-realtime-cert."},
                "record_dir": {"type": "string", "description": "Optional trace directory to pass as --record-dir."},
                "safety_config": {"type": "string", "description": "Optional SafetyLimits JSON path."},
                "device": {"type": "string", "enum": ["cpu", "cuda"], "description": "Runtime device. Default cpu for safe proof runs."},
                "samples": {"type": "integer", "description": "Number of /act samples to measure. Default 20."},
                "timeout_s": {"type": "integer", "description": "Timeout in seconds. Default 30."},
                "max_roundtrip_p95_ms": {"type": "number", "description": "Optional p95 roundtrip budget. Omit to use the control period."},
                "max_jitter_p95_minus_p50_ms": {"type": "number", "description": "Optional hard jitter budget."},
                "max_deadline_misses": {"type": "integer", "description": "Allowed deadline misses. Default 0."},
                "max_control_budget_misses": {"type": "integer", "description": "Allowed control-budget misses. Default 0."},
                "max_act_errors": {"type": "integer", "description": "Allowed /act errors. Default 0."},
                "execution_cert": {"type": "boolean", "description": "Also certify action chunk execution evidence: stale window, chunk boundary continuity, velocity discontinuity, and runtime attribution."},
                "max_stale_action_window_ms": {"type": "number", "description": "Execution cert stale-action budget. Default 100 ms."},
                "max_chunk_boundary_delta": {"type": "number", "description": "Execution cert max chunk-boundary action delta. Default 0.15."},
                "max_velocity_discontinuity": {"type": "number", "description": "Execution cert max boundary velocity jump. Default 0.2."},
                "require_phase_aware_horizon": {"type": "boolean", "description": "Require phase/low-speed transition evidence for adaptive action chunking."},
                "require_runtime_attribution": {"type": "boolean", "description": "Require scheduler/cache/adaptive-horizon attribution. Default true."},
                "offline": {"type": "boolean", "description": "Run proof in offline mode. Default true."},
                "json": {"type": "boolean", "description": "Emit realtime certificate JSON instead of human output."},
            },
            "required": ["export_dir", "control_hz"],
        },
    ),
    _tool(
        "certify_realtime_serving",
        "Build a realtime serving certificate from a deployment proof packet. Use this when the user asks whether an existing proof can run at 20 Hz, 50 Hz, realtime, or inside a robot control-loop budget. If the user gives an export instead of a proof, use prove_realtime_deployment instead.",
        {
            "properties": {
                "proof": {"type": "string", "description": "Deployment proof packet directory, or deployment-proof.json path."},
                "target": {"type": "string", "description": "Hardware/cell label to write into the certificate, e.g. agx-orin-cell-a."},
                "control_hz": {"type": "number", "description": "Robot control rate. Omit to use proof/profile evidence."},
                "max_roundtrip_p95_ms": {"type": "number", "description": "Optional p95 roundtrip budget. Omit to use the control period."},
                "max_jitter_p95_minus_p50_ms": {"type": "number", "description": "Optional hard jitter budget."},
                "max_deadline_misses": {"type": "integer", "description": "Allowed deadline misses. Default 0."},
                "max_control_budget_misses": {"type": "integer", "description": "Allowed control-budget misses. Default 0."},
                "max_act_errors": {"type": "integer", "description": "Allowed /act errors. Default 0."},
                "execution_cert": {"type": "boolean", "description": "Also certify action chunk execution evidence: stale window, chunk boundary continuity, velocity discontinuity, and runtime attribution."},
                "max_stale_action_window_ms": {"type": "number", "description": "Execution cert stale-action budget. Default 100 ms."},
                "max_chunk_boundary_delta": {"type": "number", "description": "Execution cert max chunk-boundary action delta. Default 0.15."},
                "max_velocity_discontinuity": {"type": "number", "description": "Execution cert max boundary velocity jump. Default 0.2."},
                "require_phase_aware_horizon": {"type": "boolean", "description": "Require phase/low-speed transition evidence for adaptive action chunking."},
                "require_runtime_attribution": {"type": "boolean", "description": "Require scheduler/cache/adaptive-horizon attribution. Default true."},
                "output_dir": {"type": "string", "description": "Directory for realtime-serving-cert artifacts."},
                "json": {"type": "boolean", "description": "Emit JSON instead of human output."},
            },
            "required": ["proof"],
        },
    ),
    _tool(
        "list_promotion_profiles",
        "List built-in promotion profiles and their required evidence. Use this when the user asks which profile to use.",
        {"properties": {}},
    ),
    _tool(
        "show_promotion_profile",
        "Show exact thresholds for a built-in promotion profile.",
        {
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": ["ci-default", "lab-shadow", "warehouse-safe", "contact-strict"],
                    "description": "Built-in promotion profile name.",
                },
            },
            "required": ["profile"],
        },
    ),
    _tool(
        "benchmark",
        "Measure latency/throughput of an exported model on the local machine.",
        {
            "properties": {
                "export_dir": {"type": "string"},
                "iterations": {"type": "integer", "description": "Number of /act calls (default 100)"},
                "batch_size": {"type": "integer", "description": "Concurrent requests (default 1)"},
            },
            "required": ["export_dir"],
        },
    ),
    _tool(
        "evaluate",
        "Run a LIBERO benchmark suite against an exported model. Slow (minutes-hours).",
        {
            "properties": {
                "export_dir": {"type": "string"},
                "suite": {"type": "string", "enum": ["libero_object", "libero_spatial", "libero_goal", "libero_10"]},
                "num_episodes": {"type": "integer", "description": "Episodes per task (default 50)"},
            },
            "required": ["export_dir", "suite"],
        },
    ),
    _tool(
        "list_models",
        (
            "The canonical registry of every model Tether supports — model_id, "
            "family, params, size_mb, supported devices, supported embodiments, "
            "license. This is the SOURCE OF TRUTH for any question about which "
            "models exist, what hardware they run on, or which fits a constraint. "
            "Always call this before naming a model, family, or hardware-support "
            "claim. Optional filters narrow the result set so you don't have to "
            "chain list_models + many model_info calls."
        ),
        {
            "properties": {
                "family": {
                    "type": "string",
                    "description": "Filter by family: pi0, pi05, smolvla, openvla, groot",
                },
                "device": {
                    "type": "string",
                    "description": "Filter by supported device (orin_nano, agx_orin, thor, a10g, a100, h100, h200)",
                    "enum": _DEVICE_CLASSES,
                },
                "embodiment": {
                    "type": "string",
                    "description": "Filter by supported embodiment (franka, so100, ur5)",
                },
            },
        },
    ),
    _tool(
        "pull_model",
        "Download a model checkpoint from HuggingFace into the local registry.",
        {
            "properties": {
                "model": {"type": "string", "description": "HF model ID"},
            },
            "required": ["model"],
        },
    ),
    _tool(
        "model_info",
        "Show metadata + supported targets for a single model in the registry.",
        {
            "properties": {
                "model": {"type": "string"},
            },
            "required": ["model"],
        },
    ),
    _tool(
        "list_targets",
        "List supported hardware targets and their compute/memory budgets.",
        {"properties": {}},
    ),
    _tool(
        "doctor",
        "Run diagnostic checks (ONNX runtime, CUDA, GPU memory, registry health).",
        {"properties": {}},
    ),
    _tool(
        "distill",
        "Distill a teacher VLA into a faster student via SnapFlow (training-free is target; takes hours).",
        {
            "properties": {
                "teacher": {"type": "string", "description": "HF model ID of the teacher"},
                "student_steps": {"type": "integer", "description": "Target denoise steps for student (1, 2, 4)"},
                "output": {"type": "string", "description": "Output checkpoint path"},
            },
            "required": ["teacher", "student_steps"],
        },
    ),
    _tool(
        "finetune",
        "Fine-tune a VLA on a LeRobot dataset (LoRA by default).",
        {
            "properties": {
                "model": {"type": "string"},
                "dataset": {"type": "string", "description": "HF dataset ID or local path"},
                "output": {"type": "string"},
                "lora": {"type": "boolean", "description": "Use LoRA (default true)"},
            },
            "required": ["model", "dataset"],
        },
    ),
    _tool(
        "list_traces",
        "List recent /act traces from the local trace archive (record-replay).",
        {
            "properties": {
                "since": {"type": "string", "description": "Time window, e.g. '7d', '24h', '1h'"},
                "task": {"type": "string", "description": "Filter by task name"},
                "status": {"type": "string", "enum": ["success", "failed", "any"]},
                "limit": {"type": "integer", "description": "Max rows (default 50)"},
            },
        },
    ),
    _tool(
        "replay_trace",
        "Replay a recorded JSONL trace file against a target model + show action diff.",
        {
            "properties": {
                "trace_file": {"type": "string", "description": "Path to recorded .jsonl or .jsonl.gz trace from `tether serve --record`"},
                "export_dir": {"type": "string", "description": "Path to target export dir (passed as --model to tether replay)"},
            },
            "required": ["trace_file", "export_dir"],
        },
    ),
    _tool(
        "show_status",
        "Show running Tether serve instances + their health.",
        {"properties": {}},
    ),
    _tool(
        "show_config",
        "Show the active Tether configuration (paths, defaults, registry root).",
        {"properties": {}},
    ),
    _tool(
        "show_version",
        "Show the installed Tether version + build info.",
        {"properties": {}},
    ),
]


def by_name() -> dict[str, dict[str, Any]]:
    """Return tools indexed by function name."""
    return {t["function"]["name"]: t for t in TOOLS}
