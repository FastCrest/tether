"""FastMCP server factory bound to a live ReflexServer.

Exposes 6 tools + 1 resource to MCP-compatible agents (Phase 1 + Phase 1.5):

Phase 1 (consumer-side):
- tool: `act(instruction, image_b64, state, episode_id?)` → action chunk +
  policy_version + inference_ms
- tool: `health()` → {state, model_version, uptime_seconds, cuda_graphs_active}
- tool: `models_list()` → [{id, hf_id, size_gb_fp16, hardware_fit}, ...]
- tool: `validate_dataset(dataset_path)` → {summary, checks: [...]}
- resource: `metrics://prometheus` → current Prometheus exposition text

Phase 1.5 (producer-side, agents-can-plan-without-executing):
- tool: `bench_latency(export_dir, iterations, warmup)` → p50/p95/p99 stats.
  Synchronous; defaults sized to fit MCP timeout budgets (~30s).
- tool: `export_estimate(model_id, target, precision)` → projected VRAM,
  inference latency, export time. Best-effort projection from the registry +
  HARDWARE_PROFILES table; lets agents plan exports before committing
  to long-running compute. Async export_start + export_status are Phase 2.

Usage:

    from reflex.mcp import create_mcp_server

    mcp = create_mcp_server(reflex_server=my_reflex_server)
    mcp.run(transport="stdio")  # or transport="streamable-http", host=..., port=...

Per the verb-noun CLI ADR (2026-04-24), MCP is additive to the existing REST
API — both transports share the same `ReflexServer` inference engine; the MCP
tools forward to the same methods the `/act` / `/healthz` / `/metrics` HTTP
handlers call.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from reflex.runtime.server import ReflexServer

logger = logging.getLogger(__name__)


_MCP_INSTRUCTIONS = """Reflex is a Vision-Language-Action (VLA) policy server for
robotics. This MCP surface exposes Reflex's per-chunk action prediction as agent-
callable tools. Policies are pre-trained (pi0 / pi0.5 / SmolVLA) and served via
ONNX Runtime. Action chunks are semi-fixed-shape (not token-by-token); callers
provide (instruction, image, state) and receive a chunk of actions to actuate at
the robot's control rate.

Available tools:
- act: predict one action chunk from the current observation
- health: server state (ready / warming / degraded / etc.)
- models_list: available pre-built models with hardware fit
- validate_dataset: pre-flight check a LeRobot v3.0 training dataset

Available resources:
- metrics://prometheus: current Prometheus metrics in text exposition format

Safety note: tool `act` returns actions but does NOT actuate them. The caller is
responsible for sending the action chunk to the robot's actuation controller
(SO-ARM / Trossen / ROS2 via the ros2-mcp-bridge feature, also planned).
"""


def create_mcp_server(
    reflex_server: "ReflexServer",
    *,
    name: str = "reflex",
) -> "FastMCP":
    """Build an MCP server bound to a live ReflexServer.

    Args:
        reflex_server: running `ReflexServer` (Pi05DecomposedInference-backed,
            Pi0OnnxServer, or SmolVLAOnnxServer). All tools forward to its
            public methods.
        name: MCP server name; defaults to "reflex" — appears in catalog
            listings.

    Returns:
        FastMCP instance ready to run:
        - `mcp.run(transport="stdio")` for Claude Desktop / Cursor integration
        - `mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)`
          for HTTP-based MCP clients

    Raises:
        ImportError: if `fastmcp` is not installed. Install via
            `pip install reflex-vla[mcp]` or `pip install fastmcp`.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "fastmcp not installed. Install via `pip install reflex-vla[mcp]` "
            "or `pip install fastmcp`."
        ) from exc

    mcp = FastMCP(name, instructions=_MCP_INSTRUCTIONS)
    _startup_ts = time.time()

    @mcp.tool()
    async def act(
        instruction: str,
        image_b64: str,
        state: list[float],
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Predict one action chunk for the current observation.

        Args:
            instruction: natural-language task description (e.g. "pick up the red block").
            image_b64: base64-encoded RGB image from the robot's primary camera.
            state: current proprioceptive state vector (joint positions + gripper).
                Dimensionality must match the loaded model's expected action_dim.
            episode_id: optional client-provided episode id for RTC continuity
                + record-replay tagging. Passing a new episode_id across requests
                triggers RTC's boundary reset; reusing the same id across requests
                within an episode lets RTC carry over chunk guidance.

        Returns:
            On success: {actions: [[float]], policy_version: str, inference_ms: float}.
                `actions` is a list of action vectors (chunk of e.g. 50 × action_dim).
                The caller actuates these sequentially at the robot's control rate.
            On failure: {error: {kind: str, message: str, remediation: str}}.
        """
        try:
            t0 = time.perf_counter()
            result = await reflex_server.predict_from_base64_async(
                image_b64=image_b64,
                instruction=instruction,
                state=state,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            logger.error("mcp.act error: %s: %s", type(exc).__name__, exc)
            return {
                "error": {
                    "kind": type(exc).__name__,
                    "message": str(exc),
                    "remediation": (
                        "Inspect server logs for the full traceback. Verify "
                        "(image_b64, instruction, state) match the loaded model's "
                        "expected shapes via the health tool or `reflex doctor`."
                    ),
                }
            }

        if isinstance(result, dict) and "error" in result:
            return {"error": {"kind": "DecodeError", "message": result["error"],
                              "remediation": "Check image_b64 is a valid base64-encoded "
                                             "RGB image (PNG or JPEG)."}}

        return {
            "actions": result.get("actions") if isinstance(result, dict) else None,
            "policy_version": getattr(reflex_server, "export_dir", "unknown"),
            "inference_ms": round(elapsed_ms, 2),
        }

    @mcp.tool()
    async def health() -> dict[str, Any]:
        """Server health and readiness.

        Returns:
            {state: str, model_version: str, uptime_seconds: float,
             cuda_graphs_active: bool | None}

            `state` is one of:
            - "initializing" — process just started
            - "loading" — model weights being loaded into memory
            - "warming" — first forward pass running (also triggers cuda-graph
              capture if --cuda-graphs was set)
            - "ready" — accepting /act requests
            - "warmup_failed" — init failed; server won't accept requests
            - "degraded" — consecutive crashes exceeded circuit-breaker threshold
        """
        state = getattr(reflex_server, "health_state", "initializing")
        cuda_graphs = getattr(reflex_server, "_cuda_graphs_enabled", None)
        return {
            "state": state,
            "model_version": str(getattr(reflex_server, "export_dir", "unknown")),
            "uptime_seconds": round(time.time() - _startup_ts, 2),
            "cuda_graphs_active": cuda_graphs,
        }

    @mcp.tool()
    async def models_list() -> list[dict[str, Any]]:
        """List pre-built models in the Reflex registry.

        Returns:
            List of model entries with id, hf_id, family, device fit, and
            published benchmarks. Curated set; each entry verified against
            Reflex parity tests.
        """
        try:
            from reflex.registry import filter_models
            entries = filter_models()
        except ImportError:
            return [{"error": {"kind": "ImportError",
                               "message": "reflex.registry unavailable",
                               "remediation": "reinstall reflex with the default extras."}}]

        return [
            {
                "model_id": e.model_id,
                "hf_repo": e.hf_repo,
                "family": e.family,
                "action_dim": e.action_dim,
                "size_mb": e.size_mb,
                "supported_embodiments": list(e.supported_embodiments),
                "supported_devices": list(e.supported_devices),
                "license": e.license,
                "description": e.description,
            }
            for e in entries
        ]

    @mcp.tool()
    async def validate_dataset(dataset_path: str) -> dict[str, Any]:
        """Pre-flight check a LeRobot v3.0 training dataset.

        Runs the 8 falsifiable checks from `reflex validate-dataset`:
        schema completeness, shape consistency, action-finite, embodiment match,
        episode count, timing monotonicity, etc.

        Args:
            dataset_path: filesystem path to the LeRobot dataset root.

        Returns:
            {schema_version, summary: {pass, fail, warn, skip}, decision, checks: [...]}
            Decision is one of "proceed" | "warn" | "block".
        """
        try:
            from reflex.validation import (
                DatasetContext,
                format_json,
                overall_decision,
                run_all_checks,
            )
            from pathlib import Path
            import json
        except ImportError:
            return {"error": {"kind": "ImportError",
                              "message": "reflex.validation unavailable",
                              "remediation": "reinstall reflex with the default extras."}}

        try:
            root = Path(dataset_path)
            if not root.exists():
                return {"error": {
                    "kind": "FileNotFoundError",
                    "message": f"Dataset path does not exist: {dataset_path}",
                    "remediation": "Check that dataset_path exists and is a LeRobot v3.0 dataset.",
                }}
            ctx = DatasetContext(root=root)
            results = run_all_checks(ctx)
            decision = overall_decision(results)
            payload_text = format_json(results)
            return json.loads(payload_text) | {
                "decision": decision.value if hasattr(decision, "value") else str(decision),
            }
        except Exception as exc:
            logger.error("mcp.validate_dataset error: %s: %s", type(exc).__name__, exc)
            return {"error": {"kind": type(exc).__name__,
                              "message": str(exc),
                              "remediation": "Run `reflex validate-dataset <path>` from the CLI for full diagnostics."}}

    @mcp.tool()
    async def bench_latency(
        export_dir: str,
        iterations: int = 20,
        warmup: int = 5,
    ) -> dict[str, Any]:
        """Quick latency probe for an exported model.

        Runs a synchronous warmup + iterations sweep through inference + reports
        latency stats. Designed to fit MCP tool-call timeout budgets (default
        20 iters + 5 warmup → typically <30s on GPU; up to ~2 min on CPU).

        Use cases:
        - Agent-driven hardware fit checks ("is this model fast enough on this box?")
        - Pre-deploy sanity ("did the latest export regress?")
        - Cross-checkpoint comparison without committing to a full bench run

        Args:
            export_dir: filesystem path to a Reflex export directory (the
                directory containing the .onnx file + reflex_export_meta.json).
            iterations: number of measured iterations (default 20).
                Cap at 100 to keep tool-call latency reasonable.
            warmup: warmup iterations excluded from stats (default 5).

        Returns:
            On success: {iterations, warmup_iterations, mean_ms, median_ms,
                p50_ms, p95_ms, p99_ms, min_ms, max_ms, std_ms, hz, device, export_dir}.
            On failure: {error: {kind, message, remediation}}.
        """
        if iterations < 1 or iterations > 100:
            return {"error": {
                "kind": "ValueError",
                "message": f"iterations must be in [1, 100], got {iterations}",
                "remediation": "Use a smaller iteration count for MCP-callable bench. "
                               "For full benchmarks, run `reflex bench` from the CLI.",
            }}
        if warmup < 0 or warmup > 50:
            return {"error": {
                "kind": "ValueError",
                "message": f"warmup must be in [0, 50], got {warmup}",
                "remediation": "Use 5-10 warmup iterations for typical use.",
            }}
        try:
            import time as _t
            from pathlib import Path

            from reflex.bench.methodology import compute_stats
            from reflex.runtime.server import ReflexServer
        except ImportError as exc:
            return {"error": {
                "kind": "ImportError",
                "message": f"bench helpers unavailable: {exc}",
                "remediation": "Reinstall reflex-vla with default extras (serve + bench substrate).",
            }}

        path = Path(export_dir).expanduser()
        if not path.exists():
            return {"error": {
                "kind": "FileNotFoundError",
                "message": f"export_dir does not exist: {export_dir}",
                "remediation": "Run `reflex export <model>` first or check the path.",
            }}
        if not list(path.glob("*.onnx")):
            return {"error": {
                "kind": "ValueError",
                "message": f"no ONNX file in {export_dir}",
                "remediation": "Make sure the directory points at a Reflex export output, "
                               "not the source HF model dir.",
            }}

        try:
            bench_server = ReflexServer(str(path), device="cuda", strict_providers=False)
            bench_server.load()
            if not bench_server.ready:
                return {"error": {
                    "kind": "ServerNotReady",
                    "message": "ReflexServer.load() returned without ready state",
                    "remediation": "Check the export's reflex_export_meta.json + run "
                                   "`reflex doctor --model <export_dir>` from the CLI.",
                }}
            for _ in range(int(warmup)):
                bench_server.predict()
            latencies: list[float] = []
            for _ in range(int(iterations)):
                t0 = _t.perf_counter()
                bench_server.predict()
                latencies.append((_t.perf_counter() - t0) * 1000)
            stats = compute_stats(latencies, warmup_n=0)
        except Exception as exc:  # noqa: BLE001
            logger.error("mcp.bench_latency error: %s: %s", type(exc).__name__, exc)
            return {"error": {
                "kind": type(exc).__name__,
                "message": str(exc),
                "remediation": "Run `reflex bench <export_dir>` from the CLI for "
                               "full diagnostics.",
            }}

        return {
            "iterations": int(iterations),
            "warmup_iterations": int(warmup),
            "export_dir": str(path),
            "inference_mode": getattr(bench_server, "_inference_mode", "unknown"),
            **stats.to_dict(),
        }

    @mcp.tool()
    async def export_estimate(
        model_id: str,
        target: str = "desktop",
        precision: str = "fp16",
    ) -> dict[str, Any]:
        """Estimate the resource footprint + latency of exporting a model
        without running the export.

        Lets agents plan exports before committing to long-running compute:
        will this model fit on the target hardware? Roughly how long will
        export take? What latency should we expect at inference time?

        Args:
            model_id: HuggingFace id (e.g. "lerobot/smolvla_base") or a
                Reflex-registry id (e.g. "smolvla").
            target: hardware profile slug. One of: desktop / orin /
                orin-nano / orin-64 / thor (defaults to desktop).
            precision: target precision. One of: fp32 / fp16 / fp8 / int8.

        Returns:
            {model_id, target, precision, fits_on_target, estimated_vram_gb,
             estimated_export_time_minutes, estimated_inference_ms_p50,
             notes: list[str]}.
            Estimates are best-effort projections from the registry +
            HARDWARE_PROFILES table; not a guarantee. Run a real export
            via `reflex export` for ground truth.
        """
        try:
            from reflex.config import HARDWARE_PROFILES, get_hardware_profile
            from reflex.registry import by_id
        except ImportError as exc:
            return {"error": {
                "kind": "ImportError",
                "message": f"registry unavailable: {exc}",
                "remediation": "Reinstall reflex-vla with default extras.",
            }}

        if target not in HARDWARE_PROFILES:
            return {"error": {
                "kind": "ValueError",
                "message": f"unknown target {target!r}",
                "remediation": f"Pick from: {sorted(HARDWARE_PROFILES.keys())}",
            }}
        if precision not in ("fp32", "fp16", "fp8", "int8"):
            return {"error": {
                "kind": "ValueError",
                "message": f"unsupported precision {precision!r}",
                "remediation": "Use one of: fp32, fp16, fp8, int8.",
            }}

        # Best-effort registry lookup. Falls back to a generic estimate when
        # the model isn't in the curated registry. by_id() returns ModelEntry
        # or None.
        entry = None
        try:
            entry = by_id(model_id)
        except Exception:  # noqa: BLE001
            entry = None

        hw = get_hardware_profile(target)
        notes: list[str] = []

        # VRAM estimate: registry entry's size_gb_fp16 (when present) scaled
        # by precision. Agents that ask for unknown models get a generic
        # "likely 4-8 GB at fp16" range.
        precision_scale = {"fp32": 2.0, "fp16": 1.0, "fp8": 0.5, "int8": 0.5}[precision]
        entry_size_gb = getattr(entry, "size_gb_fp16", None) if entry is not None else None
        if entry_size_gb:
            estimated_vram_gb = round(float(entry_size_gb) * precision_scale, 2)
            notes.append(f"VRAM estimate from registry size_gb_fp16={entry_size_gb}")
        else:
            estimated_vram_gb = round(4.0 * precision_scale, 2)
            notes.append(
                f"model_id {model_id!r} not in registry; using generic VRAM "
                f"estimate. Add it to reflex.registry for a tighter projection."
            )

        # Hardware fit: per HARDWARE_PROFILES.vram_gb (rough; assumes other
        # processes leave ~1 GB headroom).
        target_vram = float(getattr(hw, "vram_gb", 0)) or 0.0
        headroom_gb = 1.0
        fits = (target_vram - headroom_gb) >= estimated_vram_gb if target_vram > 0 else None
        if fits is False:
            notes.append(
                f"VRAM tight: model needs ~{estimated_vram_gb} GB; "
                f"target {target} has ~{target_vram} GB. Try fp8 or int8."
            )

        # Export-time estimate: rule-of-thumb scaled by model size.
        # Empirical ranges from prior reflex export experiments
        # (decomposed pi05 ~12 min on A100; smolvla ~3 min).
        export_minutes = max(2.0, estimated_vram_gb * 1.5)

        # Inference latency estimate: depends on hardware + precision.
        # Pull from registry's benchmarks when available (ModelEntry.benchmarks
        # is a list[ModelBenchmark]; we look for a matching device + precision).
        latency_ms_p50: float | None = None
        entry_benchmarks = getattr(entry, "benchmarks", None) if entry is not None else None
        if entry_benchmarks:
            for b in entry_benchmarks:
                if (getattr(b, "device", "") == target
                        and getattr(b, "precision", "") == precision):
                    latency_ms_p50 = float(getattr(b, "p50_ms", 0)) or None
                    if latency_ms_p50 is not None:
                        notes.append(
                            f"latency from registry benchmark device={target} "
                            f"precision={precision}"
                        )
                        break
        if latency_ms_p50 is None:
            # Fall back: scale by VRAM × precision (very rough).
            latency_ms_p50 = round(estimated_vram_gb * 30.0 / precision_scale, 1)
            notes.append(
                "no registry benchmark for this hardware/precision combo; "
                "latency is a rough estimate. Run `reflex bench` for ground truth."
            )

        return {
            "model_id": model_id,
            "target": target,
            "precision": precision,
            "fits_on_target": fits,
            "estimated_vram_gb": estimated_vram_gb,
            "estimated_export_time_minutes": round(export_minutes, 1),
            "estimated_inference_ms_p50": latency_ms_p50,
            "registry_hit": entry is not None,
            "notes": notes,
        }

    @mcp.resource("metrics://prometheus")
    async def prometheus_metrics() -> str:
        """Current Prometheus metrics in text exposition format.

        Same content as the `/metrics` HTTP endpoint. Agents can scrape this
        to monitor latency, cache hit rate, cuda-graph capture status, and
        SLO violations without needing a separate HTTP port.
        """
        try:
            from reflex.observability.prometheus import render_metrics
        except ImportError:
            return "# reflex.observability.prometheus unavailable\n"
        return render_metrics().decode("utf-8")

    return mcp
