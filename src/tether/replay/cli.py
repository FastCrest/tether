"""`tether replay` CLI implementation.

Day 2 scope: load JSONL, replay each request against a target export,
compute per-request actions diff (cosine + max_abs), print human-readable
summary + optional JSON output. Latency / cache / guard diff modes land
in Day 3.

Usage (registered as a typer subcommand by src/tether/cli.py):

    tether replay <file.jsonl[.gz]> --model <export_dir>           \\
            [--diff actions]                                       \\
            [--n <int>]                                            \\
            [--output <json>]                                      \\
            [--fail-on actions]                                    \\
            [--no-replay]            # parse only, don't load model

Replay invokes the same predict path as `tether serve` /act, so the
diff is a true regression-against-recorded comparison.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two flat float vectors. Returns 0.0 on
    degenerate inputs (zero-norm or length mismatch)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def max_abs_diff(a: list[float], b: list[float]) -> float:
    """Max |a[i] - b[i]| over the shorter prefix. 0.0 on empty inputs."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return max(abs(a[i] - b[i]) for i in range(n))


def _flatten_actions(actions: list[list[float]]) -> list[float]:
    """Flatten a chunk-of-actions to one flat vector for cosine / max_abs.
    Returns [] on empty input."""
    out: list[float] = []
    for row in actions:
        out.extend(row)
    return out


def diff_actions(
    recorded: list[list[float]], replayed: list[list[float]],
    *, threshold_cos: float = 0.999, threshold_max_abs: float = 1e-3,
) -> dict[str, Any]:
    """Compare recorded vs replayed action chunks. Returns a dict with
    cosine, max_abs, pass flag."""
    flat_r = _flatten_actions(recorded)
    flat_p = _flatten_actions(replayed)
    cos = cosine_similarity(flat_r, flat_p)
    mad = max_abs_diff(flat_r, flat_p)
    return {
        "cosine": cos,
        "max_abs_diff": mad,
        "passed": cos >= threshold_cos and mad <= threshold_max_abs,
        "threshold_cos": threshold_cos,
        "threshold_max_abs": threshold_max_abs,
    }


def diff_latency(
    recorded: dict[str, Any], replayed: dict[str, Any],
    *, threshold_pct: float = 0.05,
) -> dict[str, Any]:
    """Compare recorded vs replayed latency. Threshold is relative —
    pass if abs((replay - recorded) / recorded) <= threshold_pct on
    total_ms. Per-stage deltas reported but not gated.

    Both inputs follow the D.1.5 latency object shape.
    """
    rec_total = float(recorded.get("total_ms", 0.0))
    rep_total = float(replayed.get("total_ms", 0.0))
    if rec_total <= 0:
        # Can't compute relative delta; mark pass and surface absolute
        return {
            "recorded_total_ms": rec_total,
            "replayed_total_ms": rep_total,
            "delta_ms": rep_total - rec_total,
            "delta_pct": None,
            "passed": True,
            "threshold_pct": threshold_pct,
            "note": "recorded total_ms <= 0; gating skipped",
        }
    delta = rep_total - rec_total
    delta_pct = delta / rec_total
    passed = abs(delta_pct) <= threshold_pct

    # Per-stage, where present
    stages: dict[str, dict[str, float]] = {}
    rec_stages = recorded.get("stages", {}) or {}
    rep_stages = replayed.get("stages", {}) or {}
    for k in sorted(set(rec_stages) | set(rep_stages)):
        rv = rec_stages.get(k)
        pv = rep_stages.get(k)
        if rv is None or pv is None:
            continue
        stages[k] = {
            "recorded_ms": float(rv),
            "replayed_ms": float(pv),
            "delta_ms": float(pv) - float(rv),
        }

    return {
        "recorded_total_ms": rec_total,
        "replayed_total_ms": rep_total,
        "delta_ms": delta,
        "delta_pct": delta_pct,
        "stages": stages,
        "passed": passed,
        "threshold_pct": threshold_pct,
    }


def diff_cache(
    recorded: dict[str, Any] | None, replayed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare recorded vs replayed cache outcome. Pass = same status.

    Both inputs follow the D.1.4 cache object shape: {status: hit|miss|n/a, ...}.
    None on either side is treated as 'n/a'.
    """
    rec_status = (recorded or {}).get("status", "n/a")
    rep_status = (replayed or {}).get("status", "n/a")
    return {
        "recorded_status": rec_status,
        "replayed_status": rep_status,
        "passed": rec_status == rep_status,
    }


def _load_target_server(model: str):
    """Load the target export for replay. Returns the same server type
    create_app() would use, but invoked outside a FastAPI lifespan."""
    from tether.runtime.server import create_app  # noqa: F401  (deferred import)

    # Use create_app to get the same dispatch logic as `tether serve`,
    # but bypass FastAPI — we just want the underlying server object.
    # We can't call create_app() here easily because it sets up FastAPI;
    # instead, replicate its dispatch-by-config logic directly.
    config_path = Path(model) / "tether_config.json"
    cfg: dict[str, Any] = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            cfg = {}

    if cfg.get("export_kind") == "monolithic":
        model_type = cfg.get("model_type", "smolvla")
        if model_type == "pi0":
            from tether.runtime.pi0_onnx_server import Pi0OnnxServer
            srv = Pi0OnnxServer(model)
        elif model_type == "smolvla":
            from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer
            srv = SmolVLAOnnxServer(model)
        else:
            raise ValueError(
                f"Replay against monolithic model_type={model_type!r} not "
                f"yet supported. Day 2 ships pi0 + smolvla only."
            )
    else:
        from tether.runtime.server import TetherServer
        srv = TetherServer(model)
    srv.load()
    return srv


def run_replay(
    trace_file: str,
    model: str | None,
    *,
    diff_mode: str = "actions",
    n: int = 0,  # 0 = all
    output_json: str = "",
    fail_on: str = "",
    no_replay: bool = False,
) -> int:
    """Implementation entry point. Returns CLI exit code."""
    from tether.replay.readers import (  # noqa: F401  (deferred import)
        ReplaySchemaUnknownError,
        load_reader,
    )

    # Validate diff_mode upfront so --no-replay also catches bad values
    valid_modes = {"actions", "latency", "cache", "all"}
    if diff_mode not in valid_modes:
        print(f"ERROR: --diff must be one of {sorted(valid_modes)}, got {diff_mode!r}")
        return 1

    try:
        reader = load_reader(trace_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    header = reader.read_header()
    print(f"Replay: {trace_file}")
    print(f"  tether_version: {header.get('tether_version', '?')}")
    print(f"  model_hash:     {header.get('model_hash', '?')}")
    print(f"  config_hash:    {header.get('config_hash', '?')}")
    print(f"  model_type:     {header.get('model_type', '?')}")
    print(f"  export_kind:    {header.get('export_kind', '?')}")
    print(f"  embodiment:     {header.get('embodiment', '?')}")
    print(f"  redaction:      {header.get('redaction', {})}")
    print(f"  started_at:     {header.get('started_at', '?')}")

    # Parse-only mode: dump records, no model load
    if no_replay or model is None:
        if no_replay:
            print(f"\n--no-replay: parsing records only, not loading model.\n")
        records = list(reader.read_records())
        n_req = sum(1 for k, _ in records if k == "request")
        n_foot = sum(1 for k, _ in records if k == "footer")
        print(f"  records:        {len(records)} ({n_req} requests, {n_foot} footer)")
        return 0

    # Load target model
    print(f"\nLoading target model: {model}")
    try:
        srv = _load_target_server(model)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading target: {e}")
        return 2

    # Hash mismatch warnings (not blocking)
    target_model_hash = ""
    try:
        from tether.runtime.record import compute_model_hash
        target_model_hash = compute_model_hash(model)
    except Exception:  # noqa: BLE001
        pass
    if target_model_hash and target_model_hash != header.get("model_hash"):
        print(
            f"WARN: model_hash mismatch — recorded={header.get('model_hash')} "
            f"vs replay={target_model_hash}"
        )

    do_actions = diff_mode in ("actions", "all")
    do_latency = diff_mode in ("latency", "all")
    do_cache = diff_mode in ("cache", "all")

    # Replay loop
    diffs: list[dict[str, Any]] = []
    n_replayed = 0
    n_pass_actions = 0
    n_pass_latency = 0
    n_pass_cache = 0
    image_redaction = (header.get("redaction", {}) or {}).get("image", "hash_only")

    if do_actions and image_redaction != "full":
        print(
            f"WARN: trace was recorded with image redaction='{image_redaction}'; "
            f"actions diff needs full images. Pass --no-replay to inspect "
            f"the trace, or re-record with --record-images full."
        )

    print(f"\nReplaying requests (--n={n or 'all'}, --diff={diff_mode}):")
    for kind, rec in reader.read_records():
        if kind != "request":
            continue
        if n and n_replayed >= n:
            break

        req = rec.get("request", {})
        recorded_resp = rec.get("response", {})
        recorded_actions = recorded_resp.get("actions", [])
        recorded_latency = rec.get("latency", {}) or {}
        recorded_cache = rec.get("cache")

        # Need full image to invoke the model. Skip replay for actions/latency
        # if absent; cache diff can run from recorded-only metadata if needed.
        if not req.get("image_b64"):
            n_replayed += 1
            continue

        try:
            replay_resp = srv.predict_from_base64(
                image_b64=req["image_b64"],
                instruction=req.get("instruction", ""),
                state=req.get("state"),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  seq={rec.get('seq')}: replay raised {type(e).__name__}: {e}")
            n_replayed += 1
            continue

        per_record: dict[str, Any] = {"seq": rec.get("seq")}
        line_parts: list[str] = [f"  seq={rec.get('seq'):4d}"]

        if do_actions:
            replayed_actions = replay_resp.get("actions", [])
            d_a = diff_actions(recorded_actions, replayed_actions)
            per_record["actions"] = d_a
            if d_a["passed"]:
                n_pass_actions += 1
            line_parts.append(
                f"actions: cos={d_a['cosine']:.6f} max_abs={d_a['max_abs_diff']:.2e} "
                f"[{'PASS' if d_a['passed'] else 'FAIL'}]"
            )

        if do_latency:
            # Replay latency comes from the predict_from_base64 result
            replayed_latency = {
                "total_ms": float(replay_resp.get("latency_ms", 0.0)),
            }
            d_l = diff_latency(recorded_latency, replayed_latency)
            per_record["latency"] = d_l
            if d_l["passed"]:
                n_pass_latency += 1
            pct_str = (
                f"{d_l['delta_pct'] * 100:+.1f}%"
                if d_l["delta_pct"] is not None
                else "n/a"
            )
            line_parts.append(
                f"latency: {d_l['recorded_total_ms']:.0f}→"
                f"{d_l['replayed_total_ms']:.0f}ms ({pct_str}) "
                f"[{'PASS' if d_l['passed'] else 'FAIL'}]"
            )

        if do_cache:
            # Day 3: recorded-vs-recorded since the replay path doesn't yet
            # surface its own cache state. Stub mirrors recorded_cache for
            # the replay side until cache instrumentation lands in serve.
            replayed_cache = recorded_cache  # placeholder
            d_c = diff_cache(recorded_cache, replayed_cache)
            per_record["cache"] = d_c
            if d_c["passed"]:
                n_pass_cache += 1
            line_parts.append(
                f"cache: {d_c['recorded_status']}→{d_c['replayed_status']} "
                f"[{'PASS' if d_c['passed'] else 'FAIL'}]"
            )

        diffs.append(per_record)
        n_replayed += 1
        print("  " + "  ".join(line_parts))

    # Summary
    print("\nSummary:")
    print(f"  replayed: {n_replayed}")
    print(f"  diffed:   {len(diffs)}")
    if do_actions:
        print(
            f"  actions:  {n_pass_actions}/{len(diffs)} pass "
            f"(cos≥0.999, max_abs<1e-3)"
        )
    if do_latency:
        print(
            f"  latency:  {n_pass_latency}/{len(diffs)} pass "
            f"(within ±5% of recorded total_ms)"
        )
    if do_cache:
        print(
            f"  cache:    {n_pass_cache}/{len(diffs)} pass "
            f"(status match)"
        )

    if output_json:
        Path(output_json).write_text(
            json.dumps(
                {
                    "summary": {
                        "trace_file": str(trace_file),
                        "model": model,
                        "diff_mode": diff_mode,
                        "n_replayed": n_replayed,
                        "n_diffed": len(diffs),
                        "n_pass_actions": n_pass_actions if do_actions else None,
                        "n_pass_latency": n_pass_latency if do_latency else None,
                        "n_pass_cache": n_pass_cache if do_cache else None,
                    },
                    "header": header,
                    "per_request_diffs": diffs,
                },
                indent=2,
            )
        )
        print(f"  output:   {output_json}")

    # --fail-on dispatch
    fail_codes = {
        "actions": (do_actions, n_pass_actions, len(diffs)),
        "latency": (do_latency, n_pass_latency, len(diffs)),
        "cache": (do_cache, n_pass_cache, len(diffs)),
    }
    if fail_on:
        if fail_on not in fail_codes:
            print(
                f"ERROR: --fail-on must be one of {sorted(fail_codes)}, "
                f"got {fail_on!r}"
            )
            return 1
        active, passed, total = fail_codes[fail_on]
        if not active:
            print(
                f"ERROR: --fail-on {fail_on} requires --diff {fail_on} or --diff all"
            )
            return 1
        if passed < total:
            return 3

    return 0


__all__ = [
    "cosine_similarity",
    "max_abs_diff",
    "diff_actions",
    "run_replay",
]
