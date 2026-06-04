"""Fire the B.4 A2C2 transfer-validation gate.

Reads a trained A2C2 checkpoint (.npz) + an in-distribution held-out JSONL split
+ N out-of-distribution JSONL traces (e.g., recorded with `tether serve
--inject-latency-ms 100`). Computes per-trace MSE + the success-delta gate.
Emits a Markdown + JSON report and exits 0 on PROCEED, 1 on PAUSE, 2 on ABORT.

Path A unification 2026-04-25: pure-numpy now (no torch). Loads .npz directly.

Usage (synthetic plumbing-test, no Modal):
    python scripts/validate_a2c2_transfer.py \\
        --checkpoint outputs/a2c2_synthetic.npz \\
        --in-dist synthetic:200 \\
        --held-out synthetic:50:30 synthetic:50:60 synthetic:50:120 \\
        --task-success-on 0.94 --task-success-off 0.87 \\
        --eval-latency-ms 80 \\
        --report outputs/a2c2_transfer_report.md

Usage (real data):
    python scripts/validate_a2c2_transfer.py \\
        --checkpoint outputs/a2c2_lerobot_trained.npz \\
        --in-dist 'data/libero_held_out/*.jsonl' \\
        --held-out 'data/jetson_traces_1/*.jsonl' \\
                   'data/jetson_traces_2/*.jsonl' \\
                   'data/jetson_traces_3/*.jsonl' \\
        --task-success-on 0.94 --task-success-off 0.87 \\
        --eval-latency-ms 80 \\
        --report outputs/a2c2_transfer_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from tether.correction import (
    A2C2Head,
    GateThresholds,
    compute_gate_report,
    evaluate_mse,
)

logger = logging.getLogger("a2c2.validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _load_jsonl(path: Path) -> list[dict]:
    import gzip
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[dict] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _gather_paths(spec: str) -> list[Path]:
    if spec.startswith("synthetic:"):
        return []
    p = Path(spec)
    if p.is_dir():
        return sorted(list(p.glob("*.jsonl")) + list(p.glob("*.jsonl.gz")))
    if any(ch in spec for ch in "*?["):
        from glob import glob
        return [Path(s) for s in sorted(glob(spec))]
    return [p]


def _flatten_for_eval(records: list[dict], action_dim: int, obs_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same shape as scripts/train_a2c2._flatten_traces."""
    base_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    chunk_idx_rows: list[int] = []
    latency_rows: list[float] = []
    for rec in records:
        actions = rec.get("actions") or rec.get("response", {}).get("actions")
        state = rec.get("state") or rec.get("request", {}).get("state")
        if not actions or not state:
            continue
        latency_ms = float(rec.get("latency_ms") or rec.get("latency_total_ms") or 0.0)
        injected = float(rec.get("injected_latency_ms") or 0.0)
        observed_latency = latency_ms + injected
        for chunk_idx, action in enumerate(actions):
            a = np.asarray(action, dtype=np.float32)
            if a.shape[0] >= action_dim:
                a = a[:action_dim]
            else:
                continue
            obs_pad = np.zeros(obs_dim, dtype=np.float32)
            s = np.asarray(state[:obs_dim], dtype=np.float32)
            obs_pad[: s.shape[0]] = s
            base_rows.append(a)
            obs_rows.append(obs_pad)
            chunk_idx_rows.append(chunk_idx)
            latency_rows.append(observed_latency)
    if not base_rows:
        return (np.zeros((0, action_dim), dtype=np.float32),
                np.zeros((0, obs_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, action_dim), dtype=np.float32))
    base_arr = np.asarray(base_rows, dtype=np.float32)
    return (base_arr,
            np.asarray(obs_rows, dtype=np.float32),
            np.asarray(chunk_idx_rows, dtype=np.int64),
            np.asarray(latency_rows, dtype=np.float32),
            np.zeros_like(base_arr))  # real-data target: 0 (magnitude proxy)


def _synthetic_split(
    spec: str, action_dim: int, obs_dim: int, target_noise_std: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse 'synthetic:N' or 'synthetic:N:LAT' specs.

    N = episode count, LAT = mean latency in ms (defaults to 30).
    Also returns target_residual matching scripts/train_a2c2.py construction —
    latency-and-chunk-conditioned structured noise so the gate can see
    learn-vs-not-learn behavior.
    """
    parts = spec.split(":")
    n = int(parts[1]) if len(parts) >= 2 else 50
    lat = float(parts[2]) if len(parts) >= 3 else 30.0
    rng = np.random.default_rng(int(lat * 7919) ^ n)
    chunk_size = 16
    base = rng.standard_normal((n * chunk_size, action_dim)).astype(np.float32) * 0.3
    obs = rng.standard_normal((n * chunk_size, obs_dim)).astype(np.float32) * 0.5
    chunk_idx = np.tile(np.arange(chunk_size, dtype=np.int64), n)
    latency = rng.uniform(lat * 0.8, lat * 1.2, size=n * chunk_size).astype(np.float32)
    lat_norm = (latency - 20) / 60.0
    chunk_norm = chunk_idx.astype(np.float32) / max(chunk_size, 1)
    scale = (lat_norm * chunk_norm)[:, None]
    target_residual = (
        rng.standard_normal((n * chunk_size, action_dim)).astype(np.float32)
        * target_noise_std
        * scale
    )
    return base, obs, chunk_idx, latency, target_residual


def main() -> int:
    parser = argparse.ArgumentParser(description="Fire B.4 A2C2 transfer gate.")
    parser.add_argument("--checkpoint", required=True, help="Trained .npz from scripts/train_a2c2.py")
    parser.add_argument("--in-dist", required=True, help="JSONL spec OR 'synthetic:N[:LAT]'")
    parser.add_argument("--held-out", nargs="+", required=True,
                        help="One or more JSONL specs (each becomes one trace; "
                             "use 'synthetic:N:LAT' for synthetic plumbing tests)")
    parser.add_argument("--task-success-on", type=float, required=True,
                        help="LIBERO task-success rate with A2C2-on (0..1)")
    parser.add_argument("--task-success-off", type=float, required=True,
                        help="LIBERO task-success rate with A2C2-off (0..1)")
    parser.add_argument("--eval-latency-ms", type=float, required=True,
                        help="Latency floor for the success-delta measurement (ms)")
    parser.add_argument("--report", required=True, help="Markdown report output path")
    parser.add_argument("--report-json", default="", help="Optional JSON report output path")
    parser.add_argument("--mse-ratio-proceed-max", type=float, default=1.2)
    parser.add_argument("--mse-ratio-abort-min", type=float, default=2.0)
    parser.add_argument("--success-delta-proceed-min", type=float, default=0.05)
    parser.add_argument("--success-delta-abort-max", type=float, default=0.0)
    parser.add_argument("--eval-latency-floor", type=float, default=40.0)
    parser.add_argument("--note", action="append", default=[],
                        help="Add a note line to the report (repeatable)")
    args = parser.parse_args()

    head = A2C2Head.from_checkpoint(args.checkpoint)
    cfg = head.config
    n_params = sum(w.size + b.size for w, b in zip(head._weights, head._biases))
    logger.info(
        "loaded checkpoint: %s (%d params, ~%.1f KB FP32)",
        args.checkpoint, n_params,
        cfg.estimated_size_bytes() / 1024,
    )

    def _load_split(spec: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if spec.startswith("synthetic:"):
            return _synthetic_split(spec, cfg.action_dim, cfg.obs_dim)
        paths = _gather_paths(spec)
        if not paths:
            raise SystemExit(f"no JSONL files at {spec}")
        recs: list[dict] = []
        for p in paths:
            recs.extend(_load_jsonl(p))
        return _flatten_for_eval(recs, cfg.action_dim, cfg.obs_dim)

    in_b, in_o, in_c, in_l, in_t = _load_split(args.in_dist)
    in_dist_mse = evaluate_mse(
        head,
        base_actions=in_b, observations=in_o,
        chunk_positions=in_c, latency_ms_per_step=in_l,
        target_residuals=in_t,
    )
    logger.info("in-distribution MSE: %.6f (n=%d)", in_dist_mse, in_b.shape[0])

    held_out_mses: list[float] = []
    for spec in args.held_out:
        b, o, c, l, t = _load_split(spec)
        mse = evaluate_mse(
            head,
            base_actions=b, observations=o,
            chunk_positions=c, latency_ms_per_step=l,
            target_residuals=t,
        )
        logger.info("held-out '%s' MSE: %.6f (n=%d)", spec, mse, b.shape[0])
        held_out_mses.append(mse)

    th = GateThresholds(
        mse_ratio_proceed_max=args.mse_ratio_proceed_max,
        mse_ratio_abort_min=args.mse_ratio_abort_min,
        success_delta_proceed_min=args.success_delta_proceed_min,
        success_delta_abort_max=args.success_delta_abort_max,
        eval_latency_ms_floor=args.eval_latency_floor,
    )
    report = compute_gate_report(
        in_dist_mse=in_dist_mse,
        held_out_mses=held_out_mses,
        task_success_on=args.task_success_on,
        task_success_off=args.task_success_off,
        eval_latency_ms=args.eval_latency_ms,
        thresholds=th,
        notes=list(args.note),
    )
    report.write(args.report)
    if args.report_json:
        report.write(args.report_json)
    logger.info("report written: %s", args.report)
    logger.info("DECISION: %s", report.decision.value)
    if report.failure_reasons:
        for r in report.failure_reasons:
            logger.warning("  reason: %s", r)
    return {"PROCEED": 0, "PAUSE": 1, "ABORT": 2}[report.decision.value]


if __name__ == "__main__":
    sys.exit(main())
