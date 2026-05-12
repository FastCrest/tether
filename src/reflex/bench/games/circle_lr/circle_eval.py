# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Default evaluator for the circle game.

The default policy is an oracle calibrated tapper. Experiments can import
`evaluate_circle` and pass a trained policy later, while reusing the same
target handshake, tablet setup, run logging, and metrics.
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reflex.embodiments.so100.calibration.tapper import score_touch
from reflex.bench.games._base.circle_runtime import (
    COLLECT_MODES,
    DEFAULT_PORT,
    npos_for_mode,
)
from reflex.bench.games.circle_lr.circle_task import CircleTask
from reflex.bench.games._base.run_writer import JsonlRunWriter

print = partial(print, flush=True)

EVALS_DIR = ROOT / "data" / "circle_evals"


class OracleTapPolicy:
    """Baseline policy that uses calibration to tap the target coordinates."""

    name = "oracle_tapper"

    def __init__(self, task):
        self.task = task

    def run_episode(self, target, ep):
        return self.task.oracle_demo(
            target, ep, capture_hover=False, allow_touch_trail=True)


def summarize(records):
    attempted = [
        r for r in records
        if r.get("command_pose") is not None or r.get("result") in ("hit", "miss")
    ]
    hits = [r for r in records if r.get("result") == "hit"]
    misses = [r for r in records if r.get("result") == "miss"]
    rejects = [r for r in records if r.get("result") == "reject"]
    dists = [
        score_touch(r["touch"], r["target"])
        for r in records
        if r.get("touch") is not None and r.get("target") is not None
    ]
    return {
        "episodes": len(records),
        "attempted": len(attempted),
        "hits": len(hits),
        "misses": len(misses),
        "rejects": len(rejects),
        "success_rate": len(hits) / len(records) if records else 0.0,
        "attempt_success_rate": len(hits) / len(attempted) if attempted else 0.0,
        "mean_distance_px": sum(dists) / len(dists) if dists else None,
        "max_distance_px": max(dists) if dists else None,
    }


def evaluate_circle(args, policy_factory=None):
    npos = npos_for_mode(args.mode, args.npos)
    run_name = args.run_name or time.strftime("circle_eval_%Y%m%d_%H%M%S")
    run_dir = Path(args.evals_dir) / run_name
    writer = JsonlRunWriter(run_dir, "eval_config.json", {
        "game": "circle",
        "purpose": "eval",
        "episodes": args.episodes,
        "mode": args.mode,
        "npos": npos,
        "port": args.port,
        "policy": args.policy,
        "no_camera": args.no_camera,
        "cam_width": args.cam_width,
        "cam_height": args.cam_height,
    })

    records = []

    try:
        with CircleTask(
            port=args.port,
            npos=npos,
            frame_dir=writer.frame_dir,
            no_camera=args.no_camera,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
        ) as task:
            policy = policy_factory(task, args) if policy_factory else OracleTapPolicy(task)
            for ep in range(args.episodes):
                print(f"\n=== eval episode {ep + 1}/{args.episodes} ===")
                target, frames_or_reject = task.prepare_episode(ep)
                if target is None:
                    rec = frames_or_reject
                else:
                    rec = policy.run_episode(target, ep)
                rec.update({
                    "episode": ep,
                    "mode": args.mode,
                    "npos": npos,
                    "policy": getattr(policy, "name", args.policy),
                    "frames": {
                        "before": frames_or_reject.get("before"),
                        "hover": rec.pop("hover_frame", None),
                    },
                })
                records.append(rec)
                writer.write_episode(rec)
                print(f"  {rec['result']}: {rec['reason']}")
    except RuntimeError as e:
        print(f"ABORT: {e}")
        return 1

    summary = summarize(records)
    writer.write_json("summary.json", summary)
    print("\nEval summary:")
    print(json.dumps(summary, indent=2))
    print(f"Run: {run_dir}")
    return 0


def build_parser(default_mode="2d"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--mode",
        choices=sorted(COLLECT_MODES),
        default=default_mode,
        help="Target policy/curriculum for eval.",
    )
    parser.add_argument(
        "--npos",
        type=int,
        default=None,
        help="Raw circle-game curriculum id. Overrides --mode when set.",
    )
    parser.add_argument(
        "--policy",
        choices=["oracle"],
        default="oracle",
        help="Policy to evaluate. Experiments can import evaluate_circle for custom policies.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--evals-dir", default=str(EVALS_DIR))
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=360)
    return parser


def main(argv=None, default_mode="2d", policy_factory=None,
         default_evals_dir=None):
    parser = build_parser(default_mode=default_mode)
    if default_evals_dir is not None:
        parser.set_defaults(evals_dir=str(default_evals_dir))
    return evaluate_circle(parser.parse_args(argv), policy_factory=policy_factory)


if __name__ == "__main__":
    raise SystemExit(main())
