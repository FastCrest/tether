# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Default collection loop for the circle game.

Experiments can import `collect_circle` to reuse the standard tablet setup,
target handshake, calibrated tap, clean-touch filtering, and run writing.
"""
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reflex.embodiments.so100.calibration.tapper import (
    MAX_TRAIL_POINTS,
    MAX_TRAIL_PX,
)
from reflex.bench.games.circle_lr.demo_sanity import check_demo_episode
from reflex.bench.games._base.circle_runtime import (
    COLLECT_MODES,
    DEFAULT_PORT,
    npos_for_mode,
)
from reflex.bench.games.circle_lr.circle_task import CircleTask
from reflex.bench.games._base.run_writer import JsonlRunWriter

print = partial(print, flush=True)

RUNS_DIR = ROOT / "data" / "circle_runs"


def collect_circle(args):
    npos = npos_for_mode(args.mode, args.npos)
    run_name = args.run_name or time.strftime("circle_%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    writer = JsonlRunWriter(run_dir, "collection_config.json", {
        "game": "circle",
        "purpose": "collection",
        "format": "local_lerobot_like_jsonl",
        "episodes": args.episodes,
        "mode": args.mode,
        "npos": npos,
        "port": args.port,
        "no_camera": args.no_camera,
        "cam_width": args.cam_width,
        "cam_height": args.cam_height,
        "clean_touch": {
            "max_trail_points": MAX_TRAIL_POINTS,
            "max_trail_px": MAX_TRAIL_PX,
        },
    })

    try:
        saved = 0
        attempted = 0
        with CircleTask(
            port=args.port,
            npos=npos,
            frame_dir=writer.frame_dir,
            no_camera=args.no_camera,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
        ) as task:
            for ep in range(args.episodes):
                print(f"\n=== episode {ep + 1}/{args.episodes} ===")
                target, frames_or_reject = task.prepare_episode(ep)
                if target is None:
                    rec = frames_or_reject
                    rec.update({
                        "mode": args.mode,
                        "npos": npos,
                        "saved": False,
                    })
                    writer.write_episode(rec)
                    print(f"  reject: {rec['reason']}")
                    continue

                rec = task.oracle_demo(
                    target, ep, capture_hover=True, capture_trajectory=True)
                if rec.get("command_pose") is not None or rec["result"] in ("hit", "miss"):
                    attempted += 1
                demo_frames = rec.pop("demo_frames", [])
                sanity_ok, sanity_reasons = check_demo_episode(demo_frames)
                if rec["result"] == "hit" and not sanity_ok:
                    rec["result"] = "reject"
                    rec["reason"] = "demo sanity failed: " + "; ".join(sanity_reasons)
                rec.update({
                    "episode": ep,
                    "mode": args.mode,
                    "npos": npos,
                    "frames": {
                        "before": frames_or_reject.get("before"),
                        "hover": rec.pop("hover_frame", None),
                    },
                    "saved": rec["result"] == "hit",
                    "demo_frame_count": len(demo_frames),
                    "demo_sanity": {
                        "ok": sanity_ok,
                        "reasons": sanity_reasons,
                    },
                })
                print(f"  {rec['result']}: {rec['reason']}")
                if rec["saved"]:
                    saved += 1
                    writer.write_demo_episode(ep, demo_frames)
                writer.write_episode(rec)
    except RuntimeError as e:
        print(f"ABORT: {e}")
        return 1

    print(f"\nDone: saved {saved}/{attempted} clean hits")
    print(f"Run: {run_dir}")
    return 0


def build_parser(default_mode="random"):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--mode",
        choices=sorted(COLLECT_MODES),
        default=default_mode,
        help="Target policy. Use '2d'/'left-right' for random two-position LR.",
    )
    parser.add_argument(
        "--npos",
        type=int,
        default=None,
        help="Raw circle-game curriculum id. Overrides --mode when set.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=360)
    return parser


def main(argv=None, default_mode="random", default_runs_dir=None):
    parser = build_parser(default_mode=default_mode)
    if default_runs_dir is not None:
        parser.set_defaults(runs_dir=str(default_runs_dir))
    return collect_circle(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
