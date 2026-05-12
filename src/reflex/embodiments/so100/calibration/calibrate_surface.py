# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Old-style contact grid calibration: pan/lift grid + wrist-only taps.

This follows the useful pattern from ../auto_soarm/archive/calibrate_2d.py:
move through a pan/lift grid while coordinating wrist height for safe transit,
then tap by wrist_flex only. The original four corner samples are used only as
tablet bounds/context, not as the contact model.

Usage:
    python -u -B calibration/calibrate_surface.py
"""
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))

from reflex.embodiments.so100.edge_runtime import SOArmHardware, PAN_ID, LIFT_ID, WRIST_FLEX_ID
from reflex.embodiments.so100.calibration.model import TabletCal
from reflex.embodiments.so100.calibration.model import GENTLE_CONTACT_BACKOFF_TICKS, HOVER_MIN_WF
from reflex.embodiments.so100.calibration.calibrate_corners import (
    CONFIG_PATH, GAMES_DIR, HIT_QUEUE, HTTP_PORT,
    VERIFY_ABSOLUTE_MIN_WF, VERIFY_LIFT_DWELL_S,
    VERIFY_XY_TOLERANCE_PX,
    _adb, _drain_queue, _drain_touch_events, _format_pose_delta,
    _read_pose, _summarize_touch_events, adb_launch_url, adb_reverse,
    adb_screen_size, adb_tap, confirm, start_static_server,
)


# Raw-tick equivalents inspired by the original repo's surface sweep.
HARD_UP_WF = 850

# The old repo did not probe every row with the same shoulder lift. It swept
# much lower lifts (roughly 2100..2600) and used a deeper wrist seed there.
# These seeds are intentionally shallow-first: tap_wrist_only starts above the
# expected contact, then walks deeper and stops at first contact.
CONTACT_SEED_BY_LIFT = [
    (2200, 1740),
    (2300, 1600),
    (2400, 1460),
    (2500, 1320),
    (2605, 1160),
    (2665, 1100),
    (2705, 1035),
    (2745, 1010),
    (2785, 1005),
]

PAN_VALUES = [1525, 1625, 1725, 1825, 1925, 2025, 2125, 2225, 2325, 2425, 2525]
LIFT_VALUES = [2200, 2300, 2400, 2500, 2605, 2705, 2785]

# Relative to the row seed. Negative probes are useful because the seed is only
# an estimate; stopping at first contact keeps the tap light.
# Probe shallow to deep. Finer steps near likely first contact reduce the
# chance that the first reported touch is already a smeared press.
WF_ADJUSTS = [-120, -100, -80, -65, -50, -40, -30, -20, -10,
              0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80]
MAX_CONTACT_WF = 1850

MOVE_STEPS = 12
MOVE_STEP_DELAY = 0.08
TAP_HOLD_S = 0.18
RETRACT_HOLD_S = 0.35
GENTLE_TOUCH_MAX_EVENTS = 2
GENTLE_TOUCH_MAX_SPAN_PX = 10


def setup_tablet():
    r = _adb("devices")
    if "device\n" not in r.stdout and "device\r\n" not in r.stdout:
        print("ABORT: adb sees no devices.")
        return None
    size = adb_screen_size()
    if size is None:
        print("ABORT: could not read tablet screen size.")
        return None
    sw, sh = size
    print(f"Tablet: {sw} x {sh}")

    print(f"Starting HTTP server on Pi: 0.0.0.0:{HTTP_PORT}")
    try:
        start_static_server(port=HTTP_PORT, directory=GAMES_DIR)
    except RuntimeError as e:
        print(f"ABORT: {e}")
        return None
    adb_reverse(HTTP_PORT)
    url = f"http://localhost:{HTTP_PORT}/calibrate.html?nonce={int(time.time())}"
    print(f"Launching tablet: {url}")
    adb_launch_url(url)
    time.sleep(2.0)
    print("3-tap fullscreen...")
    for _ in range(3):
        adb_tap(sw // 2, sh // 2)
        time.sleep(0.6)
    time.sleep(1.5)
    _drain_queue(HIT_QUEUE)
    _drain_touch_events()
    return sw, sh


def real_touches(events):
    return [
        e for e in events
        if e.get("x", -1) >= 0 and e.get("y", -1) >= 0
    ]


def touch_span(touches):
    if len(touches) < 2:
        return 0.0
    xs = [e["x"] for e in touches]
    ys = [e["y"] for e in touches]
    return float(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5)


def _interp_seed_for_lift(lift):
    pts = CONTACT_SEED_BY_LIFT
    if lift <= pts[0][0]:
        return pts[0][1]
    if lift >= pts[-1][0]:
        return pts[-1][1]
    for (l0, w0), (l1, w1) in zip(pts, pts[1:]):
        if l0 <= lift <= l1:
            t = (lift - l0) / (l1 - l0)
            return w0 + (w1 - w0) * t
    return pts[-1][1]


def hover_wf_for_lift(lift):
    return int(round(min(HARD_UP_WF, _interp_seed_for_lift(lift) - 180)))


def contact_wf_for_lift(lift, adjust=0):
    return int(round(min(MAX_CONTACT_WF, _interp_seed_for_lift(lift) + adjust)))


def move_coordinated(arm, pan, lift):
    cur = _read_pose(arm)
    start_pan = cur["pan"]
    start_lift = cur["lift"]
    start_wf = cur["wrist_flex"]
    target_hover = hover_wf_for_lift(lift)
    _drain_touch_events()

    for step in range(1, MOVE_STEPS + 1):
        t = step / MOVE_STEPS
        p = int(round(start_pan + (pan - start_pan) * t))
        l = int(round(start_lift + (lift - start_lift) * t))
        w = int(round(start_wf + (target_hover - start_wf) * t))
        # Wrist first, then pan/lift, matching the old calibration pattern.
        arm.write_goal(WRIST_FLEX_ID, w)
        time.sleep(0.03)
        arm.write_goal(PAN_ID, p)
        arm.write_goal(LIFT_ID, l)
        time.sleep(MOVE_STEP_DELAY)
        touches = real_touches(_drain_touch_events())
        if touches:
            print(f"  DRAG during transit step={step}: {_summarize_touch_events(touches)}")
            arm.write_goal(WRIST_FLEX_ID, HARD_UP_WF)
            time.sleep(RETRACT_HOLD_S)
            return False
    time.sleep(0.3)
    actual = _read_pose(arm)
    print("  after coordinated move: "
          f"{_format_pose_delta(actual, {'pan': pan, 'lift': lift, 'wrist_flex': target_hover})}")
    return True


def tap_wrist_only(arm, pan, lift):
    hover = hover_wf_for_lift(lift)
    try:
        for adjust in WF_ADJUSTS:
            wf = contact_wf_for_lift(lift, adjust)
            _drain_queue(HIT_QUEUE)
            _drain_touch_events()
            arm.write_goal(WRIST_FLEX_ID, wf)
            time.sleep(TAP_HOLD_S)
            actual = _read_pose(arm)
            touches = real_touches(_drain_touch_events())
            if touches:
                span = touch_span(touches)
                clean = len(touches) <= GENTLE_TOUCH_MAX_EVENTS and span <= GENTLE_TOUCH_MAX_SPAN_PX
                touch_x = sum(float(t["x"]) for t in touches) / len(touches)
                touch_y = sum(float(t["y"]) for t in touches) / len(touches)
                print(f"  tap wf={wf} adjust={adjust}: {_summarize_touch_events(touches)}")
                if not clean:
                    print(f"    TRAIL/HEAVY contact: excluding from fit "
                          f"n_events={len(touches)} span={span:.0f}px")
                print("    actual at tap: "
                      f"{_format_pose_delta(actual, {'pan': pan, 'lift': lift, 'wrist_flex': wf})}")
                # Lower raw wrist_flex is physically higher. Store a base-layer
                # gentle command so runtime taps are not forced deeper than the
                # first-contact wrist found during calibration.
                gentle_wf = max(
                    HOVER_MIN_WF,
                    min(wf, actual["wrist_flex"]) - GENTLE_CONTACT_BACKOFF_TICKS,
                )
                return {
                    "pan": pan,
                    "lift": lift,
                    "wf_command": wf,
                    "wf_adjust": adjust,
                    "gentle_wrist_flex": gentle_wf,
                    "actual_pan": actual["pan"],
                    "actual_lift": actual["lift"],
                    "actual_wrist_flex": actual["wrist_flex"],
                    "touch_x": touch_x,
                    "touch_y": touch_y,
                    "touch_events": len(touches),
                    "touch_span_px": span,
                    "clean": clean,
                }
        print("  no wrist-only tap")
        return None
    finally:
        arm.write_goal(WRIST_FLEX_ID, hover)
        time.sleep(RETRACT_HOLD_S)


def _affine_features(samples):
    return np.array([[s["touch_x"], s["touch_y"], 1.0] for s in samples])


def _quadratic_features(samples):
    return np.array([
        [
            s["touch_x"],
            s["touch_y"],
            s["touch_x"] * s["touch_y"],
            s["touch_x"] ** 2,
            s["touch_y"] ** 2,
            1.0,
        ]
        for s in samples
    ])


def fit_touch_to_motor(samples, model="affine"):
    clean_samples = [s for s in samples if s.get("clean", True)]
    m = _quadratic_features(clean_samples) if model == "quadratic" else _affine_features(clean_samples)
    pan = np.array([s["actual_pan"] for s in clean_samples])
    lift = np.array([s["actual_lift"] for s in clean_samples])
    coef_pan, *_ = np.linalg.lstsq(m, pan, rcond=None)
    coef_lift, *_ = np.linalg.lstsq(m, lift, rcond=None)
    return coef_pan.tolist(), coef_lift.tolist(), clean_samples


def predict_with_model(sample, coef, model="affine"):
    feats = _quadratic_features([sample])[0] if model == "quadratic" else _affine_features([sample])[0]
    return float(np.dot(np.array(coef), feats))


def print_model_report(name, coef_pan, coef_lift, samples):
    print(f"\n{name} contact model:")
    print(f"  pan coefficients:  {[round(x, 6) for x in coef_pan]}")
    print(f"  lift coefficients: {[round(x, 6) for x in coef_lift]}")
    pan_errs = []
    lift_errs = []
    print("  Residuals:")
    model = "quadratic" if name.lower().startswith("quadratic") else "affine"
    for s in samples:
        pred_pan = predict_with_model(s, coef_pan, model=model)
        pred_lift = predict_with_model(s, coef_lift, model=model)
        pan_err = pred_pan - s["actual_pan"]
        lift_err = pred_lift - s["actual_lift"]
        pan_errs.append(abs(pan_err))
        lift_errs.append(abs(lift_err))
        print(f"    touch=({s['touch_x']:.0f},{s['touch_y']:.0f}) "
              f"pan_err={pan_err:+.1f} lift_err={lift_err:+.1f}")
    print(f"  MAE: pan={np.mean(pan_errs):.1f} lift={np.mean(lift_errs):.1f}; "
          f"max: pan={np.max(pan_errs):.1f} lift={np.max(lift_errs):.1f}")


def maybe_save(samples, cal):
    clean_count = sum(1 for s in samples if s.get("clean", True))
    if clean_count < 3:
        print("Not enough samples to fit.")
        return False
    coef_pan, coef_lift, clean_samples = fit_touch_to_motor(samples, model="affine")
    q_pan, q_lift, _ = fit_touch_to_motor(samples, model="quadratic")
    print(f"\nUsing {len(clean_samples)}/{len(samples)} clean samples for fit.")
    print_model_report("Affine", coef_pan, coef_lift, clean_samples)
    print_model_report("Quadratic", q_pan, q_lift, clean_samples)

    if not confirm("Save wrist-grid quadratic contact_model?", default_yes=False):
        print("Not saving.")
        return False

    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["contact_model"] = {
        "type": "quadratic_touch_to_motor_wrist_grid_v1",
        "bounds": cal.bbox,
        "surface": {
            "hard_up_wf": HARD_UP_WF,
            "contact_seed_by_lift": CONTACT_SEED_BY_LIFT,
            "max_contact_wf": MAX_CONTACT_WF,
        },
        "coefficients": {
            "pan": q_pan,
            "lift": q_lift,
        },
        "samples": samples,
        "fit_sample_count": len(clean_samples),
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"Saved contact_model to {CONFIG_PATH}")
    return True


def save_run_artifact(samples, attempts):
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "last_wrist_grid_samples.json"
    out_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pan_values": PAN_VALUES,
        "lift_values": LIFT_VALUES,
        "samples": samples,
        "attempts": attempts,
    }, indent=2))
    print(f"Wrote run samples to {out_path}")


def main():
    if not CONFIG_PATH.exists():
        print(f"ABORT: no {CONFIG_PATH}. Run calibration/calibrate_corners.py first.")
        return 1
    if setup_tablet() is None:
        return 1
    cal = TabletCal()
    print(f"Using tablet bounds: {cal.bbox}")
    print(f"Contact seed by lift: {CONTACT_SEED_BY_LIFT}")
    print(f"Grid: {len(PAN_VALUES)} pan x {len(LIFT_VALUES)} lift = "
          f"{len(PAN_VALUES) * len(LIFT_VALUES)} points")

    print("Connecting to arm...")
    arm = SOArmHardware()
    arm.home()
    time.sleep(1.0)
    samples = []
    attempts = []
    try:
        for lift in LIFT_VALUES:
            for pan in PAN_VALUES:
                print(f"\n--- grid pan={pan} lift={lift} "
                      f"hover_wf={hover_wf_for_lift(lift)} ---")
                if not move_coordinated(arm, pan, lift):
                    attempts.append({
                        "pan": pan,
                        "lift": lift,
                        "status": "drag_transit",
                    })
                    continue
                sample = tap_wrist_only(arm, pan, lift)
                if sample:
                    samples.append(sample)
                    attempts.append({
                        "pan": pan,
                        "lift": lift,
                        "status": "touch" if sample.get("clean", True) else "trail_contact",
                        "touch_x": sample["touch_x"],
                        "touch_y": sample["touch_y"],
                        "clean": sample.get("clean", True),
                        "touch_events": sample.get("touch_events"),
                        "touch_span_px": sample.get("touch_span_px"),
                    })
                else:
                    attempts.append({
                        "pan": pan,
                        "lift": lift,
                        "status": "no_touch",
                    })

        print(f"\nCollected {len(samples)} wrist-grid samples.")
        for s in samples:
            print(f"  touch=({s['touch_x']:.1f},{s['touch_y']:.1f}) "
                  f"actual_pan={s['actual_pan']} actual_lift={s['actual_lift']} "
                  f"actual_wf={s['actual_wrist_flex']}")
        save_run_artifact(samples, attempts)
        maybe_save(samples, cal)
    finally:
        arm.home()
        time.sleep(1.0)
        arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
