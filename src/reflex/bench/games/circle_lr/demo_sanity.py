# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Inline checks for local demonstration episodes before saving."""
import math

from reflex.embodiments.so100.calibration.motion_io import TAP_FRAMES

WRIST_FLEX_IDX = 3
F_SCOUT = 0
F_HOVER = 20
F_PRESS_PEAK = 26
F_RETRACT = 29


def _finite_vec(xs):
    return all(math.isfinite(float(x)) for x in xs)


def check_demo_episode(frames, tap_len=TAP_FRAMES):
    reasons = []
    if not frames:
        return False, ["empty episode"]
    if len(frames) != tap_len:
        reasons.append(f"length {len(frames)} != tap_len={tap_len}")

    required = ("observation.images.front", "observation.state", "action", "task")
    for i, rec in enumerate(frames):
        missing = [k for k in required if k not in rec]
        if missing:
            reasons.append(f"frame {i}: missing keys {missing}")
            return False, reasons
        if len(rec["observation.state"]) != 6:
            reasons.append(f"frame {i}: state length != 6")
        if len(rec["action"]) != 6:
            reasons.append(f"frame {i}: action length != 6")
        if not _finite_vec(rec["observation.state"]):
            reasons.append(f"frame {i}: non-finite state")
        if not _finite_vec(rec["action"]):
            reasons.append(f"frame {i}: non-finite action")

    if len(frames) == tap_len:
        scout_action = frames[F_SCOUT]["action"]
        scout_state = frames[F_SCOUT]["observation.state"]
        scout_err = max(abs(float(a) - float(s))
                        for a, s in zip(scout_action, scout_state))
        if scout_err > 0.5:
            reasons.append(f"scout action != state (max diff {scout_err:.2f})")

        wf = [float(rec["action"][WRIST_FLEX_IDX]) for rec in frames]
        hover = wf[F_HOVER]
        press = wf[F_PRESS_PEAK]
        retract = wf[F_RETRACT]
        if press < hover - 0.5:
            reasons.append(
                f"press wf={press:+.1f} not >= hover wf={hover:+.1f}")
        if retract > press + 0.5:
            reasons.append(
                f"retract wf={retract:+.1f} not <= press wf={press:+.1f}")

    return len(reasons) == 0, reasons
