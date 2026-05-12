# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Tablet calibration loader for soarm-bench.

Consumes `data/tablet_config.json` produced by calibration/calibrate_corners.py. Interpolates
from canvas pixel coords (sx, sy) to motor units (pan, lift, wrist_flex)
using the 4 corner samples as tablet bounds. A separate contact-plane model,
when present, maps tablet pixels to motor pan/lift at a fixed contact wrist.

Usage:
    from calibration.model import TabletCal
    cal = TabletCal()                           # loads default config path
    pose = cal.pose_for(400, 600)               # → {"pan": .., "lift": .., "wrist_flex": ..}
    scout = cal.scout_pose()                    # → centre of canvas with wrist raised
    cal.in_bounds(400, 600)                     # bool — is this pixel reachable?

Bilinear model:
    u = normalized left→right position inside sampled rectangle
    v = normalized top→bottom position inside sampled rectangle

Legacy fallback pan_lift(u, v) =
      (1-u)(1-v) * top_left
    + u(1-v)     * top_right
    + uv         * bottom_right
    + (1-u)v     * bottom_left

The legacy fallback exists only for older configs. New configs should provide
config["contact_model"], fitted from raw tablet touch coordinates while the
arm is actually contacting the screen.
"""
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "data" / "tablet_config.json"
DEFAULT_WRIST_GRID_SAMPLES = ROOT / "data" / "last_wrist_grid_samples.json"

# Wrist raise above predicted contact wf for the scout (hover) pose.
# In raw motor units; LOWER raw = wrist higher physically.
SCOUT_WF_RAISE = 200
ROW_WF_WARN_TICKS = 150
HARD_UP_WF = 850
HOVER_MIN_WF = 825
HOVER_RAISE_TICKS = 180
GENTLE_CONTACT_BACKOFF_TICKS = 10
# Extra press depth past first-contact wrist. Without this, the recorded action
# only commands the shallowest contact pose; the arm reaches contact during
# transit but the steady-state command is too high — a model trained on those
# actions plays them back smoothly and never actually presses the screen.
PRESS_DEPTH_TICKS = 5


class TabletCal:
    def __init__(self, config_path=DEFAULT_CONFIG):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No tablet config at {path}. Run calibration/calibrate_corners.py first.")
        cfg = json.loads(path.read_text())
        self.config_path = path
        self.config = cfg
        self.screen_size = tuple(cfg.get("screen_size") or (0, 0))

        corners = cfg.get("corners") or []
        if len(corners) < 4:
            raise ValueError(f"Need 4 corner samples, got {len(corners)}.")
        by_corner = {int(c["corner"]): c for c in corners}
        missing = [n for n in (1, 2, 3, 4) if n not in by_corner]
        if missing:
            raise ValueError(f"Missing corner samples: {missing}")
        self.corners = [by_corner[n] for n in (1, 2, 3, 4)]

        # Bounding box of pixel coords actually sampled.
        sxs = [c["sx"] for c in self.corners]
        sys_ = [c["sy"] for c in self.corners]
        self.bbox = {"sx_min": min(sxs), "sx_max": max(sxs),
                     "sy_min": min(sys_), "sy_max": max(sys_)}
        self.surface_wrist_flex = self._load_surface_wrist_flex(cfg)
        self.contact_model = cfg.get("contact_model") or None

    def _load_surface_wrist_flex(self, cfg):
        surface = cfg.get("surface") or {}
        if surface.get("wrist_flex") is not None:
            return int(surface["wrist_flex"])

        verified = [
            int(c["surface_wrist_flex"])
            for c in self.corners
            if c.get("surface_wrist_flex") is not None
        ]
        if verified:
            return int(round(statistics.median(verified)))

        # Fallback for old configs. Use the shallowest saved contact rather
        # than averaging bad kinesthetic wrist samples; deeper bad samples can
        # drive the stylus hard into the screen.
        return min(int(c["wrist_flex"]) for c in self.corners)

    # -- Forward prediction ---------------------------------------------------

    def pose_for(self, sx, sy):
        """Predicted contact pose for a canvas pixel."""
        if self.contact_model:
            return self._contact_model_pose_for(sx, sy)
        return self._legacy_pose_for(sx, sy)

    def _contact_model_pose_for(self, sx, sy):
        model = self.contact_model
        coeffs = model.get("coefficients") or {}
        pan = coeffs.get("pan")
        lift = coeffs.get("lift")
        if pan is None or lift is None or len(pan) not in (3, 6) or len(lift) != len(pan):
            raise ValueError("Invalid contact_model coefficients.")
        x = float(sx)
        y = float(sy)
        if len(pan) == 6:
            v = [x, y, x * y, x ** 2, y ** 2, 1.0]
        else:
            v = [x, y, 1.0]
        surface = model.get("surface") or {}
        wrist_flex = int(surface.get("wrist_flex", self.surface_wrist_flex))
        return {
            "pan": int(round(sum(pan[i] * v[i] for i in range(len(v))))),
            "lift": int(round(sum(lift[i] * v[i] for i in range(len(v))))),
            "wrist_flex": wrist_flex,
        }

    def _legacy_pose_for(self, sx, sy):
        b = self.bbox
        w = b["sx_max"] - b["sx_min"]
        h = b["sy_max"] - b["sy_min"]
        if w == 0 or h == 0:
            raise ValueError("Invalid calibration bbox with zero width/height.")
        u = (float(sx) - b["sx_min"]) / w
        v = (float(sy) - b["sy_min"]) / h

        c1, c2, c3, c4 = self.corners

        def interp(name):
            return (
                (1 - u) * (1 - v) * c1[name] +
                u * (1 - v) * c2[name] +
                u * v * c3[name] +
                (1 - u) * v * c4[name]
            )

        return {
            "pan": int(round(interp("pan"))),
            "lift": int(round(interp("lift"))),
            "wrist_flex": self.surface_wrist_flex,
        }

    def scout_pose(self):
        """Start-of-episode hover pose: canvas centre with wrist raised."""
        cx = (self.bbox["sx_min"] + self.bbox["sx_max"]) / 2
        cy = (self.bbox["sy_min"] + self.bbox["sy_max"]) / 2
        contact = self.pose_for(cx, cy)
        return {
            "pan": contact["pan"],
            "lift": contact["lift"],
            # Lower raw = wrist higher → subtract to raise.
            "wrist_flex": contact["wrist_flex"] - SCOUT_WF_RAISE,
        }

    # -- Bounds checks --------------------------------------------------------

    def in_bounds(self, sx, sy, margin=0):
        b = self.bbox
        return (b["sx_min"] - margin <= sx <= b["sx_max"] + margin and
                b["sy_min"] - margin <= sy <= b["sy_max"] + margin)

    def fit_residuals(self):
        """Return per-corner residuals for the fitted model. Useful for
        debugging — should be ~0 for an ideal kinesthetic capture."""
        out = []
        for c in self.corners:
            pred = self.pose_for(c["sx"], c["sy"])
            out.append({
                "corner": c["corner"],
                "pan_err": pred["pan"] - c["pan"],
                "lift_err": pred["lift"] - c["lift"],
                "wf_err": self.surface_wrist_flex - c["wrist_flex"],
            })
        return out

    def consistency_report(self):
        """Return simple checks for physically suspicious corner samples."""
        c1, c2, c3, c4 = self.corners
        checks = [
            ("top_row_wf_delta", abs(c1["wrist_flex"] - c2["wrist_flex"]),
             c1["wrist_flex"], c2["wrist_flex"]),
            ("bottom_row_wf_delta", abs(c4["wrist_flex"] - c3["wrist_flex"]),
             c4["wrist_flex"], c3["wrist_flex"]),
            ("left_col_wf_delta", abs(c1["wrist_flex"] - c4["wrist_flex"]),
             c1["wrist_flex"], c4["wrist_flex"]),
            ("right_col_wf_delta", abs(c2["wrist_flex"] - c3["wrist_flex"]),
             c2["wrist_flex"], c3["wrist_flex"]),
        ]
        return [
            {
                "name": name,
                "delta": delta,
                "a": a,
                "b": b,
                "warn": delta > ROW_WF_WARN_TICKS,
            }
            for name, delta, a, b in checks
        ]


class WristGridCal:
    """Quadratic touch->motor model fitted from wrist-grid samples.

    This is the model collection should use. `TabletCal` remains the loader for
    the bootstrap corner config and for saved contact models once we promote the
    wrist-grid fit into tablet_config.json.
    """

    def __init__(self, samples_path=DEFAULT_WRIST_GRID_SAMPLES):
        path = Path(samples_path)
        payload = json.loads(path.read_text())
        self.samples_path = path
        self.samples = [
            s for s in payload.get("samples", [])
            if s.get("clean", True)
        ]
        if len(self.samples) < 6:
            raise RuntimeError(f"Need >=6 clean wrist-grid samples in {path}")
        self.coef_pan = self._fit("actual_pan")
        self.coef_lift = self._fit("actual_lift")
        self.coef_wf = self._fit_wrist_flex()
        xs = [float(s["touch_x"]) for s in self.samples]
        ys = [float(s["touch_y"]) for s in self.samples]
        self.box = {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        }

    @staticmethod
    def _features(x, y):
        x = float(x)
        y = float(y)
        return np.array([x, y, x * y, x ** 2, y ** 2, 1.0], dtype=float)

    def _fit(self, key):
        mat = np.array([
            self._features(s["touch_x"], s["touch_y"])
            for s in self.samples
        ])
        vals = np.array([float(s[key]) for s in self.samples])
        coef, *_ = np.linalg.lstsq(mat, vals, rcond=None)
        return coef

    def _gentle_wrist_value(self, sample):
        if sample.get("gentle_wrist_flex") is not None:
            return float(sample["gentle_wrist_flex"])

        candidates = []
        if sample.get("wf_command") is not None:
            candidates.append(float(sample["wf_command"]))
        if sample.get("actual_wrist_flex") is not None:
            candidates.append(float(sample["actual_wrist_flex"]))
        if not candidates:
            raise ValueError("wrist-grid sample has no wrist flex value")

        # Lower raw wrist_flex is physically higher. Use the shallowest observed
        # first-contact wrist and back off slightly so runtime taps are gentle.
        return max(HOVER_MIN_WF, min(candidates) - GENTLE_CONTACT_BACKOFF_TICKS)

    def _fit_wrist_flex(self):
        mat = np.array([
            self._features(s["touch_x"], s["touch_y"])
            for s in self.samples
        ])
        vals = np.array([self._gentle_wrist_value(s) for s in self.samples])
        coef, *_ = np.linalg.lstsq(mat, vals, rcond=None)
        return coef

    def pose_for(self, sx, sy):
        f = self._features(sx, sy)
        pan = int(round(float(np.dot(self.coef_pan, f))))
        lift = int(round(float(np.dot(self.coef_lift, f))))
        wf_first_contact = int(round(float(np.dot(self.coef_wf, f))))
        # Press deeper than first-contact so the recorded action actually
        # commands a press the model can later reproduce open-loop.
        wf = wf_first_contact + PRESS_DEPTH_TICKS
        # Lower raw wrist_flex means physically higher. Hover should be near
        # contact, not clamped all the way to HARD_UP; otherwise collection has
        # to traverse hundreds of wrist ticks during the short press window and
        # many taps never reach the surface.
        hover = int(round(max(HOVER_MIN_WF, wf_first_contact - HOVER_RAISE_TICKS)))
        return {"pan": pan, "lift": lift, "wrist_flex": wf, "hover_wf": hover}

    def contains(self, sx, sy, margin=0):
        return (
            self.box["x_min"] + margin <= sx <= self.box["x_max"] - margin
            and self.box["y_min"] + margin <= sy <= self.box["y_max"] - margin
        )


if __name__ == "__main__":
    # Self-test: load + show fit + sample a few points.
    cal = TabletCal()
    print(f"Loaded: {cal.config_path}")
    print(f"Screen size: {cal.screen_size}")
    print(f"Sampled bbox: sx={cal.bbox['sx_min']}..{cal.bbox['sx_max']}, "
          f"sy={cal.bbox['sy_min']}..{cal.bbox['sy_max']}")
    print(f"Surface wrist_flex: {cal.surface_wrist_flex}")
    print(f"\nPan/lift residuals (wf_err is vs separate surface model):")
    for r in cal.fit_residuals():
        print(f"  corner {r['corner']}: pan_err={r['pan_err']:+5d} "
              f"lift_err={r['lift_err']:+5d} wf_err={r['wf_err']:+5d}")
    print(f"\nScout pose: {cal.scout_pose()}")
    print(f"Centre pose: {cal.pose_for(400, 640)}")
    print("\nConsistency checks:")
    for chk in cal.consistency_report():
        marker = "WARN" if chk["warn"] else "ok"
        print(f"  {marker}: {chk['name']} delta={chk['delta']} "
              f"({chk['a']} vs {chk['b']})")
