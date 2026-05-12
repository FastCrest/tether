# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Circle tap game — server picks targets, JS just renders.

Spawning logic lives HERE (not in JS) so:
  - There's a single source of truth for target_x/y
  - Multiple browser tabs can't race / pick different randoms
  - The collector can request a respawn and immediately know the new
    position synchronously (no JS poll round-trip)
  - The npos curriculum mode is enforced server-side regardless of what
    the URL says

The JS:
  - Polls /circle/target every 200ms and renders the current target
    (or hides it if visible=False)
  - On tap: POSTs /circle/log_tap (informational); does NOT advance
    the target
"""
import math
import random
import json
from pathlib import Path
from threading import Lock

from reflex.bench.games._base.base_game import BaseGame


class CircleGame(BaseGame):
    name = "circle"
    html_file = "circle.html"

    # canvas (browser space), not device pixels
    CANVAS_W = 602
    CANVAS_H = 962
    # Robot-perspective left/right task: shoulder_pan should change while
    # shoulder_lift stays roughly fixed. With the current tablet orientation,
    # that means the two dots are separated mostly along screen y, not screen x.
    LR_LEFT = (340.0, 160.0)
    LR_RIGHT = (340.0, 800.0)

    def __init__(self, games_dir):
        super().__init__(games_dir)
        self.state_lock = Lock()
        # Start with NO target visible. Collector POSTs /circle/respawn
        # to spawn the first one. Avoids the "stale circle on page load"
        # race that bit us before.
        self.target_x = 0
        self.target_y = 0
        self.target_r = 120         # current rendered radius (per-target)
        self.base_r = 120           # default radius for FWD positions
        self.target_color = "#00FF00"  # default green; some npos modes
                                       # set per-target colors (e.g. 34
                                       # uses red for FWD, blue for BACK).
        self.target_shape = "circle"   # "circle" or "square"; npos=36
                                       # sets per-position shapes.
        self.target_visible = False
        # Curriculum mode: 0 = full random, 1 = single fixed center,
        # 2 = left/right at mid-y, 4 = four corners.
        # Set via /circle/respawn?npos=N or default from URL.
        self.npos = 0
        self.respawn_token = 0
        # Highest token JS has confirmed it actually DREW. Lets the
        # collector/eval block until "the circle is on the screen", not
        # just "the server thinks it's visible". Closes the JS-poll-lag
        # race that hid the dot on the first few frames of every round.
        self.drawn_token = 0
        self._rng = random.Random()
        self.reachable_box = self._load_reachable_box()

    def _load_reachable_box(self):
        path = Path(__file__).resolve().parent.parent / "data" / "last_wrist_grid_samples.json"
        try:
            payload = json.loads(path.read_text())
            pts = [
                (float(s["touch_x"]), float(s["touch_y"]))
                for s in payload.get("samples", [])
                if s.get("clean", True)
            ]
        except Exception:
            pts = []
        if not pts:
            return (160.0, 80.0, 600.0, 960.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (max(0.0, min(xs)), max(0.0, min(ys)),
                min(float(self.CANVAS_W), max(xs)), min(float(self.CANVAS_H), max(ys)))

    # ---- Spawn logic (server-side) ----------------------------------

    def _spawn_random(self) -> tuple[float, float]:
        """Pick the next target position based on current npos mode."""
        if self.npos == 1:
            x0, y0, x1, y1 = self.reachable_box
            return (x0 + x1) / 2, (y0 + y1) / 2
        if self.npos == 2:
            # Left + right at mid-y (tests pan / wrist motion).
            return self.LR_LEFT if self._rng.random() < 0.5 else self.LR_RIGHT
        if self.npos == 21:
            # Left only — for fast iteration on the harder direction.
            return self.LR_LEFT
        if self.npos == 22:
            # Right only.
            return self.LR_RIGHT
        if self.npos == 23:
            # Strict alternating left ↔ right (deterministic, even split).
            # Use respawn_token parity as the alternator so we get
            # exactly N/2 of each side in N rounds.
            return self.LR_LEFT if (self.respawn_token % 2 == 0) else self.LR_RIGHT
        if self.npos == 3:
            # Forward + back at canvas (350, 240) and (350, 720).
            # Symmetric around canvas center (240+720)/2 = 480 ≈ canvas
            # center y=481. Cal: pan=1918 vs 2251 (diff 333), lift
            # constant ~2260, contact_wf reliable at both points.
            # NOTE: previous attempt at (301, 281)/(301, 681) had
            # unreliable cal contact_wf — every FWD missed contact.
            return (350.0, 240.0) if self._rng.random() < 0.5 else (350.0, 720.0)
        if self.npos == 31:
            # Forward only — for fast iteration on a single mode.
            return (350.0, 240.0)
        if self.npos == 32:
            # Back only.
            return (350.0, 720.0)
        if self.npos == 33:
            # Strict alternating fwd ↔ back (deterministic, even split).
            # Mirrors npos=23 for the LR task. Used for clean collection.
            return (350.0, 240.0) if (self.respawn_token % 2 == 0) else (350.0, 720.0)
        if self.npos == 34:
            # Same fwd/back positions as npos=33 BUT color-coded:
            # FWD = RED, BACK = BLUE. Tests if a strong visual feature
            # (color) helps the model discriminate vs position alone.
            # Color is set in _color_for() based on (x, y).
            return (350.0, 240.0) if (self.respawn_token % 2 == 0) else (350.0, 720.0)
        if self.npos == 35:
            # OCCLUSION TEST: alternating fwd/back like 33, BUT
            # the BACK circle is rendered BLACK (invisible). Tests if
            # the v54 LR-collapse was caused by BACK-mode invisibility
            # (gripper occlusion). Expected: model collapses to FWD-only
            # like the v54 LR task did, even though pan/lift/cwf cal
            # are reliable for both positions.
            return (350.0, 240.0) if (self.respawn_token % 2 == 0) else (350.0, 720.0)
        if self.npos in (36, 37, 38, 39, 40):
            # SHAPE/SIZE GRADIENT TESTS: same fwd/back positions as
            # npos=33, but the BACK target is rendered differently.
            # 36 = BACK is square (shape vs circle for FWD)
            # 37-40 = BACK is smaller circle (radius 80, 50, 30, 15)
            # All test where ACT'\''s discrimination breaks down.
            return (350.0, 240.0) if (self.respawn_token % 2 == 0) else (350.0, 720.0)
        if self.npos == 4:
            return self._rng.choice([(220, 240), (480, 240),
                                     (220, 720), (480, 720)])
        if self.npos == 44:
            # Strict alternating 4-quadrant cycle (deterministic).
            # Cycle: TL(220,240) → TR(480,240) → BL(220,720) → BR(480,720)
            quad = self.respawn_token % 4
            return [(220.0, 240.0), (480.0, 240.0),
                    (220.0, 720.0), (480.0, 720.0)][quad]
        # Full random inside the empirically observed gentle-touch box.
        x0, y0, x1, y1 = self.reachable_box
        margin = min(80.0, max(20.0, self.base_r * 0.55))
        return (
            self._rng.uniform(x0 + margin, x1 - margin),
            self._rng.uniform(y0 + margin, y1 - margin),
        )

    def _color_for(self, x: float, y: float) -> str:
        """Color for the dot at (x, y) given the current npos. Most
        modes are green; npos=34 uses red/blue per position."""
        if self.npos == 34:
            # Red for FWD (canvas y < center), Blue for BACK
            return "#FF0000" if y < 480 else "#0000FF"
        if self.npos == 35:
            # OCCLUSION TEST: FWD green visible, BACK rendered as
            # canvas-background black (invisible to camera).
            return "#00FF00" if y < 480 else "#111111"
        return "#00FF00"

    def _shape_for(self, x: float, y: float) -> str:
        """Shape for the target. Most modes circle; npos=36 makes BACK
        a square."""
        if self.npos == 36 and y >= 480:
            return "square"
        return "circle"

    def _radius_for(self, x: float, y: float) -> int:
        """Per-target radius. npos=37-40 shrink the BACK target to
        find the size threshold for ACT discrimination collapse."""
        if y < 480:
            return self.base_r  # FWD always full size (120)
        # BACK position
        if self.npos == 37:
            return 80
        if self.npos == 38:
            return 50
        if self.npos == 39:
            return 30
        if self.npos == 40:
            return 15
        return self.base_r

    # ---- HTTP -------------------------------------------------------

    def _target_payload(self):
        with self.state_lock:
            return {
                "x": self.target_x,
                "y": self.target_y,
                "r": self.target_r,
                "color": self.target_color,
                "shape": self.target_shape,
                "visible": self.target_visible,
                "token": self.respawn_token,
                "drawn_token": self.drawn_token,
            }

    def handle_get(self, path, server):
        if path in ("/target", "/circle/target"):
            server._serve_json(self._target_payload())
            return True
        return False

    def handle_post(self, path, data, server):
        # Script: "I'm at home, spawn the next circle now."
        # Optional: data={"npos": N} to switch curriculum.
        if path in ("/respawn", "/circle/respawn"):
            with self.state_lock:
                if "npos" in data:
                    self.npos = int(data["npos"])
                self.target_x, self.target_y = self._spawn_random()
                self.target_color = self._color_for(self.target_x,
                                                    self.target_y)
                self.target_shape = self._shape_for(self.target_x,
                                                    self.target_y)
                self.target_r = self._radius_for(self.target_x,
                                                 self.target_y)
                self.target_visible = True
                self.respawn_token += 1
                self.drawn_token = min(self.drawn_token, self.respawn_token - 1)
                tok = self.respawn_token
                tx, ty = self.target_x, self.target_y
                tc = self.target_color
                ts = self.target_shape
                tr = self.target_r
            print(f"[circle] /respawn → token={tok} target=({tx:.0f},{ty:.0f}) "
                  f"shape={ts} r={tr} color={tc} npos={self.npos}", flush=True)
            server._serve_json({"x": tx, "y": ty, "r": tr,
                                "color": tc, "shape": ts, "token": tok,
                                "visible": True})
            return True
        # Tap log (informational; doesn't change state).
        if path in ("/tap", "/circle/tap", "/log", "/circle/log"):
            server._log_touch(data)
            return True
        # On tap, hide the circle so the user sees a visual confirmation
        # of contact. CRITICAL: tap_done MUST include the token of the
        # circle being tapped, and we only hide if it still matches the
        # current respawn token. Otherwise a delayed tap_done from a
        # previous round can hide the FRESHLY-respawned circle of the
        # NEXT round (the bug that broke every eval).
        if path in ("/tap_done", "/circle/tap_done"):
            posted = data.get("token") if isinstance(data, dict) else None
            with self.state_lock:
                if posted is None or int(posted) == self.respawn_token:
                    self.target_visible = False
                    stale = False
                else:
                    stale = True
            if stale:
                print(f"[circle] /tap_done IGNORED stale token "
                      f"{posted} (current={self.respawn_token})", flush=True)
            server._respond_ok()
            return True
        # Draw confirmation: JS POSTs after rendering a token. Collector
        # /eval can then block on drawn_token == respawn_token to know
        # the circle is actually on the screen.
        if path in ("/drawn", "/circle/drawn"):
            posted = data.get("token") if isinstance(data, dict) else None
            if posted is not None:
                with self.state_lock:
                    if int(posted) == self.respawn_token:
                        self.drawn_token = int(posted)
            server._respond_ok()
            return True
        return False
