# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Single-script interactive calibration wizard for soarm-bench.

Self-contained: starts a tiny HTTP server on the Pi serving calibrate.html,
uses adb to launch the page in the tablet browser, reads the tablet's
screen resolution via `adb shell wm size`, computes the 4 corner positions
itself, and walks the user through 4 kinesthetic samples (move arm, ENTER).

  $ python -u -B calibration/calibrate_corners.py

You need:
  - SO-ARM connected on /dev/ttyACM0 (override with --port)
  - Tablet connected via USB with `adb` working
  - The 4 corner markers visible on the tablet after launch

Steps:
  1. Arm motor setup + home
  2. Tablet: serve calibrate.html, launch via adb, fullscreen, read resolution
  3. 4-corner kinesthetic sampling — wizard already knows each corner's
     pixel coords (computed from resolution + MARGIN), so user only has
     to move the gripper to the highlighted marker and press ENTER
  4. Save data/tablet_config.json
"""
import argparse
import http.server
import json
import queue
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# arm_hardware.py is at repo root; arm_utils.py lives in calibration/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))

CONFIG_PATH = ROOT / "data" / "tablet_config.json"
GAMES_DIR = ROOT / "games"
HTTP_PORT = 8186                        # avoid collision with the legacy
                                        # circle.py game server on 8085
MARGIN_PX = 80                          # margin from canvas edge for corners
                                        # (must match calibrate.html's MARGIN)
LOOSE_JOINTS = [1, 2, 4]                # pan, lift, wrist_flex
LOCKED_JOINTS = [3, 5, 6]               # elbow, wrist_roll, gripper

# Shared wrist/contact constants used by surface calibration and archived
# debug probes. Higher raw wf = wrist physically deeper into the surface.
VERIFY_RAISE_TICKS = 120        # lift above taught contact between taps
VERIFY_DEEPER_TICKS = 260       # how far past taught contact to search
VERIFY_MIN_SWEEP_TICKS = 320    # minimum local search span
VERIFY_STEP_TICKS = 10          # wf increment per tap iteration
VERIFY_COARSE_STEP_TICKS = 25   # first-pass wf increment before fine replay
VERIFY_ABSOLUTE_MIN_WF = 850    # hard-up transit height / lower wf clamp
VERIFY_ABSOLUTE_MAX_WF = 2300   # hard stop for verify sweeps
VERIFY_TAP_DWELL_S = 0.20       # how long to hold at target wf
VERIFY_LIFT_DWELL_S = 0.20      # how long to hold at raised wf
VERIFY_SETTLE_PAN_LIFT_S = 1.0  # initial pan/lift settle before tapping
VERIFY_XY_TOLERANCE_PX = 60     # raw touch tolerance around target marker

# Required precondition: lerobot calibration JSON must exist so that the
# arm's raw motor positions have consistent physical meaning. Without this,
# our SAFE_HOME values would not be safe.
LEROBOT_CAL_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/None.json"


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

def confirm(prompt, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        raw = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        # Non-interactive (no TTY) — fall back to default.
        print(f"{prompt} {suffix} <no tty, defaulting to {'yes' if default_yes else 'no'}>")
        return default_yes
    if not raw:
        return default_yes
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Arm helpers (kinesthetic)
# ---------------------------------------------------------------------------

def _set_torque(arm, enabled, mids=None):
    if mids is None:
        mids = [1, 2, 3, 4, 5, 6]
    for mid in mids:
        arm.ph.write1ByteTxRx(arm.port_h, mid, 40, 1 if enabled else 0)


def _read_pose(arm):
    out = {}
    for jid, name in [(1, "pan"), (2, "lift"), (3, "elbow"),
                      (4, "wrist_flex"), (5, "wrist_roll"), (6, "gripper")]:
        val, _, _ = arm.ph.read2ByteTxRx(arm.port_h, jid, 56)
        out[name] = val
    return out


def _format_pose_delta(actual, target):
    parts = []
    for name in ("pan", "lift", "wrist_flex"):
        if name in target:
            parts.append(f"{name}={actual[name]} ({actual[name]-target[name]:+d})")
        else:
            parts.append(f"{name}={actual[name]}")
    return " ".join(parts)


def _kinesthetic_sample(arm, prompt):
    _set_torque(arm, False, mids=LOOSE_JOINTS)
    print(f"\n  → {prompt}")
    print("    Pan / lift / wrist_flex are loose — move those by hand.")
    print("    DO NOT move the elbow (joint 3) — it must stay locked.")
    print("    Press ENTER when in position.")
    input()
    pose = _read_pose(arm)
    print(f"    recorded: pan={pose['pan']} lift={pose['lift']} "
          f"wrist_flex={pose['wrist_flex']}")
    name_by_jid = {1: "pan", 2: "lift", 4: "wrist_flex"}
    for jid in LOOSE_JOINTS:
        arm.ph.write2ByteTxRx(arm.port_h, jid, 42, pose[name_by_jid[jid]])
    _set_torque(arm, True, mids=LOOSE_JOINTS)
    return pose


# ---------------------------------------------------------------------------
# adb helpers
# ---------------------------------------------------------------------------

def _adb(*args, capture=True):
    return subprocess.run(["adb"] + list(args),
                          capture_output=capture, text=True)


def adb_reverse(port):
    _adb("reverse", f"tcp:{port}", f"tcp:{port}")


def adb_launch_url(url):
    # Force-stop existing browser tabs (Cloud9-specific; other browsers fall
    # through silently) so we don't get stale rendering.
    _adb("shell", "am", "force-stop", "com.amazon.cloud9", capture=False)
    time.sleep(0.4)
    safe_url = url.replace("&", "\\&")
    _adb("shell", "am", "start",
         "-a", "android.intent.action.VIEW",
         "-d", safe_url, capture=False)


def adb_tap(x, y):
    _adb("shell", "input", "tap", str(x), str(y), capture=False)


def adb_screen_size():
    """Return (width, height) of tablet display in physical pixels."""
    r = _adb("shell", "wm", "size")
    if r.returncode != 0 or not r.stdout:
        return None
    # "Physical size: 800x1280" or "Override size: 800x1280"
    for line in r.stdout.splitlines():
        if "size:" in line.lower():
            try:
                wxh = line.split(":")[-1].strip()
                w, h = wxh.split("x")
                return int(w), int(h)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Tiny HTTP server (background thread) to serve calibrate.html
# ---------------------------------------------------------------------------

# Thread-safe queues of events posted by the tablet (calibrate.html).
HIT_QUEUE: "queue.Queue[int]" = queue.Queue()
TOUCH_QUEUE: "queue.Queue[dict]" = queue.Queue()


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static file server + a single POST endpoint /hit that pushes a
    corner-number onto HIT_QUEUE. The wizard's step 3 polls this queue."""

    def log_message(self, *_a, **_k):
        pass  # silence

    def do_POST(self):
        if self.path == "/hit":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode())
                n = int(body.get("corner", 0))
            except Exception:
                n = 0
            if 1 <= n <= 4:
                HIT_QUEUE.put(n)
            self.send_response(204); self.end_headers()
            return
        if self.path == "/touch":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode())
                evt = {
                    "kind": str(body.get("kind", "")),
                    "x": float(body.get("x", -1)),
                    "y": float(body.get("y", -1)),
                    "corner": int(body.get("corner", 0)),
                    "distance": body.get("distance"),
                    "t": body.get("t"),
                    "received_s": time.time(),
                }
            except Exception:
                evt = None
            if evt is not None:
                TOUCH_QUEUE.put(evt)
            self.send_response(204); self.end_headers()
            return
        self.send_response(404); self.end_headers()


def _drain_queue(q):
    drained = []
    while not q.empty():
        try:
            drained.append(q.get_nowait())
        except queue.Empty:
            break
    return drained


def _drain_touch_events():
    return _drain_queue(TOUCH_QUEUE)


def _summarize_touch_events(events):
    touches = [e for e in events if e.get("x", -1) >= 0 and e.get("y", -1) >= 0]
    if not touches:
        return "no raw touch"
    last = touches[-1]
    dist = last.get("distance")
    dist_s = "?" if dist is None else f"{float(dist):.0f}px"
    return (f"{len(touches)} raw touch event(s), last kind={last.get('kind')} "
            f"xy=({last.get('x'):.0f},{last.get('y'):.0f}) "
            f"corner={last.get('corner')} dist={dist_s}")


def start_static_server(port=HTTP_PORT, directory=GAMES_DIR):
    handler = lambda *a, **kw: _Handler(*a, directory=str(directory), **kw)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), handler)
    except OSError as e:
        raise RuntimeError(
            f"Could not bind HTTP server to port {port}: {e}. "
            f"Pick another port with --http-port. The legacy auto_soarm "
            f"circle.py runs on 8085."
        ) from e
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def stop_static_server(httpd):
    if httpd is None:
        return
    try:
        httpd.shutdown()
        httpd.server_close()
    except Exception as e:
        print(f"WARNING: failed to stop HTTP server cleanly: {e}")


def cleanup_state(state, args=None):
    """Release process-owned resources without issuing movement commands."""
    if not state:
        return

    stop_static_server(state.pop("httpd", None))

    if args is not None and state.pop("adb_reversed", False):
        _adb("reverse", "--remove", f"tcp:{args.http_port}", capture=False)

    arm = state.pop("arm", None)
    if arm is not None:
        try:
            # Leave joints torqued at their current goals; just close serial.
            arm.close()
        except Exception as e:
            print(f"WARNING: failed to close arm serial port cleanly: {e}")


# ---------------------------------------------------------------------------
# Step 1 — arm
# ---------------------------------------------------------------------------

def step1_arm_setup(args):
    print("\n=== STEP 1: arm motor setup ===")

    # Precheck: lerobot calibration must have been done.
    if not LEROBOT_CAL_PATH.exists():
        print(f"ABORT: lerobot calibration file not found at:")
        print(f"  {LEROBOT_CAL_PATH}")
        print("Run `lerobot calibrate` first to write the homing offsets")
        print("for your specific arm. This wizard relies on raw motor")
        print("positions having consistent physical meaning, which is what")
        print("the lerobot calibration provides.")
        return None
    print(f"✓ lerobot calibration found at {LEROBOT_CAL_PATH}")

    import scservo_sdk as scs

    print(f"Opening serial port: {args.port}")
    ph_h = scs.PortHandler(args.port)
    if not ph_h.openPort():
        print(f"ABORT: could not open {args.port}")
        return None
    ph_h.setBaudRate(args.baud)
    from reflex.embodiments.so100.edge_runtime import make_packet_handler
    pkt = make_packet_handler(ph_h)

    print("Reading current motor positions (no torque change yet)…")
    pos_names = [(1, "pan"), (2, "lift"), (3, "elbow"),
                 (4, "wrist_flex"), (5, "wrist_roll"), (6, "gripper")]
    for jid, name in pos_names:
        val, comm, _ = pkt.read2ByteTxRx(ph_h, jid, 56)
        if comm != 0:
            print(f"  {name}: READ FAIL — arm offline?")
            ph_h.closePort()
            return None
        print(f"  {name:<10} (id={jid}): {val}")

    print()
    print("WARNING: this WILL move the arm to a centred home pose:")
    print("  pan ≈ 2082 (centre), lift ≈ 2089 (centre), wrist_flex = 1500 (raised)")
    print("These values rely on the lerobot calibration above being correct.")
    if not confirm("Proceed with setup_arm + home?", default_yes=False):
        ph_h.closePort()
        return None

    if args.dry_run:
        print("[dry-run] would: SOArmHardware(); arm.home()")
        ph_h.closePort()
        return {"dry_run": True}

    ph_h.closePort()
    from reflex.embodiments.so100.edge_runtime import SOArmHardware
    arm = SOArmHardware(port=args.port, baud=args.baud)
    arm.home()
    time.sleep(0.5)

    final = _read_pose(arm)
    print("\nPost-home pose:")
    for name in ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"):
        print(f"  {name:<10}: {final[name]}")

    if not confirm("Does the arm look like it's at HOME (gripper raised, "
                   "facing the canvas)?"):
        return None

    print("✓ Step 1 complete.")
    return {"arm": arm, "home_state": final}


# ---------------------------------------------------------------------------
# Step 2 — tablet (HTTP + adb)
# ---------------------------------------------------------------------------

def step2_tablet_launch(state, args):
    print("\n=== STEP 2: tablet (auto via adb) ===")

    # 2a. Make sure adb sees a device.
    r = _adb("devices")
    if "device\n" not in r.stdout and "device\r\n" not in r.stdout:
        print("ABORT: adb sees no devices. Is the tablet plugged in + authorized?")
        print(r.stdout)
        return None

    # 2b. Read screen size BEFORE launching (gives us authoritative resolution).
    size = adb_screen_size()
    if size is None:
        print("ABORT: could not read tablet screen size via `adb shell wm size`.")
        return None
    sw, sh = size
    print(f"Tablet screen size: {sw} × {sh}")

    # 2c. Start the static HTTP server on Pi.
    print(f"Starting HTTP server on Pi: 0.0.0.0:{args.http_port}")
    state["httpd"] = start_static_server(port=args.http_port,
                                         directory=GAMES_DIR)

    # 2d. Reverse the port so tablet can hit Pi's localhost.
    adb_reverse(args.http_port)
    state["adb_reversed"] = True

    # 2e. Launch URL.
    nonce = int(time.time())
    url = f"http://localhost:{args.http_port}/calibrate.html?nonce={nonce}"
    print(f"Launching on tablet: {url}")
    adb_launch_url(url)
    time.sleep(2.0)

    # 2f. 3-tap dance to engage fullscreen.
    print("Engaging fullscreen via 3 taps in the centre of the screen…")
    for _ in range(3):
        adb_tap(sw // 2, sh // 2)
        time.sleep(0.6)
    time.sleep(1.5)

    # 2g. Compute the 4 corner positions (these MUST match calibrate.html's MARGIN).
    corners = [
        (1, "top-left",     MARGIN_PX,      MARGIN_PX),
        (2, "top-right",    sw - MARGIN_PX, MARGIN_PX),
        (3, "bottom-right", sw - MARGIN_PX, sh - MARGIN_PX),
        (4, "bottom-left",  MARGIN_PX,      sh - MARGIN_PX),
    ]
    print("\nThe wizard will guide you through 4 corners with these pixel coords:")
    for n, name, x, y in corners:
        print(f"  {n}. {name:<14} ({x}, {y})")

    if not confirm("\nCan you see the 4 numbered green markers on the tablet?"):
        print("ABORT — make sure the calibrate.html page is visible + fullscreen.")
        return None

    state.update({"screen_size": (sw, sh), "corners_plan": corners})
    print("✓ Step 2 complete.")
    return state


# ---------------------------------------------------------------------------
# Step 3 — kinesthetic
# ---------------------------------------------------------------------------

def step3_corner_taps(state, args):
    """Wait for tablet hit events; record arm pose at each.

    Pan / lift / wrist_flex stay loose throughout — user moves smoothly
    between corners. No ENTER prompt — the tablet's touch event triggers
    pose capture.
    """
    print("\n=== STEP 3: 4-corner kinesthetic sampling (touch-driven) ===")
    arm = state.get("arm")
    if arm is None:
        print("ABORT: no arm in state.")
        return None

    plan = {n: (name, sx, sy) for n, name, sx, sy in state["corners_plan"]}

    print("\nMove the gripper around the tablet and TOUCH each numbered")
    print("green marker. The tablet flashes HIT/MISS. On HIT, the wizard")
    print("automatically records the arm pose.")
    print()
    print("Pan / lift / wrist_flex will be LOOSE for the whole step.")
    print("DO NOT move the elbow (joint 3) — it must stay locked.")
    print("Drain any pending hits from prior runs…")
    while not HIT_QUEUE.empty():
        try: HIT_QUEUE.get_nowait()
        except queue.Empty: break

    _set_torque(arm, False, mids=LOOSE_JOINTS)
    samples = {}
    try:
        while len(samples) < 4:
            print(f"  waiting for tap (got {len(samples)}/4 corners)…", flush=True)
            try:
                n = HIT_QUEUE.get(timeout=300)  # 5-min stall timeout
            except queue.Empty:
                print("ABORT: no hit received in 5 minutes — re-run.")
                return None
            if n in samples:
                # Tablet shouldn't re-fire on already-hit, but be safe.
                print(f"  (corner {n} already recorded, ignoring)")
                continue
            pose = _read_pose(arm)
            name, sx, sy = plan[n]
            samples[n] = {"corner": n, "name": name, "sx": sx, "sy": sy,
                          "pan": pose["pan"], "lift": pose["lift"],
                          "wrist_flex": pose["wrist_flex"]}
            print(f"  ✓ corner {n} ({name}): pan={pose['pan']} "
                  f"lift={pose['lift']} wrist_flex={pose['wrist_flex']}")
    finally:
        # Lock loose joints back at whatever pose the user left them in.
        final = _read_pose(arm)
        name_by_jid = {1: "pan", 2: "lift", 4: "wrist_flex"}
        for jid in LOOSE_JOINTS:
            arm.ph.write2ByteTxRx(arm.port_h, jid, 42, final[name_by_jid[jid]])
        _set_torque(arm, True, mids=LOOSE_JOINTS)

    ordered = [samples[n] for n in (1, 2, 3, 4)]
    state["corners"] = ordered
    pans = [s["pan"] for s in ordered]
    lifts = [s["lift"] for s in ordered]
    wfs = [s["wrist_flex"] for s in ordered]
    print(f"\n✓ pan range: {min(pans)} → {max(pans)}")
    print(f"  lift range: {min(lifts)} → {max(lifts)}")
    print(f"  wrist_flex range: {min(wfs)} → {max(wfs)}")
    return state


# ---------------------------------------------------------------------------
# Step 4 — save
# ---------------------------------------------------------------------------

def step5_save(state, args):
    print("\n=== STEP 4: save tablet_config.json ===")
    cfg = {
        "version": 1,
        "screen_size": state.get("screen_size"),
        "home_state": state.get("home_state", {}),
        "corners": state.get("corners", []),
        "notes": (
            "Calibration via 4-corner kinesthetic sampling. These corners "
            "define the tablet bounds used to constrain later surface/grid "
            "calibration; contact height and accurate tap mapping are handled "
            "by calibration/calibrate_surface.py."
        ),
    }
    if args.dry_run:
        print("[dry-run] would write:")
        print(json.dumps(cfg, indent=2))
        return state
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"✓ wrote {CONFIG_PATH}")
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = [
    (1, step1_arm_setup),
    (2, step2_tablet_launch),
    (3, step3_corner_taps),
    (4, step5_save),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0", help="Arm serial port")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--http-port", type=int, default=HTTP_PORT,
                   help="Pi-side HTTP server for tablet")
    p.add_argument("--step", type=int, default=1, help="Start at this step.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    state = {}
    completed = False
    try:
        for idx, fn in STEPS:
            if idx < args.step:
                continue
            if idx == 1:
                state = fn(args) or {}
                if not state:
                    return 1
            else:
                res = fn(state, args)
                if res is None:
                    return 1
                state = res
        completed = True
        print("\n🎉 Calibration complete.")
        return 0
    except KeyboardInterrupt:
        print("\nABORT: interrupted by operator.")
        return 130
    finally:
        cleanup_state(state, args)


if __name__ == "__main__":
    sys.exit(main())
