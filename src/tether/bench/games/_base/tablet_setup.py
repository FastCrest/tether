# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Bring the tablet's circle game to a known good state.

Steps:
  1. adb reverse the touch_server port back to the Pi
  2. Launch the circle game URL in the tablet browser
  3. Issue N taps via `adb shell input tap` to:
       - dismiss any "tap to wake / dismiss notification" overlay
       - satisfy the user-gesture requirement for requestFullscreen()
       - re-trigger fullscreen if it was lost
  4. Verify /circle/target returns valid JSON

Usage (from another script):
    from games.tablet_setup import ensure_circle_fullscreen
    ensure_circle_fullscreen(port=8085)
"""
import json
import subprocess
import time
import urllib.request
from typing import Optional

DEFAULT_TAPS = 3
TAP_X, TAP_Y = 400, 640      # middle of an 800x1280 screen
TAP_GAP_S = 0.6
SETTLE_S = 1.5


def _adb(*args, capture=True):
    res = subprocess.run(["adb"] + list(args),
                         capture_output=capture, text=True)
    return res


def adb_tap(x: int = TAP_X, y: int = TAP_Y):
    _adb("shell", "input", "tap", str(x), str(y))


def adb_keyevent(code: int):
    _adb("shell", "input", "keyevent", str(code))


def adb_launch_url(url: str):
    # KILL any existing browser tabs first. Cloud9 leaks tabs on every
    # URL launch — over a session we accumulate dozens, each running
    # its own setInterval and POSTing /update_target. Multiple tabs
    # racing meant the wrong target value won every time.
    _adb("shell", "am", "force-stop", "com.amazon.cloud9")
    time.sleep(0.5)
    # Escape & for adb shell (it parses & as command separator).
    safe_url = url.replace("&", "\\&")
    _adb("shell", "am", "start",
         "-a", "android.intent.action.VIEW",
         "-d", safe_url)


def adb_reverse(port: int):
    _adb("reverse", f"tcp:{port}", f"tcp:{port}")


def _check_screen_size(port: int) -> tuple[int, int] | None:
    """Read latest browser-reported canvas size from data/touches.jsonl.

    Returns (sw, sh) or None if no recent entry. We use this to verify
    fullscreen actually engaged (sh==962) vs partial (sh==777 = chrome bar).
    """
    try:
        from pathlib import Path
        path = Path("data/touches.jsonl")
        if not path.exists():
            return None
        with path.open() as f:
            lines = f.readlines()
        for line in reversed(lines[-20:]):
            try:
                d = json.loads(line)
                sw = d.get("screen_w"); sh = d.get("screen_h")
                if sw and sh:
                    return (int(sw), int(sh))
            except Exception:
                continue
    except Exception:
        return None
    return None


def _check_target(port: int, path: str = "/circle/target",
                  timeout_s: float = 2.0) -> Optional[dict]:
    url = f"http://localhost:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            d = json.loads(r.read())
            if isinstance(d, dict) and "x" in d:
                return d
    except Exception:
        return None
    return None


def ensure_circle_fullscreen(port: int = 8085, taps: int = DEFAULT_TAPS,
                             relaunch: bool = True, verbose: bool = True,
                             npos: int = 0):
    """Make sure the circle game is foreground + fullscreen on the tablet.

    Idempotent — safe to call before every collection / eval run.
    Returns True on success, False if /circle/target never responded.
    """
    def _say(msg):
        if verbose:
            print(f"[tablet_setup] {msg}", flush=True)

    adb_reverse(port)
    if relaunch:
        # Cache-busting nonce so the browser fetches fresh circle.html
        # every launch. Without this, tablet's Cloud9 caches old JS and
        # we see stale behavior (phantom circles, wrong respawns).
        nonce = int(time.time())
        url = f"http://localhost:{port}/circle?nonce={nonce}"
        if npos > 0:
            url += f"&npos={npos}"
        _say(f"launching {url}")
        adb_launch_url(url)
        time.sleep(SETTLE_S)

    # NOTE: do NOT tap or F11 here. Fullscreen engagement must be the LAST
    # thing done before the round loop starts — otherwise downstream setup
    # (camera, dataset, listener, etc.) can disturb it. Callers (collect /
    # eval) should do their own 3-tap fullscreen at the very end of setup.

    # Verify endpoint
    for attempt in range(5):
        d = _check_target(port)
        if d:
            _say(f"target endpoint OK: x={d.get('x'):.0f} y={d.get('y'):.0f} r={d.get('r')}")
            return True
        time.sleep(0.5)
    _say("WARNING: /circle/target never responded with valid JSON")
    return False


def request_respawn(port: int = 8085, npos: int = 0,
                    draw_timeout_s: float = 2.5,
                    require_drawn: bool = True) -> Optional[dict]:
    """POST /circle/respawn and BLOCK until the JS browser confirms it
    has drawn the new token (handshake). Returns the new target dict.

    The handshake closes the JS-poll-lag race that otherwise left the
    first ~5 frames of every round capturing a screen with no circle
    drawn yet. Returns None on POST failure or if draw was not confirmed
    within draw_timeout_s.
    """
    body = json.dumps({"npos": npos}).encode()
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(
                f"http://localhost:{port}/circle/respawn",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"),
            timeout=2).read()
        data = json.loads(resp)
    except Exception as e:
        print(f"[tablet_setup] respawn POST failed: {e}", flush=True)
        return None
    new_token = int(data.get("token", 0))
    # Block until /circle/target says drawn_token >= new_token.
    deadline = time.time() + draw_timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://localhost:{port}/circle/target", timeout=1) as r:
                t = json.loads(r.read())
            if (int(t.get("drawn_token", 0)) >= new_token
                    and bool(t.get("visible", False))):
                return {
                    "x": data["x"],
                    "y": data["y"],
                    "r": data.get("r", t.get("r", 120)),
                    "token": new_token,
                    "color": data.get("color"),
                    "shape": data.get("shape"),
                }
        except Exception:
            pass
        time.sleep(0.05)
    print(f"[tablet_setup] WARN: draw confirmation timeout for token {new_token}", flush=True)
    if require_drawn:
        return None
    return {
        "x": data["x"],
        "y": data["y"],
        "r": data.get("r", 120),
        "token": new_token,
        "color": data.get("color"),
        "shape": data.get("shape"),
    }


def reset(port: int = 8085, npos: int = 0, verbose: bool = True
          ) -> Optional[dict]:
    """One-shot 'put the tablet in a known good state for the next run.'

    Steps (atomically):
      1. adb reverse the touch_server port
      2. force-stop + relaunch Cloud9 with cache-busting URL
      3. verify /circle/target endpoint responds
      4. POST /circle/respawn so the FIRST circle is on screen before
         the caller starts capturing frames

    Returns the initial target dict, or None if any step failed. Callers
    should treat None as "skip this run / abort".
    """
    if not ensure_circle_fullscreen(port=port, npos=npos, verbose=verbose):
        return None
    target = request_respawn(port=port, npos=npos)
    if target is None:
        return None
    if verbose:
        print(f"[tablet_setup] reset OK — initial target "
              f"({target['x']:.0f},{target['y']:.0f})", flush=True)
    return target


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--taps", type=int, default=DEFAULT_TAPS)
    p.add_argument("--no-relaunch", action="store_true")
    p.add_argument("--reset", action="store_true",
                   help="Full reset incl. initial respawn")
    p.add_argument("--npos", type=int, default=0)
    args = p.parse_args()
    if args.reset:
        ok = reset(port=args.port, npos=args.npos) is not None
    else:
        ok = ensure_circle_fullscreen(port=args.port, taps=args.taps,
                                      relaunch=not args.no_relaunch)
    raise SystemExit(0 if ok else 1)
