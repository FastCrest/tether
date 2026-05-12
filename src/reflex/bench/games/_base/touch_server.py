# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Local touch server for circle collection."""

import argparse
import json
import math
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from threading import Lock, Thread
from urllib.parse import parse_qs, urlparse

from reflex.bench.games.circle_lr.circle import CircleGame, SHARED_HEAD, TOUCH_HANDLER_JS

print = partial(print, flush=True)

ROOT = Path(__file__).parent
LOG_FILE = ROOT / "data" / "touches.jsonl"
GAMES_DIR = ROOT / "games"
GAMES = {"circle": CircleGame(GAMES_DIR)}
GAME_DISPATCH_ORDER = ["circle"]

latest_touch = None
latest_touch_lock = Lock()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def compute_trail(path):
    if len(path) < 2:
        return len(path), 0.0
    dx = path[-1]["x"] - path[0]["x"]
    dy = path[-1]["y"] - path[0]["y"]
    return len(path), math.sqrt(dx * dx + dy * dy)


def render_game_html(game):
    return game.html_template().replace("__SHARED_HEAD__", SHARED_HEAD).replace(
        "__TOUCH_HANDLER_JS__", TOUCH_HANDLER_JS
    )


class TouchHandler(BaseHTTPRequestHandler):
    def _serve_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _respond_ok(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def _serve_404(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        if path == "/":
            self._serve_html(render_game_html(GAMES["circle"]))
            return
        stripped = path.lstrip("/")
        if stripped in GAMES:
            self._serve_html(render_game_html(GAMES[stripped]))
            return
        if path == "/latest_touch":
            with latest_touch_lock:
                self._serve_json(latest_touch or {})
            return
        if path == "/touches":
            since = int(qs.get("since", [0])[0])
            entries = []
            if LOG_FILE.exists():
                with LOG_FILE.open() as f:
                    for i, line in enumerate(f):
                        if i >= since:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            self._serve_json({"since": since, "count": len(entries), "touches": entries})
            return

        for name in GAME_DISPATCH_ORDER:
            if GAMES[name].handle_get(path, self):
                return
        self._serve_404()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length)) if length else {}
        except json.JSONDecodeError:
            data = {}
        path = urlparse(self.path).path
        if path == "/log_touch":
            self._log_touch(data)
            return
        for name in GAME_DISPATCH_ORDER:
            if GAMES[name].handle_post(path, data, self):
                return
        self._serve_404()

    def _log_touch(self, data):
        global latest_touch
        server_time_s = time.time()
        client_time_s = data.get("timestamp", server_time_s)
        offset_s = server_time_s - client_time_s
        path = data.get("path", [])
        if not path and "touch_x" in data:
            path = [{"x": data["touch_x"], "y": data["touch_y"]}]
        for p in path:
            if "t" in p:
                p["t"] = p["t"] / 1000.0 + offset_s
        trail_n, trail_px = compute_trail(path)
        entry = dict(data)
        entry.update({
            "touch_x": data.get("touch_x", path[0]["x"] if path else 0),
            "touch_y": data.get("touch_y", path[0]["y"] if path else 0),
            "path": path,
            "trail_n": trail_n,
            "trail_px": round(trail_px, 1),
            "timestamp": server_time_s,
            "screen_w": data.get("screen_w", 0),
            "screen_h": data.get("screen_h", 0),
        })
        for game in GAMES.values():
            extra = game.on_touch(entry)
            if extra:
                entry.update(extra)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        with latest_touch_lock:
            latest_touch = entry
        label = "TRAIL" if trail_n > 2 or trail_px > 10.0 else "clean"
        print(f"  touch ({entry['touch_x']:.0f},{entry['touch_y']:.0f}) | "
              f"{label} pts={trail_n} trail={trail_px:.0f}px")
        self._respond_ok()

    def log_message(self, *_):
        pass


def start_background_server(port=8186):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", port), TouchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8186)
    args = parser.parse_args()
    server = start_background_server(args.port)
    print(f"Touch server at http://0.0.0.0:{args.port}/circle")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
