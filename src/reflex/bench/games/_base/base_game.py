# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Shared game HTML helpers for the local touch server."""

from pathlib import Path


TOUCH_HANDLER_JS = """
let _touchPath = [];
let _touchStartTime = 0;

document.addEventListener('fullscreenchange', () => {
  fetch('/log_touch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      _fs_event: true,
      fullscreen: !!document.fullscreenElement,
      screen_w: window.innerWidth,
      screen_h: window.innerHeight,
      timestamp: Date.now() / 1000,
    })
  }).catch(()=>{});
});

function _initTouchHandler(canvas) {
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(()=>{});
    }
    const touch = e.touches[0];
    _touchStartTime = Date.now();
    _touchPath = [{x: touch.clientX, y: touch.clientY, t: _touchStartTime}];
    if (typeof onTouchTap === 'function') {
      onTouchTap(touch.clientX, touch.clientY);
    }
  }, {passive: false});

  canvas.addEventListener('touchmove', function(e) {
    e.preventDefault();
    const touch = e.touches[0];
    _touchPath.push({x: touch.clientX, y: touch.clientY, t: Date.now()});
    if (typeof onTouchMove === 'function') {
      onTouchMove(touch.clientX, touch.clientY);
    }
  }, {passive: false});

  canvas.addEventListener('touchend', function(e) {
    e.preventDefault();
    const first = _touchPath[0] || {x:0, y:0};
    const extras = (typeof getLogExtras === 'function') ? getLogExtras() : {};
    fetch('/log_touch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.assign({
        touch_x: first.x,
        touch_y: first.y,
        path: _touchPath,
        timestamp: Date.now() / 1000,
        screen_w: canvas.width,
        screen_h: canvas.height,
      }, extras))
    }).catch(()=>{});
    if (typeof onTouchEnd === 'function') {
      onTouchEnd(_touchPath);
    }
    _touchPath = [];
  }, {passive: false});
}
"""


SHARED_HEAD = """<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<style>
  * { margin: 0; padding: 0; }
  body { background: #111; width: 100vw; height: 100vh; overflow: hidden;
         touch-action: none; -webkit-user-select: none; user-select: none; }
  canvas { display: block; width: 100vw; height: 100vh; }
  #score { position: fixed; top: 10px; left: 10px; color: #888;
           font: 24px monospace; z-index: 10; display: none; }
  #feedback { position: fixed; top: 50%; left: 50%;
              transform: translate(-50%, -50%);
              color: #fff; font: 80px monospace; z-index: 10;
              opacity: 0; transition: opacity 0.3s; pointer-events: none; }
</style>"""


class BaseGame:
    name = ""
    html_file = ""

    def __init__(self, games_dir: Path):
        self.games_dir = games_dir

    def html_template(self) -> str:
        return (self.games_dir / self.html_file).read_text()

    def handle_get(self, path: str, server) -> bool:
        return False

    def handle_post(self, path: str, data: dict, server) -> bool:
        return False

    def on_touch(self, touch_data: dict) -> dict:
        return {}
