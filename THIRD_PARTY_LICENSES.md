# Third-party licenses

Reflex VLA bundles or vendors code from the following projects. Each
project's full license text is reproduced below in alphabetical order.

## auto_soarm (MIT)

Source: https://github.com/0o8o0-blip/auto_soarm
Vendored: 2026-05-06 (initial lift)
Scope: tablet-tap automation rig for SO-ARM 100 + Android tablet.

Lifted into:
- `src/reflex/embodiments/so100/calibration/` — calibration scripts +
  motion + tapper logic
- `src/reflex/embodiments/so100/edge_runtime.py` — no-torch arm driver
  (`arm_hardware.py`)
- `src/reflex/bench/games/_base/` — generic game scaffold
  (`base_game.py`, `tablet_setup.py`, `circle_runtime.py`,
  `touch_server.py`)
- `src/reflex/bench/games/circle_lr/` — canonical circle-tap benchmark
- `examples/01_circle_tap_so100/` — end-to-end recipe (collect + train + eval)

Per-file headers identify ported files:
```
# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
```

Architectural decisions: `reflex_context/01_decisions/2026-05-06-vendor-auto-soarm.md`.

### MIT License

```
MIT License

Copyright (c) 2026 soarm-bench contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
