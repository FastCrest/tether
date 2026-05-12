"""Canonical tablet circle-tap benchmark (left/right circle).

Per the upstream auto_soarm circle_lr experiment: the tablet displays a
circle on the left or right; the robot taps it with a capacitive stylus;
the bench scores hits + misses + per-trial reaction time.

Modules:
    circle         CircleGame HTML page + state
    circle_task    state-machine driver
    circle_collect data collection runner (records LeRobot dataset)
    circle_eval    real-arm eval runner with a trained policy
    demo_sanity    quick check that a recorded episode looks reasonable
"""
from __future__ import annotations

# Surface the public bits without importing the heavy modules at package load.
# Touch_server.py + circle_collect.py + circle_eval.py pull in cv2 and other
# heavy deps that callers may not need just to see the package.
