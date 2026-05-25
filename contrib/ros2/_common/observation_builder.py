"""Shared observation builder for ROS2 robot adapters.

Handles sensor synchronization + observation dict construction
for calling reflex serve's /act endpoint.

Ported from FluxVLA aloha_operator.py + ur_operator.py common patterns
(Apache-2.0, LimX Dynamics).
"""
from __future__ import annotations

import collections
import time
from typing import Any

import numpy as np


class ObservationBuilder:
    """Synchronizes multi-sensor data into a single observation dict.

    Buffers incoming sensor readings (images, joint states, etc.) and
    constructs the observation dict by matching timestamps within a
    tolerance window.

    Args:
        image_keys: Names of camera topics to expect (e.g. ["cam_high", "cam_left_wrist"]).
        state_dim: Expected state vector dimension.
        buffer_size: Max readings to buffer per sensor before dropping old ones.
        sync_tolerance_ms: Max timestamp difference (ms) for sensor synchronization.
    """

    def __init__(
        self,
        image_keys: list[str],
        state_dim: int = 14,
        buffer_size: int = 30,
        sync_tolerance_ms: float = 50.0,
    ) -> None:
        self.image_keys = image_keys
        self.state_dim = state_dim
        self.sync_tolerance_ms = sync_tolerance_ms

        # Per-sensor buffers: deque of (timestamp_sec, data)
        self._image_buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=buffer_size) for k in image_keys
        }
        self._state_buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._last_obs: dict[str, Any] | None = None

    def push_image(self, key: str, image: np.ndarray, timestamp: float | None = None) -> None:
        """Buffer a new camera frame."""
        if key not in self._image_buffers:
            self._image_buffers[key] = collections.deque(maxlen=30)
        ts = timestamp if timestamp is not None else time.time()
        self._image_buffers[key].append((ts, image))

    def push_state(self, state: np.ndarray, timestamp: float | None = None) -> None:
        """Buffer a new joint state reading."""
        ts = timestamp if timestamp is not None else time.time()
        self._state_buffer.append((ts, state))

    def build(self, instruction: str = "") -> dict[str, Any] | None:
        """Build a synchronized observation dict.

        Returns None if any sensor is missing data. Otherwise returns a
        dict matching reflex serve's /act request schema.
        """
        # Check all sensors have data
        if not self._state_buffer:
            return None
        for key in self.image_keys:
            if key not in self._image_buffers or not self._image_buffers[key]:
                return None

        # Use the latest state as anchor timestamp
        anchor_ts, anchor_state = self._state_buffer[-1]

        # Find closest image for each camera within tolerance
        images = {}
        for key in self.image_keys:
            buf = self._image_buffers[key]
            best_img = None
            best_dt = float("inf")
            for ts, img in buf:
                dt = abs(ts - anchor_ts) * 1000  # ms
                if dt < best_dt:
                    best_dt = dt
                    best_img = img
            if best_img is None or best_dt > self.sync_tolerance_ms:
                return None
            images[key] = best_img

        obs: dict[str, Any] = {}
        for key, img in images.items():
            obs[f"observation.images.{key}"] = img
        obs["observation.state"] = anchor_state
        if instruction:
            obs["task"] = [instruction]

        self._last_obs = obs
        return obs

    @property
    def last_observation(self) -> dict[str, Any] | None:
        return self._last_obs


__all__ = ["ObservationBuilder"]
