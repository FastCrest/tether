"""Shared action publisher for ROS2 robot adapters.

Converts reflex serve's action chunk [chunk_size, action_dim] into
robot-specific joint commands, handling replan logic (action deque)
and rate limiting.

Ported from FluxVLA operator patterns (Apache-2.0, LimX Dynamics).
"""
from __future__ import annotations

import collections
import time
from typing import Any

import numpy as np


class ActionPublisher:
    """Manages action chunk → per-step joint command publishing.

    Maintains a deque of pending actions from the last chunk. Each call
    to ``next_action()`` pops one action; when the deque is empty,
    ``needs_replan`` returns True.

    Args:
        action_dim: Robot action dimension (e.g. 7 for single-arm, 14 for bimanual).
        replan_steps: How many actions to execute before replanning.
            Typically 5-10 (shorter = more responsive, longer = smoother).
        control_hz: Control loop frequency. Used for rate limiting.
    """

    def __init__(
        self,
        action_dim: int = 7,
        replan_steps: int = 5,
        control_hz: float = 50.0,
    ) -> None:
        self.action_dim = action_dim
        self.replan_steps = replan_steps
        self.control_hz = control_hz
        self._action_deque: collections.deque = collections.deque()
        self._last_action_time: float = 0.0
        self._total_actions_published: int = 0

    def set_chunk(self, actions: np.ndarray) -> None:
        """Load a new action chunk from reflex serve.

        Args:
            actions: [chunk_size, action_dim] or [1, chunk_size, action_dim].
                Only the first ``replan_steps`` actions are used.
        """
        if actions.ndim == 3:
            actions = actions[0]  # drop batch dim
        # Trim to action_dim (reflex may pad to max_action_dim=32)
        actions = actions[:, :self.action_dim]
        self._action_deque.clear()
        for i in range(min(self.replan_steps, len(actions))):
            self._action_deque.append(actions[i])

    @property
    def needs_replan(self) -> bool:
        return len(self._action_deque) == 0

    def next_action(self) -> np.ndarray | None:
        """Pop the next action from the deque.

        Returns None if the deque is empty (needs_replan is True).
        Rate-limits to ``control_hz``.
        """
        if not self._action_deque:
            return None

        # Rate limit
        now = time.time()
        dt = now - self._last_action_time
        min_dt = 1.0 / self.control_hz
        if dt < min_dt:
            time.sleep(min_dt - dt)

        action = self._action_deque.popleft()
        self._last_action_time = time.time()
        self._total_actions_published += 1
        return action

    @property
    def pending_count(self) -> int:
        return len(self._action_deque)

    @property
    def total_published(self) -> int:
        return self._total_actions_published


__all__ = ["ActionPublisher"]
