# Portions adapted from LeRobot (Apache-2.0), lerobot==0.6.1:
#   lerobot/policies/rtc/modeling_rtc.py  RTCProcessor.denoise_step,
#   RTCProcessor.get_prefix_weights and its helpers.
#   Copyright 2025 The HuggingFace Inc. team.
#   Licensed under the Apache License, Version 2.0; see THIRD_PARTY_LICENSES.md.
"""Numpy port of LeRobot 0.6.1's RTC guidance step, for per-step ONNX loops.

The torch runtimes (SmolVLA native) call LeRobot's ``RTCProcessor`` directly.
The pi0.5 per-step ONNX expert drives its Euler loop in numpy and cannot hand
LeRobot a torch callable, so this module reproduces ``RTCProcessor.denoise_step``
on numpy arrays. ``tests/test_rtc_guidance_oracle.py`` checks it against
LeRobot on matched seeds.

Behaviour matches LeRobot, not the RTC paper. The paper (arXiv 2506.07339,
Eq. 2, Algorithm 1) and its reference code (Physical-Intelligence/
real-time-chunking-kinetix ``src/model.py`` L241-246) multiply the weighted
prefix error by the denoiser's vector-Jacobian product. LeRobot 0.6.1 evaluates
the denoiser before ``x_t`` joins the autograd graph (``modeling_rtc.py``
L213-219), so its Jacobian is the identity and the correction equals the
weighted error. That is what this port does. It needs per-step velocity
access only, not a differentiable denoiser. See
``docs/research/rtc-guidance-adr-20261027.md`` for the difference and its
measured size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SCHEDULES = ("ZEROS", "ONES", "LINEAR", "EXP")


@dataclass(frozen=True)
class RtcGuidance:
    """Guidance inputs for one chunk.

    ``prev_chunk_left_over`` is the unexecuted tail of the previous chunk in the
    policy's normalized action space, shape ``(T_prev, A_prev)``; ``None`` means
    there is nothing to guide toward and the chunk is sampled unguided.
    """

    inference_delay: int
    execution_horizon: int
    prev_chunk_left_over: np.ndarray | None
    schedule: str = "LINEAR"
    max_guidance_weight: float = 10.0

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.inference_delay < 0 or self.execution_horizon < 0:
            raise ValueError("inference_delay and execution_horizon must be >= 0")
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
        if self.prev_chunk_left_over is not None and self.prev_chunk_left_over.ndim != 2:
            raise ValueError("prev_chunk_left_over must be 2-D (T, A)")


def prefix_weights(start: int, end: int, total: int, schedule: str) -> np.ndarray:
    """Port of ``RTCProcessor.get_prefix_weights`` (float32, shape ``(total,)``)."""
    start = min(start, end)
    if schedule == "ZEROS":
        w = np.zeros(total, dtype=np.float32)
        w[:start] = 1.0
        return w
    if schedule == "ONES":
        w = np.ones(total, dtype=np.float32)
        w[end:] = 0.0
        return w
    if schedule not in ("LINEAR", "EXP"):
        raise ValueError(f"unknown schedule {schedule!r}")
    steps = total - max(total - end, 0) - start
    lin: np.ndarray
    if end <= start or steps <= 0:
        lin = np.zeros(0, dtype=np.float32)
    else:
        # torch.linspace(1, 0, steps + 2)[1:-1], float32. Computed in float64
        # then rounded; the oracle test bounds the difference from torch's
        # float32 linspace.
        lin = np.linspace(1.0, 0.0, steps + 2, dtype=np.float64)[1:-1].astype(np.float32)
    if schedule == "EXP":
        lin = (lin * (np.expm1(lin) / np.float32(math.e - 1))).astype(np.float32)
    w = np.concatenate([lin, np.zeros(max(total - end, 0), dtype=np.float32)])
    return np.concatenate([np.ones(min(start, total), dtype=np.float32), w])


def guidance_weight(time: float, max_guidance_weight: float) -> np.float32:
    """LeRobot's clipped weight min(beta, c * inv_r2) with tau = 1 - time."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.float32(1.0 - time)
        one_minus = np.float32(1.0) - tau
        inv_r2 = (one_minus**2 + tau**2) / (one_minus**2)
        c = np.nan_to_num(one_minus / tau, posinf=max_guidance_weight)
        w = np.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    return np.float32(min(w, np.float32(max_guidance_weight)))


def guide_velocity(
    x_t: np.ndarray,
    v_t: np.ndarray,
    time: float,
    guidance: RtcGuidance,
) -> np.ndarray:
    """Return LeRobot's guided velocity ``v_t - w * err`` for one Euler step.

    ``x_t``/``v_t``: ``(B, T, A)`` float32, LeRobot time convention
    (time 1 = noise, 0 = actions). ``err = (Y - (x_t - time * v_t)) * W`` with
    the leftover ``Y`` right-padded with zeros to ``(T, A)``, as LeRobot pads.
    """
    prev = guidance.prev_chunk_left_over
    if prev is None:
        return v_t
    _, total, action_dim = x_t.shape
    horizon = min(guidance.execution_horizon, prev.shape[0])
    y = np.zeros((total, action_dim), dtype=np.float32)
    rows, cols = min(prev.shape[0], total), min(prev.shape[1], action_dim)
    y[:rows, :cols] = prev[:rows, :cols]
    w = prefix_weights(guidance.inference_delay, horizon, total, guidance.schedule)
    x1 = x_t - np.float32(time) * v_t
    err = (y[None] - x1) * w[None, :, None]
    return (v_t - guidance_weight(time, guidance.max_guidance_weight) * err).astype(np.float32)


__all__ = ["RtcGuidance", "SCHEDULES", "guidance_weight", "guide_velocity", "prefix_weights"]
