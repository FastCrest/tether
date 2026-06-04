"""Adapters that expose Tether's VLA inference to external evaluation harnesses.

Each adapter is a thin wrapper around :class:`tether.runtime.TetherServer` —
the real inference (VLM prefix + expert denoising + safety + deadline) lives
in TetherServer, and adapters only translate between observation/action
schemas. When a bug exists in the denoising loop or VLM wiring, it is fixed
in TetherServer, not in the adapters.

Available adapters:
    vla_eval — AllenAI's vla-evaluation-harness (LIBERO, SimplerEnv, ManiSkill)
"""
from __future__ import annotations

__all__: list[str] = []
