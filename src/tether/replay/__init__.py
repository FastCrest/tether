"""Replay subsystem — reads recorded JSONL traces and re-runs them against
a target model for regression testing + bug repro.

Schema is at TECHNICAL_PLAN.md §D.1. Writer lives at
`src/tether/runtime/record.py`. Readers per schema version live at
`replay/readers/v<N>.py` and dispatch by reading the first header line.
"""
from __future__ import annotations
