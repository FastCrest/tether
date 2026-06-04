"""Tests for the per-step expert + RTC config-time compatibility guard.

Validates the helper at ``rtc_adapter.assert_rtc_compatible_with_num_steps``
that rejects 1-NFE + RTC combinations at config-time. The math-level
singularity is documented in the research sidecar at
``reflex_context/features/03_export/per-step-expert-export_research.md``
Lens 2 FM-4/FM-6.

This test is part of gate 2 of the per-step expert ONNX export ship
sequence (gate 1 is local CPU smoke; gate 2 is this unit test; gate 3
onwards are Modal-paid).
"""
from __future__ import annotations

import pytest

from tether.runtime.rtc_adapter import assert_rtc_compatible_with_num_steps


class TestRtcCompatibleWithNumSteps:
    """The guard fires loud at config-time, not silently mid-inference."""

    def test_one_nfe_rejects(self):
        with pytest.raises(ValueError, match=r"num_steps=1"):
            assert_rtc_compatible_with_num_steps(1)

    def test_two_steps_passes(self):
        # tau=1-time at step 0 is 0, but step 1 has time=0.5 so tau=0.5.
        # Whether the singularity hits at step 0 of a 2-step run is open
        # (research sidecar honest unknown #5). For now, allow it — the
        # bench gate (artifact 4) will catch any runtime crash.
        assert_rtc_compatible_with_num_steps(2) is None

    def test_ten_steps_passes(self):
        assert_rtc_compatible_with_num_steps(10) is None

    def test_none_passes(self):
        # Unknown num_steps (e.g. monolithic exports without that field
        # in tether_config.json) bypasses the guard. Caller's responsibility
        # to supply None when num_steps is unknown rather than asserting
        # against a default.
        assert_rtc_compatible_with_num_steps(None) is None

    def test_error_message_includes_research_pointer(self):
        # The error message must point users at the research sidecar so
        # they understand WHY the failure mode exists, not just THAT it does.
        with pytest.raises(ValueError) as exc_info:
            assert_rtc_compatible_with_num_steps(1)
        assert "per-step-expert-export_research.md" in str(exc_info.value)
        assert "FM-4" in str(exc_info.value)

    def test_error_message_suggests_fix(self):
        # The error must suggest a concrete remediation per CLAUDE.md
        # "Each failure has an inline pip install remediation hint."
        with pytest.raises(ValueError) as exc_info:
            assert_rtc_compatible_with_num_steps(1)
        msg = str(exc_info.value)
        assert "num_steps >= 2" in msg or "disable RTC" in msg
