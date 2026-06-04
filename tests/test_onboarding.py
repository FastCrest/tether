"""Tests for src/tether/onboarding.py — first-run consent + recurring prompts."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tether.onboarding import (
    ONBOARDING_VERSION,
    OnboardingState,
    get_onboarding_state,
    maybe_onboard,
    maybe_prompt_contribution,
    set_contribute_data,
    set_telemetry_enabled,
)


@pytest.fixture
def onboarding_path(tmp_path):
    """Temporary onboarding.json path."""
    return tmp_path / "onboarding.json"


# ── Test 1: First run creates state file ──────────────────────────────

def test_first_run_creates_state(onboarding_path):
    """First run in interactive mode creates the state file."""
    def prompt_fn(text, is_contribution):
        if is_contribution:
            return False
        return True  # accept telemetry

    state = maybe_onboard(
        path=onboarding_path, interactive=True, prompt_fn=prompt_fn,
    )
    assert state.completed_at is not None
    assert state.telemetry_enabled is True
    assert state.contribute_data is False
    assert onboarding_path.exists()


# ── Test 2: Non-interactive skips prompt ──────────────────────────────

def test_non_interactive_skips(onboarding_path):
    """Non-interactive mode returns defaults without saving."""
    state = maybe_onboard(path=onboarding_path, interactive=False)
    assert state.completed_at is None
    assert state.telemetry_enabled is True  # default


# ── Test 3: Idempotent after completion ───────────────────────────────

def test_idempotent_after_completion(onboarding_path):
    """Second call returns existing state without prompting."""
    def prompt_fn(text, is_contribution):
        return True

    state1 = maybe_onboard(
        path=onboarding_path, interactive=True, prompt_fn=prompt_fn,
    )
    state2 = maybe_onboard(
        path=onboarding_path, interactive=True, prompt_fn=prompt_fn,
    )
    assert state1.completed_at == state2.completed_at


# ── Test 4: Telemetry opt-out via prompt ──────────────────────────────

def test_telemetry_opt_out(onboarding_path):
    """User can opt out of telemetry at first run."""
    call_count = 0

    def prompt_fn(text, is_contribution):
        nonlocal call_count
        call_count += 1
        return False  # reject both telemetry and contribution

    state = maybe_onboard(
        path=onboarding_path, interactive=True, prompt_fn=prompt_fn,
    )
    assert state.telemetry_enabled is False
    assert state.contribute_data is False
    assert call_count == 2  # telemetry prompt + contribution prompt


# ── Test 5: Contribution opt-in via prompt ────────────────────────────

def test_contribution_opt_in(onboarding_path):
    """User can opt into data contribution at first run."""
    def prompt_fn(text, is_contribution):
        return True  # accept both

    state = maybe_onboard(
        path=onboarding_path, interactive=True, prompt_fn=prompt_fn,
    )
    assert state.telemetry_enabled is True
    assert state.contribute_data is True


# ── Test 6: Ctrl+C handling ───────────────────────────────────────────

def test_ctrlc_handling(onboarding_path):
    """Ctrl+C (None return) is treated as dismissal, saves defaults."""
    def prompt_fn(text, is_contribution):
        return None  # simulate Ctrl+C

    state = maybe_onboard(
        path=onboarding_path, interactive=True, prompt_fn=prompt_fn,
    )
    # Ctrl+C on first prompt: defaults saved, not completed
    assert state.completed_at is None
    assert state.telemetry_enabled is True  # default


# ── Test 7: State serialization round-trip ────────────────────────────

def test_state_round_trip():
    """OnboardingState serializes and deserializes correctly."""
    state = OnboardingState(
        onboarding_version=ONBOARDING_VERSION,
        completed_at="2026-05-04T12:00:00.000000Z",
        telemetry_enabled=False,
        contribute_data=True,
        prompts_dismissed=3,
        dont_ask_again=True,
        last_prompt_at=1234567890.0,
        last_version_prompted="0.8.0",
        episode_milestone_prompted=True,
    )
    d = state.to_dict()
    restored = OnboardingState.from_dict(d)
    assert restored.telemetry_enabled is False
    assert restored.contribute_data is True
    assert restored.prompts_dismissed == 3
    assert restored.dont_ask_again is True
    assert restored.episode_milestone_prompted is True


# ── Test 8: set_telemetry_enabled ─────────────────────────────────────

def test_set_telemetry_enabled(onboarding_path):
    """set_telemetry_enabled updates state file."""
    state = set_telemetry_enabled(False, path=onboarding_path)
    assert state.telemetry_enabled is False
    assert onboarding_path.exists()

    state = set_telemetry_enabled(True, path=onboarding_path)
    assert state.telemetry_enabled is True


# ── Test 9: set_contribute_data ───────────────────────────────────────

def test_set_contribute_data(onboarding_path):
    """set_contribute_data updates state file."""
    state = set_contribute_data(True, path=onboarding_path)
    assert state.contribute_data is True

    state = set_contribute_data(False, path=onboarding_path)
    assert state.contribute_data is False


# ── Test 10: Recurring prompt respects dont_ask_again ─────────────────

def test_recurring_prompt_respects_dont_ask_again(onboarding_path):
    """Recurring prompts are suppressed when dont_ask_again is True."""
    state = OnboardingState(
        completed_at="2026-05-04T12:00:00.000000Z",
        contribute_data=False,
        dont_ask_again=True,
    )
    onboarding_path.parent.mkdir(parents=True, exist_ok=True)
    onboarding_path.write_text(json.dumps(state.to_dict()))

    result = maybe_prompt_contribution(
        path=onboarding_path,
        trigger="monthly",
        interactive=True,
    )
    assert result is None  # no prompt shown


# ── Test 11: Recurring prompt fires on episode milestone ──────────────

def test_recurring_prompt_episode_milestone(onboarding_path):
    """Recurring prompt fires after 100 episodes."""
    state = OnboardingState(
        completed_at="2026-05-04T12:00:00.000000Z",
        contribute_data=False,
        dont_ask_again=False,
        episode_milestone_prompted=False,
    )
    onboarding_path.parent.mkdir(parents=True, exist_ok=True)
    onboarding_path.write_text(json.dumps(state.to_dict()))

    def prompt_fn(text, is_contribution):
        return False  # dismiss

    result = maybe_prompt_contribution(
        path=onboarding_path,
        trigger="episodes",
        episode_count=150,
        interactive=True,
        prompt_fn=prompt_fn,
    )
    assert result is not None
    assert result.prompts_dismissed == 1
    assert result.episode_milestone_prompted is True
