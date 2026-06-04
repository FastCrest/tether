"""First-run onboarding + recurring contribution prompts.

Fires on the very first CLI invocation (before any command) to explain:
1. Anonymous telemetry (opt-out, default ON): model name, hardware, latency
   stats, errors, episode counts. No images, no trajectories, no PII.
2. Episode data contribution (opt-in): full episode data anonymized locally
   before upload to improve Tether VLA models.

State persisted at ``~/.tether/onboarding.json`` with the following fields:
- onboarding_version (int): schema version for the onboarding state
- completed_at (str | None): ISO 8601 UTC timestamp of initial completion
- telemetry_enabled (bool): True by default (opt-out)
- contribute_data (bool): False by default (opt-in)
- prompts_dismissed (int): number of recurring prompts dismissed
- dont_ask_again (bool): permanent suppress for recurring prompts

Recurring prompts trigger when contribute_data is false AND dont_ask_again
is false, at these moments:
- After 100 episodes (flag from serve runtime)
- After successful finetune
- After version upgrade
- Monthly (30 days since last prompt or completion)

All prompts skip in non-interactive contexts (CI, pipes, daemon mode).
Ctrl+C safe: KeyboardInterrupt is caught and treated as dismissal.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bumped on breaking changes to onboarding semantics.
ONBOARDING_VERSION = 1

# State file path. Never synced to servers.
DEFAULT_ONBOARDING_PATH = "~/.tether/onboarding.json"

# Recurring prompt cooldown: 30 days (monthly).
_MONTHLY_COOLDOWN_S = 30 * 24 * 60 * 60

# Episode milestone for contribution prompt.
_EPISODE_MILESTONE = 100


@dataclass
class OnboardingState:
    """Persisted onboarding state. Mutable (updated on each prompt interaction)."""

    onboarding_version: int = ONBOARDING_VERSION
    completed_at: str | None = None
    telemetry_enabled: bool = True
    contribute_data: bool = False
    prompts_dismissed: int = 0
    dont_ask_again: bool = False
    last_prompt_at: float = 0.0
    last_version_prompted: str = ""
    episode_milestone_prompted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OnboardingState":
        return cls(
            onboarding_version=int(d.get("onboarding_version", ONBOARDING_VERSION)),
            completed_at=d.get("completed_at"),
            telemetry_enabled=bool(d.get("telemetry_enabled", True)),
            contribute_data=bool(d.get("contribute_data", False)),
            prompts_dismissed=int(d.get("prompts_dismissed", 0)),
            dont_ask_again=bool(d.get("dont_ask_again", False)),
            last_prompt_at=float(d.get("last_prompt_at", 0.0)),
            last_version_prompted=str(d.get("last_version_prompted", "")),
            episode_milestone_prompted=bool(d.get("episode_milestone_prompted", False)),
        )


def _is_interactive() -> bool:
    """True iff stdin and stdout are connected to a TTY."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_state(path: Path) -> OnboardingState | None:
    """Load existing onboarding state. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return OnboardingState.from_dict(data)
    except Exception as exc:
        logger.debug("Failed to load onboarding state: %s", exc)
        return None


def _save_state(state: OnboardingState, path: Path) -> None:
    """Atomically write onboarding state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.debug("Failed to save onboarding state: %s", exc)


def _welcome_prompt_text() -> str:
    """Text shown on very first run."""
    return """\
================================================================
  Welcome to Tether VLA
================================================================

Tether collects anonymous usage telemetry by default to help us
improve the project. This includes: model name, hardware info,
latency stats, error counts, and episode counts.

  NO images, trajectories, or personally identifiable info.

You can opt out any time:
  - Set TETHER_NO_TELEMETRY=1
  - Run: tether config set telemetry off

Separately, you can opt IN to contribute episode data (images,
states, actions) to help train better VLA models. All data is
anonymized locally before upload.

================================================================
"""


def _contribution_prompt_text() -> str:
    """Text shown for recurring contribution prompts."""
    return """\
================================================================
  Help improve Tether VLA models
================================================================

You can contribute anonymized episode data to help train better
VLA models. All data is anonymized locally before upload.

  - No raw images leave your machine without anonymization
  - You can review pending uploads: tether data review --pending
  - Revoke any time: tether data revoke

================================================================
"""


def _prompt_yes_no(text: str, default_no: bool = True) -> bool | None:
    """Prompt user for yes/no. Returns True/False, or None on Ctrl+C.

    Ctrl+C safe: catches KeyboardInterrupt and returns None.
    """
    try:
        print(text)
        suffix = "[y/N]" if default_no else "[Y/n]"
        response = input(f"  {suffix}: ").strip().lower()
        if default_no:
            return response in ("y", "yes")
        return response not in ("n", "no")
    except (KeyboardInterrupt, EOFError):
        print()  # newline after ^C
        return None


def _prompt_dont_ask_again() -> bool:
    """Ask if user wants to permanently suppress contribution prompts."""
    try:
        response = input("  Don't ask again? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def maybe_onboard(
    *,
    path: str | Path = DEFAULT_ONBOARDING_PATH,
    interactive: bool | None = None,
    prompt_fn=None,
) -> OnboardingState:
    """Run onboarding on first CLI invocation. Returns the state.

    Idempotent: if onboarding already completed, returns existing state
    immediately. Only shows prompts in interactive TTY contexts.
    Ctrl+C safe: treats interruption as dismissal.

    Args:
        path: state file location.
        interactive: True/False to force; None auto-detects.
        prompt_fn: testable injection. Receives (text, is_contribution)
            and returns True (accepted) / False (rejected) / None (dismissed).
    """
    path_obj = Path(path).expanduser()
    state = _load_state(path_obj)

    if state is not None and state.completed_at is not None:
        return state

    # Not yet completed. Skip if non-interactive.
    if interactive is None:
        interactive = _is_interactive()
    if not interactive:
        # Return defaults without saving (will prompt on next interactive run)
        return state if state is not None else OnboardingState()

    # Show welcome prompt
    if prompt_fn is not None:
        result = prompt_fn(_welcome_prompt_text(), False)
    else:
        result = _prompt_yes_no(
            _welcome_prompt_text() + "\nEnable anonymous telemetry?",
            default_no=False,
        )

    state = OnboardingState()
    if result is None:
        # Ctrl+C: save defaults, mark incomplete
        _save_state(state, path_obj)
        return state

    state.telemetry_enabled = bool(result)

    # Ask about data contribution
    if prompt_fn is not None:
        contrib_result = prompt_fn(_contribution_prompt_text(), True)
    else:
        contrib_result = _prompt_yes_no(
            "\nWould you like to contribute anonymized episode data?",
            default_no=True,
        )

    if contrib_result is not None:
        state.contribute_data = bool(contrib_result)

    state.completed_at = _utc_now_iso()
    _save_state(state, path_obj)
    return state


def maybe_prompt_contribution(
    *,
    path: str | Path = DEFAULT_ONBOARDING_PATH,
    trigger: str = "monthly",
    current_version: str = "",
    episode_count: int = 0,
    interactive: bool | None = None,
    prompt_fn=None,
) -> OnboardingState | None:
    """Show a recurring contribution prompt if conditions are met.

    Triggers:
    - "episodes": after episode_count >= 100
    - "finetune": after successful finetune
    - "upgrade": after version upgrade
    - "monthly": 30 days since last prompt

    Returns updated state, or None if no prompt was shown.
    Ctrl+C safe. Skips in non-interactive contexts.
    """
    path_obj = Path(path).expanduser()
    state = _load_state(path_obj)

    if state is None or state.completed_at is None:
        return None  # onboarding not yet done
    if state.contribute_data:
        return None  # already contributing
    if state.dont_ask_again:
        return None  # permanently suppressed

    if interactive is None:
        interactive = _is_interactive()
    if not interactive:
        return None

    now = time.time()
    should_prompt = False

    if trigger == "episodes":
        if episode_count >= _EPISODE_MILESTONE and not state.episode_milestone_prompted:
            should_prompt = True
    elif trigger == "finetune":
        should_prompt = True
    elif trigger == "upgrade":
        if current_version and current_version != state.last_version_prompted:
            should_prompt = True
    elif trigger == "monthly":
        if (now - state.last_prompt_at) >= _MONTHLY_COOLDOWN_S:
            should_prompt = True

    if not should_prompt:
        return None

    # Show prompt
    if prompt_fn is not None:
        result = prompt_fn(_contribution_prompt_text(), True)
    else:
        result = _prompt_yes_no(
            _contribution_prompt_text() + "\nContribute anonymized episode data?",
            default_no=True,
        )

    if result is True:
        state.contribute_data = True
    else:
        state.prompts_dismissed += 1
        if prompt_fn is None:
            dont_ask = _prompt_dont_ask_again()
        else:
            dont_ask = False
        if dont_ask:
            state.dont_ask_again = True

    state.last_prompt_at = now
    if trigger == "episodes":
        state.episode_milestone_prompted = True
    if trigger == "upgrade" and current_version:
        state.last_version_prompted = current_version

    _save_state(state, path_obj)
    return state


def get_onboarding_state(
    path: str | Path = DEFAULT_ONBOARDING_PATH,
) -> OnboardingState:
    """Load the current onboarding state (no prompts). Returns defaults if missing."""
    path_obj = Path(path).expanduser()
    state = _load_state(path_obj)
    return state if state is not None else OnboardingState()


def set_telemetry_enabled(
    enabled: bool,
    path: str | Path = DEFAULT_ONBOARDING_PATH,
) -> OnboardingState:
    """Programmatically set telemetry on/off."""
    path_obj = Path(path).expanduser()
    state = _load_state(path_obj) or OnboardingState()
    state.telemetry_enabled = enabled
    if state.completed_at is None:
        state.completed_at = _utc_now_iso()
    _save_state(state, path_obj)
    return state


def set_contribute_data(
    enabled: bool,
    path: str | Path = DEFAULT_ONBOARDING_PATH,
) -> OnboardingState:
    """Programmatically set data contribution on/off."""
    path_obj = Path(path).expanduser()
    state = _load_state(path_obj) or OnboardingState()
    state.contribute_data = enabled
    if state.completed_at is None:
        state.completed_at = _utc_now_iso()
    _save_state(state, path_obj)
    return state


__all__ = [
    "ONBOARDING_VERSION",
    "DEFAULT_ONBOARDING_PATH",
    "OnboardingState",
    "get_onboarding_state",
    "maybe_onboard",
    "maybe_prompt_contribution",
    "set_contribute_data",
    "set_telemetry_enabled",
]
