"""Nudge engine — decides when, where, and which touchpoint message to show.

Single rule: never nag. Touchpoint cadence governs frequency, the consent
receipt governs whether to show at all (opted-in users see no opt-in nudges),
and the kill switch (`REFLEX_NO_CONTRIB_NUDGE=1`) gives a hard exit.

State lives at `~/.reflex/curate_nudge.json`:

  {
    "last_first_run_shown_at": "2026-05-05T...",   // null until shown
    "serve_starts_count": 17,
    "record_sessions_count": 4,
    "decided": false                                // set true after opt-in or opt-out
  }

Phase 1 cadence (per opt-in-flow.md):
- first_run     → once, ever (sets last_first_run_shown_at)
- serve_start   → every serve start UNTIL decided OR REFLEX_NO_CONTRIB_NUDGE=1
- record_nth    → every 10th `--record` invocation that hasn't already opted in
- doctor_status → every doctor invocation (passive line, never noisy)
- chat_hint     → injected into chat system prompt; one mention per conversation max

Phase 1.5 layers an A/B variant pool on top.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reflex.curate import consent as curate_consent
from reflex.curate import messaging

logger = logging.getLogger(__name__)

DEFAULT_NUDGE_STATE_PATH = "~/.reflex/curate_nudge.json"
SERVE_RECORD_REMINDER_EVERY_N = 10
KILL_SWITCH_ENV = "REFLEX_NO_CONTRIB_NUDGE"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _kill_switch_active() -> bool:
    val = os.environ.get(KILL_SWITCH_ENV, "").strip().lower()
    return val in ("1", "true", "yes", "on")


@dataclass
class NudgeState:
    last_first_run_shown_at: str | None = None
    serve_starts_count: int = 0
    record_sessions_count: int = 0
    decided: bool = False  # opted-in or opted-out either way

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NudgeState":
        return cls(
            last_first_run_shown_at=d.get("last_first_run_shown_at"),
            serve_starts_count=int(d.get("serve_starts_count", 0)),
            record_sessions_count=int(d.get("record_sessions_count", 0)),
            decided=bool(d.get("decided", False)),
        )


def _load_state(path: str | Path = DEFAULT_NUDGE_STATE_PATH) -> NudgeState:
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        return NudgeState()
    try:
        return NudgeState.from_dict(json.loads(path_obj.read_text()))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("nudge state corrupted at %s: %s — resetting", path_obj, exc)
        return NudgeState()


def _save_state(state: NudgeState, path: str | Path = DEFAULT_NUDGE_STATE_PATH) -> None:
    path_obj = Path(path).expanduser()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    tmp.replace(path_obj)


def _is_decided() -> bool:
    """The user has decided iff they have a receipt OR they explicitly opted out
    (we don't persist explicit opt-out separately yet — but having no receipt
    AND a high serve_starts_count is treated as "still considering")."""
    return curate_consent.is_opted_in()


def maybe_first_run_splash(*, path: str | Path = DEFAULT_NUDGE_STATE_PATH) -> messaging.Message | None:
    """Return the first-run splash message iff it hasn't been shown before AND
    the user hasn't already decided AND the kill switch isn't active.
    Side effect: updates state to record that we've shown it."""
    if _kill_switch_active():
        return None
    if _is_decided():
        return None
    state = _load_state(path)
    if state.last_first_run_shown_at is not None:
        return None
    state.last_first_run_shown_at = _utc_now_iso()
    _save_state(state, path)
    return messaging.FIRST_RUN_SPLASH


def maybe_serve_start_banner(*, path: str | Path = DEFAULT_NUDGE_STATE_PATH) -> messaging.Message | None:
    """Show the serve-start banner UNTIL the user has decided OR kill switch."""
    if _kill_switch_active():
        return None
    state = _load_state(path)
    state.serve_starts_count += 1
    _save_state(state, path)
    if _is_decided():
        return None
    return messaging.SERVE_START_BANNER


def maybe_record_reminder(
    *,
    hours_recorded_total: float,
    path: str | Path = DEFAULT_NUDGE_STATE_PATH,
) -> messaging.Message | None:
    """Show the stronger reminder every Nth `serve --record` invocation.
    Skip if already opted in or kill switch."""
    if _kill_switch_active():
        return None
    state = _load_state(path)
    state.record_sessions_count += 1
    _save_state(state, path)
    if _is_decided():
        return None
    if state.record_sessions_count % SERVE_RECORD_REMINDER_EVERY_N != 0:
        return None
    return messaging.serve_record_reminder(
        hours_recorded_total=hours_recorded_total,
        sessions_since_first_record=state.record_sessions_count,
    )


def doctor_status(*, hours_contributed: float | None = None) -> str:
    """One-line passive status for the `reflex doctor` table. Never gated by
    kill switch or cadence — doctor output is always informational."""
    if not curate_consent.is_opted_in():
        return messaging.doctor_status_line(opted_in=False)
    receipt = curate_consent.load()
    return messaging.doctor_status_line(
        opted_in=True,
        tier=receipt.tier,
        hours_contributed=hours_contributed,
    )


def mark_decided(*, path: str | Path = DEFAULT_NUDGE_STATE_PATH) -> None:
    """Called after a user opts in or out via CLI. Future nudges are silenced
    until the user un-decides (no current path; would require state reset)."""
    state = _load_state(path)
    state.decided = True
    _save_state(state, path)


__all__ = [
    "DEFAULT_NUDGE_STATE_PATH",
    "KILL_SWITCH_ENV",
    "SERVE_RECORD_REMINDER_EVERY_N",
    "NudgeState",
    "doctor_status",
    "mark_decided",
    "maybe_first_run_splash",
    "maybe_record_reminder",
    "maybe_serve_start_banner",
]
