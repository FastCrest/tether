"""Chat session persistence.

One JSONL file per session, written to $TETHER_HOME/chat_history/.
Each line is one message in OpenAI chat-completion format. Truncates the system
message on save to save bytes (it's regenerated from SYSTEM_PROMPT on load).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def history_dir() -> Path:
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".cache" / "tether"))
    d = home / "chat_history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_path() -> Path:
    return history_dir() / f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"


def save_session(path: Path, messages: list[dict[str, Any]]) -> None:
    """Write the full conversation. Overwrites — atomicity not load-bearing here."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str))
            f.write("\n")
    tmp.replace(path)


def load_session(path: Path) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs.append(json.loads(line))
    return msgs


def latest_session_path() -> Path | None:
    """Return the most-recently-modified session file, or None if no history."""
    d = history_dir()
    sessions = sorted(d.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0] if sessions else None


def list_sessions(limit: int = 20) -> list[Path]:
    d = history_dir()
    return sorted(d.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
