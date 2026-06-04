"""Plain-console chat UI (REPL). Streams tokens live as they arrive."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from tether.chat.backends import ChatBackend, ProxyError, RateLimitError
from tether.chat.loop import LoopState

console = Console()


@dataclass
class _UIState:
    """Mutable per-turn UI state. Tracks whether we've started a streaming reply
    so we can print a leading separator on the first token, and a trailing
    newline when the turn finishes."""

    streaming_reply: bool = False
    _stream_started: bool = False

    def on_token(self, text: str) -> None:
        if not self._stream_started:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._stream_started = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_stream(self) -> None:
        if self._stream_started:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
        self._stream_started = False
        self.streaming_reply = False


def _make_event_handler(ui: _UIState):
    def handler(evt: dict[str, Any]) -> None:
        kind = evt["kind"]
        if kind == "turn_start":
            # New round-trip; reset stream tracking.
            ui._stream_started = False
        elif kind == "token":
            ui.on_token(evt["text"])
        elif kind == "tool_start":
            # If we were mid-stream (model wrote some thinking text before the
            # tool call), close that out cleanly first.
            if ui._stream_started:
                ui.end_stream()
            args_preview = evt.get("args") or {}
            console.print(f"[cyan]→ {evt['name']}[/cyan] [dim]{args_preview}[/dim]")
        elif kind == "tool_end":
            result = evt.get("result") or {}
            ec = result.get("exit_code")
            color = "green" if ec == 0 else "red"
            console.print(f"  [{color}]exit_code={ec}[/{color}] [dim]{result.get('command','')}[/dim]")
        elif kind == "final":
            ui.end_stream()
    return handler


def _format_history(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = ["[bold]Conversation so far[/bold]\n"]
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue  # too long + always the same; skip
        if role == "tool":
            preview = (m.get("content") or "")[:100].replace("\n", " ")
            lines.append(f"  [yellow]tool[/yellow]  {preview}")
            continue
        content = m.get("content") or ""
        if not content and m.get("tool_calls"):
            names = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
            lines.append(f"  [cyan]bot[/cyan]   → {names}")
        else:
            tag = "you" if role == "user" else "bot"
            color = "green" if role == "user" else "white"
            preview = content[:200].replace("\n", " ")
            lines.append(f"  [{color}]{tag}[/{color}]   {preview}")
    return "\n".join(lines)


def _handle_slash(cmd: str, state: LoopState) -> bool:
    """Return True if the command was handled (REPL should continue without LLM)."""
    from tether.chat.welcome import SLASH_HELP, TOUR_BLOCK, tools_listing
    cmd = cmd.lower().strip()
    if cmd in {"/help", "/?"}:
        console.print(SLASH_HELP)
        return True
    if cmd == "/tools":
        console.print(tools_listing())
        return True
    if cmd == "/history":
        console.print(_format_history(state.messages))
        return True
    if cmd == "/clear":
        console.clear()
        return True
    if cmd == "/reset":
        state.reset()
        console.print("[dim]conversation cleared[/dim]")
        return True
    if cmd == "/tour":
        console.print(TOUR_BLOCK)
        return True
    return False


def run_repl(
    proxy_url: str | None = None,
    dry_run: bool = False,
    no_stream: bool = False,
    resume: bool = False,
) -> None:
    from tether.chat.welcome import (
        WELCOME_CARD, SHORT_BANNER,
        has_been_welcomed, mark_welcomed,
    )
    from tether.chat.history import latest_session_path, load_session, new_session_path

    backend = ChatBackend(proxy_url=proxy_url) if proxy_url else ChatBackend()
    try:
        h = backend.health()
        console.print(f"[dim]connected: {backend.proxy_url} (model={h.get('model','?')})[/dim]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]warning: health check failed ({e}); will try requests anyway[/yellow]")

    ui = _UIState()
    state = LoopState(
        backend=backend,
        on_event=_make_event_handler(ui),
        dry_run=dry_run,
        streaming=not no_stream,
    )

    if resume:
        prev = latest_session_path()
        if prev is None:
            console.print("[yellow]no previous chat session found; starting fresh[/yellow]")
            state.reset()
        else:
            try:
                state.messages = load_session(prev)
                console.print(f"[dim]resumed session: {prev.name} ({len(state.messages)} messages)[/dim]")
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]could not resume {prev.name} ({e}); starting fresh[/yellow]")
                state.reset()
    else:
        state.reset()

    if not has_been_welcomed():
        console.print()
        console.print(WELCOME_CARD)
        mark_welcomed()
    else:
        console.print(SHORT_BANNER)

    # Auto-save this session as it grows (one file per session).
    session_path = new_session_path()

    while True:
        try:
            user = console.input("[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            _save_session(session_path, state.messages)
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            _save_session(session_path, state.messages)
            return
        if user.startswith("/"):
            if _handle_slash(user, state):
                continue
            console.print(f"[yellow]unknown command: {user}[/yellow]  (try /help)")
            continue

        try:
            reply = state.send(user)
        except RateLimitError as e:
            ui.end_stream()
            console.print(f"[red]rate limit:[/red] {e}")
            continue
        except ProxyError as e:
            ui.end_stream()
            console.print(f"[red]proxy error:[/red] {e}")
            continue

        # Streaming path already printed the reply token-by-token; non-streaming
        # path returns the full reply here, so render it once.
        if no_stream:
            console.print()
            console.print(reply or "_(empty reply)_")
            console.print()

        # Persist after every turn so a Ctrl+C never loses context.
        _save_session(session_path, state.messages)


def _save_session(path: "Path", messages: list[dict[str, Any]]) -> None:
    from tether.chat.history import save_session
    try:
        save_session(path, messages)
    except Exception:  # noqa: BLE001
        pass  # not load-bearing — never crash the REPL on a disk hiccup
