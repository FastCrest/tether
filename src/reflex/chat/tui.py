"""Textual TUI for reflex chat.

Optional — install with `pip install 'reflex-vla[tui]'` (adds textual). Falls
back to `reflex.chat.console.run_repl` (Rich REPL) if textual isn't installed.

Layout (4 panels):

  ┌─────────────────────────── header ────────────────────────────┐
  │                                                                │
  │  transcript (scrollable, Markdown-rendered)                    │
  │                                                                │
  ├─────────────────────────── tool calls ─────────────────────────┤
  │  ⏳ list_targets({})           ✓ exit=0 (0.2s)                 │
  │  ⏳ pull_model({"model":...})   ⠋ running...                   │
  ├─────────────────────────── input ──────────────────────────────┤
  │ you › ▌                                                        │
  └────── tokens 1,247 │ tools 6 │ status: ready ─────────────────┘
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, Input, Label, Markdown, Static
    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    TEXTUAL_AVAILABLE = False

from reflex.chat.backends import ChatBackend, ProxyError, RateLimitError
from reflex.chat.executor import execute, format_tool_result
from reflex.chat.loop import LoopState, SYSTEM_PROMPT, build_system_prompt
from reflex.chat.schema import TOOLS


@dataclass
class _ToolCallRow:
    name: str
    args: dict[str, Any]
    started_at: float = field(default_factory=lambda: datetime.now().timestamp())
    finished_at: float | None = None
    exit_code: int | None = None
    command: str = ""

    def status_line(self) -> str:
        from rich.markup import escape as _esc
        if self.finished_at is None:
            return f"[yellow]⏳ {_esc(self.name)}[/yellow] [dim]{_esc(json.dumps(self.args))}[/dim]"
        elapsed = self.finished_at - self.started_at
        color = "green" if self.exit_code == 0 else "red"
        return (
            f"[{color}]✓ {_esc(self.name)}[/{color}] "
            f"[dim]exit={self.exit_code} ({elapsed:.1f}s) {_esc(self.command)}[/dim]"
        )


if TEXTUAL_AVAILABLE:

    class ChatTUI(App):
        """4-panel Textual chat for the reflex agent."""

        CSS = """
        Screen { layout: vertical; }
        #transcript { height: 1fr; padding: 1 2; border: solid $primary; }
        #tool_calls { height: 8; padding: 0 2; border: solid $accent; }
        #input_row { height: 3; padding: 0 1; border: solid $secondary; }
        #status_bar { height: 1; padding: 0 2; background: $boost; color: $text; }
        Input { width: 1fr; }
        """

        BINDINGS = [
            Binding("ctrl+c,ctrl+q", "quit", "Quit"),
            Binding("ctrl+l", "clear_transcript", "Clear"),
            Binding("ctrl+r", "reset_conversation", "Reset"),
        ]

        def __init__(
            self,
            backend: ChatBackend,
            dry_run: bool = False,
            model_label: str = "?",
        ) -> None:
            super().__init__()
            self.backend = backend
            self.dry_run = dry_run
            self.model_label = model_label
            self.messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}]
            self.token_count = 0
            self.tool_call_count = 0
            self.tool_rows: list[_ToolCallRow] = []
            self.busy = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield VerticalScroll(id="transcript")
            yield Vertical(id="tool_calls")
            yield Container(Input(placeholder="ask anything…", id="input"), id="input_row")
            yield Static(self._status_line(), id="status_bar")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "reflex chat"
            self.sub_title = f"{self.backend.proxy_url} · {self.model_label}"
            self.query_one(Input).focus()

        def _status_line(self) -> str:
            state = "thinking…" if self.busy else "ready"
            return f"tokens {self.token_count:,} · tool calls {self.tool_call_count} · {state}"

        def _refresh_status(self) -> None:
            self.query_one("#status_bar", Static).update(self._status_line())

        def _refresh_tool_panel(self) -> None:
            panel = self.query_one("#tool_calls", Vertical)
            panel.remove_children()
            # Show the last 6 tool calls; older ones implied by tool_call_count badge.
            for row in self.tool_rows[-6:]:
                panel.mount(Label(row.status_line(), markup=True))

        async def _append_transcript(self, who: str, text: str) -> None:
            ts = self.query_one("#transcript", VerticalScroll)
            md_block = f"**{who}**\n\n{text}\n\n---\n"
            ts.mount(Markdown(md_block))
            ts.scroll_end(animate=False)

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text or self.busy:
                return
            inp = self.query_one(Input)
            inp.value = ""

            if text.lower() in {"/quit", "/exit", ":q"}:
                self.exit()
                return
            if text.lower() == "/reset":
                self.messages = [{"role": "system", "content": build_system_prompt()}]
                self.tool_rows.clear()
                self.tool_call_count = 0
                self.token_count = 0
                ts = self.query_one("#transcript", VerticalScroll)
                ts.remove_children()
                self._refresh_tool_panel()
                self._refresh_status()
                return

            await self._append_transcript("you", text)
            self.messages.append({"role": "user", "content": text})
            self.busy = True
            self._refresh_status()
            try:
                await self._agent_loop()
            except RateLimitError as e:
                await self._append_transcript("error", f"rate limit: {e}")
            except ProxyError as e:
                await self._append_transcript("error", f"proxy error: {e}")
            finally:
                self.busy = False
                self._refresh_status()

        async def _agent_loop(self) -> None:
            for _ in range(16):
                msg = await self._one_turn(tools=TOOLS)
                self.messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    content = (msg.get("content") or "").strip()
                    if content:
                        await self._append_transcript("bot", content)
                    return
                for tc in tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    row = _ToolCallRow(name=name, args=args)
                    self.tool_rows.append(row)
                    self.tool_call_count += 1
                    self._refresh_tool_panel()
                    self._refresh_status()

                    # Run the tool off the UI thread so the panel keeps refreshing.
                    result = await asyncio.to_thread(execute, name, args, None, self.dry_run)
                    row.finished_at = datetime.now().timestamp()
                    row.exit_code = result.get("exit_code")
                    row.command = result.get("command", "")
                    self._refresh_tool_panel()
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": format_tool_result(name, result),
                    })

            # Cap reached.
            self.messages.append({"role": "user", "content": "[system] tool-call cap reached; summarize and stop calling tools."})
            msg = await self._one_turn(tools=None)
            content = (msg.get("content") or "").strip()
            if content:
                await self._append_transcript("bot", content)

        async def _one_turn(self, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
            """One LLM round-trip. Streams tokens into the live Markdown widget."""
            from reflex.chat.backends import assemble_stream

            ts = self.query_one("#transcript", VerticalScroll)
            live_block = Markdown("**bot**\n\n")
            ts.mount(live_block)
            ts.scroll_end(animate=False)
            buffer = []

            def on_token(t: str) -> None:
                buffer.append(t)
                self.token_count += len(t.split())  # rough word count
                # Schedule UI refresh on the event loop without blocking the producer.
                self.call_from_thread(self._update_live_block, live_block, "".join(buffer))

            chunks = self.backend.chat_stream(self.messages, tools=tools, tool_choice="auto" if tools else "none")
            msg = await asyncio.to_thread(assemble_stream, chunks, on_token, None)

            # Final flush + scroll
            self._update_live_block(live_block, "**bot**\n\n" + (msg.get("content") or "").strip())
            self._refresh_status()
            ts.scroll_end(animate=False)

            # If the assistant emitted ONLY tool_calls (no content), drop the empty placeholder.
            if not (msg.get("content") or "").strip():
                live_block.remove()

            return msg

        def _update_live_block(self, widget: "Markdown", text: str) -> None:
            try:
                widget.update("**bot**\n\n" + text)
            except Exception:  # noqa: BLE001
                pass  # widget may have been removed

        def action_clear_transcript(self) -> None:
            ts = self.query_one("#transcript", VerticalScroll)
            ts.remove_children()

        def action_reset_conversation(self) -> None:
            self.messages = [{"role": "system", "content": build_system_prompt()}]
            self.tool_rows.clear()
            self.tool_call_count = 0
            self.token_count = 0
            self._refresh_tool_panel()
            self._refresh_status()
            self.action_clear_transcript()


def run_tui(proxy_url: str | None = None, dry_run: bool = False) -> None:
    """Entry point: launches the Textual TUI. Falls back to the Rich REPL if
    textual isn't installed (so `reflex chat --tui` is safe even without
    the [tui] extra)."""
    if not TEXTUAL_AVAILABLE:
        print(
            "Textual not installed — falling back to the Rich REPL.\n"
            "Install the optional extra to use the TUI: pip install 'reflex-vla[tui]'\n",
            flush=True,
        )
        from reflex.chat.console import run_repl
        run_repl(proxy_url=proxy_url, dry_run=dry_run)
        return

    backend = ChatBackend(proxy_url=proxy_url) if proxy_url else ChatBackend()
    model_label = "gpt-5-mini"
    try:
        h = backend.health()
        model_label = h.get("model", model_label)
    except Exception:  # noqa: BLE001
        pass
    ChatTUI(backend=backend, dry_run=dry_run, model_label=model_label).run()
