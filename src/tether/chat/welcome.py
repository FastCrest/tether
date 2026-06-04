"""Welcome card + slash-command catalog for `tether chat`.

The welcome card is shown once per machine (cached in $TETHER_HOME/.welcomed).
Slash commands are recognized in the REPL/TUI loop and short-circuit the LLM.
"""

from __future__ import annotations

import os
from pathlib import Path

WELCOME_CARD = """[bold]tether chat[/bold] — natural language for VLA deployment

What I can do for you:
  • Deploy a model       [dim]"deploy smolvla to my mac"[/dim]
  • Browse models        [dim]"what models are available?"[/dim]
  • Diagnose problems    [dim]"why is my install broken?"[/dim]
  • Inspect traces       [dim]"show my recent traces from this week"[/dim]
  • Run benchmarks       [dim]"benchmark pi05 on my desktop"[/dim]

Useful commands:
  [green]/help[/green]    list everything I can do          [green]/tools[/green]   show my tool catalog
  [green]/clear[/green]   clear the screen                   [green]/reset[/green]   start a fresh conversation
  [green]/history[/green] show this session's turns          [green]/tour[/green]    suggested first prompts
  [green]exit[/green]    quit (or Ctrl+C)
"""

SHORT_BANNER = """[bold]tether chat[/bold] — type a question, [green]/help[/green] for commands, [green]exit[/green] to quit
"""

TOUR_PROMPTS = [
    'what version of tether am i on?',
    'what models can i deploy and which is smallest?',
    'check my install for problems',
    "i'm on a mac with no gpu — what can i actually do?",
    'show me my recent traces',
]


SLASH_HELP = """[bold]Slash commands[/bold]

  [green]/help[/green]      this message
  [green]/tools[/green]     list every tool the assistant can call
  [green]/history[/green]   show the conversation so far
  [green]/clear[/green]     clear the screen (keeps conversation context)
  [green]/reset[/green]     start a fresh conversation (drops context)
  [green]/tour[/green]      show 5 example prompts to copy-paste
  [green]exit[/green]       quit

[bold]What chat can do[/bold]
  Natural-language prompts route to one of 17 tools that wrap the [cyan]tether[/cyan] CLI.
  Examples: "deploy smolvla to my mac", "show running serve processes",
  "list traces from yesterday", "benchmark pi05 on my desktop".

[bold]Conversation persistence[/bold]
  Sessions auto-save to ~/.cache/tether/chat_history/. Resume the most
  recent with: [cyan]tether chat --resume[/cyan]
"""

TOUR_BLOCK = """[bold]Try one of these[/bold] — copy-paste any line:

  what version of tether am i on?
  what models can i deploy and which is smallest?
  check my install for problems
  i'm on a mac with no gpu — what can i actually do?
  show me my recent traces

[dim]Tip: the assistant can also chain operations — try "deploy smolvla to my orin nano".[/dim]
"""


def welcomed_path() -> Path:
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".cache" / "tether"))
    home.mkdir(parents=True, exist_ok=True)
    return home / ".welcomed"


def has_been_welcomed() -> bool:
    return welcomed_path().exists()


def mark_welcomed() -> None:
    try:
        welcomed_path().write_text("1")
    except OSError:
        pass  # not load-bearing


def tools_listing() -> str:
    """Markdown-style listing of all chat tools, grouped by category."""
    from tether.chat.schema import TOOLS
    groups: dict[str, list[tuple[str, str]]] = {
        "Deploy": [],
        "Models": [],
        "Train": [],
        "Inspect": [],
        "Status": [],
    }
    rules = {
        "deploy_one_command": "Deploy",
        "export_model": "Deploy",
        "serve_model": "Deploy",
        "list_models": "Models",
        "pull_model": "Models",
        "model_info": "Models",
        "list_targets": "Models",
        "distill": "Train",
        "finetune": "Train",
        "benchmark": "Inspect",
        "evaluate": "Inspect",
        "list_traces": "Inspect",
        "replay_trace": "Inspect",
        "doctor": "Inspect",
        "show_status": "Status",
        "show_config": "Status",
        "show_version": "Status",
    }
    for tool in TOOLS:
        name = tool["function"]["name"]
        desc = tool["function"]["description"]
        # Trim long descriptions to one line for the listing.
        first_line = desc.split(".")[0].strip()[:80]
        cat = rules.get(name, "Other")
        groups.setdefault(cat, []).append((name, first_line))
    lines: list[str] = ["[bold]Chat tool catalog[/bold] — what the assistant can call on your behalf\n"]
    for cat, items in groups.items():
        if not items:
            continue
        lines.append(f"[bold cyan]{cat}[/bold cyan]")
        for name, desc in items:
            lines.append(f"  [green]{name:22}[/green] {desc}")
        lines.append("")
    return "\n".join(lines)
