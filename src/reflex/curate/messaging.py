"""Curate touchpoint message templates.

The persuasion design (per opt-in-flow.md): three framings combined in
priority order — mission > reciprocity > privacy. Roboticists are
mission-aligned; reciprocity addresses "what do I get"; privacy disarms
the obvious objection without leading with anxiety.

Each touchpoint has a `Message` with:
  - `body`: the rendered text (with rich markup)
  - `prompt`: optional Y/N prompt for interactive touchpoints (None when passive)
  - `cta`: optional call-to-action line (e.g. "run `reflex contribute --opt-in`")

Phase 1 ships ONE variant per touchpoint (the opinionated launch). Phase 1.5
adds variant pools + A/B routing in `nudge_engine.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A renderable touchpoint message."""

    body: str
    prompt: str | None = None
    cta: str | None = None


# ─── Mission / reciprocity / privacy framings ──────────────────────────────────

MISSION_FRAMING = (
    "Real-world robot data trains the next generation of robot AI. "
    "Today's models are trained on academic datasets that don't reflect "
    "production. Your deployments help close that gap."
)

RECIPROCITY_FRAMING_FREE = (
    "Free contributors earn Pro tier credits: 3 months free at 100 contributed "
    "hours, 12 months free at 500 hours. Your data flows back to you in better "
    "models, distilled into the next Reflex release."
)

RECIPROCITY_FRAMING_PRO = (
    "Pro contributors earn 10% revenue share on datasets that include their "
    "data. We track contribution hours; when a dataset is sold, your share "
    "lands in your account automatically."
)

PRIVACY_FRAMING = (
    "Face-blurred and instruction-hashed at ingest. Aggregated across 5+ "
    "contributors before anything is published. You can revoke anytime — "
    "your data is removed from all derived datasets within 30 days."
)


# ─── Touchpoint 1: first-run splash ────────────────────────────────────────────

FIRST_RUN_SPLASH = Message(
    body=f"""
═══════════════════════════════════════════════════════════════════════
[bold]Reflex Data Contribution Program[/bold]
═══════════════════════════════════════════════════════════════════════

{MISSION_FRAMING}

{RECIPROCITY_FRAMING_FREE}

{PRIVACY_FRAMING}

Opt-in is voluntary and reversible. Recordings are only contributed when
you explicitly run `reflex serve --record`.
""",
    prompt="Opt in now? [y/N]: ",
    cta="Opt in later: [cyan]reflex contribute --opt-in[/cyan]",
)


# ─── Touchpoint 2: serve-start banner ──────────────────────────────────────────

SERVE_START_BANNER = Message(
    body=(
        "[dim]reflex contribute:[/dim] join the data corpus to earn Pro credits. "
        "[cyan]reflex contribute --opt-in[/cyan] to learn more, or "
        "[cyan]REFLEX_NO_CONTRIB_NUDGE=1[/cyan] to silence."
    ),
    prompt=None,
    cta=None,
)


# ─── Touchpoint 3: every Nth recorded session ──────────────────────────────────

def serve_record_reminder(
    *, hours_recorded_total: float, sessions_since_first_record: int
) -> Message:
    """Stronger reminder shown every Nth `reflex serve --record` invocation.
    Mentions hours-recorded threshold to nudge toward Pro credit."""
    body = (
        f"[bold]You've recorded {hours_recorded_total:.1f} hours of robot data "
        f"({sessions_since_first_record} sessions).[/bold]\n"
        f"Contributing your recordings would earn Pro credits — "
        f"3 months free at 100 hours, 12 months at 500.\n"
        f"[cyan]reflex contribute --opt-in[/cyan] to join, "
        f"[cyan]reflex contribute --info[/cyan] to read the privacy details first."
    )
    return Message(body=body, prompt=None, cta=None)


# ─── Touchpoint 4: doctor status line ──────────────────────────────────────────

def doctor_status_line(
    *,
    opted_in: bool,
    tier: str | None = None,
    hours_contributed: float | None = None,
    pro_credits_progress: float | None = None,
) -> str:
    """One-line status string for the `reflex doctor` table.

    Returns plain text suitable for table cell rendering (no rich markup)."""
    if not opted_in:
        return "not opted in (run `reflex contribute --opt-in`)"
    parts = [f"opted in ({tier or 'free'})"]
    if hours_contributed is not None:
        parts.append(f"{hours_contributed:.1f}h contributed")
    if pro_credits_progress is not None:
        parts.append(f"{pro_credits_progress * 100:.0f}% to next credit")
    return " · ".join(parts)


# ─── Touchpoint 5: chat agent system-prompt addition ───────────────────────────

CHAT_SYSTEM_PROMPT_ADDITION = (
    "Reflex runs an opt-in data contribution program. If the user mentions "
    "data, recordings, training, or contribution and they are NOT yet opted "
    "in, briefly mention `reflex contribute --opt-in` and the Pro-credits "
    "incentive. Do not nag — mention once per conversation, only when "
    "topical. If the user is already opted in (you'll be told via "
    "system-injected status), do not bring it up unless they ask."
)


# ─── Opt-in confirmation messages ──────────────────────────────────────────────

def opt_in_success(*, tier: str, contributor_id: str) -> str:
    return (
        f"[green]✓[/green] Opted in to Reflex Data Contribution program ({tier}).\n"
        f"[green]✓[/green] contributor_id: [dim]{contributor_id}[/dim]\n"
        f"[green]✓[/green] Future recordings (when [cyan]--record[/cyan] is used) "
        f"will be cleaned, anonymized, and contributed to the corpus.\n"
        f"[green]✓[/green] Track progress: [cyan]reflex contribute --status[/cyan]\n"
        f"[green]✓[/green] Revoke anytime: [cyan]reflex contribute --opt-out[/cyan]"
    )


def opt_out_success() -> str:
    return (
        "[yellow]✓[/yellow] Opted out. Future recordings will not be contributed.\n"
        "[dim]Note: data already contributed remains in derived datasets unless you "
        "also run [cyan]reflex contribute --revoke[/cyan].[/dim]"
    )


def revoke_warning() -> str:
    return (
        "[yellow]⚠[/yellow]  This will remove your historical contributions from all "
        "derived datasets within 30 days.\n"
        "[dim]Datasets already sold cannot be recalled, but your contribution will "
        "not appear in v2+ releases.[/dim]"
    )


def revoke_success(*, contributor_id: str) -> str:
    return (
        f"[green]✓[/green] Revocation submitted for contributor [dim]{contributor_id}[/dim].\n"
        f"[green]✓[/green] Local consent receipt removed.\n"
        f"[green]✓[/green] Server-side cascade purge will complete within 30 days "
        f"per GDPR Article 17."
    )


__all__ = [
    "MISSION_FRAMING",
    "RECIPROCITY_FRAMING_FREE",
    "RECIPROCITY_FRAMING_PRO",
    "PRIVACY_FRAMING",
    "Message",
    "FIRST_RUN_SPLASH",
    "SERVE_START_BANNER",
    "serve_record_reminder",
    "doctor_status_line",
    "CHAT_SYSTEM_PROMPT_ADDITION",
    "opt_in_success",
    "opt_out_success",
    "revoke_warning",
    "revoke_success",
]
