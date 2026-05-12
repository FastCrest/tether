"""Reflex Curate wedge — cross-customer real-world VLA data collection.

The 8th wedge. Per the architecture ADR (`reflex_context/01_decisions/
2026-05-05-data-curate-wedge-architecture.md`):

- Free tier: opt-in via `reflex contribute --opt-in`. Earns Pro credits.
- Pro tier: opt-in by default with transparent disclosure. Earns 10% revenue
  share on datasets that include their data.
- Enterprise tier: negotiated per-deal (data exclusivity, cash %).

Substrate composition:

    consent.py        — CurateConsent receipt at ~/.reflex/consent.json
    messaging.py      — touchpoint message templates (mission/reciprocity/privacy)
    nudge_engine.py   — when/where/which message to show
    opt_in_cli.py     — `reflex contribute` typer subcommand
    free_collector.py — Free tier extension of pro.data_collection.ProDataCollector
    uploader.py       — background R2 uploader (queue → Cloudflare R2)

Companion infra in `infra/contribution-worker/` (Cloudflare Worker).
"""
from __future__ import annotations

__all__: list[str] = []
