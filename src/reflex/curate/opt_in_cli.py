"""`reflex contribute` CLI subcommand.

Surface:
  reflex contribute                  → equivalent to --status (zero-friction default)
  reflex contribute --opt-in         → create receipt + show success
  reflex contribute --opt-out        → delete receipt locally
  reflex contribute --revoke         → opt-out + cascade purge through server
  reflex contribute --status         → show current consent + contribution stats
  reflex contribute --info           → show privacy/legal details before deciding
  reflex contribute --inspect        → list episodes in the local upload queue
  reflex contribute --purge ID       → drop a specific queued episode before upload

Phase 1 wires --opt-in / --opt-out / --revoke / --status / --info. Inspect
and purge land alongside the uploader spec (those need the queue).
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from reflex.curate import consent as curate_consent
from reflex.curate import messaging

logger = logging.getLogger(__name__)

contribute_app = typer.Typer(
    name="contribute",
    help="Reflex Data Contribution — opt in, opt out, check status, or revoke.",
    invoke_without_command=True,
)
console = Console()


def _detect_pro_tier() -> tuple[str, str | None]:
    """Return (tier, customer_id_if_pro). Pro tier is detected when a valid
    Pro license file exists; otherwise free tier with no customer_id.

    Anonymous Free contributors get a derived contributor_id in `consent.save()`
    (via `derive_contributor_id()`). Pro contributors use the license-bound
    customer_id so their revenue share lands in the right billing record.
    """
    try:
        from reflex.pro.activate import DEFAULT_LICENSE_PATH
        path = Path(DEFAULT_LICENSE_PATH).expanduser()
        if not path.exists():
            return "free", None
        import json
        data = json.loads(path.read_text())
        customer_id = data.get("customer_id")
        if not customer_id:
            return "free", None
        return "pro", str(customer_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pro_tier detect failed, defaulting to free: %s", exc)
        return "free", None


@contribute_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    opt_in: bool = typer.Option(
        False, "--opt-in", help="Opt in to data contribution.",
    ),
    opt_out: bool = typer.Option(
        False, "--opt-out", help="Opt out (local receipt only; historical data stays).",
    ),
    revoke: bool = typer.Option(
        False, "--revoke",
        help="Opt out + request server-side cascade purge of historical data.",
    ),
    status: bool = typer.Option(
        False, "--status", help="Show current consent + contribution stats.",
    ),
    info: bool = typer.Option(
        False, "--info", help="Show full privacy + legal details before deciding.",
    ),
    inspect: bool = typer.Option(
        False, "--inspect",
        help="List files in the local upload queue with episode counts + sizes.",
    ),
    purge: str = typer.Option(
        "", "--purge",
        help="Remove a specific queued file by name (run --inspect first to see names).",
    ),
    revoke_status: str = typer.Option(
        "", "--revoke-status",
        help="Check cascade progress for a revoke request_id (returned from --revoke).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Auto-confirm prompts (for --revoke / --purge). Required in non-interactive contexts.",
    ),
) -> None:
    """Reflex Data Contribution program — opt in, opt out, or check status."""
    if ctx.invoked_subcommand is not None:
        return

    flags = [opt_in, opt_out, revoke, status, info, inspect, bool(purge), bool(revoke_status)]
    set_count = sum(1 for f in flags if f)
    if set_count > 1:
        console.print(
            "[red]Pick one of:[/red] --opt-in, --opt-out, --revoke, --status, "
            "--info, --inspect, --purge, --revoke-status"
        )
        raise typer.Exit(2)

    if info:
        _cmd_info()
        return
    if opt_in:
        _cmd_opt_in()
        return
    if opt_out:
        _cmd_opt_out()
        return
    if revoke:
        _cmd_revoke(yes=yes)
        return
    if inspect:
        _cmd_inspect()
        return
    if purge:
        _cmd_purge(target=purge, yes=yes)
        return
    if revoke_status:
        _cmd_revoke_status(request_id=revoke_status)
        return
    # No flag set OR --status set: show status
    _cmd_status()


def _cmd_info() -> None:
    """Print the privacy + legal context before the operator decides."""
    body = f"""
[bold]Reflex Data Contribution — what you're agreeing to[/bold]

[bold cyan]Mission[/bold cyan]
{messaging.MISSION_FRAMING}

[bold cyan]What you get[/bold cyan]
Free tier: {messaging.RECIPROCITY_FRAMING_FREE}

Pro tier:  {messaging.RECIPROCITY_FRAMING_PRO}

[bold cyan]What's contributed[/bold cyan]
- Robot recordings made with [cyan]reflex serve --record[/cyan]
- Episode metadata (timestamps, embodiment, task labels)
- Camera frames (face-blurred at source)
- Action chunks (the policy's outputs)
- Instruction text (SHA-256 hashed by default; raw on explicit opt-in)

What is NOT contributed:
- Recordings made without --record
- Recordings made BEFORE you opted in (no retroactive contribution)
- Anything you mark sensitive via [cyan]reflex inspect traces --mark-sensitive ID[/cyan]
- Raw images, unless you explicitly enable [cyan]privacy_mode=raw_opt_in[/cyan]

[bold cyan]Privacy[/bold cyan]
{messaging.PRIVACY_FRAMING}

[bold cyan]Revoke anytime[/bold cyan]
[cyan]reflex contribute --opt-out[/cyan]   stop future contribution
[cyan]reflex contribute --revoke[/cyan]    cascade purge historical contributions
                              (30-day SLA, GDPR Article 17 compliant)

[bold cyan]Decide[/bold cyan]
[cyan]reflex contribute --opt-in[/cyan]    join the program
"""
    console.print(body)


def _cmd_opt_in() -> None:
    if curate_consent.is_opted_in():
        receipt = curate_consent.load()
        console.print(
            f"[yellow]Already opted in[/yellow] as [dim]{receipt.contributor_id}[/dim] "
            f"({receipt.tier}, since {receipt.opted_in_at}).\n"
            f"Run [cyan]reflex contribute --status[/cyan] to see contribution stats, "
            f"or [cyan]reflex contribute --opt-out[/cyan] to stop."
        )
        return

    tier, license_customer_id = _detect_pro_tier()
    receipt = curate_consent.save(
        tier=tier,
        contributor_id=license_customer_id,  # None → anonymous derive for Free
    )
    console.print(messaging.opt_in_success(
        tier=receipt.tier, contributor_id=receipt.contributor_id,
    ))


def _cmd_opt_out() -> None:
    removed = curate_consent.revoke()
    if not removed:
        console.print(
            "[dim]Already opted out[/dim] — no consent receipt found at "
            f"[cyan]{Path(curate_consent.DEFAULT_CONSENT_PATH).expanduser()}[/cyan]."
        )
        return
    console.print(messaging.opt_out_success())


def _cmd_revoke(*, yes: bool) -> None:
    try:
        receipt = curate_consent.load()
    except curate_consent.ConsentNotFound:
        console.print(
            "[dim]No active consent receipt — nothing to revoke.[/dim] "
            "If you previously contributed before clearing the receipt, "
            "contact [cyan]privacy@fastcrest.com[/cyan] with your contributor_id."
        )
        return
    except curate_consent.ConsentCorrupted as exc:
        console.print(f"[red]Receipt corrupted:[/red] {exc}")
        raise typer.Exit(2)

    console.print(messaging.revoke_warning())
    if not yes:
        try:
            ans = typer.confirm("Confirm revoke?", default=False)
        except (EOFError, OSError):
            console.print(
                "[red]Non-interactive context — pass --yes to confirm.[/red]"
            )
            raise typer.Exit(2)
        if not ans:
            console.print("[dim]Revoke cancelled.[/dim]")
            return

    # Local: remove the receipt now.
    curate_consent.revoke()

    # Server-side cascade: POST to /v1/revoke/cascade.
    contributor_id = receipt.contributor_id
    logger.info("curate revoke requested for contributor_id=%s", contributor_id)
    server_request_id = None
    try:
        import httpx
        from reflex.curate.uploader import _worker_url, HTTP_TIMEOUT_S
        r = httpx.post(
            f"{_worker_url()}/v1/revoke/cascade",
            json={"contributor_id": contributor_id, "scope": "all"},
            timeout=HTTP_TIMEOUT_S,
        )
        if r.status_code == 200:
            server_request_id = r.json().get("request_id")
        else:
            logger.warning(
                "revoke cascade returned status=%d body=%s", r.status_code, r.text[:200],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("revoke cascade POST failed: %s", exc)

    console.print(messaging.revoke_success(contributor_id=contributor_id))
    if server_request_id:
        console.print(f"[dim]Server-side cascade request_id: {server_request_id}[/dim]")
        console.print(
            f"[dim]Track progress: [cyan]reflex contribute --revoke-status {server_request_id}[/cyan][/dim]"
        )
    else:
        console.print(
            "[yellow]⚠[/yellow]  Could not reach the contribution worker — local "
            "receipt was still removed. To trigger server-side cascade manually, "
            "email [cyan]privacy@fastcrest.com[/cyan] with your contributor_id above."
        )


def _cmd_status() -> None:
    if not curate_consent.is_opted_in():
        console.print(
            "[bold]Data contribution:[/bold] [yellow]not opted in[/yellow]\n"
            "[dim]Run [cyan]reflex contribute --opt-in[/cyan] to join, or "
            "[cyan]reflex contribute --info[/cyan] for privacy details first.[/dim]"
        )
        return

    receipt = curate_consent.load()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Status:", "[green]✓ opted in[/green]")
    table.add_row("Tier:", receipt.tier)
    table.add_row("contributor_id:", f"[dim]{receipt.contributor_id}[/dim]")
    table.add_row("Opted in at:", receipt.opted_in_at)
    table.add_row("Privacy mode:", receipt.privacy_mode)
    table.add_row("Terms version:", receipt.accepted_terms_version)
    table.add_row(
        "Receipt path:",
        f"[dim]{Path(curate_consent.DEFAULT_CONSENT_PATH).expanduser()}[/dim]",
    )
    console.print(table)
    console.print()

    # Live contribution stats from the worker.
    stats = _fetch_live_stats(receipt.contributor_id)
    if stats is None:
        console.print(
            "[dim]Could not reach the contribution worker — live counts "
            "unavailable. Local queue: [cyan]reflex contribute --inspect[/cyan].[/dim]"
        )
        return
    if stats.get("error") == "not_found":
        console.print(
            "[dim]No uploads recorded server-side yet — once your first session "
            "uploads, totals will appear here. Local queue: "
            "[cyan]reflex contribute --inspect[/cyan].[/dim]"
        )
        return

    live_table = Table(show_header=False, box=None, pad_edge=False, title="Server-side totals")
    live_table.add_column(style="bold")
    live_table.add_column()
    live_table.add_row("Episodes contributed:", str(stats.get("total_episodes", 0)))
    live_table.add_row("Total uploads:", str(stats.get("total_uploads", 0)))
    total_bytes = int(stats.get("total_bytes", 0))
    if total_bytes >= 1_000_000_000:
        size_str = f"{total_bytes / 1_000_000_000:.2f} GB"
    elif total_bytes >= 1_000_000:
        size_str = f"{total_bytes / 1_000_000:.1f} MB"
    else:
        size_str = f"{total_bytes / 1024:.1f} KB"
    live_table.add_row("Total bytes:", size_str)
    last_active = stats.get("last_active_at")
    if last_active:
        live_table.add_row("Last upload:", str(last_active))
    if stats.get("revoked_at"):
        live_table.add_row("Revoked at:", f"[red]{stats['revoked_at']}[/red]")
    console.print(live_table)


def _fetch_live_stats(contributor_id: str) -> dict | None:
    """GET /v1/contributors/<id>/stats. Returns None on connection error,
    {error: 'not_found'} on 404, or the worker's response dict on 200."""
    try:
        import httpx
        from reflex.curate.uploader import _worker_url, HTTP_TIMEOUT_S
        r = httpx.get(
            f"{_worker_url()}/v1/contributors/{contributor_id}/stats",
            timeout=HTTP_TIMEOUT_S,
        )
        if r.status_code == 404:
            return {"error": "not_found"}
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("fetch live stats failed: %s", exc)
        return None


def _cmd_inspect() -> None:
    """List files in ~/.reflex/contribute/queue/ + sibling subdirs."""
    from collections import defaultdict
    from reflex.curate.uploader import (
        DEFAULT_QUEUE_DIR,
        DEFAULT_REJECTED_DIR,
        DEFAULT_UPLOADED_DIR,
    )

    def _summarize(label: str, dir_path: Path) -> tuple[int, int, list[tuple[str, int, int]]]:
        if not dir_path.exists():
            return 0, 0, []
        entries: list[tuple[str, int, int]] = []
        total_bytes = 0
        for f in sorted(dir_path.glob("*.jsonl")):
            size = f.stat().st_size
            try:
                # Cheap line count for episode-ish granularity
                with open(f) as fh:
                    line_count = sum(1 for _ in fh)
            except OSError:
                line_count = -1
            entries.append((f.name, line_count, size))
            total_bytes += size
        return len(entries), total_bytes, entries

    queue = Path(DEFAULT_QUEUE_DIR).expanduser()
    uploaded = Path(DEFAULT_UPLOADED_DIR).expanduser()
    rejected = Path(DEFAULT_REJECTED_DIR).expanduser()

    n_q, b_q, files_q = _summarize("queue", queue)
    n_u, b_u, _ = _summarize("uploaded", uploaded)
    n_r, b_r, _ = _summarize("rejected", rejected)

    summary = Table(show_header=True, title="Contribution queue")
    summary.add_column("Bucket")
    summary.add_column("Files", justify="right")
    summary.add_column("Size", justify="right")
    summary.add_row(f"queue ({queue})", str(n_q), f"{b_q / 1024:.1f} KB")
    summary.add_row(f"uploaded ({uploaded})", str(n_u), f"{b_u / 1024:.1f} KB")
    summary.add_row(f"rejected ({rejected})", str(n_r), f"{b_r / 1024:.1f} KB")
    console.print(summary)

    if files_q:
        details = Table(show_header=True, title="Queued files")
        details.add_column("Name")
        details.add_column("Lines", justify="right")
        details.add_column("Size", justify="right")
        for name, lines, size in files_q:
            line_str = "?" if lines < 0 else str(lines)
            details.add_row(name, line_str, f"{size / 1024:.1f} KB")
        console.print(details)
        console.print(
            "[dim]Remove a queued file with: "
            "[cyan]reflex contribute --purge <name>[/cyan][/dim]"
        )
    else:
        console.print("[dim]No files currently in the upload queue.[/dim]")


def _cmd_purge(*, target: str, yes: bool) -> None:
    """Remove a specific queued file before upload."""
    from reflex.curate.uploader import DEFAULT_QUEUE_DIR

    queue = Path(DEFAULT_QUEUE_DIR).expanduser()
    if not queue.exists():
        console.print("[dim]No queue directory exists yet — nothing to purge.[/dim]")
        return
    target_path = queue / target
    if not target_path.exists() or not target_path.is_file():
        console.print(
            f"[red]No queued file named[/red] [cyan]{target}[/cyan] [red]in[/red] "
            f"{queue}.\n"
            f"[dim]Run [cyan]reflex contribute --inspect[/cyan] to see queued files.[/dim]"
        )
        raise typer.Exit(2)

    if not yes:
        try:
            ok = typer.confirm(f"Remove {target_path}?", default=False)
        except (EOFError, OSError):
            console.print("[red]Non-interactive — pass --yes to confirm.[/red]")
            raise typer.Exit(2)
        if not ok:
            console.print("[dim]Purge cancelled.[/dim]")
            return

    target_path.unlink()
    console.print(f"[green]✓[/green] Removed [cyan]{target_path}[/cyan].")


def _cmd_revoke_status(*, request_id: str) -> None:
    """Fetch cascade status from the contribution worker and render."""
    try:
        import httpx
        from reflex.curate.uploader import HTTP_TIMEOUT_S, _worker_url
        r = httpx.get(
            f"{_worker_url()}/v1/revoke/cascade-status/{request_id}",
            timeout=HTTP_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to reach worker:[/red] {exc}")
        raise typer.Exit(2)

    if r.status_code == 404:
        console.print(
            f"[red]Request not found:[/red] [cyan]{request_id}[/cyan]\n"
            f"[dim]Either the request_id is wrong, or it predates the cascade-status "
            f"endpoint deploy. Email privacy@fastcrest.com if you need help.[/dim]"
        )
        raise typer.Exit(2)
    if r.status_code != 200:
        console.print(f"[red]Worker returned status={r.status_code}:[/red] {r.text[:300]}")
        raise typer.Exit(2)

    data = r.json()
    overall = data.get("overall_status", "unknown")
    overall_color = "green" if overall == "completed" else "yellow"
    console.print(
        f"[bold]Revoke cascade:[/bold] [{overall_color}]{overall}[/{overall_color}]"
    )
    console.print(
        f"  request_id:     [dim]{data['request_id']}[/dim]"
    )
    console.print(
        f"  contributor_id: [dim]{data['contributor_id']}[/dim]"
    )
    console.print(f"  requested_at:   {data['requested_at']}")
    if data.get("completed_at"):
        console.print(f"  completed_at:   {data['completed_at']}")
    console.print()

    table = Table(title="Cascade stages", show_header=True)
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Completed at", overflow="fold")
    table.add_column("Detail")

    for stage in data.get("stages", []):
        name = stage.get("name", "?")
        st = stage.get("status", "?")
        st_color = {
            "completed": "green",
            "in_progress": "yellow",
            "pending": "dim",
        }.get(st, "white")
        at = stage.get("at") or "—"
        detail_parts = []
        if stage.get("objects_purged") is not None:
            detail_parts.append(f"{stage['objects_purged']} R2 objects")
        if stage.get("datasets_rebuilt") is not None:
            detail_parts.append(f"{stage['datasets_rebuilt']} datasets")
        if stage.get("notifications_sent") is not None:
            detail_parts.append(f"{stage['notifications_sent']} notifications")
        detail = " · ".join(detail_parts) or ""
        table.add_row(name, f"[{st_color}]{st}[/{st_color}]", at, detail)
    console.print(table)

    if overall != "completed":
        console.print(
            f"\n[dim]Cascade SLA: {data.get('sla_days', 30)} days. "
            f"Re-run [cyan]reflex contribute --revoke-status {request_id}[/cyan] "
            f"to refresh.[/dim]"
        )


__all__ = ["contribute_app"]
