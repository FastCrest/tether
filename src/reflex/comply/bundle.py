"""Conformity-bundle assembler (Reflex Comply).

Takes what a Reflex deployment produces on the robot — the signed parity cert
(``VERIFICATION.md``), the export metadata, the tamper-evident audit log, and the
generated SBOM — hashes each piece of evidence, summarizes the audit log, and maps
it all to the EU AI Act / Machinery Reg / CRA / GDPR checklist with an honest
per-requirement status (met / partial / gap / customer-responsibility).

Output: a structured ``conformity.json`` + a human-readable ``CONFORMITY.md`` + the
``sbom.json`` — the bundle a deployer hands their notified body. Pure stdlib; runs
on the robot, no network.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from reflex.comply.checklist import ARTIFACTS, RequirementStatus, evaluate
from reflex.comply.sbom import generate_sbom

# Runtime capabilities present in the Reflex serve path (the deployer must ENABLE
# them in their deployment; the report flags that explicitly).
DEFAULT_CAPABILITIES = frozenset({"actionguard", "anonymization", "erasure", "safety_log"})
DEFAULT_FRAMEWORKS = ("ai_act", "eu_mr", "cra", "gdpr")
_HASH_CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize_audit_log(path: str | Path) -> dict[str, Any]:
    """Summarize a tamper-evident audit log without trusting its contents blindly.

    Reports record count, the model-hash(es) seen (consistency is the tamper
    signal), the time range, and how many safety events were logged.
    """
    path = Path(path)
    n = 0
    model_hashes: set[str] = set()
    first_ts: str | None = None
    last_ts: str | None = None
    safety_events = 0
    for rec in _read_jsonl(path):
        n += 1
        mh = rec.get("model_hash") or (rec.get("meta") or {}).get("model_hash")
        if mh:
            model_hashes.add(str(mh))
        ts = rec.get("timestamp") or rec.get("ts")
        if ts:
            first_ts = first_ts or str(ts)
            last_ts = str(ts)
        # ActionGuard violations may be flagged inline; count defensively.
        if rec.get("guard_clamped") or rec.get("safety_violation") or rec.get("guard_violations"):
            safety_events += 1
    return {
        "records": n,
        "model_hashes": sorted(model_hashes),
        "model_hash_consistent": len(model_hashes) <= 1,
        "time_range": [first_ts, last_ts],
        "safety_events": safety_events,
        "file_sha256": _sha256_file(path) if path.exists() else None,
    }


@dataclass
class ComplyReport:
    generated_at: str
    frameworks: list[str]
    artifacts: dict[str, dict[str, Any]]
    audit_summary: dict[str, Any] | None
    checklist: list[dict[str, Any]]
    counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "frameworks": self.frameworks,
            "artifacts": self.artifacts,
            "audit_summary": self.audit_summary,
            "checklist": self.checklist,
            "counts": self.counts,
            "notes": self.notes,
            "disclaimer": (
                "Reflex produces verifiable evidence; it does not issue the CE mark. "
                "The deployer + notified body assemble this into the conformity file and sign."
            ),
        }


def build_conformity_bundle(
    export_dir: str | Path,
    audit_log_path: str | Path | None = None,
    *,
    frameworks: tuple[str, ...] = DEFAULT_FRAMEWORKS,
    capabilities: frozenset[str] = DEFAULT_CAPABILITIES,
    out_dir: str | Path | None = None,
) -> ComplyReport:
    """Assemble the conformity bundle for one deployment / export directory."""
    export_dir = Path(export_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    available: set[str] = set()

    def _add(key: str, present: bool, path: Path | None = None) -> None:
        entry: dict[str, Any] = {"present": present, "description": ARTIFACTS.get(key, "")}
        if present and path is not None and path.exists():
            entry["path"] = str(path)
            entry["sha256"] = _sha256_file(path)
        artifacts[key] = entry
        if present:
            available.add(key)

    # file-detected evidence
    cert = export_dir / "VERIFICATION.md"
    _add("parity_cert", cert.exists(), cert)
    cfg = export_dir / "reflex_config.json"
    _add("model_metadata", cfg.exists(), cfg)
    audit_summary: dict[str, Any] | None = None
    if audit_log_path is not None and Path(audit_log_path).exists():
        audit_summary = summarize_audit_log(audit_log_path)
        _add("audit_log", True, Path(audit_log_path))
    else:
        _add("audit_log", False)

    # generated evidence
    sbom = generate_sbom()
    artifacts["sbom"] = {
        "present": True,
        "description": ARTIFACTS["sbom"],
        "components": len(sbom.get("components", [])),
    }
    available.add("sbom")

    # runtime-capability evidence (present in the runtime; deployer must enable)
    notes: list[str] = []
    for cap in ("actionguard", "anonymization", "erasure", "safety_log"):
        present = cap in capabilities
        artifacts[cap] = {"present": present, "description": ARTIFACTS[cap], "kind": "runtime-capability"}
        if present:
            available.add(cap)
            notes.append(f"'{cap}' is a runtime capability — confirm it is ENABLED in your deployment config.")

    statuses: list[RequirementStatus] = evaluate(available, frameworks)
    checklist = [s.to_dict() for s in statuses]
    counts = {"met": 0, "partial": 0, "gap": 0, "customer-responsibility": 0}
    for s in statuses:
        counts[s.status] = counts.get(s.status, 0) + 1

    report = ComplyReport(
        generated_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        frameworks=list(frameworks),
        artifacts=artifacts,
        audit_summary=audit_summary,
        checklist=checklist,
        counts=counts,
        notes=notes,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "conformity.json").write_text(json.dumps(report.to_dict(), indent=2))
        (out / "sbom.json").write_text(json.dumps(sbom, indent=2))
        (out / "CONFORMITY.md").write_text(render_markdown(report))

    return report


def render_markdown(report: ComplyReport) -> str:
    c = report.counts
    lines = [
        "# Reflex Comply — Conformity Evidence Bundle",
        "",
        f"Generated: {report.generated_at}",
        f"Frameworks: {', '.join(report.frameworks)}",
        "",
        "> **Disclaimer:** Reflex produces verifiable, tamper-evident *evidence*. It does "
        "**not** issue the CE mark. The deployer and a notified body assemble this into the "
        "conformity file and sign it.",
        "",
        "## Summary",
        "",
        f"- ✅ met: **{c.get('met', 0)}**",
        f"- 🟡 partial (evidence provided, deployer completes): **{c.get('partial', 0)}**",
        f"- ❌ gap (action needed): **{c.get('gap', 0)}**",
        f"- 👤 deployer / notified-body responsibility: **{c.get('customer-responsibility', 0)}**",
        "",
        "## Evidence artifacts",
        "",
        "| Artifact | Present | Detail |",
        "|---|---|---|",
    ]
    for key, a in report.artifacts.items():
        detail = a.get("sha256", a.get("components", a.get("kind", "")))
        if isinstance(detail, int):
            detail = f"{detail} components"
        elif isinstance(detail, str) and len(detail) == 64:
            detail = f"sha256 {detail[:12]}…"
        lines.append(f"| `{key}` | {'✅' if a['present'] else '❌'} | {detail} |")

    if report.audit_summary:
        s = report.audit_summary
        lines += [
            "",
            "## Audit log",
            "",
            f"- records: **{s['records']}**",
            f"- model-hash consistent (tamper signal): **{s['model_hash_consistent']}** ({', '.join(s['model_hashes']) or 'none'})",
            f"- time range: {s['time_range'][0]} → {s['time_range'][1]}",
            f"- safety events logged: **{s['safety_events']}**",
        ]

    lines += ["", "## Requirement checklist", "", "| Framework | Ref | Requirement | Status | Note |", "|---|---|---|---|---|"]
    icon = {"met": "✅", "partial": "🟡", "gap": "❌", "customer-responsibility": "👤"}
    for r in report.checklist:
        status = r["status"]
        cell = f"{icon.get(status, '')} {status}"
        if r["missing"]:
            cell += f" (missing: {', '.join(r['missing'])})"
        lines.append(f"| {r['framework']} | {r['ref']} | {r['title']} | {cell} | {r['note']} |")

    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    lines.append("")
    return "\n".join(lines)
