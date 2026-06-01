"""Conformity-bundle export and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reflex.comply.collect import collect_evidence, copy_source_artifacts
from reflex.comply.mapping import build_gap_report, build_regulatory_mapping
from reflex.comply.pdf import write_text_pdf
from reflex.comply.sbom import generate_sbom, write_sbom
from reflex.comply.schemas import DeploymentMetadata, SCHEMA_VERSION, utc_now_iso
from reflex.comply.signing import sign_payload, verify_payload_signature
from reflex.parity_cert import verify_parity_cert_signature
from reflex.verification_report import _sha256

CONFORMITY_JSON = "conformity.json"
CONFORMITY_SIG = "conformity.sig"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _manifest(root: Path, *, exclude_names: set[str] | None = None) -> list[dict[str, Any]]:
    exclude = exclude_names or set()
    out: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in exclude:
            continue
        rel = p.relative_to(root).as_posix()
        out.append({
            "path": rel,
            "sha256": _sha256(p),
            "size_bytes": p.stat().st_size,
        })
    return out


def _write_safety_violations(path: Path, audit_summary: dict[str, Any]) -> Path:
    samples = audit_summary.get("safety_violations_sample")
    if not isinstance(samples, list):
        samples = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in samples))
    return path


def _write_model_hashes(path: Path, evidence: dict[str, Any]) -> Path:
    audit = evidence.get("audit_summary") or {}
    cert = evidence.get("parity_cert") or {}
    payload = {
        "from_audit_log": {
            "model_hashes": audit.get("model_hashes", []),
            "config_hashes": audit.get("config_hashes", []),
        },
        "from_parity_cert": {
            "optimized_ref": cert.get("optimized_ref"),
            "original_ref": cert.get("original_ref"),
            "target": cert.get("target"),
            "artifacts": cert.get("artifacts", {}),
        },
    }
    return _write_json(path, payload)


def _render_status(status: str) -> str:
    if status == "covered_by_reflex_evidence":
        return "Covered by Reflex evidence"
    if status == "partial_customer_action_required":
        return "Partial - customer action required"
    return "Gap"


def _render_technical_file(
    *,
    deployment: DeploymentMetadata,
    evidence: dict[str, Any],
    mapping: list[dict[str, Any]],
    generated_at: str,
) -> str:
    cert = evidence.get("parity_cert") or {}
    audit = evidence.get("audit_summary") or {}
    actionguard = evidence.get("actionguard") or {}
    lines = [
        "# Reflex EU Technical Documentation Evidence File",
        "",
        f"Generated: {generated_at}",
        "",
        "## Scope",
        "",
        "This file is an evidence pack for a robot deployment running Reflex. Reflex does not declare CE conformity. The manufacturer remains responsible for the system-level risk assessment, intended-purpose documentation, conformity assessment route, and notified-body interactions.",
        "",
        "## Deployment",
        "",
        f"- Product/deployment: {deployment.product_name}",
        f"- Deployment ID: {deployment.deployment_id or 'not provided'}",
        f"- Robot ID: {deployment.robot_id or 'not provided'}",
        f"- Manufacturer: {deployment.manufacturer or 'not provided'}",
        f"- Operator: {deployment.operator or 'not provided'}",
        f"- Data residency: {deployment.data_residency}",
        f"- Retention days: {deployment.retention_days}",
        "",
        "## Model Verification",
        "",
        f"- Parity verdict: {cert.get('verdict', 'missing')}",
        f"- Target: {cert.get('target', 'missing')}",
        f"- Optimized model/export ref: {cert.get('optimized_ref', 'missing')}",
        f"- Original/reference ref: {cert.get('original_ref', 'missing')}",
        f"- Signed parity certificate: {'yes' if cert.get('signature') else 'no'}",
        f"- Parity certificate signature valid: {evidence.get('parity_cert_signature_valid')}",
        "",
        "## Runtime Audit Feed",
        "",
        f"- Audit present: {audit.get('present', False)}",
        f"- Request/event count: {audit.get('request_count', 0)}",
        f"- Time range: {audit.get('first_timestamp', '')} to {audit.get('last_timestamp', '')}",
        f"- Model hashes observed: {', '.join(audit.get('model_hashes', [])) or 'none'}",
        f"- Config hashes observed: {', '.join(audit.get('config_hashes', [])) or 'none'}",
        f"- Tamper-evidence head: {(audit.get('tamper_evidence') or {}).get('head', '')}",
        f"- Safety violations: {audit.get('safety_violation_count', 0)}",
        f"- Errors: {audit.get('error_count', 0)}",
        "",
        "## Safety Function",
        "",
        f"- ActionGuard config present: {actionguard.get('present', False)}",
        f"- Joint count: {actionguard.get('joint_count', 0)}",
        f"- Position limits: {actionguard.get('has_position_limits', False)}",
        f"- Velocity limits: {actionguard.get('has_velocity_limits', False)}",
        f"- Workspace limits: {actionguard.get('has_workspace_limits', False)}",
        "",
        "## Regulatory Mapping",
        "",
        "| Control | Regulation | Status | Reflex evidence | Customer action |",
        "|---|---|---|---|---|",
    ]
    for control in mapping:
        lines.append(
            "| {control_id} | {regulation} {article} | {status} | {evidence} | {gap} |".format(
                control_id=control["control_id"],
                regulation=control["regulation"],
                article=control["article"],
                status=_render_status(control["status"]),
                evidence=", ".join(control["reflex_evidence"]),
                gap=control["customer_gap"] or "None",
            )
        )
    lines.extend([
        "",
        "## Auditor Verification",
        "",
        "Run `reflex comply verify-bundle <bundle_dir>` to check the conformity JSON signature, parity certificate signature, and artifact hashes.",
        "",
    ])
    return "\n".join(lines)


def _render_gap_report(gaps: list[dict[str, Any]]) -> str:
    lines = [
        "# Reflex Comply Gap Report",
        "",
        "Reflex evidence reduces the compliance-documentation workload, but it does not replace the manufacturer's legal, safety, and quality-system obligations.",
        "",
    ]
    if not gaps:
        lines.append("No open gaps detected by Reflex Comply. A qualified reviewer should still inspect the full technical file.")
        return "\n".join(lines) + "\n"
    for gap in gaps:
        lines.extend([
            f"## {gap['control_id']}",
            "",
            f"- Regulation: {gap['regulation']}",
            f"- Article/control: {gap['article']}",
            f"- Status: {_render_status(gap['status'])}",
            f"- Customer still needs: {gap['customer_gap']}",
            "",
        ])
    return "\n".join(lines)


def _render_trust_page(deployment: DeploymentMetadata, evidence: dict[str, Any]) -> str:
    audit = evidence.get("audit_summary") or {}
    redaction = audit.get("redaction") if isinstance(audit.get("redaction"), dict) else {}
    return "\n".join([
        "# Reflex Deployment Trust Page",
        "",
        "## Data Flow",
        "",
        "Robot -> Reflex runtime -> on-device ActionGuard -> local/customer-controlled audit log -> exported conformity bundle.",
        "",
        "## Stored Data",
        "",
        "- Model/config hashes",
        "- Timestamped action decisions",
        "- Latency and error counters",
        "- Safety-violation events",
        "- Image SHA-256 by default when recording is configured for hash-only redaction",
        "",
        "## Privacy Controls",
        "",
        f"- Image hashes observed: {redaction.get('image_sha256_count', 0)}",
        f"- Raw images observed in audit: {redaction.get('image_b64_count', 0)}",
        f"- Instruction hashes observed: {redaction.get('instruction_hash_count', 0)}",
        f"- Raw instructions observed: {redaction.get('raw_instruction_count', 0)}",
        "",
        "## Residency, Retention, Deletion",
        "",
        f"- Data residency: {deployment.data_residency}",
        f"- Nominal retention: {deployment.retention_days} days",
        "- GDPR erasure handling: delete/revoke customer trace files and regenerate bundles without revoked records.",
        "",
        "## Encryption",
        "",
        "Reflex writes local files. The deployment owner is responsible for disk encryption, access control, and backup policy in the target environment.",
        "",
    ])


def _render_security_whitepaper(deployment: DeploymentMetadata) -> str:
    return "\n".join([
        "# Reflex Security Whitepaper",
        "",
        "## Secure-by-Design Position",
        "",
        "Reflex minimizes cloud dependency by running model serving, ActionGuard, and audit logging on the robot or customer-controlled infrastructure.",
        "",
        "## Integrity Controls",
        "",
        "- Signed parity certificate proves the validated model/export identity.",
        "- Conformity bundle signature seals the exported evidence set.",
        "- Artifact manifest records SHA-256 for every included file.",
        "- Audit summary includes a hash-chain head over canonical JSONL records.",
        "",
        "## Runtime Controls",
        "",
        "- ActionGuard enforces deterministic joint/workspace/velocity bounds.",
        "- Non-finite action detection prevents NaN/Inf actions from reaching actuators.",
        "- Safety-violation counts are included in audit evidence.",
        "",
        "## Customer Responsibilities",
        "",
        "- Secure boot and OS hardening on the robot computer.",
        "- Secrets management for signing keys.",
        "- Network segmentation and authenticated access to any exposed API.",
        "- Vulnerability intake and patch deployment.",
        "",
        f"Security contact for this deployment: {deployment.vulnerability_contact}",
        "",
    ])


def _render_vulnerability_manifest(deployment: DeploymentMetadata) -> str:
    return "\n".join([
        "# Vulnerability Handling Manifest",
        "",
        "## Contact",
        "",
        f"- Security contact: {deployment.vulnerability_contact}",
        "",
        "## Intake",
        "",
        "- Accept vulnerability reports through the security contact above.",
        "- Record report timestamp, affected component, affected version, exploitability, and reporter contact.",
        "- Assign severity using CVSS or an equivalent documented rubric.",
        "",
        "## Triage SLA",
        "",
        "- Critical: initial triage within 1 business day.",
        "- High: initial triage within 3 business days.",
        "- Medium/Low: initial triage within 10 business days.",
        "",
        "## CVE Tracking",
        "",
        "| Component | Version | CVE | Severity | Status | Fix version | Notes |",
        "|---|---|---|---|---|---|---|",
        "| reflex-vla | see SBOM | TBD | TBD | monitoring | TBD | Fill during operations |",
        "",
        "## Update Handling",
        "",
        "When a model, runtime, or safety config changes, rerun `reflex verify` and `reflex comply export` so the technical file contains the updated parity certificate and artifact hashes.",
        "",
    ])


def export_conformity_bundle(
    *,
    verify_dir: str | Path,
    out_dir: str | Path,
    audit_log: str | Path | None = None,
    actionguard: str | Path | None = None,
    deployment: DeploymentMetadata | None = None,
    signing_key: str = "",
    key_id: str = "",
    include_environment_sbom: bool = True,
) -> dict[str, Any]:
    out = Path(out_dir)
    artifacts_dir = out / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    deployment = deployment or DeploymentMetadata()
    generated_at = utc_now_iso()

    collection = collect_evidence(
        verify_dir=verify_dir,
        audit_log=audit_log,
        actionguard=actionguard,
    )
    evidence = collection.to_dict()
    copied_artifacts = copy_source_artifacts(collection, artifacts_dir)

    audit_summary_path = _write_json(artifacts_dir / "audit_summary.json", collection.audit_summary)
    safety_path = _write_safety_violations(artifacts_dir / "safety_violations.jsonl", collection.audit_summary)
    model_hashes_path = _write_model_hashes(artifacts_dir / "model_hashes.json", evidence)

    sbom = generate_sbom(
        artifact_paths=copied_artifacts + [audit_summary_path, safety_path, model_hashes_path],
        include_environment=include_environment_sbom,
    )
    sbom_path = write_sbom(out / "SBOM.cyclonedx.json", sbom)
    mapping = [c.to_dict() for c in build_regulatory_mapping(evidence, sbom_present=sbom_path.exists())]
    gaps = build_gap_report(build_regulatory_mapping(evidence, sbom_present=sbom_path.exists()))

    technical_md = _render_technical_file(
        deployment=deployment,
        evidence=evidence,
        mapping=mapping,
        generated_at=generated_at,
    )
    (out / "TECHNICAL_FILE.md").write_text(technical_md + "\n")
    write_text_pdf(out / "TECHNICAL_FILE.pdf", title="Reflex EU Technical Documentation Evidence File", text=technical_md)
    (out / "GAP_REPORT.md").write_text(_render_gap_report(gaps))
    (out / "TRUST_PAGE.md").write_text(_render_trust_page(deployment, evidence))
    (out / "SECURITY_WHITEPAPER.md").write_text(_render_security_whitepaper(deployment))
    (out / "VULNERABILITY_HANDLING.md").write_text(_render_vulnerability_manifest(deployment))

    manifest = _manifest(out, exclude_names={CONFORMITY_JSON, CONFORMITY_SIG})
    conformity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "deployment": deployment.to_dict(),
        "legal_notice": "Reflex Comply produces evidence for a manufacturer's technical file; it does not certify or declare conformity.",
        "evidence": evidence,
        "regulatory_mapping": mapping,
        "gap_report": gaps,
        "artifact_manifest": manifest,
    }
    if signing_key:
        conformity = sign_payload(conformity, signing_key=signing_key, key_id=key_id)
        (out / CONFORMITY_SIG).write_text(conformity["signature"]["sig"] + "\n")
    else:
        conformity["signature_status"] = "unsigned"
    conformity_path = _write_json(out / CONFORMITY_JSON, conformity)

    return {
        "bundle_dir": str(out),
        "conformity_json": str(conformity_path),
        "conformity_sig": str(out / CONFORMITY_SIG) if signing_key else "",
        "technical_file_md": str(out / "TECHNICAL_FILE.md"),
        "technical_file_pdf": str(out / "TECHNICAL_FILE.pdf"),
        "sbom": str(sbom_path),
        "gap_report": str(out / "GAP_REPORT.md"),
        "signed": bool(signing_key),
        "gaps": gaps,
    }


def verify_conformity_bundle(bundle_dir: str | Path, *, require_signature: bool = False) -> dict[str, Any]:
    root = Path(bundle_dir)
    issues: list[str] = []
    conformity_path = root / CONFORMITY_JSON
    if not conformity_path.exists():
        return {"passed": False, "issues": [f"missing {CONFORMITY_JSON}"], "bundle_dir": str(root)}
    try:
        conformity = json.loads(conformity_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "issues": [f"unreadable {CONFORMITY_JSON}: {exc}"], "bundle_dir": str(root)}

    if conformity.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"unexpected schema_version: {conformity.get('schema_version')!r}")

    if "signature" in conformity:
        try:
            verify_payload_signature(conformity)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"conformity signature invalid: {exc}")
    elif require_signature:
        issues.append("conformity signature missing")

    for item in conformity.get("artifact_manifest", []):
        rel = item.get("path")
        if not rel:
            issues.append("manifest entry missing path")
            continue
        p = root / rel
        if not p.exists():
            issues.append(f"manifest file missing: {rel}")
            continue
        actual = _sha256(p)
        if actual != item.get("sha256"):
            issues.append(f"manifest hash mismatch: {rel}")

    parity_cert_path = root / "artifacts" / "parity.cert.json"
    if parity_cert_path.exists():
        try:
            cert = json.loads(parity_cert_path.read_text())
            if isinstance(cert.get("signature"), dict):
                verify_parity_cert_signature(cert)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"parity cert invalid: {exc}")
    else:
        issues.append("missing artifacts/parity.cert.json")

    return {
        "passed": not issues,
        "issues": issues,
        "bundle_dir": str(root),
        "schema_version": conformity.get("schema_version"),
        "signed": "signature" in conformity,
        "artifact_count": len(conformity.get("artifact_manifest", [])),
    }


__all__ = ["export_conformity_bundle", "verify_conformity_bundle"]
