"""Regulatory-control mapping for Reflex Comply bundles."""

from __future__ import annotations

from typing import Any

from reflex.comply.schemas import RegulatoryControl


def _status(condition: bool, *, customer_gap: str = "") -> str:
    if condition and not customer_gap:
        return "covered_by_reflex_evidence"
    if condition:
        return "partial_customer_action_required"
    return "gap"


def build_regulatory_mapping(evidence: dict[str, Any], *, sbom_present: bool) -> list[RegulatoryControl]:
    parity_cert = evidence.get("parity_cert") or {}
    audit = evidence.get("audit_summary") or {}
    actionguard = evidence.get("actionguard") or {}
    audit_present = bool(audit.get("present") and audit.get("record_count", 0) > 0)
    actionguard_present = bool(actionguard.get("present"))
    parity_present = bool(parity_cert)
    parity_signed = bool(parity_cert.get("signature"))
    redaction = audit.get("redaction") if isinstance(audit.get("redaction"), dict) else {}
    privacy_evidence = bool(
        redaction.get("image_sha256_count", 0) or redaction.get("instruction_hash_count", 0)
    )

    return [
        RegulatoryControl(
            control_id="eu-ai-act.art-11",
            regulation="EU AI Act (Regulation (EU) 2024/1689)",
            article="Article 11 - technical documentation",
            requirement="Maintain technical documentation showing model identity, validation, deployment configuration, and evidence supporting conformity.",
            reflex_evidence=["parity.cert.json", "PARITY.md", "conformity.json"],
            customer_gap="Manufacturer must add intended purpose, risk-management file, training/data governance documentation, and notified-body submission context.",
            status=_status(parity_present, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="eu-ai-act.art-12",
            regulation="EU AI Act (Regulation (EU) 2024/1689)",
            article="Article 12 - record keeping / logging",
            requirement="Keep automatic logs sufficient for traceability of high-risk AI operation.",
            reflex_evidence=["audit_summary.json", "tamper-evidence hash-chain head", "model/config hashes"],
            customer_gap="Manufacturer/operator must define retention policy, access controls, and operational log-review procedure.",
            status=_status(audit_present, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="eu-ai-act.art-14",
            regulation="EU AI Act (Regulation (EU) 2024/1689)",
            article="Article 14 - human oversight",
            requirement="Enable natural persons to oversee, interpret, interrupt, or stop AI operation where needed.",
            reflex_evidence=["ActionGuard config", "safety-violation log", "webhook/event counters when enabled"],
            customer_gap="Customer must document responsible operator roles, stop procedure, escalation policy, and training.",
            status=_status(actionguard_present, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="eu-ai-act.art-15",
            regulation="EU AI Act (Regulation (EU) 2024/1689)",
            article="Article 15 - accuracy, robustness, cybersecurity",
            requirement="Design for appropriate accuracy, robustness, resilience, cybersecurity, and error handling.",
            reflex_evidence=["signed parity cert", "ActionGuard limits", "SBOM", "audit error/safety counters"],
            customer_gap="Customer must add system-level validation, threat model, secure update process, and production QA evidence.",
            status=_status(parity_present and actionguard_present and sbom_present, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="eu-machinery-reg.self-evolving-safety",
            regulation="EU Machinery Regulation (EU) 2023/1230",
            article="Safety function evidence for software/self-evolving behaviour",
            requirement="Document deterministic safety functions and re-validation evidence when software/model updates alter behaviour.",
            reflex_evidence=["ActionGuard as deterministic safety boundary", "parity cert for deployed model hash", "re-certify-on-update evidence"],
            customer_gap="Manufacturer must integrate this evidence into the machine technical file, risk assessment, and CE conformity route.",
            status=_status(actionguard_present and parity_present, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="cra.sbom",
            regulation="Cyber Resilience Act",
            article="Secure-by-design / software supply chain evidence",
            requirement="Maintain software component inventory and vulnerability-management evidence for products with digital elements.",
            reflex_evidence=["SBOM.cyclonedx.json", "VULNERABILITY_HANDLING.md"],
            customer_gap="Customer must operate vulnerability intake, triage SLAs, coordinated disclosure, and update distribution.",
            status=_status(sbom_present, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="gdpr.privacy",
            regulation="GDPR",
            article="Consent, data minimization, anonymization, erasure",
            requirement="Limit personal data, document consent/legal basis, enable deletion, and avoid unnecessary raw biometric/video transfer.",
            reflex_evidence=["edge image hashing/redaction", "instruction hashing where enabled", "TRUST_PAGE.md deletion/retention section"],
            customer_gap="Customer must provide legal basis, notices, consent records where required, DPA/ROPA, and data-subject request process.",
            status=_status(privacy_evidence, customer_gap="yes"),
        ),
        RegulatoryControl(
            control_id="reflex.parity.identity",
            regulation="Reflex deployment control",
            article="Deployed model equals validated model",
            requirement="Show the model/config being served is the same model/config that passed verification.",
            reflex_evidence=["parity.cert.json signature", "model hash", "config hash"],
            customer_gap="" if parity_signed else "Use --signing-key for production evidence accepted by third-party auditors.",
            status=_status(parity_present and parity_signed, customer_gap="" if parity_signed else "yes"),
        ),
    ]


def build_gap_report(mapping: list[RegulatoryControl]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for control in mapping:
        if control.status == "covered_by_reflex_evidence":
            continue
        gaps.append({
            "control_id": control.control_id,
            "regulation": control.regulation,
            "article": control.article,
            "status": control.status,
            "customer_gap": control.customer_gap,
        })
    return gaps


__all__ = ["build_gap_report", "build_regulatory_mapping"]
