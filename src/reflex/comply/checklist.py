"""Regulatory requirement → Reflex-evidence mapping (Reflex Comply).

Maps the concrete requirements of the EU AI Act, the EU Machinery Regulation
2023/1230, the Cyber Resilience Act, and GDPR to the artifacts the Reflex runtime
produces, with an HONEST role per requirement:

- ``satisfied``  — Reflex emits the evidence that meets this requirement.
- ``partial``    — Reflex provides part of the evidence; the deployer completes it.
- ``customer``   — entirely the deployer's / notified body's responsibility;
                   Reflex makes no claim (CE marking, the overall risk assessment, etc.).

This is deliberately not a compliance rubber-stamp: Reflex produces verifiable,
tamper-evident *evidence*; the deployer + a notified body assemble it into the
conformity file and sign the mark. The honesty IS the product — an auditor can
read this mapping and the source that backs each artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Evidence-artifact keys (what the bundle looks for on the robot / in the export).
ARTIFACTS = {
    "audit_log": "Tamper-evident per-action JSONL audit log (action + safety events + model-hash)",
    "parity_cert": "Signed VERIFICATION.md / parity cert (deployed model = validated model)",
    "actionguard": "ActionGuard safety bounds (joint/velocity/workspace limits, NaN detection)",
    "safety_log": "Logged ActionGuard safety violations / interventions",
    "sbom": "Software Bill of Materials (CycloneDX)",
    "anonymization": "Edge-side anonymization (image-hash by default, face blur, instruction hashing)",
    "erasure": "Right-to-erasure mechanism (reflex data revoke)",
    "model_metadata": "Export metadata (model id, target, opset, denoise steps, hashes)",
}

SATISFIED = "satisfied"
PARTIAL = "partial"
CUSTOMER = "customer"


@dataclass(frozen=True)
class RegRequirement:
    framework: str          # "EU AI Act" | "EU Machinery Reg 2023/1230" | "CRA" | "GDPR"
    ref: str                # e.g. "Art 12"
    title: str
    role: str               # SATISFIED | PARTIAL | CUSTOMER
    evidence: tuple[str, ...] = ()   # artifact keys Reflex contributes
    note: str = ""


# The mapping. References are to the operative articles; roles are intentionally
# conservative (we under-claim, never over-claim).
REQUIREMENTS: tuple[RegRequirement, ...] = (
    # ---------------- EU AI Act (high-risk AI systems) ----------------
    RegRequirement(
        "EU AI Act", "Art 9", "Risk management system", PARTIAL,
        ("actionguard", "audit_log"),
        "Reflex provides the deterministic safety bounds + the behavioral record; the overall risk-management *process* is the deployer's.",
    ),
    RegRequirement(
        "EU AI Act", "Art 10", "Data & data governance", PARTIAL,
        ("anonymization", "erasure"),
        "Reflex anonymizes captured data at the edge + supports erasure; training-data governance is the deployer's.",
    ),
    RegRequirement(
        "EU AI Act", "Art 11", "Technical documentation", PARTIAL,
        ("parity_cert", "model_metadata", "sbom"),
        "Reflex supplies the deployment tech docs (cert + export metadata + SBOM) into the Annex IV file; the full file is the deployer's.",
    ),
    RegRequirement(
        "EU AI Act", "Art 12", "Record-keeping (logging)", SATISFIED,
        ("audit_log",),
        "The tamper-evident per-action audit log is the Art-12 record.",
    ),
    RegRequirement(
        "EU AI Act", "Art 13", "Transparency & instructions for use", CUSTOMER,
        (),
        "User-facing instructions are the deployer's responsibility.",
    ),
    RegRequirement(
        "EU AI Act", "Art 14", "Human oversight", PARTIAL,
        ("actionguard", "safety_log"),
        "ActionGuard bounds + intervention/violation logging support oversight; the operational oversight process is the deployer's.",
    ),
    RegRequirement(
        "EU AI Act", "Art 15", "Accuracy, robustness & cybersecurity", PARTIAL,
        ("parity_cert", "sbom"),
        "The parity cert proves the deployed model matches the validated one (deployment robustness) + SBOM covers cybersecurity; model-level accuracy is the deployer's.",
    ),
    # ---------------- EU Machinery Regulation 2023/1230 (from 2027-01-20) ----------------
    RegRequirement(
        "EU Machinery Reg 2023/1230", "Annex III §1.2", "Safety & reliability of control systems (self-evolving behaviour)", PARTIAL,
        ("actionguard", "safety_log"),
        "ActionGuard is the deterministic safety function bounding the AI; the machine-level safety design is the manufacturer's.",
    ),
    RegRequirement(
        "EU Machinery Reg 2023/1230", "Art 10 / Annex IV", "Documented safety proofs for self-evolving behaviour", SATISFIED,
        ("audit_log", "parity_cert"),
        "The audit log + parity cert are the documented evidence that the AI behaviour was bounded and the deployed model was the validated one.",
    ),
    RegRequirement(
        "EU Machinery Reg 2023/1230", "Art 18", "Re-conformity on substantial modification (model update)", SATISFIED,
        ("parity_cert", "model_metadata"),
        "A fresh signed cert per model version is the re-conformity evidence when the policy is updated.",
    ),
    RegRequirement(
        "EU Machinery Reg 2023/1230", "Annex V", "Risk assessment & CE marking", CUSTOMER,
        (),
        "The risk assessment, technical file sign-off, and CE marking are the manufacturer's / notified body's.",
    ),
    # ---------------- Cyber Resilience Act ----------------
    RegRequirement(
        "CRA", "Annex I Part II §1", "Software Bill of Materials", SATISFIED,
        ("sbom",),
        "Auto-generated CycloneDX SBOM of the runtime.",
    ),
    RegRequirement(
        "CRA", "Annex I Part II §2", "Vulnerability handling", PARTIAL,
        ("sbom",),
        "Reflex provides the SBOM to drive vuln tracking; the product-level vulnerability-handling process is the deployer's.",
    ),
    RegRequirement(
        "CRA", "Art 13", "Conformity assessment", CUSTOMER,
        (),
        "The product conformity assessment is the manufacturer's.",
    ),
    # ---------------- GDPR ----------------
    RegRequirement(
        "GDPR", "Art 5(1)(c)", "Data minimisation", SATISFIED,
        ("anonymization",),
        "Image-hash-by-default + edge anonymization means raw footage never leaves the device.",
    ),
    RegRequirement(
        "GDPR", "Art 17", "Right to erasure", SATISFIED,
        ("erasure",),
        "reflex data revoke deletes a contributor's uploads.",
    ),
    RegRequirement(
        "GDPR", "Art 6 / Art 28", "Lawful basis & processor agreement", CUSTOMER,
        (),
        "Lawful basis + any DPA are the deployer's.",
    ),
)

FRAMEWORK_KEYS = {
    "ai_act": "EU AI Act",
    "eu_mr": "EU Machinery Reg 2023/1230",
    "cra": "CRA",
    "gdpr": "GDPR",
}


@dataclass
class RequirementStatus:
    requirement: RegRequirement
    status: str              # "met" | "partial" | "gap" | "customer-responsibility"
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        r = self.requirement
        return {
            "framework": r.framework,
            "ref": r.ref,
            "title": r.title,
            "reflex_role": r.role,
            "status": self.status,
            "evidence": list(r.evidence),
            "missing": self.missing,
            "note": r.note,
        }


def evaluate_requirement(req: RegRequirement, available: set[str]) -> RequirementStatus:
    """Resolve a requirement's status against the artifacts actually present."""
    if req.role == CUSTOMER:
        return RequirementStatus(req, "customer-responsibility")
    missing = [a for a in req.evidence if a not in available]
    if missing:
        return RequirementStatus(req, "gap", missing)
    return RequirementStatus(req, "met" if req.role == SATISFIED else "partial")


def evaluate(
    available: set[str], frameworks: tuple[str, ...] | None = None
) -> list[RequirementStatus]:
    """Evaluate every requirement (optionally filtered to ``frameworks`` keys)."""
    wanted = None
    if frameworks:
        wanted = {FRAMEWORK_KEYS[f] for f in frameworks if f in FRAMEWORK_KEYS}
    out = []
    for req in REQUIREMENTS:
        if wanted is not None and req.framework not in wanted:
            continue
        out.append(evaluate_requirement(req, available))
    return out
