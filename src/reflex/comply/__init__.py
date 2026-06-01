"""reflex.comply — Reflex Comply: the EU conformity evidence pack.

Maps the Reflex runtime's tamper-evident artifacts (audit log, signed parity cert,
ActionGuard, auto-SBOM) to the EU AI Act / Machinery Reg 2023/1230 / CRA / GDPR
checklist and assembles a conformity-evidence bundle a deployer hands their
notified body. Runs on the robot, pure stdlib, no network — and source-available,
so an auditor can read exactly what each piece of evidence means.

Reflex produces verifiable evidence; it does NOT issue the CE mark.
"""
from __future__ import annotations

from reflex.comply.bundle import (
    DEFAULT_CAPABILITIES,
    DEFAULT_FRAMEWORKS,
    ComplyReport,
    build_conformity_bundle,
    render_markdown,
    summarize_audit_log,
)
from reflex.comply.checklist import (
    ARTIFACTS,
    REQUIREMENTS,
    RegRequirement,
    RequirementStatus,
    evaluate,
    evaluate_requirement,
)
from reflex.comply.sbom import generate_sbom, sbom_json

__all__ = [
    "ARTIFACTS",
    "ComplyReport",
    "DEFAULT_CAPABILITIES",
    "DEFAULT_FRAMEWORKS",
    "REQUIREMENTS",
    "RegRequirement",
    "RequirementStatus",
    "build_conformity_bundle",
    "evaluate",
    "evaluate_requirement",
    "generate_sbom",
    "render_markdown",
    "sbom_json",
    "summarize_audit_log",
]
